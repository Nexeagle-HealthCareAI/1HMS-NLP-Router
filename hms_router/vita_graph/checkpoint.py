"""LangGraph checkpointer on plain Redis (redis:7, no RedisJSON/Search modules, so langgraph-checkpoint-redis does not
apply). Each Vita conversation is one LangGraph thread (thread_id = the Vita session id); every graph step saves a
checkpoint, so a conversation's state survives restarts and every turn can be inspected or replayed.

Layout per thread and namespace (TTL refreshed on every write, default 7 days):
    vita:ckpt:{thread}:{ns}:ids          sorted set of checkpoint ids (score = save order), newest last
    vita:ckpt:{thread}:{ns}:c:{id}       the checkpoint + metadata + parent id (serialized, base64)
    vita:ckpt:{thread}:{ns}:w:{id}       hash of pending writes for that checkpoint
Only the newest KEEP checkpoints per thread are kept. Values are serialized by LangGraph's own serde, base64-encoded
because the app's Redis client decodes responses as text.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Iterator, Sequence

from langgraph.checkpoint.base import (WRITES_IDX_MAP, BaseCheckpointSaver, ChannelVersions, Checkpoint,
                                       CheckpointMetadata, CheckpointTuple, get_checkpoint_id,
                                       get_checkpoint_metadata)

KEEP = 40
TTL_S = 7 * 24 * 3600


def _enc(typed: tuple[str, bytes]) -> list[str]:
    return [typed[0], base64.b64encode(typed[1]).decode("ascii")]


def _dec(pair: list[str]) -> tuple[str, bytes]:
    return pair[0], base64.b64decode(pair[1])


class RedisCheckpointSaver(BaseCheckpointSaver):
    def __init__(self, redis_client, *, keep: int = KEEP, ttl_s: int = TTL_S, serde=None):
        super().__init__(serde=serde)
        self.r, self.keep, self.ttl = redis_client, keep, ttl_s

    # ---------------------------------------------------------------- keys
    @staticmethod
    def _base(thread: str, ns: str) -> str:
        return f"vita:ckpt:{thread}:{ns}"

    def _touch(self, *keys: str) -> None:
        for k in keys:
            self.r.expire(k, self.ttl)

    # ---------------------------------------------------------------- write
    def put(self, config, checkpoint: Checkpoint, metadata: CheckpointMetadata,
            new_versions: ChannelVersions) -> dict:
        thread = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        base = self._base(thread, ns)
        record = {"checkpoint": _enc(self.serde.dumps_typed(checkpoint)),
                  "metadata": _enc(self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))),
                  "parent": config["configurable"].get("checkpoint_id")}
        ids = f"{base}:ids"
        order = self.r.incr(f"{base}:seq")
        self.r.set(f"{base}:c:{checkpoint['id']}", json.dumps(record), ex=self.ttl)
        self.r.zadd(ids, {checkpoint["id"]: order})
        self._touch(ids, f"{base}:seq")
        extra = self.r.zcard(ids) - self.keep
        if extra > 0:                                              # forget the oldest
            for old in self.r.zrange(ids, 0, extra - 1):
                self.r.delete(f"{base}:c:{old}", f"{base}:w:{old}")
            self.r.zremrangebyrank(ids, 0, extra - 1)
        return {"configurable": {"thread_id": thread, "checkpoint_ns": ns, "checkpoint_id": checkpoint["id"]}}

    def put_writes(self, config, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        thread = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = config["configurable"]["checkpoint_id"]
        key = f"{self._base(thread, ns)}:w:{cid}"
        for idx, (channel, value) in enumerate(writes):
            inner = f"{task_id}|{WRITES_IDX_MAP.get(channel, idx)}"
            if WRITES_IDX_MAP.get(channel, idx) >= 0 and self.r.hexists(key, inner):
                continue
            self.r.hset(key, inner, json.dumps({"task": task_id, "channel": channel, "path": task_path,
                                                "value": _enc(self.serde.dumps_typed(value))}))
        self._touch(key)

    # ---------------------------------------------------------------- read
    def _tuple(self, thread: str, ns: str, cid: str) -> CheckpointTuple | None:
        base = self._base(thread, ns)
        raw = self.r.get(f"{base}:c:{cid}")
        if not raw:
            return None
        rec = json.loads(raw)
        writes = []
        for inner, w in sorted((self.r.hgetall(f"{base}:w:{cid}") or {}).items()):
            w = json.loads(w)
            writes.append((w["task"], w["channel"], self.serde.loads_typed(_dec(w["value"]))))
        parent = rec.get("parent")
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread, "checkpoint_ns": ns, "checkpoint_id": cid}},
            checkpoint=self.serde.loads_typed(_dec(rec["checkpoint"])),
            metadata=self.serde.loads_typed(_dec(rec["metadata"])),
            pending_writes=writes,
            parent_config={"configurable": {"thread_id": thread, "checkpoint_ns": ns, "checkpoint_id": parent}}
            if parent else None)

    def get_tuple(self, config) -> CheckpointTuple | None:
        thread = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = get_checkpoint_id(config)
        if not cid:
            newest = self.r.zrange(f"{self._base(thread, ns)}:ids", -1, -1)
            if not newest:
                return None
            cid = newest[0]
        return self._tuple(thread, ns, cid)

    def list(self, config, *, filter: dict | None = None, before=None, limit: int | None = None
             ) -> Iterator[CheckpointTuple]:
        if not config:
            return
        thread = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        before_id = get_checkpoint_id(before) if before else None
        n = 0
        for cid in reversed(self.r.zrange(f"{self._base(thread, ns)}:ids", 0, -1)):
            if before_id and cid >= before_id:
                continue
            t = self._tuple(thread, ns, cid)
            if t is None or (filter and not all(t.metadata.get(k) == v for k, v in filter.items())):
                continue
            yield t
            n += 1
            if limit and n >= limit:
                return

    def delete_thread(self, thread_id: str) -> None:
        for key in self.r.scan_iter(f"vita:ckpt:{thread_id}:*"):
            self.r.delete(key)
