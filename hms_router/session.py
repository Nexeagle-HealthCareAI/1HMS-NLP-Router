"""Step 1 - session manager on Redis (HSET session:{id}, 30-min sliding TTL).

Keys (all expire together, TTL refreshed on every message):
  session:{id}          HASH  created_at, last_active
  session:{id}:turns    INT   turn counter
  session:{id}:buffer   LIST  JSON message records (raw + normalized + language)
  session:{id}:context  HASH  language, script (Step 4 adds intensity)
The doc lists `session:{id}:ttl`; a separate key isn't needed - EXPIRE on each key does that job.
"""
from __future__ import annotations

import json
import secrets
import time


class SessionManager:
    def __init__(self, redis_client, ttl_seconds: int = 1800):
        self.r = redis_client           # create with decode_responses=True
        self.ttl = ttl_seconds

    @staticmethod
    def _k(sid: str, suffix: str = "") -> str:
        return f"session:{sid}{suffix}"

    def open(self, session_id: str | None) -> tuple[str, bool]:
        """Continue an active session, or start a new one (also when the old one expired)."""
        if session_id and self.r.exists(self._k(session_id)):
            return session_id, False
        sid = "sess_" + secrets.token_hex(6)
        now = time.time()
        self.r.hset(self._k(sid), mapping={"created_at": now, "last_active": now})
        self.r.expire(self._k(sid), self.ttl)
        return sid, True

    def append(self, sid: str, record: dict) -> int:
        """Append a message to the conversation buffer; returns the new turn number."""
        pipe = self.r.pipeline(transaction=True)
        pipe.incr(self._k(sid, ":turns"))
        pipe.rpush(self._k(sid, ":buffer"), json.dumps(record, ensure_ascii=False))
        pipe.hset(self._k(sid), "last_active", time.time())
        for suffix in ("", ":turns", ":buffer", ":context"):
            pipe.expire(self._k(sid, suffix), self.ttl)     # sliding window
        return int(pipe.execute()[0])

    def set_context(self, sid: str, **fields: str) -> None:
        self.r.hset(self._k(sid, ":context"), mapping=fields)
        self.r.expire(self._k(sid, ":context"), self.ttl)

    def history(self, sid: str) -> list[dict]:
        return [json.loads(x) for x in self.r.lrange(self._k(sid, ":buffer"), 0, -1)]
