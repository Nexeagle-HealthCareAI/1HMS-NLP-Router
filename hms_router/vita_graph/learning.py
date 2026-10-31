"""How Vita grows: every turn is traced, conversations get outcomes, staff turn good (or corrected) turns into
examples, and active examples are recalled into Vita's prompts for similar patients.

    traces        vita:trace:{sid} (list, one JSON per turn) + vita:conv (sorted set: sid by last activity) +
                  vita:conv:{sid} (hash: language, turns, outcome flags). Patient text is redacted. 30 days.
    outcomes      booked (a booking finished), emergency, thumbs up / down (POST /v2/agent/feedback)
    examples      vita:examples:draft / vita:examples:active (hashes id -> JSON) + vita:examples:log (history).
                  Staff approve a good turn as it is, or correct its reply and/or its specialist. Drafts go live only
                  through the promotion gate (evals.py): the scenario suite must not score worse with them.
    recall        the k active examples most similar to the patient's message (TF-IDF, character n-grams),
                  shown to the LLM as "how Vita answered similar patients".
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid

log = logging.getLogger("hms_router.vita_learning")

TRACE_TTL_S = 30 * 24 * 3600
OUTCOMES = ("booked", "emergency", "up", "down")
_PHONE = re.compile(r"(?<!\d)(\+?\d[\d\s-]{8,}\d)(?!\d)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_OTP = re.compile(r"(?<!\d)\d{4,6}(?!\d)")


def redact(text: str) -> str:
    text = _EMAIL.sub("[email]", text or "")
    text = _PHONE.sub("[phone]", text)
    return _OTP.sub("[code]", text)


class TraceStore:
    def __init__(self, redis_client):
        self.r = redis_client

    def record(self, sid: str, entry: dict) -> None:
        if not sid:
            return
        try:
            entry = dict(entry, at=int(time.time()))
            entry["patient"] = redact(entry.get("patient", ""))[:500]
            self.r.rpush(f"vita:trace:{sid}", json.dumps(entry, ensure_ascii=False, default=str))
            self.r.expire(f"vita:trace:{sid}", TRACE_TTL_S)
            meta = f"vita:conv:{sid}"
            self.r.hincrby(meta, "turns", 1)
            self.r.hset(meta, mapping={"language": entry.get("language") or "", "lastAt": entry["at"]})
            if entry.get("kind") == "booked":
                self.r.hset(meta, "booked", 1)
            if entry.get("kind") == "emergency":
                self.r.hset(meta, "emergency", 1)
            self.r.expire(meta, TRACE_TTL_S)
            self.r.zadd("vita:conv", {sid: entry["at"]})
        except Exception:
            log.exception("trace not recorded")

    def feedback(self, sid: str, rating: str, comment: str | None = None) -> bool:
        if rating not in ("up", "down") or not self.r.exists(f"vita:conv:{sid}"):
            return False
        self.r.hset(f"vita:conv:{sid}", mapping={"rating": rating, "comment": redact(comment or "")[:300]})
        return True

    def turns(self, sid: str) -> list[dict]:
        return [json.loads(x) for x in self.r.lrange(f"vita:trace:{sid}", 0, -1)]

    def conversations(self, outcome: str | None = None, limit: int = 50) -> list[dict]:
        out = []
        for sid in self.r.zrevrange("vita:conv", 0, 500):
            meta = self.r.hgetall(f"vita:conv:{sid}") or {}
            if not meta:
                continue
            if outcome in ("booked", "emergency") and not meta.get(outcome):
                continue
            if outcome in ("up", "down") and meta.get("rating") != outcome:
                continue
            out.append({"sessionId": sid, **meta})
            if len(out) >= limit:
                break
        return out


from .meaning import concepts, similarity  # noqa: E402


class ExampleStore:
    """Approved examples, in two sets: draft (staff approved or corrected) and active (in Vita's prompts)."""

    def __init__(self, redis_client):
        self.r = redis_client
        self._index = None            # (version, ids, vectorizer, matrix)

    # ---------------------------------------------------------------- staff
    def add_draft(self, *, patient: str, reply: str | None = None, specialist: str | None = None,
                  language: str | None = None, note: str | None = None, source: str = "approved",
                  session_id: str | None = None, turn: int | None = None, by: str | None = None) -> dict:
        ex = {"id": uuid.uuid4().hex[:12], "patient": redact(patient)[:500], "reply": (reply or "")[:700],
              "specialist": specialist, "language": language, "note": (note or "")[:300], "source": source,
              "sessionId": session_id, "turn": turn, "by": by, "createdAt": int(time.time())}
        self.r.hset("vita:examples:draft", ex["id"], json.dumps(ex, ensure_ascii=False))
        return ex

    def drafts(self) -> list[dict]:
        return [json.loads(v) for v in (self.r.hgetall("vita:examples:draft") or {}).values()]

    def active(self) -> list[dict]:
        return [json.loads(v) for v in (self.r.hgetall("vita:examples:active") or {}).values()]

    def remove(self, ex_id: str) -> bool:
        return bool(self.r.hdel("vita:examples:draft", ex_id) or self.r.hdel("vita:examples:active", ex_id))

    def promote(self, report: dict) -> int:
        """Move every draft to active (called by the gate only when the suite is not worse)."""
        drafts = self.r.hgetall("vita:examples:draft") or {}
        if drafts:
            self.r.hset("vita:examples:active", mapping=drafts)
            self.r.delete("vita:examples:draft")
        self.r.incr("vita:examples:version")
        self.r.lpush("vita:examples:log", json.dumps({"at": int(time.time()), "promoted": len(drafts),
                                                      "report": report}, default=str))
        self.r.ltrim("vita:examples:log", 0, 99)
        return len(drafts)

    def version(self) -> int:
        return int(self.r.get("vita:examples:version") or 0)

    # ---------------------------------------------------------------- recall
    def similar(self, text: str, k: int = 3, examples: list[dict] | None = None) -> list[dict]:
        """The k examples most like this message (active set unless `examples` is given)."""
        pool = examples if examples is not None else self.active()
        pool = [e for e in pool if e.get("patient")]
        if not pool or not (text or "").strip():
            return []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            key = (self.version(), tuple(sorted(e["id"] for e in pool)))
            if examples is not None or self._index is None or self._index[0] != key:
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
                mat = vec.fit_transform([e["patient"] for e in pool])
                index = (key, pool, vec, mat)
                if examples is None:
                    self._index = index
            else:
                index = self._index
            _, items, vec, mat = index
            spelling = cosine_similarity(vec.transform([text]), mat)[0]
            # meaning first ("pet me jalan" ~ "acidity hoti hai"), spelling second ("bukhar" ~ "bukhaar")
            q = concepts(text)
            meant = [similarity(q, concepts(e["patient"])) for e in items]
            score = [0.65 * m + 0.35 * float(sp) if q else float(sp) for m, sp in zip(meant, spelling)]
            best = sorted(range(len(items)), key=lambda i: -score[i])[:k]
            return [dict(items[i], similarity=round(score[i], 3), meaning=round(meant[i], 3))
                    for i in best if score[i] >= 0.25]
        except Exception:
            log.exception("example recall failed")
            return []
