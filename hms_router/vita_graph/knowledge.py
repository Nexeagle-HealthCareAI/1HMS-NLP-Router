"""Vita's knowledge base: facts about Doctor Dekho and each hospital that Vita may state (RAG for facts).

  seed     hms_router/data/knowledge/*.md, one "## title" per fact (confirmed facts only)
  staff    added and removed through /v2/agent/admin/knowledge; "scope" is a hospital's name, or "all"
  search   meaning (vita_graph/meaning.py concepts) + words (TF-IDF over title, tags and text); a hospital's own
           facts first when that hospital is in the conversation

Vita answers factual questions only from what search() returns (the reply prompt and the planner say so); when nothing
matches it says it will check with the team. Never medical advice: these are service facts, and checker.py still
checks every reply.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path

from .meaning import concepts, similarity

log = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
KEY = "vita:knowledge"
MIN_SCORE = 0.22           # below this nothing is "known": Vita says it will check


def load_seed(directory: Path = SEED_DIR) -> list[dict]:
    out: list[dict] = []
    for path in sorted(directory.glob("*.md")) if directory.exists() else []:
        text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
        for block in re.split(r"^## ", text, flags=re.M)[1:]:
            title, _, body = block.partition("\n")
            tags = ""
            m = re.match(r"\s*tags:(.*)\n", body)
            if m:
                tags, body = m.group(1).strip(), body[m.end():]
            body = " ".join(body.split())
            if title.strip() and body:
                out.append({"id": f"seed-{path.stem}-{len(out) + 1}", "title": title.strip(), "tags": tags,
                            "text": body, "scope": "all", "source": "seed"})
    return out


class KnowledgeBase:
    def __init__(self, redis_client=None, seed: list[dict] | None = None):
        self.r = redis_client
        self.seed = load_seed() if seed is None else seed
        self._index = None                       # (signature, entries, vectorizer, matrix)

    # ---------------------------------------------------------------- staff
    def entries(self) -> list[dict]:
        staff = []
        if self.r is not None:
            try:
                staff = [json.loads(v) for v in (self.r.hgetall(KEY) or {}).values()]
            except Exception:
                log.exception("knowledge unreadable")
        return self.seed + sorted(staff, key=lambda e: e.get("updatedAt", 0))

    def add(self, *, title: str, text: str, scope: str = "all", tags: str = "", by: str | None = None) -> dict:
        e = {"id": uuid.uuid4().hex[:12], "title": title.strip()[:200], "text": " ".join(text.split())[:2000],
             "tags": tags.strip()[:300], "scope": (scope or "all").strip()[:120] or "all", "source": "staff",
             "by": by, "updatedAt": int(time.time())}
        self.r.hset(KEY, e["id"], json.dumps(e, ensure_ascii=False))
        self._index = None
        return e

    def remove(self, entry_id: str) -> bool:
        self._index = None
        return bool(self.r.hdel(KEY, entry_id)) if self.r is not None else False

    # ---------------------------------------------------------------- search
    def search(self, question: str, *, hospital: str | None = None, k: int = 3) -> list[dict]:
        """The facts that answer this question, best first (score >= MIN_SCORE). A hospital's own facts are
        searched with the general ones and win a tie; other hospitals' facts are left out."""
        if not (question or "").strip():
            return []
        pool = [e for e in self.entries() if _in_scope(e, hospital)]
        if not pool:
            return []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            sig = tuple(e["id"] for e in pool)
            if self._index is None or self._index[0] != sig:
                docs = [f"{e['title']} {e.get('tags', '')} {e.get('tags', '')} {e['text']}" for e in pool]
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
                self._index = (sig, pool, vec, vec.fit_transform(docs))
            _, items, vec, mat = self._index
            words = cosine_similarity(vec.transform([question]), mat)[0]
        except Exception:
            log.exception("knowledge search failed")
            return []
        q = concepts(question)
        scored = []
        for e, w in zip(items, words):
            meant = similarity(q, concepts(f"{e['title']} {e.get('tags', '')}")) if q else 0.0
            s = 0.6 * float(w) + 0.4 * meant
            if hospital and e.get("scope") not in (None, "", "all"):
                s += 0.05                                     # this hospital's own fact beats a general one
            scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        return [{"title": e["title"], "text": e["text"], "scope": e.get("scope", "all"), "id": e["id"],
                 "score": round(s, 3)} for s, e in scored[:k] if s >= MIN_SCORE]


def _in_scope(entry: dict, hospital: str | None) -> bool:
    scope = (entry.get("scope") or "all").lower()
    if scope == "all":
        return True
    if not hospital:
        return False
    h = hospital.lower()
    return scope in h or h in scope
