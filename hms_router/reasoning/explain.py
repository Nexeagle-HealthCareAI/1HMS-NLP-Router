"""Deterministic one-line reasons, built only from words the patient actually used.

A reason may never mention a symptom the patient did not say, so it quotes the patient's own words that
also occur in the retrieved evidence - not the evidence sentences themselves. English or Hinglish,
following the patient's language.
"""
from __future__ import annotations

from ..domain.models import Candidate, Query
from ..textnorm import PERSIST_TOKEN, STOPWORDS, preprocess


class Explainer:
    def __init__(self, max_words: int = 4):
        self.max_words = max_words

    def reason(self, query: Query, candidate: Candidate | None) -> str | None:
        if candidate is None:
            return None
        hinglish = query.language in ("hinglish", "hindi")
        evidence_words: set[str] = set()
        for e in candidate.evidence:
            evidence_words.update(preprocess(e.text).split())
        shared = [w for w in dict.fromkeys(query.normalized.split())
                  if len(w) >= 3 and w not in STOPWORDS and w != PERSIST_TOKEN and w in evidence_words]
        name = candidate.name
        if not shared:
            return (f"Aapki baat {name} ke cases se sabse zyada milti hai." if hinglish
                    else f"Closest to {name} cases we have seen.")
        words = shared[: self.max_words]
        quoted = ", ".join(f"'{w}'" for w in words)
        if hinglish:
            return f"Aapke shabd {quoted} {name} ke milte-julte cases se match karte hain."
        verb = "matches" if len(words) == 1 else "match"
        return f"Suggested {name} because {quoted} {verb} similar {name} cases."
