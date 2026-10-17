"""Layer 2 "Keyword Extractor" / doc section "Unique Keyword Tagging Strategy".

For every user entry, tag each token with how *discriminative* it is for choosing a symptom cluster:

    CRITICAL  symptom that is on the emergency red-flag list            (aankhon andhera -> vision loss)
    HIGH      symptom that appears in few specialist clusters (rare = it points somewhere)   (chakkar)
    MEDIUM    symptom shared by many clusters (fever, pain ...) - supports but doesn't decide
    CONTEXT   body part
    TEMPORAL  time / onset words                                        (subah, achanak)
    MODIFIER  intensity words                                           (bahut, thoda)
    FILLER    particles and function words                              (aaya, hai)
    UNKNOWN   not in the lexicon (also queued for Step 8 review)

Discriminativeness is an IDF over the specialist corpus (doc: "TF-IDF discriminative token scoring per entry"),
so it updates itself whenever the corpus grows.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .severity import EMERGENCY_CANONICAL
from .specialist_rag import Cluster
from .spelling import CorrectionResult

HIGH_MAX_CLUSTER_SHARE = 0.35        # symptom in <= 35% of clusters counts as discriminative


@dataclass
class KeywordTag:
    token: str                 # what the user typed (normalized)
    canonical: str | None
    level: str                 # CRITICAL | HIGH | MEDIUM | CONTEXT | TEMPORAL | MODIFIER | FILLER | UNKNOWN
    discriminative: bool
    idf: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class KeywordTagger:
    def __init__(self, corpus: list[Cluster], emergency: frozenset[str] | set[str] | None = None):
        self.n = max(1, len(corpus))
        df: dict[str, int] = {}
        for c in corpus:
            for s in set(c.symptoms):
                df[s] = df.get(s, 0) + 1
        self.df = df
        self.emergency = set(emergency) if emergency is not None else set(EMERGENCY_CANONICAL)

    def idf(self, symptom: str) -> float:
        return math.log((1 + self.n) / (1 + self.df.get(symptom, 0))) + 1

    def tag(self, correction: CorrectionResult) -> list[KeywordTag]:
        out: list[KeywordTag] = []
        for t in correction.tokens:
            cat, canon = t.category, t.canonical
            if cat == "symptom" and canon:
                idf = self.idf(canon)
                if canon in self.emergency:
                    out.append(KeywordTag(t.source, canon, "CRITICAL", True, idf))
                elif self.df.get(canon, 0) / self.n <= HIGH_MAX_CLUSTER_SHARE:
                    out.append(KeywordTag(t.source, canon, "HIGH", True, idf))
                else:
                    out.append(KeywordTag(t.source, canon, "MEDIUM", False, idf))
            elif cat == "body_part":
                out.append(KeywordTag(t.source, canon, "CONTEXT", False))
            elif cat == "time":
                out.append(KeywordTag(t.source, canon, "TEMPORAL", True))
            elif cat == "intensity":
                out.append(KeywordTag(t.source, canon, "MODIFIER", False))
            elif cat in ("particle", "function", "number"):
                out.append(KeywordTag(t.source, canon, "FILLER", False))
            else:
                out.append(KeywordTag(t.source, None, "UNKNOWN", False))
        return out
