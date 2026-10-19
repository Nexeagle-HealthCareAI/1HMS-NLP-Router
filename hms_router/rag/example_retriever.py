"""Sentence-level retrieval: the k labelled example sentences closest to the query (cosine on TF-IDF).

This is the "R" of the RAG pipeline. It returns Evidence - real labelled sentences - which the vote
turns into specialty scores, the reasoner may cite, and the API can show back to a reviewer.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..domain.models import Evidence, Query
from .encoder import TextEncoder


class ExampleRetriever:
    def __init__(self, encoder: TextEncoder, unit_matrix, examples: list[tuple[str, str, str]]):
        """unit_matrix: L2-normalized TF-IDF rows, one per example; examples: (id, text, specialtyId)."""
        if unit_matrix.shape[0] != len(examples):
            raise ValueError("index and examples are out of sync")
        self.encoder, self.matrix, self.examples = encoder, unit_matrix, examples

    def retrieve(self, query: Query, k: int) -> list[Evidence]:
        q = self.encoder.encode_unit(query)
        sims = (self.matrix @ q.T).toarray().ravel()
        if not sims.size:
            return []
        k = min(k, sims.size)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        out = []
        for i in top:
            s = float(sims[i])
            if s <= 0:
                break
            ex_id, text, sid = self.examples[i]
            out.append(Evidence(ex_id, text, sid, round(min(1.0, s), 4)))
        return out


class EvidenceVoteScorer:
    """Specialty scores from retrieved evidence: each sentence votes similarity**power for its specialty."""

    def __init__(self, retriever: ExampleRetriever, k: int = 25, power: float = 2.0):
        self.retriever, self.k, self.power = retriever, k, power

    def score_evidence(self, evidence: list[Evidence]) -> dict[str, float]:
        votes: dict[str, float] = defaultdict(float)
        for e in evidence:
            votes[e.specialtyId] += e.similarity ** self.power
        total = sum(votes.values())
        return {s: v / total for s, v in votes.items()} if total else {}

    def score(self, query: Query) -> dict[str, float]:
        return self.score_evidence(self.retriever.retrieve(query, self.k))
