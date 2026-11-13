"""The two RAG ports. Any implementation (TF-IDF today; ChromaDB, a multilingual embedder or a
cross-encoder tomorrow) plugs in without touching RouteSymptomService.

Contract every implementation must keep (checked by tests/test_rag_contract.py):
  Retriever.retrieve -> at most k Evidence, similarity in [0, 1], best first, specialtyId in the taxonomy
  Scorer.score       -> {specialtyId: score}, scores >= 0, summing to 1 (or {} when it has no opinion)
"""
from __future__ import annotations

from typing import Protocol

from ..domain.models import Evidence, Query


class Retriever(Protocol):
    def retrieve(self, query: Query, k: int) -> list[Evidence]: ...


class Scorer(Protocol):
    def score(self, query: Query) -> dict[str, float]: ...
