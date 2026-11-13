"""Reasoning ports. A Reasoner proposes; the Verifier decides whether the proposal may be used."""
from __future__ import annotations

from typing import Protocol

from ..domain.models import Candidate, Query, ReasonerResult


class Reasoner(Protocol):
    def decide(self, query: Query, candidates: list[Candidate]) -> ReasonerResult | None:
        """Pick from `candidates` only, citing their evidence ids. None = no opinion (keep the ranking)."""
        ...


class NoReasoner:
    """The default: no LLM. The deterministic ranking stands."""

    def decide(self, query: Query, candidates: list[Candidate]) -> ReasonerResult | None:
        return None
