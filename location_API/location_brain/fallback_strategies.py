"""
location_brain/fallback_strategies.py
--------------------------------------
Open/Closed-compliant fallback resolution for district-name matching.

Adding a new strategy (e.g. phonetic/Soundex) requires ONLY creating a new
class that satisfies `FallbackStrategy` — no existing code needs editing.
"""

from __future__ import annotations

import difflib
from typing import List, Dict, Protocol, runtime_checkable

from .repositories import PincodeRepository


@runtime_checkable
class FallbackStrategy(Protocol):
    """
    A strategy that attempts to resolve pincode records for a query that
    had no exact match in the PincodeRepository.
    """

    def resolve(self, query_norm: str, repo: PincodeRepository) -> List[Dict]:
        """
        Return matching pincode records, or an empty list if the strategy
        cannot find anything useful.
        """
        ...


class SubstringFallback:
    """
    Matches districts whose normalised name *contains* the query, or that
    the query *contains* (e.g. `"mumbai"` matches `"mumbai suburban"`).
    """

    def resolve(self, query_norm: str, repo: PincodeRepository) -> List[Dict]:
        matches: List[Dict] = []
        for d_norm in repo.all_districts():
            if query_norm in d_norm or d_norm in query_norm:
                matches.extend(repo.by_district(d_norm))
        return matches


class FuzzyFallback:
    """
    Uses difflib to catch slight spelling variations
    (e.g. `"Bangalor"` → `"Bangalore"`).
    """

    def __init__(self, cutoff: float = 0.7, n: int = 3) -> None:
        self._cutoff = cutoff
        self._n = n

    def resolve(self, query_norm: str, repo: PincodeRepository) -> List[Dict]:
        close = difflib.get_close_matches(
            query_norm, repo.all_districts(), n=self._n, cutoff=self._cutoff
        )
        matches: List[Dict] = []
        for c in close:
            matches.extend(repo.by_district(c))
        return matches


class FallbackChain:
    """
    Runs a list of FallbackStrategy instances in order, stopping as soon as
    one returns results.  This preserves the original waterfall behaviour
    while keeping each strategy independently testable and replaceable.
    """

    def __init__(self, strategies: List[FallbackStrategy]) -> None:
        self._strategies = strategies

    def resolve(self, query_norm: str, repo: PincodeRepository) -> List[Dict]:
        for strategy in self._strategies:
            results = strategy.resolve(query_norm, repo)
            if results:
                return results
        return []


def default_fallback_chain() -> FallbackChain:
    """
    Factory that returns the standard two-step fallback chain used by the app.
    Import and call this instead of constructing the chain manually.
    """
    return FallbackChain(
        strategies=[
            SubstringFallback(),
            FuzzyFallback(cutoff=0.7, n=3),
        ]
    )
