"""
tests/location_API/test_fallback_strategies.py
-----------------------------------------------
Unit tests for the OCP-compliant fallback chain.
Each strategy is tested in isolation with a mock repository.
"""

import pytest
from unittest.mock import MagicMock

from location_API.location_brain.fallback_strategies import (
    SubstringFallback,
    FuzzyFallback,
    FallbackChain,
    default_fallback_chain,
)


# ---------------------------------------------------------------------------
# Mock repository helper
# ---------------------------------------------------------------------------

def _make_repo(districts: dict):
    """Build a mock PincodeRepository from a {district_norm: [records]} dict."""
    repo = MagicMock()
    repo.by_district.side_effect = lambda k: districts.get(k, [])
    repo.all_districts.return_value = list(districts.keys())
    return repo


# ---------------------------------------------------------------------------
# SubstringFallback
# ---------------------------------------------------------------------------

class TestSubstringFallback:
    def test_matches_when_query_is_substring_of_district(self):
        repo = _make_repo({
            "mumbai suburban": [{"Pincode": "400001", "District": "Mumbai Suburban", "StateName": "Maharashtra"}],
            "delhi": [{"Pincode": "110001", "District": "Delhi", "StateName": "Delhi"}],
        })
        strategy = SubstringFallback()
        results = strategy.resolve("mumbai", repo)
        assert len(results) == 1
        assert results[0]["Pincode"] == "400001"

    def test_matches_when_district_is_substring_of_query(self):
        repo = _make_repo({
            "new delhi": [{"Pincode": "110001", "District": "New Delhi", "StateName": "Delhi"}],
        })
        strategy = SubstringFallback()
        results = strategy.resolve("new delhi central", repo)
        assert len(results) == 1

    def test_returns_empty_when_no_substring_match(self):
        repo = _make_repo({
            "bangalore": [{"Pincode": "560001", "District": "Bangalore", "StateName": "Karnataka"}],
        })
        strategy = SubstringFallback()
        results = strategy.resolve("mumbai", repo)
        assert results == []


# ---------------------------------------------------------------------------
# FuzzyFallback
# ---------------------------------------------------------------------------

class TestFuzzyFallback:
    def test_catches_spelling_variation(self):
        repo = _make_repo({
            "bangalore": [{"Pincode": "560001", "District": "Bangalore", "StateName": "Karnataka"}],
        })
        strategy = FuzzyFallback(cutoff=0.7)
        results = strategy.resolve("bangalor", repo)   # one char dropped
        assert len(results) == 1
        assert results[0]["Pincode"] == "560001"

    def test_returns_empty_when_no_close_match(self):
        repo = _make_repo({
            "bangalore": [{"Pincode": "560001", "District": "Bangalore", "StateName": "Karnataka"}],
        })
        strategy = FuzzyFallback(cutoff=0.9)
        results = strategy.resolve("xyz", repo)
        assert results == []


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_stops_at_first_successful_strategy(self):
        first = MagicMock()
        first.resolve.return_value = [{"Pincode": "111"}]
        second = MagicMock()
        second.resolve.return_value = [{"Pincode": "222"}]

        chain = FallbackChain([first, second])
        results = chain.resolve("any", MagicMock())

        first.resolve.assert_called_once()
        second.resolve.assert_not_called()
        assert results[0]["Pincode"] == "111"

    def test_falls_through_to_second_strategy_when_first_returns_empty(self):
        first = MagicMock()
        first.resolve.return_value = []
        second = MagicMock()
        second.resolve.return_value = [{"Pincode": "222"}]

        chain = FallbackChain([first, second])
        results = chain.resolve("any", MagicMock())

        assert results[0]["Pincode"] == "222"

    def test_returns_empty_when_all_strategies_fail(self):
        first = MagicMock()
        first.resolve.return_value = []
        second = MagicMock()
        second.resolve.return_value = []

        chain = FallbackChain([first, second])
        assert chain.resolve("any", MagicMock()) == []

    def test_default_chain_has_two_strategies(self):
        chain = default_fallback_chain()
        assert len(chain._strategies) == 2
        assert isinstance(chain._strategies[0], SubstringFallback)
        assert isinstance(chain._strategies[1], FuzzyFallback)
