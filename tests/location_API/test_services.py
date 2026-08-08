"""
tests/location_API/test_services.py
-------------------------------------
Unit tests for the three service classes, using mock repositories.
No CSV files touched — purely tests business logic.
"""

import csv
import pytest
from unittest.mock import MagicMock

from location_API.location_brain.fallback_strategies import FallbackChain
from location_API.location_brain.services import (
    CitySearchService,
    CoordinateLookupService,
    PincodeLookupService,
)


# ---------------------------------------------------------------------------
# Shared mock-repo builders
# ---------------------------------------------------------------------------

def _cities_repo(data: dict, norms: list = None):
    repo = MagicMock()
    repo.by_norm.side_effect = lambda k: data.get(k, [])
    repo.all_norms.return_value = norms or list(data.keys())
    repo.all_norms_set.return_value = set(data.keys())
    return repo


def _pincode_repo(data: dict):
    repo = MagicMock()
    repo.by_district.side_effect = lambda k: data.get(k, [])
    repo.all_districts.return_value = list(data.keys())
    repo.all_districts_set.return_value = set(data.keys())
    return repo


MUMBAI_CITY = {"City": "Mumbai", "State": "Maharashtra", "Lat": "19.076", "Long": "72.877"}
DELHI_CITY  = {"City": "Delhi",  "State": "Delhi",       "Lat": "28.704", "Long": "77.102"}
MUMBAI_PIN  = {"Pincode": "400001", "District": "Mumbai", "StateName": "Maharashtra"}
DELHI_PIN   = {"Pincode": "110001", "District": "New Delhi", "StateName": "Delhi"}


# ---------------------------------------------------------------------------
# CitySearchService
# ---------------------------------------------------------------------------

class TestCitySearchService:
    def _service(self):
        cities = _cities_repo({"mumbai": [MUMBAI_CITY], "delhi": [DELHI_CITY]})
        pincodes = _pincode_repo({"mumbai": [MUMBAI_PIN], "new delhi": [DELHI_PIN]})
        return CitySearchService(cities, pincodes)

    def test_exact_match_returns_first(self):
        svc = self._service()
        results = svc.search_cities("mumbai")
        assert any(r["city"] == "Mumbai" for r in results)

    def test_empty_query_returns_empty_list(self):
        svc = self._service()
        assert svc.search_cities("") == []

    def test_whitespace_only_returns_empty_list(self):
        svc = self._service()
        assert svc.search_cities("   ") == []

    def test_limit_is_respected(self):
        cities = _cities_repo({"mumbai": [MUMBAI_CITY], "delhi": [DELHI_CITY]})
        pincodes = _pincode_repo({"mumbai": [MUMBAI_PIN], "new delhi": [DELHI_PIN]})
        svc = CitySearchService(cities, pincodes)
        results = svc.search_cities("m", limit=1)
        assert len(results) <= 1

    def test_result_has_city_and_state_keys(self):
        svc = self._service()
        results = svc.search_cities("mumbai")
        assert results
        assert "city" in results[0]
        assert "state" in results[0]


# ---------------------------------------------------------------------------
# PincodeLookupService
# ---------------------------------------------------------------------------

class TestPincodeLookupService:
    def _service(self, fallback=None):
        cities = _cities_repo({"mumbai": [MUMBAI_CITY], "delhi": [DELHI_CITY]})
        pincodes = _pincode_repo({"mumbai": [MUMBAI_PIN]})
        chain = fallback or FallbackChain([])  # no fallback by default
        return PincodeLookupService(cities, pincodes, chain)

    def test_exact_city_match_returns_found(self):
        svc = self._service()
        result = svc.find("mumbai")
        assert result["found"] is True
        assert "400001" in result["pincodes"]

    def test_empty_query_returns_not_found_with_message(self):
        svc = self._service()
        result = svc.find("")
        assert result["found"] is False
        assert result["message"] == "Please enter a city name."

    def test_unknown_city_returns_not_found(self):
        svc = self._service()
        result = svc.find("unknowncity12345")
        assert result["found"] is False

    def test_state_filter_narrows_results(self):
        cities = _cities_repo({
            "mumbai": [
                {"City": "Mumbai", "State": "Maharashtra", "Lat": "19.076", "Long": "72.877"},
                {"City": "Mumbai", "State": "Goa", "Lat": "15.0", "Long": "74.0"},
            ]
        })
        pincodes = _pincode_repo({
            "mumbai": [
                {"Pincode": "400001", "District": "Mumbai", "StateName": "Maharashtra"},
                {"Pincode": "403001", "District": "Mumbai", "StateName": "Goa"},
            ]
        })
        svc = PincodeLookupService(cities, pincodes, FallbackChain([]))
        result = svc.find("mumbai", state="Goa")
        assert result["found"] is True
        assert result["pincodes"] == ["403001"]

    def test_details_contain_pincode_district_state(self):
        svc = self._service()
        result = svc.find("mumbai")
        assert result["details"]
        detail = result["details"][0]
        assert "pincode" in detail
        assert "district" in detail
        assert "state" in detail

    def test_message_empty_on_success(self):
        svc = self._service()
        result = svc.find("mumbai")
        assert result["message"] == ""


# ---------------------------------------------------------------------------
# CoordinateLookupService
# ---------------------------------------------------------------------------

class TestCoordinateLookupService:
    def _service(self):
        cities = _cities_repo(
            {"mumbai": [MUMBAI_CITY], "delhi": [DELHI_CITY]},
            norms=["mumbai", "delhi"],
        )
        pincodes = _pincode_repo({})
        return CoordinateLookupService(cities, pincodes)

    def test_exact_city_returns_coordinates(self):
        svc = self._service()
        result = svc.get_coordinates("mumbai")
        assert result["found"] is True
        assert result["latitude"] == pytest.approx(19.076)
        assert result["longitude"] == pytest.approx(72.877)

    def test_matched_city_and_state_populated(self):
        svc = self._service()
        result = svc.get_coordinates("mumbai")
        assert result["matched_city"] == "Mumbai"
        assert result["state"] == "Maharashtra"

    def test_empty_query_returns_not_found(self):
        svc = self._service()
        result = svc.get_coordinates("")
        assert result["found"] is False
        assert result["message"] == "Please enter a city name."

    def test_unknown_city_returns_not_found(self):
        svc = self._service()
        result = svc.get_coordinates("unknowncity99")
        assert result["found"] is False

    def test_city_with_no_coords_returns_not_found(self):
        cities = _cities_repo({
            "nocoords": [{"City": "NoCoords", "State": "X", "Lat": "", "Long": ""}],
        }, norms=["nocoords"])
        svc = CoordinateLookupService(cities, _pincode_repo({}))
        result = svc.get_coordinates("nocoords")
        assert result["found"] is False
        assert "no coordinate data" in result["message"]
