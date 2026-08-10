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
    SmartLocationService,
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


# ---------------------------------------------------------------------------
# SmartLocationService
# ---------------------------------------------------------------------------

def _towns_repo(data: dict):
    repo = MagicMock()
    repo.by_norm.side_effect = lambda k: data.get(k, [])
    repo.all_norms_set.return_value = set(data.keys())
    return repo


def _gov_pincode_repo(by_pincode: dict = None, by_district: dict = None, nearest_result: list = None):
    repo = MagicMock()
    repo.by_pincode.side_effect = lambda k: (by_pincode or {}).get(k.strip(), [])
    repo.by_district.side_effect = lambda k: (by_district or {}).get(k, [])
    repo.all_districts.return_value = list((by_district or {}).keys())
    repo.all_districts_set.return_value = set((by_district or {}).keys())
    repo.nearest.return_value = nearest_result or []
    return repo


MUMBAI_OFFICE = {
    "Pincode": "400070", "OfficeName": "Kurla West SO", "OfficeType": "PO",
    "District": "MUMBAI SUBURBAN", "StateName": "MAHARASHTRA",
    "Latitude": 19.068833, "Longitude": 72.877783,
}
GANGANAGAR_TOWN = {"Name": "Ganganagar", "UrbanStatus": "M.Cl.", "District": "Ganganagar", "State": "Rajasthan"}
GANGANAGAR_OFFICE = {
    "Pincode": "335001", "OfficeName": "Ganganagar HO", "OfficeType": "HO",
    "District": "Ganganagar", "StateName": "Rajasthan", "Latitude": None, "Longitude": None,
}


class TestSmartLocationService:
    def test_detects_pincode_query_type(self):
        svc = SmartLocationService(
            _cities_repo({}), _towns_repo({}), _gov_pincode_repo(by_pincode={"400070": [MUMBAI_OFFICE]}),
        )
        result = svc.locate("400070")
        assert result["queryType"] == "pincode"
        assert result["found"] is True
        assert result["district"] == "Mumbai Suburban"
        assert result["pincodes"] == ["400070"]

    def test_unknown_pincode_returns_not_found(self):
        svc = SmartLocationService(_cities_repo({}), _towns_repo({}), _gov_pincode_repo())
        result = svc.locate("999999")
        assert result["queryType"] == "pincode"
        assert result["found"] is False

    def test_detects_coordinates_query_type(self):
        gov = _gov_pincode_repo(nearest_result=[{**MUMBAI_OFFICE, "DistanceKm": 0.1}])
        svc = SmartLocationService(_cities_repo({}), _towns_repo({}), gov)

        result = svc.locate("19.0760, 72.8777")

        assert result["queryType"] == "coordinates"
        assert result["found"] is True
        assert result["matched"] == "Kurla West SO"
        gov.nearest.assert_called_once_with(19.0760, 72.8777, limit=5)

    def test_coordinates_without_space_after_comma_still_detected(self):
        gov = _gov_pincode_repo(nearest_result=[{**MUMBAI_OFFICE, "DistanceKm": 0.1}])
        svc = SmartLocationService(_cities_repo({}), _towns_repo({}), gov)
        assert svc.locate("19.0760,72.8777")["queryType"] == "coordinates"

    def test_no_coordinate_data_returns_not_found(self):
        svc = SmartLocationService(_cities_repo({}), _towns_repo({}), _gov_pincode_repo(nearest_result=[]))
        result = svc.locate("19.0760,72.8777")
        assert result["found"] is False

    def test_detects_text_query_via_city_and_includes_coordinates(self):
        cities = _cities_repo({"mumbai": [MUMBAI_CITY]}, norms=["mumbai"])
        gov = _gov_pincode_repo(by_district={"mumbai": [MUMBAI_OFFICE]})
        svc = SmartLocationService(cities, _towns_repo({}), gov)

        result = svc.locate("Mumbai")

        assert result["queryType"] == "text"
        assert result["found"] is True
        assert result["matched"] == "Mumbai"
        assert result["coordinates"] == {"latitude": 19.076, "longitude": 72.877}

    def test_detects_text_query_via_town_not_present_in_cities(self):
        # This is the whole point of TownsRepository: broader coverage than
        # CitiesRepository's curated ~213 major cities.
        towns = _towns_repo({"ganganagar": [GANGANAGAR_TOWN]})
        gov = _gov_pincode_repo(by_district={"ganganagar": [GANGANAGAR_OFFICE]})
        svc = SmartLocationService(_cities_repo({}), towns, gov)

        result = svc.locate("Ganganagar")

        assert result["queryType"] == "text"
        assert result["found"] is True
        assert result["matched"] == "Ganganagar"
        assert result["district"] == "Ganganagar"
        assert result["pincodes"] == ["335001"]
        assert result["coordinates"] is None  # not in CitiesRepository -- no coords available

    def test_unknown_text_query_returns_not_found_with_message(self):
        svc = SmartLocationService(_cities_repo({}), _towns_repo({}), _gov_pincode_repo())
        result = svc.locate("zzznotarealplace")
        assert result["found"] is False
        assert "No location found" in result["message"]

    def test_empty_query_returns_not_found_with_message(self):
        svc = SmartLocationService(_cities_repo({}), _towns_repo({}), _gov_pincode_repo())
        result = svc.locate("   ")
        assert result["found"] is False
        assert result["queryType"] is None
        assert "Please enter" in result["message"]
