"""
tests/location_API/test_repositories.py
---------------------------------------
Unit tests for the data-access layer.  No FastAPI, no services — just the
repositories against tiny in-memory CSV fixtures.
"""

import csv
import pytest

from location_API.location_brain.repositories import (
    CitiesRepository,
    GovPincodeRepository,
    PincodeRepository,
    TownsRepository,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cities_csv(tmp_path):
    p = tmp_path / "cities.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["City", "State", "Lat", "Long"])
        w.writerow(["Mumbai", "Maharashtra", "19.076", "72.877"])
        w.writerow(["Delhi", "Delhi", "28.704", "77.102"])
        w.writerow(["Mumbai", "Goa", "15.299", "74.124"])   # duplicate city, different state
    return str(p)


@pytest.fixture
def pincodes_csv(tmp_path):
    p = tmp_path / "pincodes.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Pincode", "District", "StateName"])
        w.writerow(["400001", "Mumbai", "Maharashtra"])
        w.writerow(["110001", "New Delhi", "Delhi"])
        w.writerow(["400002", "Mumbai", "Maharashtra"])
    return str(p)


# ---------------------------------------------------------------------------
# CitiesRepository
# ---------------------------------------------------------------------------

class TestCitiesRepository:
    def test_by_norm_exact_single(self, cities_csv):
        repo = CitiesRepository(cities_csv)
        records = repo.by_norm("delhi")
        assert len(records) == 1
        assert records[0]["City"] == "Delhi"
        assert records[0]["State"] == "Delhi"

    def test_by_norm_returns_all_duplicate_cities(self, cities_csv):
        """Same city name in two states must both be returned."""
        repo = CitiesRepository(cities_csv)
        records = repo.by_norm("mumbai")
        assert len(records) == 2
        states = {r["State"] for r in records}
        assert states == {"Maharashtra", "Goa"}

    def test_by_norm_case_insensitive_storage(self, cities_csv):
        repo = CitiesRepository(cities_csv)
        # stored as lowercase norm — uppercase lookup should work via caller normalising
        assert repo.by_norm("mumbai") != []

    def test_by_norm_missing_returns_empty(self, cities_csv):
        repo = CitiesRepository(cities_csv)
        assert repo.by_norm("nonexistent") == []

    def test_all_norms_set_contains_all_cities(self, cities_csv):
        repo = CitiesRepository(cities_csv)
        norms = repo.all_norms_set()
        assert "mumbai" in norms
        assert "delhi" in norms

    def test_all_norms_list_length(self, cities_csv):
        """all_norms() includes duplicates (3 rows => 3 entries)."""
        repo = CitiesRepository(cities_csv)
        assert len(repo.all_norms()) == 3

    def test_lat_long_stored(self, cities_csv):
        repo = CitiesRepository(cities_csv)
        record = repo.by_norm("delhi")[0]
        assert record["Lat"] == "28.704"
        assert record["Long"] == "77.102"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Cities dataset not found"):
            CitiesRepository(str(tmp_path / "missing.csv"))


# ---------------------------------------------------------------------------
# PincodeRepository
# ---------------------------------------------------------------------------

class TestPincodeRepository:
    def test_by_district_exact(self, pincodes_csv):
        repo = PincodeRepository(pincodes_csv)
        records = repo.by_district("mumbai")
        assert len(records) == 2          # two Mumbai pincodes
        pincodes = {r["Pincode"] for r in records}
        assert pincodes == {"400001", "400002"}

    def test_by_district_missing_returns_empty(self, pincodes_csv):
        repo = PincodeRepository(pincodes_csv)
        assert repo.by_district("nonexistent") == []

    def test_all_districts_set_contains_all(self, pincodes_csv):
        repo = PincodeRepository(pincodes_csv)
        districts = repo.all_districts_set()
        assert "mumbai" in districts
        assert "new delhi" in districts

    def test_all_districts_list_distinct_keys(self, pincodes_csv):
        """all_districts() returns distinct district keys (2 unique districts)."""
        repo = PincodeRepository(pincodes_csv)
        assert len(repo.all_districts()) == 2

    def test_state_name_stored(self, pincodes_csv):
        repo = PincodeRepository(pincodes_csv)
        record = repo.by_district("new delhi")[0]
        assert record["StateName"] == "Delhi"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Pincode dataset not found"):
            PincodeRepository(str(tmp_path / "missing.csv"))


# ---------------------------------------------------------------------------
# GovPincodeRepository
# ---------------------------------------------------------------------------

@pytest.fixture
def gov_pincodes_csv(tmp_path):
    p = tmp_path / "gov_pincodes.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["circlename", "regionname", "divisionname", "officename", "pincode",
                    "officetype", "delivery", "district", "statename", "latitude", "longitude"])
        w.writerow(["Maharashtra Circle", "Mumbai Region", "Mumbai Division", "Kurla West SO",
                    "400070", "PO", "Delivery", "MUMBAI SUBURBAN", "MAHARASHTRA", "19.068833", "72.877783"])
        w.writerow(["Maharashtra Circle", "Mumbai Region", "Mumbai Division", "Netaji Nagar SO",
                    "400070", "PO", "Delivery", "MUMBAI SUBURBAN", "MAHARASHTRA", "19.068181", "72.877839"])
        w.writerow(["Delhi Circle", "Delhi Region", "New Delhi Division", "Rail Bhawan SO",
                    "110001", "PO", "Delivery", "NEW DELHI", "DELHI", "28.6158611", "77.2113889"])
    return str(p)


class TestGovPincodeRepository:
    def test_by_pincode_returns_all_offices_sharing_it(self, gov_pincodes_csv):
        repo = GovPincodeRepository(gov_pincodes_csv)
        offices = repo.by_pincode("400070")
        assert len(offices) == 2
        assert {o["OfficeName"] for o in offices} == {"Kurla West SO", "Netaji Nagar SO"}

    def test_by_pincode_missing_returns_empty(self, gov_pincodes_csv):
        repo = GovPincodeRepository(gov_pincodes_csv)
        assert repo.by_pincode("999999") == []

    def test_by_district_exact(self, gov_pincodes_csv):
        repo = GovPincodeRepository(gov_pincodes_csv)
        assert len(repo.by_district("mumbai suburban")) == 2

    def test_all_districts_set_contains_all(self, gov_pincodes_csv):
        repo = GovPincodeRepository(gov_pincodes_csv)
        assert repo.all_districts_set() == {"mumbai suburban", "new delhi"}

    def test_nearest_orders_closest_first(self, gov_pincodes_csv):
        repo = GovPincodeRepository(gov_pincodes_csv)
        nearest = repo.nearest(19.0760, 72.8777, limit=3)
        assert len(nearest) == 3
        assert nearest[0]["District"] == "MUMBAI SUBURBAN"
        assert nearest[0]["DistanceKm"] <= nearest[1]["DistanceKm"] <= nearest[2]["DistanceKm"]

    def test_nearest_respects_limit(self, gov_pincodes_csv):
        repo = GovPincodeRepository(gov_pincodes_csv)
        assert len(repo.nearest(19.0760, 72.8777, limit=1)) == 1

    def test_nearest_still_returns_something_far_from_every_known_point(self, gov_pincodes_csv):
        # No fixture point is anywhere near (0, 0) -- nearest() must still
        # return the closest of what it has, not an empty list.
        repo = GovPincodeRepository(gov_pincodes_csv)
        assert len(repo.nearest(0.0, 0.0, limit=1)) == 1

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Government pincode dataset not found"):
            GovPincodeRepository(str(tmp_path / "missing.csv"))


# ---------------------------------------------------------------------------
# TownsRepository
# ---------------------------------------------------------------------------

@pytest.fixture
def towns_csv(tmp_path):
    """Mirrors the real source file's actual quirks (verified against
    Cities_Towns_District_State_India.csv directly): a blank first row, a
    header row whose cells are wrapped in literal single-quote characters
    (not real CSV quoting), a blank row after the header, and a state
    column name with a long run of internal whitespace."""
    p = tmp_path / "towns.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "", "", "", "", "", ""])
        w.writerow(["'Sl. No.'", "'City/Town'", "'Urban Status'", "'State Code'",
                    "'State/                Union territory*'", "'District Code'", "'District'"])
        w.writerow(["", "", "", "", "", "", ""])
        w.writerow(["1", "Ganganagar", "M.Cl.", "8", "Rajasthan", "1", "Ganganagar"])
        w.writerow(["2", "Sri Vijaynagar", "C.T.", "8", "Rajasthan", "1", "Ganganagar"])
    return str(p)


class TestTownsRepository:
    def test_parses_the_quirky_source_header(self, towns_csv):
        repo = TownsRepository(towns_csv)
        records = repo.by_norm("ganganagar")
        assert len(records) == 1
        assert records[0]["Name"] == "Ganganagar"
        assert records[0]["District"] == "Ganganagar"
        assert records[0]["State"] == "Rajasthan"
        assert records[0]["UrbanStatus"] == "M.Cl."

    def test_by_norm_missing_returns_empty(self, towns_csv):
        repo = TownsRepository(towns_csv)
        assert repo.by_norm("nonexistent") == []

    def test_all_norms_set_contains_all_towns(self, towns_csv):
        repo = TownsRepository(towns_csv)
        assert repo.all_norms_set() == {"ganganagar", "sri vijaynagar"}

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Towns dataset not found"):
            TownsRepository(str(tmp_path / "missing.csv"))

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Towns dataset is empty"):
            TownsRepository(str(p))
