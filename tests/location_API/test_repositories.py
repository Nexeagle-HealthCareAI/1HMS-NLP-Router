"""
tests/location_API/test_repositories.py
---------------------------------------
Unit tests for the data-access layer.  No FastAPI, no services — just the
repositories against tiny in-memory CSV fixtures.
"""

import csv
import pytest

from location_API.location_brain.repositories import CitiesRepository, PincodeRepository


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
