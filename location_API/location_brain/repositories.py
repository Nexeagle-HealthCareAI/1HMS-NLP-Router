"""
location_brain/repositories.py
-------------------------------
Single-Responsibility data-access layer.

Each repository owns exactly ONE CSV file: loading, normalising, and
providing read-only access to its data.  No business logic lives here.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


class CitiesRepository:
    """
    Loads and indexes `Indian_Cities_Database.csv`.

    Provides O(1) lookup by normalised city name and the full set of
    known normalised names for fuzzy matching.
    """

    def __init__(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cities dataset not found: {csv_path}")

        self._by_norm: Dict[str, List[Dict]] = defaultdict(list)
        self._norms: List[str] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("City") or "").strip()
                if not name:
                    continue
                norm = name.lower()
                self._by_norm[norm].append(
                    {
                        "City": name,
                        "State": (row.get("State") or "").strip(),
                        "Lat": (row.get("Lat") or "").strip(),
                        "Long": (row.get("Long") or "").strip(),
                    }
                )
                self._norms.append(norm)

    def by_norm(self, name_norm: str) -> List[Dict]:
        """Return all city records whose normalised name matches exactly."""
        return self._by_norm.get(name_norm, [])

    def all_norms(self) -> List[str]:
        """Return every known normalised city name (with duplicates)."""
        return self._norms

    def all_norms_set(self) -> set:
        """Return the de-duplicated set of normalised city names."""
        return set(self._by_norm.keys())


class PincodeRepository:
    """
    Loads and indexes `pincode-dataset.csv`.

    Provides O(1) lookup by normalised district name and the full set of
    known district norms for fuzzy matching.
    """

    def __init__(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Pincode dataset not found: {csv_path}")

        self._by_district: Dict[str, List[Dict]] = defaultdict(list)

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                district = (row.get("District") or "").strip()
                pincode = (row.get("Pincode") or "").strip()
                state = (row.get("StateName") or "").strip()
                if not district or not pincode:
                    continue
                norm = district.lower()
                self._by_district[norm].append(
                    {"Pincode": pincode, "District": district, "StateName": state}
                )

    def by_district(self, district_norm: str) -> List[Dict]:
        """Return all pincode records whose normalised district name matches exactly."""
        return self._by_district.get(district_norm, [])

    def all_districts(self) -> List[str]:
        """Return a list of every known normalised district name."""
        return list(self._by_district.keys())

    def all_districts_set(self) -> set:
        """Return the de-duplicated set of normalised district names."""
        return set(self._by_district.keys())


class GovPincodeRepository:
    """
    Loads and indexes `pincodes_gov.csv` -- the official India Post pincode
    directory (~165K rows, post-office-level granularity, WITH
    coordinates). Far richer than PincodeRepository's `pincode-dataset.csv`
    (district-level only, no coordinates, no office detail); this is what
    SmartLocationService uses for exact-pincode and coordinate-based
    (nearest-office) lookups.
    """

    def __init__(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Government pincode dataset not found: {csv_path}")

        self._by_pincode: Dict[str, List[Dict]] = defaultdict(list)
        self._by_district: Dict[str, List[Dict]] = defaultdict(list)
        geo_records: List[Dict] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pincode = (row.get("pincode") or "").strip()
                district = (row.get("district") or "").strip()
                if not pincode or not district:
                    continue

                try:
                    lat = float(row.get("latitude") or "")
                    lon = float(row.get("longitude") or "")
                except ValueError:
                    lat = lon = None

                record = {
                    "Pincode": pincode,
                    "OfficeName": (row.get("officename") or "").strip(),
                    "OfficeType": (row.get("officetype") or "").strip(),
                    "District": district,
                    "StateName": (row.get("statename") or "").strip(),
                    "Latitude": lat,
                    "Longitude": lon,
                }
                self._by_pincode[pincode].append(record)
                self._by_district[district.lower()].append(record)
                if lat is not None:
                    geo_records.append(record)

        # Precomputed once (not per-request) so nearest() is a single
        # vectorized numpy call regardless of dataset size -- a per-row
        # Python loop over ~165K records here would repeat exactly the
        # kind of O(N)-per-request mistake that made /route-symptom
        # regress to ~50ms/request elsewhere in this project (see
        # nlp_brain/matching.py's normalize_matrix() docstring).
        self._geo_records = geo_records
        self._lat_rad = np.radians(np.array([r["Latitude"] for r in geo_records]))
        self._lon_rad = np.radians(np.array([r["Longitude"] for r in geo_records]))

    def by_pincode(self, pincode: str) -> List[Dict]:
        """Return every post office record for an exact pincode."""
        return self._by_pincode.get(pincode.strip(), [])

    def by_district(self, district_norm: str) -> List[Dict]:
        """Return all post office records whose normalised district matches exactly."""
        return self._by_district.get(district_norm, [])

    def all_districts(self) -> List[str]:
        return list(self._by_district.keys())

    def all_districts_set(self) -> set:
        return set(self._by_district.keys())

    def nearest(self, lat: float, lon: float, limit: int = 5) -> List[Dict]:
        """The `limit` nearest post offices to (lat, lon) by great-circle
        (haversine) distance, closest first. Each result gets a
        `DistanceKm` field. Returns [] if no record in the dataset has
        coordinates (shouldn't happen with the current data, but a future
        swap-in dataset might omit them)."""
        if self._lat_rad.size == 0:
            return []

        lat_r, lon_r = np.radians(lat), np.radians(lon)
        dlat = self._lat_rad - lat_r
        dlon = self._lon_rad - lon_r
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(self._lat_rad) * np.sin(dlon / 2) ** 2
        distance_km = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

        nearest_idx = np.argsort(distance_km)[:limit]
        return [
            {**self._geo_records[i], "DistanceKm": round(float(distance_km[i]), 2)}
            for i in nearest_idx
        ]


class TownsRepository:
    """
    Loads and indexes `Cities_Towns_District_State_India.csv` -- a much
    broader city/town list (~5,193 entries, incl. small towns) than
    CitiesRepository's ~213 major cities, though without coordinates.

    Handles the source file's quirks directly so no other code has to know
    about them: a blank first row, a header row whose cells are wrapped in
    literal single-quote characters (not real CSV quoting -- an artifact of
    however this was exported), and a state column name mangled with a
    long run of internal whitespace.
    """

    def __init__(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Towns dataset not found: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [row for row in csv.reader(f) if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError(f"Towns dataset is empty: {csv_path}")

        # Strip the literal quote characters and collapse the mangled
        # internal whitespace (e.g. "State/         Union territory*" ->
        # "State/ Union territory*") so header lookups below are reliable.
        header = [" ".join(cell.strip().strip("'").split()) for cell in rows[0]]

        self._by_norm: Dict[str, List[Dict]] = defaultdict(list)
        for raw_row in rows[1:]:
            row = dict(zip(header, raw_row))
            name = (row.get("City/Town") or "").strip()
            if not name:
                continue
            state = next((v for k, v in row.items() if k.startswith("State/")), "")
            norm = name.lower()
            self._by_norm[norm].append({
                "Name": name,
                "UrbanStatus": (row.get("Urban Status") or "").strip(),
                "District": (row.get("District") or "").strip(),
                "State": state.strip(),
            })

    def by_norm(self, name_norm: str) -> List[Dict]:
        """Return all town records whose normalised name matches exactly."""
        return self._by_norm.get(name_norm, [])

    def all_norms(self) -> List[str]:
        return list(self._by_norm.keys())

    def all_norms_set(self) -> set:
        return set(self._by_norm.keys())
