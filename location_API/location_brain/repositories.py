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
