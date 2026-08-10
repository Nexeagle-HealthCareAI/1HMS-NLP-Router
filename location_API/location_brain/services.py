"""
location_brain/services.py
---------------------------
Business-logic layer — one class, one responsibility.

Each service class:
  - Implements exactly ONE narrow interface from interfaces.py (ISP + LSP).
  - Receives its dependencies via the constructor (DIP).
  - Contains NO CSV I/O or presentation logic (SRP).
  - Is independently unit-testable with mock repositories.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional

from .interfaces import ICitySearcher, ICoordinateFinder, IPincodeFinder, ISmartLocator
from .fallback_strategies import FallbackChain, default_fallback_chain
from .repositories import CitiesRepository, GovPincodeRepository, PincodeRepository, TownsRepository


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _suggest_cities(
    query_norm: str,
    cities_repo: CitiesRepository,
    pincode_repo: PincodeRepository,
    n: int = 5,
) -> List[str]:
    """Fuzzy 'did you mean' suggestions drawn from both datasets."""
    all_names = cities_repo.all_norms_set() | pincode_repo.all_districts_set()
    close = difflib.get_close_matches(query_norm, list(all_names), n=n, cutoff=0.5)

    seen: set = set()
    suggestions: List[str] = []
    for c in close:
        for rec in cities_repo.by_norm(c):
            if rec["City"] not in seen:
                seen.add(rec["City"])
                suggestions.append(rec["City"])
        for rec in pincode_repo.by_district(c):
            title = rec["District"].title()
            if title not in seen:
                seen.add(title)
                suggestions.append(title)
    return suggestions[:n]


# ---------------------------------------------------------------------------
# ICitySearcher
# ---------------------------------------------------------------------------

class CitySearchService(ICitySearcher):
    """
    Searches for matching cities across BOTH datasets and returns a ranked,
    deduplicated list.
    """

    def __init__(
        self,
        cities_repo: CitiesRepository,
        pincode_repo: PincodeRepository,
    ) -> None:
        self._cities = cities_repo
        self._pincodes = pincode_repo

    def search_cities(self, query: str, limit: int = 10) -> List[Dict]:
        query_norm = " ".join(query.strip().split()).lower()
        if not query_norm:
            return []

        all_names = self._cities.all_norms_set() | self._pincodes.all_districts_set()
        scored: List[tuple] = []
        seen: set = set()

        def _add(name_norm: str, score: int) -> None:
            for rec in self._cities.by_norm(name_norm):
                key = (rec["City"], rec["State"])
                if key not in seen:
                    seen.add(key)
                    scored.append((score, rec["City"], rec["State"]))
            for rec in self._pincodes.by_district(name_norm):
                key = (rec["District"].title(), rec["StateName"].title())
                if key not in seen:
                    seen.add(key)
                    scored.append((score, key[0], key[1]))

        # 1. Exact match
        if query_norm in all_names:
            _add(query_norm, 0)

        # 2. Prefix / substring match
        for name_norm in all_names:
            if name_norm == query_norm:
                continue
            if name_norm.startswith(query_norm):
                _add(name_norm, 1)
            elif query_norm in name_norm:
                _add(name_norm, 2)

        # 3. Fuzzy typo match
        close = difflib.get_close_matches(query_norm, list(all_names), n=limit * 2, cutoff=0.6)
        for name_norm in close:
            _add(name_norm, 3)

        scored.sort(key=lambda t: (t[0], t[1]))
        return [{"city": c, "state": s} for _, c, s in scored[:limit]]


# ---------------------------------------------------------------------------
# IPincodeFinder
# ---------------------------------------------------------------------------

class PincodeLookupService(IPincodeFinder):
    """
    Resolves pincodes for a given city / district query using the provided
    fallback chain for partial / fuzzy district matching.
    """

    def __init__(
        self,
        cities_repo: CitiesRepository,
        pincode_repo: PincodeRepository,
        fallback: Optional[FallbackChain] = None,
    ) -> None:
        self._cities = cities_repo
        self._pincodes = pincode_repo
        self._fallback = fallback or default_fallback_chain()

    def _pincodes_for_district(
        self, district_norm: str, state: Optional[str] = None
    ) -> List[Dict]:
        matches = self._pincodes.by_district(district_norm)

        if not matches:
            matches = self._fallback.resolve(district_norm, self._pincodes)

        if state and matches:
            state_norm = state.strip().lower()
            state_matches = [
                m for m in matches if m["StateName"].strip().lower() == state_norm
            ]
            if state_matches:
                return state_matches

        return matches

    def find(self, city_query: str, state: Optional[str] = None) -> Dict:
        raw = city_query
        query_norm = " ".join(city_query.strip().split()).lower()

        result: Dict = {
            "query": raw,
            "matched_city": None,
            "states": [],
            "found": False,
            "pincodes": [],
            "details": [],
            "suggestions": [],
            "message": "",
        }

        if not query_norm:
            result["message"] = "Please enter a city name."
            return result

        # Step 1: Resolve canonical city name and optional state hint
        city_records = self._cities.by_norm(query_norm)
        state_hint = state.strip() if state else None
        canonical_name = raw.strip()

        if city_records:
            if state_hint:
                scoped = [
                    r for r in city_records
                    if r["State"].strip().lower() == state_hint.lower()
                ]
                if scoped:
                    city_records = scoped
            canonical_name = city_records[0]["City"]
            result["matched_city"] = canonical_name
            result["states"] = sorted({r["State"] for r in city_records if r["State"]})
            if not state_hint and len(city_records) == 1:
                state_hint = city_records[0]["State"]

        # Step 2: Look up pincodes via fallback chain
        pin_matches = self._pincodes_for_district(query_norm, state=state_hint)

        if not pin_matches:
            result["suggestions"] = _suggest_cities(query_norm, self._cities, self._pincodes)
            result["message"] = (
                f"No matching pincodes were found in the dataset for '{canonical_name}'."
            )
            return result

        details = [
            {"pincode": m["Pincode"], "district": m["District"], "state": m["StateName"]}
            for m in pin_matches
        ]
        unique_pincodes = sorted({d["pincode"] for d in details})

        result["found"] = True
        result["matched_city"] = (
            canonical_name if result["matched_city"] else pin_matches[0]["District"]
        )
        result["states"] = sorted({m["StateName"] for m in pin_matches if m["StateName"]})
        result["pincodes"] = unique_pincodes
        result["details"] = details
        result["message"] = ""
        return result


# ---------------------------------------------------------------------------
# ICoordinateFinder
# ---------------------------------------------------------------------------

class CoordinateLookupService(ICoordinateFinder):
    """
    Returns latitude / longitude for a city, sourced exclusively from
    `Indian_Cities_Database.csv` (~213 major Indian cities).
    """

    def __init__(self, cities_repo: CitiesRepository, pincode_repo: Optional[PincodeRepository] = None) -> None:
        self._cities = cities_repo
        self._pincodes = pincode_repo  # used only for suggestions

    def get_coordinates(self, city_query: str) -> Dict:
        raw = city_query
        query_norm = " ".join(city_query.strip().split()).lower()

        result: Dict = {
            "query": raw,
            "found": False,
            "matched_city": None,
            "state": None,
            "latitude": None,
            "longitude": None,
            "suggestions": [],
            "message": "",
        }

        if not query_norm:
            result["message"] = "Please enter a city name."
            return result

        city_records = self._cities.by_norm(query_norm)

        # Smart fallback for slight spelling variations
        if not city_records:
            close = difflib.get_close_matches(
                query_norm, self._cities.all_norms(), n=1, cutoff=0.7
            )
            if close:
                city_records = self._cities.by_norm(close[0])

        if not city_records:
            if self._pincodes:
                result["suggestions"] = _suggest_cities(
                    query_norm, self._cities, self._pincodes
                )
            result["message"] = (
                "Coordinates are only available for the ~213 major cities "
                "in Indian_Cities_Database.csv."
            )
            return result

        record = city_records[0]
        try:
            lat = float(record["Lat"]) if record["Lat"] else None
            lon = float(record["Long"]) if record["Long"] else None
        except ValueError:
            lat = lon = None

        if lat is None or lon is None:
            result["message"] = f"'{record['City']}' was found but has no coordinate data."
            return result

        result["found"] = True
        result["matched_city"] = record["City"]
        result["state"] = record["State"] or None
        result["latitude"] = lat
        result["longitude"] = lon
        return result


# ---------------------------------------------------------------------------
# ISmartLocator
# ---------------------------------------------------------------------------

_PINCODE_RE = re.compile(r"^\d{6}$")
_COORDINATES_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")

# A district/office listing can run into the hundreds for a large district
# -- cap the raw supporting records returned so one query can't return a
# multi-hundred-entry payload. `pincodes` (deduped) has no such cap.
_MAX_DETAILS = 20


class SmartLocationService(ISmartLocator):
    """
    Detects what KIND of location query it's given -- a 6-digit pincode, a
    "lat,lon" coordinate pair, or free-text (city/town/district name) --
    and routes to the appropriate repository, always returning the SAME
    response shape regardless of which kind it was. Built on
    GovPincodeRepository (the ~165K-row official India Post directory,
    with coordinates) and TownsRepository (~5,193 towns) for far broader
    coverage than CitiesRepository's ~213 curated major cities alone.
    """

    def __init__(
        self,
        cities_repo: CitiesRepository,
        towns_repo: TownsRepository,
        gov_pincode_repo: GovPincodeRepository,
        fallback: Optional[FallbackChain] = None,
    ) -> None:
        self._cities = cities_repo
        self._towns = towns_repo
        self._gov_pincodes = gov_pincode_repo
        # Reused as-is against GovPincodeRepository: SubstringFallback/
        # FuzzyFallback only ever call .by_district()/.all_districts(),
        # which GovPincodeRepository implements with the same shape as
        # PincodeRepository -- no new fallback code needed (OCP/DIP).
        self._fallback = fallback or default_fallback_chain()

    def locate(self, query: str) -> Dict:
        raw = query
        stripped = query.strip()

        result: Dict = {
            "query": raw,
            "queryType": None,
            "found": False,
            "matched": None,
            "district": None,
            "state": None,
            "pincodes": [],
            "coordinates": None,
            "details": [],
            "suggestions": [],
            "message": "",
        }

        if not stripped:
            result["message"] = "Please enter a pincode, coordinates (\"lat,lon\"), or a city/town name."
            return result

        pincode_match = _PINCODE_RE.match(stripped)
        coord_match = _COORDINATES_RE.match(stripped)

        if pincode_match:
            return self._locate_by_pincode(stripped, result)
        if coord_match:
            return self._locate_by_coordinates(coord_match, result)
        return self._locate_by_text(raw, stripped, result)

    def _locate_by_pincode(self, pincode: str, result: Dict) -> Dict:
        result["queryType"] = "pincode"
        offices = self._gov_pincodes.by_pincode(pincode)
        if not offices:
            result["message"] = f"No location found for pincode '{pincode}'."
            return result

        primary = offices[0]
        result["found"] = True
        result["matched"] = primary["District"].title()
        result["district"] = ", ".join(sorted({o["District"].title() for o in offices}))
        result["state"] = ", ".join(sorted({o["StateName"].title() for o in offices}))
        result["pincodes"] = [pincode]
        if primary["Latitude"] is not None:
            result["coordinates"] = {"latitude": primary["Latitude"], "longitude": primary["Longitude"]}
        result["details"] = [_office_detail(o) for o in offices[:_MAX_DETAILS]]
        return result

    def _locate_by_coordinates(self, match: "re.Match[str]", result: Dict) -> Dict:
        """KNOWN LIMITATION: pincodes_gov.csv itself contains occasional
        bad coordinates (verified -- one Madhya Pradesh record's listed
        lat/long exactly coincides with a real Mumbai location, ~800m
        closer than genuine Mumbai post offices in the same dataset). This
        is a source-data quality issue, not something resolvable by
        querying it differently. Returning the top-5 nearest (not just 1,
        see `details`) is a partial mitigation -- a caller comparing
        multiple candidates is more likely to notice one is an outlier
        than to blindly trust a single `matched` value."""
        result["queryType"] = "coordinates"
        lat, lon = float(match.group(1)), float(match.group(2))
        nearest = self._gov_pincodes.nearest(lat, lon, limit=5)
        if not nearest:
            result["message"] = "No location data available for reverse coordinate lookup."
            return result

        primary = nearest[0]
        result["found"] = True
        result["matched"] = primary["OfficeName"]
        result["district"] = primary["District"].title()
        result["state"] = primary["StateName"].title()
        result["pincodes"] = sorted({o["Pincode"] for o in nearest})
        result["coordinates"] = {"latitude": primary["Latitude"], "longitude": primary["Longitude"]}
        result["details"] = [{**_office_detail(o), "distanceKm": o["DistanceKm"]} for o in nearest]
        return result

    def _locate_by_text(self, raw: str, query: str, result: Dict) -> Dict:
        result["queryType"] = "text"
        query_norm = " ".join(query.split()).lower()

        city_records = self._cities.by_norm(query_norm)
        town_records = self._towns.by_norm(query_norm)

        canonical = None
        district_hint = None
        state_hint = None
        coordinates = None

        if city_records:
            rec = city_records[0]
            canonical = rec["City"]
            state_hint = rec["State"] or None
            coordinates = _parse_coordinates(rec["Lat"], rec["Long"])
        if town_records:
            rec = town_records[0]
            canonical = canonical or rec["Name"]
            district_hint = rec["District"] or None
            state_hint = state_hint or (rec["State"] or None)

        district_query = (district_hint or canonical or query).strip().lower()
        offices = self._gov_pincodes.by_district(district_query)
        if not offices:
            offices = self._fallback.resolve(district_query, self._gov_pincodes)

        if not canonical and not offices:
            result["suggestions"] = self._suggest(query_norm)
            result["message"] = f"No location found for '{raw.strip()}'."
            return result

        result["found"] = True
        result["matched"] = canonical or offices[0]["District"].title()
        result["district"] = district_hint or (offices[0]["District"].title() if offices else None)
        result["state"] = state_hint or (offices[0]["StateName"].title() if offices else None)
        result["coordinates"] = coordinates
        if offices:
            result["pincodes"] = sorted({o["Pincode"] for o in offices})
            result["details"] = [_office_detail(o) for o in offices[:_MAX_DETAILS]]
        return result

    def _suggest(self, query_norm: str, n: int = 5) -> List[str]:
        all_names = (
            self._towns.all_norms_set()
            | self._cities.all_norms_set()
            | self._gov_pincodes.all_districts_set()
        )
        close = difflib.get_close_matches(query_norm, list(all_names), n=n, cutoff=0.5)
        return [c.title() for c in close]


def _office_detail(office: Dict) -> Dict:
    return {
        "officeName": office["OfficeName"],
        "officeType": office["OfficeType"],
        "district": office["District"].title(),
        "state": office["StateName"].title(),
        "pincode": office["Pincode"],
        "latitude": office["Latitude"],
        "longitude": office["Longitude"],
    }


def _parse_coordinates(lat_str: str, lon_str: str) -> Optional[Dict]:
    try:
        if lat_str and lon_str:
            return {"latitude": float(lat_str), "longitude": float(lon_str)}
    except ValueError:
        pass
    return None
