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

# City results rank above towns, which rank above bare districts, when tied
# on match quality -- cities are the curated/major-place dataset.
_SEARCH_TYPE_ORDER = {"city": 0, "town": 1, "district": 2}

# Autocomplete results stay light -- a large district can have hundreds of
# pincodes; the caller can always follow up with /locate for the full list.
_SEARCH_PINCODE_CAP = 5


class CitySearchService(ICitySearcher):
    """
    Autocomplete-style search across every available dataset -- curated
    cities (~213, with coordinates), pincode-dataset districts (~19K rows),
    and, when provided, the much broader towns list (~5,193) and the
    official government pincode directory's districts (~165K post offices)
    -- merged into one ranked, deduplicated, typed result list ("city" /
    "town" / "district"), each enriched with state/district and a small
    sample of pincodes/coordinates where available.

    `towns_repo`/`gov_pincode_repo` are optional so existing callers built
    with just the original two repos keep working, just with narrower
    coverage (no "town" results, no pincode/coordinate enrichment from the
    government dataset) -- same optionality pattern as
    `PincodeFinder.locate()`'s gov_pincodes_csv/towns_csv.
    """

    def __init__(
        self,
        cities_repo: CitiesRepository,
        pincode_repo: PincodeRepository,
        towns_repo: Optional[TownsRepository] = None,
        gov_pincode_repo: Optional[GovPincodeRepository] = None,
    ) -> None:
        self._cities = cities_repo
        self._pincodes = pincode_repo
        self._towns = towns_repo
        self._gov_pincodes = gov_pincode_repo

    def search_cities(self, query: str, limit: int = 10) -> List[Dict]:
        query_norm = " ".join(query.strip().split()).lower()
        if not query_norm:
            return []

        pool = self._name_pool()

        # (type, name_norm) -> best score seen (0 exact / 1 prefix / 2
        # substring / 3 fuzzy) -- computed over the whole pool first, since
        # it's cheap; only the winning entries get enriched with
        # pincodes/coordinates below.
        scored: Dict[tuple, int] = {}

        def _consider(name_norm: str, kind: str, score: int) -> None:
            key = (kind, name_norm)
            if key not in scored or score < scored[key]:
                scored[key] = score

        for name_norm, kind in pool:
            if name_norm == query_norm:
                _consider(name_norm, kind, 0)
            elif name_norm.startswith(query_norm):
                _consider(name_norm, kind, 1)
            elif query_norm in name_norm:
                _consider(name_norm, kind, 2)

        pool_names = [name for name, _kind in pool]
        for name_norm in difflib.get_close_matches(query_norm, pool_names, n=limit * 3, cutoff=0.6):
            for kind in {k for n, k in pool if n == name_norm}:
                _consider(name_norm, kind, 3)

        records = self._typed_records(scored)
        records.sort(key=lambda r: (r[0], _SEARCH_TYPE_ORDER.get(r[1], 9), r[2]))

        return [self._enrich(kind, name, state, district) for _, kind, name, state, district in records[:limit]]

    def _name_pool(self) -> List[tuple]:
        """Every searchable (normalised_name, type) pair across all
        available datasets."""
        pool = [(n, "city") for n in self._cities.all_norms_set()]
        pool += [(n, "district") for n in self._pincodes.all_districts_set()]
        if self._towns is not None:
            pool += [(n, "town") for n in self._towns.all_norms_set()]
        if self._gov_pincodes is not None:
            pool += [(n, "district") for n in self._gov_pincodes.all_districts_set()]
        return pool

    def _typed_records(self, scored: Dict[tuple, int]) -> List[tuple]:
        """One display record per winning (type, name_norm) key:
        (score, type, name, state, district). Cheap lookups only --
        pincode/coordinate enrichment happens later, just for the final
        top-`limit` slice."""
        seen: set = set()
        records: List[tuple] = []

        for (kind, name_norm), score in scored.items():
            if kind == "city":
                for rec in self._cities.by_norm(name_norm):
                    key = ("city", rec["City"], rec["State"])
                    if key not in seen:
                        seen.add(key)
                        records.append((score, "city", rec["City"], rec["State"] or None, None))
            elif kind == "town" and self._towns is not None:
                for rec in self._towns.by_norm(name_norm):
                    key = ("town", rec["Name"], rec["District"])
                    if key not in seen:
                        seen.add(key)
                        records.append((score, "town", rec["Name"], rec["State"] or None, rec["District"] or None))
            else:  # district (from pincode_repo and/or gov_pincode_repo -- deduped by title-cased name)
                title = name_norm.title()
                key = ("district", title)
                if key not in seen:
                    seen.add(key)
                    records.append((score, "district", title, self._district_state(name_norm), None))

        return records

    def _district_state(self, district_norm: str) -> Optional[str]:
        if self._gov_pincodes is not None:
            offices = self._gov_pincodes.by_district(district_norm)
            if offices:
                return offices[0]["StateName"].title()
        matches = self._pincodes.by_district(district_norm)
        if matches and matches[0]["StateName"]:
            return matches[0]["StateName"].title()
        return None

    def _enrich(self, kind: str, name: str, state: Optional[str], district: Optional[str]) -> Dict:
        coordinates: Optional[Dict] = None
        if kind == "city":
            city_records = self._cities.by_norm(name.strip().lower())
            if city_records:
                coordinates = _parse_coordinates(city_records[0]["Lat"], city_records[0]["Long"])

        lookup_norm = (district or name).strip().lower()
        pincodes: List[str] = []

        if self._gov_pincodes is not None:
            offices = self._gov_pincodes.by_district(lookup_norm)
            if offices:
                pincodes = sorted({o["Pincode"] for o in offices})[:_SEARCH_PINCODE_CAP]
                if coordinates is None and offices[0]["Latitude"] is not None:
                    coordinates = {"latitude": offices[0]["Latitude"], "longitude": offices[0]["Longitude"]}
        if not pincodes:
            matches = self._pincodes.by_district(lookup_norm)
            if matches:
                pincodes = sorted({m["Pincode"] for m in matches})[:_SEARCH_PINCODE_CAP]

        return {
            "name": name,
            "type": kind,
            "state": state,
            "district": district,
            "pincodes": pincodes,
            "coordinates": coordinates,
        }


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
