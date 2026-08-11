import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from location_brain.Pincode_final import PincodeFinder, _script_dir

CITIES_CSV = os.environ.get("CITIES_CSV", "Indian_Cities_Database.csv")
PINCODES_CSV = os.environ.get("PINCODES_CSV", "pincode-dataset.csv")
# Power GET/POST /locate (ISmartLocator) -- the ~165K-row official India
# Post directory (with coordinates) and the ~5,193-town list. Both were
# committed to the repo from the start but unused by any code path until
# now (verified by grep before wiring them in).
GOV_PINCODES_CSV = os.environ.get("GOV_PINCODES_CSV", "pincodes_gov.csv")
TOWNS_CSV = os.environ.get("TOWNS_CSV", "Cities_Towns_District_State_India.csv")

BASE_DIR = Path(_script_dir())
CITIES_CSV_PATH = BASE_DIR / CITIES_CSV
PINCODES_CSV_PATH = BASE_DIR / PINCODES_CSV
GOV_PINCODES_CSV_PATH = BASE_DIR / GOV_PINCODES_CSV
TOWNS_CSV_PATH = BASE_DIR / TOWNS_CSV

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    finder = PincodeFinder(
        str(CITIES_CSV_PATH), str(PINCODES_CSV_PATH),
        gov_pincodes_csv=str(GOV_PINCODES_CSV_PATH), towns_csv=str(TOWNS_CSV_PATH),
    )
    _state.update(finder=finder)
    yield
    _state.clear()


app = FastAPI(title="City Pincode Lookup API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class PincodeRequest(BaseModel):
    city: str
    state: Optional[str] = None


class PincodeDetail(BaseModel):
    pincode: str
    district: str
    state: str


class PincodeResponse(BaseModel):
    query: str
    matchedCity: Optional[str]
    states: List[str]
    found: bool
    pincodes: List[str]
    details: List[PincodeDetail]
    suggestions: List[str]
    message: str


class Coordinates(BaseModel):
    latitude: float
    longitude: float


class SearchMatch(BaseModel):
    name: str
    # "city" (curated ~213, has coordinates) | "town" (broader ~5,193 list,
    # district/state only) | "district" (from either pincode dataset)
    type: str
    state: Optional[str] = None
    district: Optional[str] = None
    pincodes: List[str] = []
    coordinates: Optional[Coordinates] = None


class SearchResponse(BaseModel):
    query: str
    matches: List[SearchMatch]


class CoordinatesResponse(BaseModel):
    query: str
    found: bool
    matchedCity: Optional[str]
    state: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    suggestions: List[str]
    message: str


class LocateDetail(BaseModel):
    officeName: str
    officeType: str
    district: str
    state: str
    pincode: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distanceKm: Optional[float] = None


class LocateRequest(BaseModel):
    query: str


class LocateResponse(BaseModel):
    query: str
    # "pincode" | "coordinates" | "text" -- which kind of query this was
    # detected as; see location_brain.services.SmartLocationService.locate().
    queryType: Optional[str]
    found: bool
    matched: Optional[str]
    district: Optional[str]
    state: Optional[str]
    pincodes: List[str]
    coordinates: Optional[Coordinates]
    details: List[LocateDetail]
    suggestions: List[str]
    message: str


@app.get("/health")
def health():
    return {"status": "ok", "ready": bool(_state)}


def _lookup(city: str, state: Optional[str] = None) -> PincodeResponse:
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="City name must not be empty.")

    finder: PincodeFinder = _state["finder"]
    result: Dict = finder.find(city, state=state)

    return PincodeResponse(
        query=result["query"],
        matchedCity=result["matched_city"],
        states=result["states"],
        found=result["found"],
        pincodes=result["pincodes"],
        details=[
            PincodeDetail(pincode=d["pincode"], district=d["district"], state=d["state"])
            for d in result["details"]
        ],
        suggestions=result["suggestions"],
        message=result["message"],
    )


def _search(city: str, limit: int) -> SearchResponse:
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="City name must not be empty.")

    finder: PincodeFinder = _state["finder"]
    matches = finder.search_cities(city, limit=limit)

    return SearchResponse(
        query=city,
        matches=[
            SearchMatch(
                name=m["name"],
                type=m["type"],
                state=m["state"],
                district=m["district"],
                pincodes=m["pincodes"],
                coordinates=Coordinates(**m["coordinates"]) if m["coordinates"] else None,
            )
            for m in matches
        ],
    )


def _lookup_coordinates(city: str) -> CoordinatesResponse:
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="City name must not be empty.")

    finder: PincodeFinder = _state["finder"]
    result: Dict = finder.get_coordinates(city)

    return CoordinatesResponse(
        query=result["query"],
        found=result["found"],
        matchedCity=result["matched_city"],
        state=result["state"],
        latitude=result["latitude"],
        longitude=result["longitude"],
        suggestions=result["suggestions"],
        message=result["message"],
    )


@app.get("/search", response_model=SearchResponse)
@limiter.limit("60/minute")
def search_cities(
    request: Request,
    city: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    """Autocomplete-style "did you mean" search as the user types. Searches
    across ALL datasets -- the curated ~213 cities, ~19K pincode-dataset
    districts, ~5,193 towns, and the ~165K-row government pincode
    directory's districts (the same broad coverage /locate draws on) --
    merged into one ranked, typed ("city"/"town"/"district") result list,
    each with state/district and a small capped sample of pincodes/
    coordinates when known. For a single best-effort answer to one query
    (rather than a list of candidates), use /locate instead."""
    return _search(city, limit)


@app.post("/find-pincode", response_model=PincodeResponse)
@limiter.limit("60/minute")
def find_pincode(request: Request, req: PincodeRequest):
    return _lookup(req.city, state=req.state)


@app.get("/find-pincode", response_model=PincodeResponse)
@limiter.limit("60/minute")
def find_pincode_get(
    request: Request,
    city: str = Query(..., min_length=1),
    state: Optional[str] = Query(None),
):
    return _lookup(city, state=state)


@app.post("/coordinates", response_model=CoordinatesResponse)
@limiter.limit("60/minute")
def get_coordinates(request: Request, req: PincodeRequest):
    return _lookup_coordinates(req.city)


@app.get("/coordinates", response_model=CoordinatesResponse)
@limiter.limit("60/minute")
def get_coordinates_get(request: Request, city: str = Query(..., min_length=1)):
    return _lookup_coordinates(city)


def _locate(query: str) -> LocateResponse:
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty -- pass a pincode, \"lat,lon\" coordinates, or a city/town name.",
        )

    finder: PincodeFinder = _state["finder"]
    result: Dict = finder.locate(query)

    return LocateResponse(
        query=result["query"],
        queryType=result["queryType"],
        found=result["found"],
        matched=result["matched"],
        district=result["district"],
        state=result["state"],
        pincodes=result["pincodes"],
        coordinates=Coordinates(**result["coordinates"]) if result["coordinates"] else None,
        details=[LocateDetail(**d) for d in result["details"]],
        suggestions=result["suggestions"],
        message=result["message"],
    )


@app.get("/locate", response_model=LocateResponse)
@limiter.limit("60/minute")
def locate_get(request: Request, q: str = Query(..., min_length=1)):
    """Smart unified search: pass a 6-digit pincode, "lat,lon" coordinates,
    or a free-text city/town/district name -- the query type is detected
    automatically and a SINGLE best-effort result is returned (with a few
    supporting candidates in `details`). Broader coverage than
    /find-pincode and /coordinates (which only cover the curated ~213
    cities), and the only endpoint that supports reverse coordinate lookup.
    For a ranked LIST of candidates as the user types, use /search instead
    (same underlying data coverage as this endpoint)."""
    return _locate(q)


@app.post("/locate", response_model=LocateResponse)
@limiter.limit("60/minute")
def locate_post(request: Request, req: LocateRequest):
    return _locate(req.query)
