import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from location_API.location_brain.Pincode_final import _script_dir
from location_API.location_brain.finder import PincodeFinder
from location_API.location_brain.interfaces import ICitySearcher, ICoordinateFinder, IPincodeFinder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CITIES_CSV = os.environ.get("CITIES_CSV", "Indian_Cities_Database.csv")
PINCODES_CSV = os.environ.get("PINCODES_CSV", "pincode-dataset.csv")

BASE_DIR = Path(_script_dir())
CITIES_CSV_PATH = BASE_DIR / CITIES_CSV
PINCODES_CSV_PATH = BASE_DIR / PINCODES_CSV


# ---------------------------------------------------------------------------
# App lifespan — store the finder on app.state (not a mutable global dict)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    finder = PincodeFinder(str(CITIES_CSV_PATH), str(PINCODES_CSV_PATH))
    _app.state.finder = finder
    yield
    del _app.state.finder


app = FastAPI(title="City Pincode Lookup API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Dependency providers (DIP — routes depend on interfaces, not concretions)
# ---------------------------------------------------------------------------

def get_city_searcher(request: Request) -> ICitySearcher:
    """Provide the ICitySearcher for the /search endpoint."""
    finder: ICitySearcher | None = getattr(request.app.state, "finder", None)
    if finder is None:
        raise HTTPException(status_code=503, detail="Service not ready — finder not initialised.")
    return finder


def get_pincode_finder(request: Request) -> IPincodeFinder:
    """Provide the IPincodeFinder for the /find-pincode endpoint."""
    finder: IPincodeFinder | None = getattr(request.app.state, "finder", None)
    if finder is None:
        raise HTTPException(status_code=503, detail="Service not ready — finder not initialised.")
    return finder


def get_coordinate_finder(request: Request) -> ICoordinateFinder:
    """Provide the ICoordinateFinder for the /coordinates endpoint."""
    finder: ICoordinateFinder | None = getattr(request.app.state, "finder", None)
    if finder is None:
        raise HTTPException(status_code=503, detail="Service not ready — finder not initialised.")
    return finder


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

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


class CityMatch(BaseModel):
    city: str
    state: Optional[str]


class SearchResponse(BaseModel):
    query: str
    matches: List[CityMatch]


class CoordinatesResponse(BaseModel):
    query: str
    found: bool
    matchedCity: Optional[str]
    state: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    suggestions: List[str]
    message: str


# ---------------------------------------------------------------------------
# Internal helpers (thin — no business logic, just shape-mapping)
# ---------------------------------------------------------------------------

def _build_pincode_response(city: str, state: Optional[str], finder: IPincodeFinder) -> PincodeResponse:
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="City name must not be empty.")
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


def _build_search_response(city: str, limit: int, searcher: ICitySearcher) -> SearchResponse:
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="City name must not be empty.")
    matches = searcher.search_cities(city, limit=limit)
    return SearchResponse(
        query=city,
        matches=[CityMatch(city=m["city"], state=m["state"]) for m in matches],
    )


def _build_coordinates_response(city: str, finder: ICoordinateFinder) -> CoordinatesResponse:
    if not city or not city.strip():
        raise HTTPException(status_code=400, detail="City name must not be empty.")
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health(request: Request):
    ready = hasattr(request.app.state, "finder")
    return {"status": "ok", "ready": ready}


@app.get("/search", response_model=SearchResponse)
@limiter.limit("60/minute")
def search_cities(
    request: Request,
    city: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    searcher: ICitySearcher = Depends(get_city_searcher),
):
    return _build_search_response(city, limit, searcher)


@app.post("/find-pincode", response_model=PincodeResponse)
@limiter.limit("60/minute")
def find_pincode(
    request: Request,
    req: PincodeRequest,
    finder: IPincodeFinder = Depends(get_pincode_finder),
):
    return _build_pincode_response(req.city, state=req.state, finder=finder)


@app.get("/find-pincode", response_model=PincodeResponse)
@limiter.limit("60/minute")
def find_pincode_get(
    request: Request,
    city: str = Query(..., min_length=1),
    state: Optional[str] = Query(None),
    finder: IPincodeFinder = Depends(get_pincode_finder),
):
    return _build_pincode_response(city, state=state, finder=finder)


@app.post("/coordinates", response_model=CoordinatesResponse)
@limiter.limit("60/minute")
def get_coordinates(
    request: Request,
    req: PincodeRequest,
    finder: ICoordinateFinder = Depends(get_coordinate_finder),
):
    return _build_coordinates_response(req.city, finder)


@app.get("/coordinates", response_model=CoordinatesResponse)
@limiter.limit("60/minute")
def get_coordinates_get(
    request: Request,
    city: str = Query(..., min_length=1),
    finder: ICoordinateFinder = Depends(get_coordinate_finder),
):
    return _build_coordinates_response(city, finder)
