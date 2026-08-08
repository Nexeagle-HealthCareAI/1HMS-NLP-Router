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

BASE_DIR = Path(_script_dir())
CITIES_CSV_PATH = BASE_DIR / CITIES_CSV
PINCODES_CSV_PATH = BASE_DIR / PINCODES_CSV

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    finder = PincodeFinder(str(CITIES_CSV_PATH), str(PINCODES_CSV_PATH))
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
        matches=[CityMatch(city=m["city"], state=m["state"]) for m in matches],
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
