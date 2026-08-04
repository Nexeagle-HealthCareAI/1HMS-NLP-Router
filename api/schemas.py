"""HTTP request/response contracts. Data shapes only -- no behavior."""
from typing import Optional

from pydantic import BaseModel


class RouteRequest(BaseModel):
    query: str


class RouteResponse(BaseModel):
    specialtyIds: list[str]
    # True when the NLP Brain's gibberish check or cosine-similarity coverage
    # check rejected the input, i.e. there is no specialist to return at all.
    # No default-specialist fallback -- an honest "no match" beats a guess.
    noMatch: bool
    method: Optional[str]
    # Cosine similarity (0-1) between the query and the closest known
    # training example's TF-IDF vector -- NOT a classifier probability.
    confidence: Optional[float]
    modelVersion: Optional[str]
    raw: dict
