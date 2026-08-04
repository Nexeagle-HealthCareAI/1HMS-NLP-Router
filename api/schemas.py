"""HTTP request/response contracts. Data shapes only -- no behavior."""
from typing import Optional

from pydantic import BaseModel, Field

# Generous headroom over any real symptom description (the longest sample
# query is well under 100 chars) while capping the worst case -- an
# unbounded string field is free amplification for whoever's paying the
# ~50ms/query coverage-gate cost (see nlp_brain.matching), gibberish or not.
MAX_QUERY_LENGTH = 1000


class RouteRequest(BaseModel):
    query: str = Field(max_length=MAX_QUERY_LENGTH)


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
