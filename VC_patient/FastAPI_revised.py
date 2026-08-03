#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
FastAPI wrapper around Model_1_Final's Hinglish symptom -> specialist classifier.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Import dependencies from your central brain file
from Model_1_Final import (
    MODEL_OUT,
    MATCH_THRESHOLD,
    NO_MATCH_MESSAGE,
    is_gibberish,
    clean_text,
    best_match,
)

# Ensure this mapping file exists in your directory
try:
    from Model_1_Final import LABEL_TO_NEXEAGLE_SPECIALTY_ID
except ImportError:
    # Fallback if the mapping file is not present
    LABEL_TO_NEXEAGLE_SPECIALTY_ID = {}

MODEL_META_PATH = Path(__file__).parent / "model_meta.json"
MODEL_BUNDLE_PATH = Path(__file__).parent / MODEL_OUT

_state: dict = {}

def _read_model_version() -> str | None:
    try:
        with open(MODEL_META_PATH, encoding="utf-8") as f:
            return json.load(f).get("modelVersion")
    except (FileNotFoundError, json.JSONDecodeError):
        return None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    _state.update(bundle=bundle, model_version=_read_model_version())
    yield
    _state.clear()

app = FastAPI(title="EasyHMS NLP Symptom Router", lifespan=lifespan)

# Per-IP rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class RouteRequest(BaseModel):
    query: str

class RouteResponse(BaseModel):
    specialtyIds: list[str]
    noMatch: bool
    method: str | None
    confidence: float | None
    modelVersion: str | None
    raw: dict

@app.get("/health")
def health():
    return {"status": "ok", "ready": bool(_state)}

@app.get("/model-info")
def model_info():
    try:
        with open(MODEL_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"modelVersion": "unknown", "lastRetrainedAt": None, "validationMetrics": None}

def _predict_with_coverage(query: str, bundle: dict, threshold: float = MATCH_THRESHOLD) -> dict:
    cleaned = clean_text(query)

    if len(cleaned) < 3:
        return {
            "specialist": None, "matchRatio": None, "closestKnownExample": None,
            "flaggedGibberish": False, "noMatch": True,
        }

    # Updated to use the standalone is_gibberish function from Model_1_Final
    if is_gibberish(cleaned):
        return {
            "specialist": None, "matchRatio": None, "closestKnownExample": None,
            "flaggedGibberish": True, "noMatch": True,
        }

    ratio, closest = best_match(cleaned, bundle["features"], bundle["texts_matrix"], bundle["texts"])
    if ratio < threshold:
        return {
            "specialist": None, "matchRatio": ratio, "closestKnownExample": closest,
            "flaggedGibberish": False, "noMatch": True,
        }

    feats = bundle["features"].transform([cleaned])
    pred = bundle["model"].predict(feats)[0]
    return {
        "specialist": pred, "matchRatio": ratio, "closestKnownExample": closest,
        "flaggedGibberish": False, "noMatch": False,
    }

@app.post("/route-symptom", response_model=RouteResponse)
@limiter.limit("30/minute")
def route_symptom(request: Request, req: RouteRequest):
    query = (req.query or "").strip()
    if not query:
        return RouteResponse(
            specialtyIds=[], noMatch=True, method=None, confidence=None,
            modelVersion=_state.get("model_version"),
            raw={"specialist": None, "matchRatio": None, "closestKnownExample": None,
                 "flaggedGibberish": False, "message": NO_MATCH_MESSAGE},
        )

    result = _predict_with_coverage(query, _state["bundle"])

    specialty_ids: list[str] = []
    if result["specialist"]:
        slug = LABEL_TO_NEXEAGLE_SPECIALTY_ID.get(result["specialist"], result["specialist"])
        if slug:
            specialty_ids.append(slug)

    return RouteResponse(
        specialtyIds=specialty_ids,
        noMatch=result["noMatch"],
        method="classifier" if result["specialist"] else None,
        confidence=result["matchRatio"],
        modelVersion=_state.get("model_version"),
        raw={
            "specialist": result["specialist"],
            "matchRatio": result["matchRatio"],
            "closestKnownExample": result["closestKnownExample"],
            "flaggedGibberish": result["flaggedGibberish"],
            "message": None if result["specialist"] else NO_MATCH_MESSAGE,
        },
    )




