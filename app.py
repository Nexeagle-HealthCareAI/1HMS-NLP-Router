"""
FastAPI wrapper around Model_1_revised's Hinglish symptom -> specialist classifier.

CHANGED FROM THE PREVIOUS VERSION (which wrapped Model_1_Doctor_Dekho):
  - Model_1_revised has no multi-symptom segmentation. One query -> at most
    one specialist. `specialtyIds` will therefore contain 0 or 1 items, never
    more, and `raw["segments"]` no longer exists.
  - Model_1_revised has no DEFAULT_SPECIALIST fallback. Low-confidence /
    gibberish input now returns Model_1_revised.NO_MATCH_MESSAGE instead of
    being routed to "General Physician". The old `usedDefault` field is
    replaced with `noMatch` — this is a BREAKING response-shape change for
    any caller that was relying on always getting back at least one
    specialtyId.
  - "Confidence" is no longer a classifier probability or cosine similarity;
    it's the word-overlap coverage ratio (see Model_1_revised.word_match_ratio)
    between the query and the closest known training example.
  - Model_1_revised has no LABEL_ALIASES normalization step, so specialist
    label spelling/duplication is whatever the training CSV contains as-is.
    If the training data introduced new/renamed labels, specialty_mapping.py
    may need new entries or those queries will silently map to no
    specialtyIds.

DEPLOYMENT REQUIREMENTS (new):
  - This service now loads a pre-trained joblib artifact (Model_1_revised.MODEL_OUT)
    instead of training from the CSV on every startup. That artifact is expected
    to be produced by an offline run of Model_1_revised.main() (i.e. your
    retrain pipeline), the same way model_meta.json already is.
  - If Model_1_revised's gibberish pre-check is in use, big.model (or whatever
    GIBBERISH_MODEL_PATH points at) and the `gibberish-detector` package must
    both be present, or that check silently no-ops (see Model_1_revised's own
    warning prints).
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from Model_1_revised import (
    MODEL_OUT,
    MATCH_THRESHOLD,
    NO_MATCH_MESSAGE,
    GIBBERISH_DETECTOR,
    clean_text,
    best_match,
)
from specialty_mapping import LABEL_TO_NEXEAGLE_SPECIALTY_ID

# Written only by data_pipeline/retrain_pipeline.py on a successful promotion — the live
# service just reads it, never writes it.
MODEL_META_PATH = Path(__file__).parent / "model_meta.json"

# The trained pipeline + label list + raw training texts, produced by an
# offline run of Model_1_revised.main(). The live service loads this once at
# startup and never retrains in-process.
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


class RouteRequest(BaseModel):
    query: str


class RouteResponse(BaseModel):
    specialtyIds: list[str]
    # True when Model_1_revised's gibberish check or word-overlap coverage
    # check rejected the input (Model_1_revised.NO_MATCH_MESSAGE), i.e. there
    # is no specialist to return at all. Replaces the old `usedDefault`
    # field — there is no default specialist to fall back to anymore.
    noMatch: bool
    method: str | None
    # Word-overlap coverage ratio (0-1) against the closest known training
    # example — NOT a classifier probability or cosine similarity anymore.
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
    """Mirrors Model_1_revised.predict()'s logic step-by-step, but also
    surfaces the intermediate word-overlap ratio and closest matching
    training example, which the original predict() computes internally but
    doesn't return (it only returns the final label or NO_MATCH_MESSAGE)."""
    cleaned = clean_text(query)

    if len(cleaned) < 3:
        return {
            "specialist": None, "matchRatio": None, "closestKnownExample": None,
            "flaggedGibberish": False, "noMatch": True,
        }

    if GIBBERISH_DETECTOR is not None and GIBBERISH_DETECTOR.is_gibberish(cleaned):
        return {
            "specialist": None, "matchRatio": None, "closestKnownExample": None,
            "flaggedGibberish": True, "noMatch": True,
        }

    ratio, closest = best_match(cleaned, bundle["texts"])
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
def route_symptom(req: RouteRequest):
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
        slug = LABEL_TO_NEXEAGLE_SPECIALTY_ID.get(result["specialist"])
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