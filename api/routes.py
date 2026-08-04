"""HTTP contract + orchestration only -- no ML logic. Translates HTTP
requests into nlp_brain.SymptomClassifier.predict() calls and formats the
result into the response shape NexEagleWebsite expects."""
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request

from nlp_brain import NO_MATCH_MESSAGE, PredictionResult
from specialty_mapping import LABEL_TO_NEXEAGLE_SPECIALTY_ID

from .rate_limit import limiter
from .schemas import RouteRequest, RouteResponse

router = APIRouter()

# Written only by data_pipeline/retrain_pipeline.py on a successful promotion
# -- this layer just reads it, never writes it.
MODEL_META_PATH = Path(__file__).resolve().parent.parent / "model_meta.json"


def _read_model_meta() -> dict:
    try:
        with open(MODEL_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"modelVersion": "unknown", "lastRetrainedAt": None, "validationMetrics": None}


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "ready": getattr(request.app.state, "classifier", None) is not None}


@router.get("/model-info")
def model_info():
    return _read_model_meta()


def _to_response(result: PredictionResult, model_version: Optional[str]) -> RouteResponse:
    specialty_ids: list[str] = []
    if result.specialist:
        slug = LABEL_TO_NEXEAGLE_SPECIALTY_ID.get(result.specialist)
        if slug:
            specialty_ids.append(slug)

    return RouteResponse(
        specialtyIds=specialty_ids,
        noMatch=result.no_match,
        method="classifier" if result.specialist else None,
        confidence=result.match_ratio,
        modelVersion=model_version,
        raw={
            "specialist": result.specialist,
            "matchRatio": result.match_ratio,
            "closestKnownExample": result.closest_known_example,
            "flaggedGibberish": result.flagged_gibberish,
            "message": None if result.specialist else NO_MATCH_MESSAGE,
        },
    )


@router.post("/route-symptom", response_model=RouteResponse)
@limiter.limit("30/minute")
def route_symptom(request: Request, req: RouteRequest):
    query = (req.query or "").strip()
    result = request.app.state.classifier.predict(query)
    return _to_response(result, request.app.state.model_version)
