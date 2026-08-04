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
    """Reads model_meta.json fresh on every /model-info call (unlike the
    modelVersion cached at startup in main.py -- see main._read_model_version)
    so a monitoring dashboard hitting /model-info repeatedly always sees the
    latest promoted metrics, not a stale in-memory snapshot from process
    startup. Falls back to an "unknown" placeholder if the file is missing
    or malformed, so this endpoint never 500s."""
    try:
        with open(MODEL_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"modelVersion": "unknown", "lastRetrainedAt": None, "validationMetrics": None}


@router.get("/health")
def health(request: Request):
    """Liveness/readiness probe -- what deploy-nlp.yml's health-check step
    and any container orchestrator's health check should call. `ready` is
    False only in the narrow window before startup's lifespan() has
    finished loading the classifier (or after shutdown has cleared it), not
    on ordinary request errors."""
    return {"status": "ok", "ready": getattr(request.app.state, "classifier", None) is not None}


@router.get("/model-info")
def model_info():
    """What's currently deployed: model version, when it was last
    retrained, and its recorded validation metrics. Call this before
    trusting a batch of /route-symptom results if you need to know exactly
    which model produced them (e.g. when comparing behavior across a
    retrain promotion)."""
    return _read_model_meta()


def _to_response(result: PredictionResult, model_version: Optional[str]) -> RouteResponse:
    """Maps a PredictionResult (nlp_brain's internal representation) onto
    the RouteResponse contract callers actually depend on -- in particular,
    translating the Brain's internal specialist label (e.g. "Cardiologist
    (Heart)") into NexEagleWebsite's specialtyId slug (e.g. "cardiology")
    via specialty_mapping.py. If a label has no corresponding slug (e.g.
    the training data introduced a new/renamed label that specialty_mapping.py
    hasn't caught up with yet), specialtyIds comes back empty rather than
    raising -- the raw.specialist field still carries the internal label for
    debugging."""
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
    """The main endpoint: given a Hinglish symptom description, returns the
    specialist to route the patient to (or an honest "no match" -- see
    RouteResponse.noMatch -- if the input is gibberish or too dissimilar
    from anything the model was trained on). This is the only endpoint
    voice/api_client.SymptomRouterClient calls, and the only one
    NexEagleWebsite should call for routing decisions; /health and
    /model-info are operational, not part of the routing flow. Rate-limited
    to 30 requests/minute per client IP (see rate_limit.py)."""
    query = (req.query or "").strip()
    result = request.app.state.classifier.predict(query)
    return _to_response(result, request.app.state.model_version)
