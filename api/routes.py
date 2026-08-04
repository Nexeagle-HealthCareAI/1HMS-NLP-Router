"""HTTP contract + orchestration only -- no ML logic. Translates HTTP
requests into nlp_brain.SymptomClassifier.predict() calls and formats the
result into the response shape NexEagleWebsite expects."""
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from nlp_brain import NO_MATCH_MESSAGE, PredictionResult
from specialty_mapping import LABEL_TO_NEXEAGLE_SPECIALTY_ID
from speech import (
    AudioFormatError,
    AudioUnintelligible,
    SpeechServiceError,
    TranscriptionError,
    transcribe_audio_file,
)

from .rate_limit import limiter
from .schemas import MAX_AUDIO_BYTES, RouteRequest, RouteResponse, RouteResponseWithTranscript

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
    translating the Brain's internal specialist label(s) (e.g. "Cardiologist
    (Heart)") into NexEagleWebsite's specialtyId slug(s) (e.g. "cardiology")
    via specialty_mapping.py. specialtyIds preserves result.candidates'
    order (most-confident first) and may have more than one entry when a
    close runner-up was surfaced (see nlp_brain.candidates.build_candidates)
    -- deduped by slug rather than by internal label, since two internal
    labels can map to the same NexEagleWebsite specialty (see
    specialty_mapping.py's GI/Surgical Gastroenterologist note). A candidate
    label with no corresponding slug (e.g. the training data introduced a
    new/renamed label that specialty_mapping.py hasn't caught up with yet)
    is skipped rather than raising -- the raw.specialist field still carries
    the internal top-pick label for debugging."""
    specialty_ids: list[str] = []
    for label in result.candidates:
        slug = LABEL_TO_NEXEAGLE_SPECIALTY_ID.get(label)
        if slug and slug not in specialty_ids:
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


def _reject_if_too_large(audio: UploadFile) -> None:
    """UploadFile wraps a SpooledTemporaryFile, so its size can be checked
    synchronously by seeking -- no need to buffer the whole thing into
    memory first just to measure it. Raises before any ffmpeg/STT work
    happens, so an oversized upload is cheap to reject."""
    audio.file.seek(0, 2)  # SEEK_END
    size = audio.file.tell()
    audio.file.seek(0)
    if size > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({size} bytes, max {MAX_AUDIO_BYTES}).",
        )


@router.post("/route-symptom-audio", response_model=RouteResponseWithTranscript)
@limiter.limit("10/minute")
def route_symptom_audio(request: Request, audio: UploadFile = File(...)):
    """Voice counterpart to /route-symptom: accepts an uploaded audio
    recording (any format ffmpeg can decode -- webm/opus from a browser's
    MediaRecorder, wav, m4a, etc.), transcribes and transliterates it
    server-side (see speech/transcription.py), then routes the resulting
    text exactly like /route-symptom would. Intended for a web/mobile
    client that records audio itself (e.g. a Next.js frontend using
    MediaRecorder) rather than running Python -- voice/'s own CLI still
    uses a live microphone directly and doesn't need this endpoint.

    Rate-limited tighter than /route-symptom (10/min vs. 30/min per IP):
    ffmpeg decoding + a speech-recognition API call is real work per
    request, on top of the same classification cost the text endpoint
    already pays.
    """
    _reject_if_too_large(audio)

    try:
        transcript = transcribe_audio_file(audio.file)
    except AudioFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SpeechServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except AudioUnintelligible:
        return RouteResponseWithTranscript(
            transcript=None,
            specialtyIds=[], noMatch=True, method=None, confidence=None,
            modelVersion=request.app.state.model_version,
            raw={"specialist": None, "matchRatio": None, "closestKnownExample": None,
                 "flaggedGibberish": False, "message": "Could not understand the audio."},
        )
    except TranscriptionError as e:
        # Package not installed, etc. -- an operational/deployment problem,
        # not something a client retry fixes.
        raise HTTPException(status_code=503, detail=str(e))

    result = request.app.state.classifier.predict(transcript)
    response = _to_response(result, request.app.state.model_version)
    return RouteResponseWithTranscript(transcript=transcript, **response.model_dump())
