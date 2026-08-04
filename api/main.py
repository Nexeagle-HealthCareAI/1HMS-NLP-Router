"""FastAPI app wiring: creates the app, loads the NLP Brain once at startup,
and attaches cross-cutting HTTP concerns (rate limiting). No ML logic lives
here -- see nlp_brain/ for that, and routes.py for the HTTP contract.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from nlp_brain import MODEL_OUT, SymptomClassifier

from .rate_limit import limiter
from .routes import router

MODEL_BUNDLE_PATH = Path(__file__).resolve().parent.parent / MODEL_OUT
MODEL_META_PATH = Path(__file__).resolve().parent.parent / "model_meta.json"


def _read_model_version() -> Optional[str]:
    """Reads just the modelVersion field out of model_meta.json, to stamp
    onto every /route-symptom response. Called once at startup (see
    lifespan() below) rather than per-request, since model_meta.json only
    changes when data_pipeline/retrain_pipeline.py promotes a new model --
    which always triggers a redeploy (and thus a fresh startup) anyway.
    Returns None if the file is missing or malformed, so a fresh
    environment without a trained model yet still boots instead of
    crashing."""
    try:
        with open(MODEL_META_PATH, encoding="utf-8") as f:
            return json.load(f).get("modelVersion")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup/shutdown hook: loads the trained SymptomClassifier
    bundle into app.state ONCE when the process starts (not per-request --
    loading takes ~0.2-2s depending on disk cache), and clears it on
    shutdown. Route handlers read app.state.classifier (see routes.py);
    nothing else in this codebase should call SymptomClassifier.load()
    directly in a request path."""
    app.state.classifier = SymptomClassifier.load(str(MODEL_BUNDLE_PATH))
    app.state.model_version = _read_model_version()
    yield
    app.state.classifier = None


app = FastAPI(title="EasyHMS NLP Symptom Router", lifespan=lifespan)

# Per-IP rate limiting -- /route-symptom has no auth, and a single query
# still takes real CPU time (TF-IDF transform + classifier + cosine-
# similarity coverage check), so an unauthenticated client hammering it is a
# cheap way to degrade the service for everyone else. Keyed on remote
# address, so this only works as intended if the client's real IP reaches
# the app unmodified (no IP-rewriting reverse proxy in front) -- revisit if
# one gets added later.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(router)
