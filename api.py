"""FastAPI gateway for the HMS NLP Router.

Endpoints
---------
POST /chat                  Send a symptom message, get a reply (follow-up / emergency / recommendation)
POST /feedback              Submit user feedback (correct_match | wrong_specialist | new_symptom_phrase)
GET  /session/{session_id}  Inspect accumulated state for an active session
DELETE /session/{session_id} Clear (reset) a session
GET  /health                Liveness + Redis ping

Running
-------
    pip install fastapi uvicorn
    uvicorn api:app --reload                        # uses fakeredis (dev)
    REDIS_URL=redis://localhost:6379 uvicorn api:app --reload  # real Redis

Environment variables
---------------------
    REDIS_URL           Full Redis connection URL (default: fakeredis in-memory)
    HMS_SESSION_TTL     Session idle timeout in seconds (default: 1800 = 30 min)
    HMS_FASTTEXT_LID    Path to lid.176.bin for accurate language detection (optional)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hms_router import HMSRouterAgent
from hms_router.accumulator import SymptomAccumulator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("hms_router.api")

# ---------------------------------------------------------------------------
# Redis client (built once at startup)
# ---------------------------------------------------------------------------
_redis_client = None


def _build_redis():
    redis_url = os.getenv("REDIS_URL", "")
    if redis_url:
        import redis  # type: ignore
        log.info("Connecting to Redis at %s", redis_url)
        return redis.Redis.from_url(redis_url, decode_responses=True)
    else:
        import fakeredis  # type: ignore
        log.warning("REDIS_URL not set - using fakeredis (in-memory, dev only)")
        return fakeredis.FakeRedis(decode_responses=True)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
_agent: HMSRouterAgent | None = None
_accumulator: SymptomAccumulator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _redis_client, _accumulator
    _redis_client = _build_redis()
    _agent = HMSRouterAgent(_redis_client)
    _accumulator = _agent.accumulator
    log.info("HMSRouterAgent ready")
    yield
    log.info("HMSRouterAgent shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HMS NLP Router API",
    description=(
        "Hinglish symptom text -> clinical intent -> specialist routing -> emergency flagging. "
        "**Triage support only - not a diagnosis.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000,
                      description="Patient message in Hinglish / Hindi / English")
    session_id: str | None = Field(
        None,
        description="Session token returned by a previous /chat call. "
                    "Omit (or pass null) to start a new session.",
    )


class FeedbackRequest(BaseModel):
    kind: str = Field(
        ...,
        description="One of: correct_match | wrong_specialist | new_symptom_phrase",
    )
    session_id: str | None = Field(None, description="Session the feedback refers to")
    cluster_id: str | None = Field(None, description="cluster_id from the AgentReply card output")
    specialist: str | None = Field(None, description="Specialist that was shown (for wrong_specialist)")
    symptoms: list[str] | None = Field(None, description="Symptoms that were active (for wrong_specialist)")
    suggested_specialist: str | None = Field(None, description="What the user thinks would be correct")
    phrase: str | None = Field(None, description="The unrecognised phrase (for new_symptom_phrase)")


# ---------------------------------------------------------------------------
# Helper: agent guard
# ---------------------------------------------------------------------------
def _get_agent() -> HMSRouterAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet")
    return _agent


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"], summary="Liveness check")
def health():
    """Returns 200 when the service is up. Pings Redis to confirm connectivity."""
    try:
        _redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"status": "ok", "redis": "ok" if redis_ok else "unreachable"}


@app.post(
    "/chat",
    tags=["Router"],
    summary="Send a symptom message",
    response_description=(
        "AgentReply with kind=follow_up|emergency|recommendation, "
        "the ready-to-send message, and full diagnostic context."
    ),
)
def chat(req: ChatRequest) -> dict[str, Any]:
    """Main conversational endpoint.

    - Pass `session_id=null` (or omit it) for the **first message**.
    - Echo the returned `session_id` back on every subsequent message to keep
      the multi-turn context alive.
    - Sessions auto-expire after 30 minutes of inactivity (sliding TTL).
    """
    agent = _get_agent()
    try:
        reply = agent.handle(req.text.strip(), req.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal routing error") from exc
    return reply.to_dict()


@app.post(
    "/feedback",
    tags=["Router"],
    summary="Submit user feedback",
    status_code=status.HTTP_200_OK,
)
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    """Step 8 - record whether the routing was helpful.

    | kind                 | required fields                             |
    |----------------------|---------------------------------------------|
    | correct_match        | cluster_id                                  |
    | wrong_specialist     | cluster_id, specialist, symptoms            |
    | new_symptom_phrase   | phrase                                      |

    Duplicate votes within the same session+kind are silently ignored.
    """
    agent = _get_agent()
    try:
        result = agent.feedback.submit(
            req.kind,
            session_id=req.session_id,
            cluster_id=req.cluster_id,
            specialist=req.specialist,
            symptoms=req.symptoms,
            phrase=req.phrase,
            suggested_specialist=req.suggested_specialist,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Unhandled error in /feedback")
        raise HTTPException(status_code=500, detail="Internal error") from exc
    return result


@app.get(
    "/session/{session_id}",
    tags=["Session"],
    summary="Inspect accumulated session state",
)
def get_session(
    session_id: str = Path(..., description="Session token from /chat"),
) -> dict[str, Any]:
    """Returns the merged multi-turn symptom state for a live session.

    Returns 404 if the session has expired or never existed.
    """
    if _accumulator is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet")

    snap = _accumulator.snapshot(session_id)

    # snapshot() returns an empty AccumulatedState when the session is gone
    if snap.turn == 0 and not snap.symptoms:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or has expired",
        )
    return snap.to_dict()


@app.delete(
    "/session/{session_id}",
    tags=["Session"],
    summary="Clear / reset a session",
    status_code=status.HTTP_200_OK,
)
def delete_session(
    session_id: str = Path(..., description="Session token from /chat"),
) -> dict[str, str]:
    """Deletes all Redis keys for this session (symptoms, context, turns).

    Useful for testing or when a user explicitly wants to restart the conversation.
    """
    if _accumulator is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet")
    _accumulator.clear(session_id)
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Global exception handler - always return JSON, never an HTML 500 page
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
