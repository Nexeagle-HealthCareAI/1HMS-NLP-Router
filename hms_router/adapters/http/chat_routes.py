"""/chat, /feedback, /session - moved here from api.py unchanged in behaviour (tests/test_chat_contract.py).

Only additions: the auth + rate-limit guard (scope "chat"; a no-op while HMS_API_KEYS is unset) and
problem+json error bodies. Status codes and success bodies are exactly as before.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from pydantic import BaseModel, Field

log = logging.getLogger("hms_router.api")


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Patient message in Hinglish / Hindi / English")
    session_id: str | None = Field(None, description="Session token returned by a previous /chat call. "
                                                     "Omit (or pass null) to start a new session.")


class FeedbackRequest(BaseModel):
    kind: str = Field(..., description="One of: correct_match | wrong_specialist | new_symptom_phrase")
    session_id: str | None = Field(None, description="Session the feedback refers to")
    cluster_id: str | None = Field(None, description="cluster_id from the AgentReply card output")
    specialist: str | None = Field(None, description="Specialist that was shown (for wrong_specialist)")
    symptoms: list[str] | None = Field(None, description="Symptoms that were active (for wrong_specialist)")
    suggested_specialist: str | None = Field(None, description="What the user thinks would be correct")
    phrase: str | None = Field(None, description="The unrecognised phrase (for new_symptom_phrase)")


def build_router(guard) -> APIRouter:
    router = APIRouter()

    def service(request: Request):
        svc = getattr(request.app.state, "chat", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Agent not initialised yet")
        return svc

    @router.post("/chat", tags=["Router"], summary="Send a symptom message",
                 response_description="AgentReply with kind=follow_up|emergency|recommendation, "
                                      "the ready-to-send message, and full diagnostic context.")
    def chat(req: ChatRequest, request: Request, _client=Depends(guard)) -> dict[str, Any]:
        """Conversational triage. Omit `session_id` on the first message, echo it back after; sessions
        expire after 30 minutes idle."""
        svc = service(request)
        try:
            reply = svc.handle(req.text.strip(), req.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Unhandled error in /chat")
            raise HTTPException(status_code=500, detail="Internal routing error") from exc
        return reply.to_dict()

    @router.post("/feedback", tags=["Router"], summary="Submit user feedback", status_code=status.HTTP_200_OK)
    def feedback(req: FeedbackRequest, request: Request, _client=Depends(guard)) -> dict[str, Any]:
        """correct_match (cluster_id) | wrong_specialist (cluster_id, specialist, symptoms) |
        new_symptom_phrase (phrase). Duplicate votes within the same session+kind are ignored."""
        svc = service(request)
        try:
            return svc.feedback(req.kind, session_id=req.session_id, cluster_id=req.cluster_id,
                                specialist=req.specialist, symptoms=req.symptoms, phrase=req.phrase,
                                suggested_specialist=req.suggested_specialist)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Unhandled error in /feedback")
            raise HTTPException(status_code=500, detail="Internal error") from exc

    @router.get("/session/{session_id}", tags=["Session"], summary="Inspect accumulated session state")
    def get_session(request: Request, session_id: str = Path(..., description="Session token from /chat"),
                    _client=Depends(guard)) -> dict[str, Any]:
        snap = service(request).snapshot(session_id)
        if snap.turn == 0 and not snap.symptoms:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or has expired")
        return snap.to_dict()

    @router.delete("/session/{session_id}", tags=["Session"], summary="Clear / reset a session",
                   status_code=status.HTTP_200_OK)
    def delete_session(request: Request, session_id: str = Path(..., description="Session token from /chat"),
                       _client=Depends(guard)) -> dict[str, str]:
        service(request).clear(session_id)
        return {"deleted": session_id}

    return router
