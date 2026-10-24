"""/chat, /feedback, /session - moved here from api.py unchanged in behaviour (tests/test_chat_contract.py).

Only additions: the auth + rate-limit guard (scope "chat"; a no-op while HMS_API_KEYS is unset),
problem+json error bodies, and - only while HMS_DOCTORS_MCP_URL/KEY are set - Vita doctor search
(application/doctor_chat.py): asks by specialty, name, degree or distance answer with kind=doctor_list,
a confident recommendation carries a `doctors` block, select_doctor_id gives kind=doctor_selected.
With lookups off, status codes and success bodies are exactly as before.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from pydantic import BaseModel, Field

from ...application.doctor_chat import DoctorChat
from ...doctors import DoctorDirectory
from ...telemetry import DECISIONS, FEEDBACK, chat_record, feedback_record

log = logging.getLogger("hms_router.api")


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Patient message in Hinglish / Hindi / English")
    session_id: str | None = Field(None, description="Session token returned by a previous /chat call. "
                                                     "Omit (or pass null) to start a new session.")
    city: str | None = Field(None, max_length=100, description="Patient's city, for the doctors list (optional)")
    lat: float | None = Field(None, ge=-90, le=90, description="Patient latitude, with lng: nearest doctors first")
    lng: float | None = Field(None, ge=-180, le=180, description="Patient longitude, with lat")
    hospital_id: str | None = Field(None, max_length=64, pattern=r"^[0-9a-fA-F-]{32,36}$",
                                    description="Vita in a hospital's own channel: its doctors are listed first")
    select_doctor_id: str | None = Field(None, max_length=64, pattern=r"^[0-9a-fA-F-]{32,36}$",
                                         description="The doctor the patient tapped 'Choose' on; must be one "
                                                     "shown in this session (else 422)")


class FeedbackRequest(BaseModel):
    kind: str = Field(..., description="One of: correct_match | wrong_specialist | new_symptom_phrase")
    session_id: str | None = Field(None, description="Session the feedback refers to")
    cluster_id: str | None = Field(None, description="cluster_id from the AgentReply card output")
    specialist: str | None = Field(None, description="Specialist that was shown (for wrong_specialist)")
    symptoms: list[str] | None = Field(None, description="Symptoms that were active (for wrong_specialist)")
    suggested_specialist: str | None = Field(None, description="What the user thinks would be correct")
    phrase: str | None = Field(None, description="The unrecognised phrase (for new_symptom_phrase)")


def _doctor_chat(request: Request, svc) -> DoctorChat:
    """Built per request: app.state.doctors can be swapped at runtime (settings reload, tests)."""
    directory = getattr(request.app.state, "doctors", None) or DoctorDirectory(None)
    return DoctorChat(svc.agent, directory, request.app.state.redis)


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
        t0 = time.perf_counter()
        directory = getattr(request.app.state, "doctors", None)
        searching = (directory is not None and directory.enabled) or req.select_doctor_id is not None
        try:
            if searching:                     # Vita doctor search (application/doctor_chat.py)
                lat, lng = (req.lat, req.lng) if req.lat is not None and req.lng is not None else (None, None)
                out = _doctor_chat(request, svc).handle(
                    req.text.strip(), req.session_id, city=req.city, lat=lat, lng=lng,
                    hospital_id=req.hospital_id, select_doctor_id=req.select_doctor_id)
            else:                             # lookups off: exactly the original /chat
                out = svc.handle(req.text.strip(), req.session_id).to_dict()
        except ValueError as exc:             # includes SelectionError
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Unhandled error in /chat")
            raise HTTPException(status_code=500, detail="Internal routing error") from exc
        rec = getattr(request.app.state, "telemetry", None)
        if rec is not None and rec.enabled:
            try:
                rec.record(DECISIONS, chat_record(out, recorder=rec, caller=getattr(request.state, "client", "unknown"),
                                                  latency_ms=(time.perf_counter() - t0) * 1000, text=req.text.strip(),
                                                  model_version="chat-agent"))
            except Exception:
                log.exception("telemetry record failed")
        return out

    @router.post("/feedback", tags=["Router"], summary="Submit user feedback", status_code=status.HTTP_200_OK)
    def feedback(req: FeedbackRequest, request: Request, _client=Depends(guard)) -> dict[str, Any]:
        """correct_match (cluster_id) | wrong_specialist (cluster_id, specialist, symptoms) |
        new_symptom_phrase (phrase). Duplicate votes within the same session+kind are ignored."""
        svc = service(request)
        rec = getattr(request.app.state, "telemetry", None)
        if rec is not None and rec.enabled and req.kind in ("wrong_specialist", "new_symptom_phrase", "correct_match"):
            try:
                rec.record(FEEDBACK, feedback_record(req.kind, recorder=rec, caller=getattr(request.state, "client", "unknown"),
                                                     specialist=req.specialist, suggested_specialist=req.suggested_specialist,
                                                     symptoms=req.symptoms, phrase=req.phrase))
            except Exception:
                log.exception("telemetry record failed")
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
        svc = service(request)
        snap = svc.snapshot(session_id)
        doctor = _doctor_chat(request, svc).session_view(session_id)
        if snap.turn == 0 and not snap.symptoms and doctor is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or has expired")
        out = snap.to_dict()
        if doctor is not None:
            out["doctorSearch"] = doctor
            out["selectedDoctorId"] = doctor["selectedDoctorId"]
        return out

    @router.delete("/session/{session_id}", tags=["Session"], summary="Clear / reset a session",
                   status_code=status.HTTP_200_OK)
    def delete_session(request: Request, session_id: str = Path(..., description="Session token from /chat"),
                       _client=Depends(guard)) -> dict[str, str]:
        svc = service(request)
        svc.clear(session_id)
        try:
            request.app.state.redis.delete(DoctorChat._key(session_id))
        except Exception:
            log.exception("doctor session state not cleared")
        return {"deleted": session_id}

    return router
