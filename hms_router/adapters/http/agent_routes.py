"""POST /v2/agent/chat - Vita, the patient agent (NLP Router Agentic AI vision; Vita Chat Part 2).

A new endpoint, so /chat keeps its contract unchanged (vision: "Today's /chat and /v1/route-symptom keep
working unchanged; the agent is a new endpoint that uses them"). It finds doctors by symptom, specialty,
name, degree, hospital or distance, asks at most two questions before showing something, lets the
patient choose, and puts an emergency ahead of everything (application/doctor_chat.py).

Auth and rate limits use the "chat" scope, so a key allowed to call /chat can call this too.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from pydantic import BaseModel, Field

from ...application.doctor_chat import DoctorChat
from ...doctors import DoctorDirectory
from ...language import ui_language
from ...telemetry import DECISIONS, chat_record

log = logging.getLogger("hms_router.agent")

_UUID = r"^[0-9a-fA-F-]{32,36}$"


class AgentChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Patient message in Hinglish / Hindi / English")
    session_id: str | None = Field(None, description="From the previous reply; omit on the first message")
    city: str | None = Field(None, max_length=100, description="Patient's city, used when no location is shared")
    lat: float | None = Field(None, ge=-90, le=90, description="Patient latitude, with lng (this search only)")
    lng: float | None = Field(None, ge=-180, le=180, description="Patient longitude, with lat")
    hospital_id: str | None = Field(None, max_length=64, pattern=_UUID,
                                    description="Vita in a hospital's own channel: its doctors are listed first")
    select_doctor_id: str | None = Field(None, max_length=64, pattern=_UUID,
                                         description="The doctor the patient tapped 'Choose' on; must be one shown "
                                                     "in this session (else 422)")
    page: int = Field(1, ge=1, le=10, description="'Show more doctors': the next page of the last list")
    ui_language: str | None = Field(None, max_length=16,
                                    description="The channel's UI language (en, hi, hinglish, bn). Used only until the "
                                                "patient's own words show a language; Vita then mirrors the patient")
    new_search: bool = Field(False, description="Search with exactly the asks in `text`, not merged into the last "
                                                "search (the screen removed a filter chip)")


def _flow(request: Request) -> DoctorChat:
    agent = getattr(request.app.state, "vita_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet")
    directory = getattr(request.app.state, "doctors", None) or DoctorDirectory(None)
    return DoctorChat(agent, directory, request.app.state.redis)


def build_router(guard) -> APIRouter:
    router = APIRouter(prefix="/v2/agent", tags=["Vita agent"])

    @router.post("/chat", summary="Talk to Vita: find a doctor, choose one",
                 response_description="kind = follow_up | emergency | recommendation | doctor_list | doctor_selected "
                                      "| info, a ready-to-send message, tap options in `clarify`, doctors in `doctors`")
    def agent_chat(req: AgentChatRequest, request: Request, _client=Depends(guard)) -> dict[str, Any]:
        flow = _flow(request)
        t0 = time.perf_counter()
        lat, lng = (req.lat, req.lng) if req.lat is not None and req.lng is not None else (None, None)
        try:
            out = flow.handle(req.text.strip(), req.session_id, city=req.city, lat=lat, lng=lng,
                              hospital_id=req.hospital_id, select_doctor_id=req.select_doctor_id, page=req.page,
                              new_search=req.new_search, ui_language=ui_language(req.ui_language))
        except ValueError as exc:              # includes SelectionError
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Unhandled error in /v2/agent/chat")
            raise HTTPException(status_code=500, detail="Internal agent error") from exc
        rec = getattr(request.app.state, "telemetry", None)
        if rec is not None and rec.enabled:
            try:
                record = chat_record(out, recorder=rec, caller=getattr(request.state, "client", "unknown"),
                                     latency_ms=(time.perf_counter() - t0) * 1000, text=req.text.strip(),
                                     model_version="vita-agent-v2")
                rec.record(DECISIONS, record)
            except Exception:
                log.exception("telemetry record failed")
        return out

    @router.get("/session/{session_id}", summary="What Vita remembers about this conversation")
    def get_session(request: Request, session_id: str = Path(...), _client=Depends(guard)) -> dict[str, Any]:
        flow = _flow(request)
        snap = flow.agent.accumulator.snapshot(session_id)
        doctor = flow.session_view(session_id)
        if snap.turn == 0 and not snap.symptoms and doctor is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or has expired")
        out = snap.to_dict()
        out["doctorSearch"] = doctor
        out["selectedDoctorId"] = (doctor or {}).get("selectedDoctorId")
        return out

    @router.delete("/session/{session_id}", summary="Forget this conversation", status_code=status.HTTP_200_OK)
    def delete_session(request: Request, session_id: str = Path(...), _client=Depends(guard)) -> dict[str, str]:
        flow = _flow(request)
        flow.agent.accumulator.clear(session_id)
        flow.clear(session_id)
        return {"deleted": session_id}

    return router
