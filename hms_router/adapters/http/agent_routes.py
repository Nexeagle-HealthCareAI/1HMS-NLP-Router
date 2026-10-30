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
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from pydantic import BaseModel, Field

from ...application.doctor_chat import DoctorChat
from ...vita_graph.engine import VitaEngine
from ...channels import whatsapp
from ...doctors import DoctorDirectory
from ...language import ui_language
from ...telemetry import DECISIONS, chat_record

log = logging.getLogger("hms_router.agent")

_UUID = r"^[0-9a-fA-F-]{32,36}$"


class AgentChatRequest(BaseModel):
    text: str | None = Field(None, max_length=1000,
                             description="Patient message in Hinglish / Hindi / English. May be left out when "
                                         "`reply_id` (a tapped WhatsApp row/button) or a shared location is sent")
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
    channel: Literal["web", "whatsapp"] = Field("web", description="whatsapp: the reply also carries `whatsapp`, "
                                                                     "ready-to-send WhatsApp Cloud API messages")
    reply_id: str | None = Field(None, max_length=256,
                                 description="WhatsApp: the id of the list row or button the patient tapped, exactly "
                                             "as Vita wrote it (doc:<id>, more, refine:<key>, say:<text>)")
    client_ip: str | None = Field(None, max_length=64,
                                  description="Booking: the visitor's own IP, so 1HMS limits each visitor, not the "
                                              "router (needs HMS_PROXY_FORWARDING_SECRET)")
    patient_token: str | None = Field(None, max_length=4096,
                                      description="Booking: the channel's signed-in patient token (1HMS patient JWT). "
                                                  "With it Vita skips the OTP and offers the saved profile and family. "
                                                  "Server-to-server only; never logged or echoed")
    verified_mobile: str | None = Field(None, max_length=16, pattern=r"^\+?\d{10,13}$",
                                        description="Booking: a number the channel already verified (the WhatsApp "
                                                    "sender). Honoured only for API keys with the verified_mobile scope")
    visit_appointment_id: str | None = Field(None, pattern=r"^[0-9a-fA-F-]{36}$",
                                             description="After booking: the appointment the patient is looking at (the "
                                                         "website's visit page or its latest booking), for directions, "
                                                         "\"I've reached\", check-in and token status")
    new_search: bool = Field(False, description="Search with exactly the asks in `text`, not merged into the last "
                                                "search (the screen removed a filter chip)")
    remember: bool = Field(False, description="The identified patient agreed that Vita may remember who they book for, "
                                              "past specialties and doctors (language and city are kept without it)")


class FeedbackIn(BaseModel):
    session_id: str = Field(..., max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(None, max_length=500)


class ForgetIn(BaseModel):
    patient_token: str | None = Field(None, max_length=4096)
    verified_mobile: str | None = Field(None, max_length=16, pattern=r"^\+?\d{10,13}$")


class ExampleIn(BaseModel):
    session_id: str | None = Field(None, max_length=64, description="Approve a turn of this conversation...")
    turn: int | None = Field(None, ge=0, description="...this turn (0-based, from the conversation's traces)")
    patient: str | None = Field(None, max_length=500, description="Or write the patient message yourself")
    reply: str | None = Field(None, max_length=700, description="The reply Vita should give (a correction), or empty "
                                                               "to keep Vita's own reply")
    specialist: str | None = Field(None, max_length=60, description="The right doctor (a correction)")
    note: str | None = Field(None, max_length=300)


class ControlIn(BaseModel):
    mode: Literal["agent", "rules"] = "agent"
    channels: dict[str, Literal["agent", "rules"]] = Field(default_factory=dict)


def _flow(request: Request) -> VitaEngine:
    agent = getattr(request.app.state, "vita_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet")
    directory = getattr(request.app.state, "doctors", None) or DoctorDirectory(None)
    flow = DoctorChat(agent, directory, request.app.state.redis,
                      booking_client=getattr(request.app.state, "booking", None))
    # the LangGraph agent when VITA_LLM_PROVIDER is set (hms_router/vita_graph), else the rules engine as before
    return VitaEngine(flow, getattr(request.app.state, "vita_runtime", None))


def build_router(guard, admin_guard=None) -> APIRouter:
    router = APIRouter(prefix="/v2/agent", tags=["Vita agent"])

    @router.post("/chat", summary="Talk to Vita: find a doctor, choose one",
                 response_description="kind = follow_up | emergency | recommendation | doctor_list | doctor_selected "
                                      "| info, a ready-to-send message, tap options in `clarify`, doctors in `doctors`")
    def agent_chat(req: AgentChatRequest, request: Request, _client=Depends(guard)) -> dict[str, Any]:
        flow = _flow(request)
        t0 = time.perf_counter()
        lat, lng = (req.lat, req.lng) if req.lat is not None and req.lng is not None else (None, None)
        text, select = (req.text or "").strip(), req.select_doctor_id
        if req.reply_id:                                    # a tapped WhatsApp row / button
            tap = whatsapp.read_reply_id(req.reply_id, flow.load(req.session_id).get("language"))
            text, select = tap["text"], tap.get("select_doctor_id") or select
        elif not text and lat is not None:                  # a location shared on its own
            text = "location"
        if not text:
            raise HTTPException(status_code=422, detail="text, reply_id or a shared location is required")
        trusted = "verified_mobile" in getattr(_client, "scopes", ())
        verified = (req.verified_mobile or "")[-10:] if trusted and req.verified_mobile else None
        try:
            out = flow.handle(text, req.session_id, city=req.city, lat=lat, lng=lng,
                              client_ip=req.client_ip, patient_token=req.patient_token, verified_mobile=verified,
                              hospital_id=req.hospital_id, select_doctor_id=select, page=req.page,
                              new_search=req.new_search, ui_language=ui_language(req.ui_language),
                              channel=req.channel, visit_appointment_id=req.visit_appointment_id,
                              remember=req.remember)
        except ValueError as exc:              # includes SelectionError
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Unhandled error in /v2/agent/chat")
            raise HTTPException(status_code=500, detail="Internal agent error") from exc
        session = out.pop("patient_session", None)                 # a credential: kept out of telemetry
        booking_turn = (out.get("kind") == "booked" or str((out.get("clarify") or {}).get("kind") or "").startswith("book_"))
        rec = getattr(request.app.state, "telemetry", None)
        if rec is not None and rec.enabled:
            try:
                record = chat_record(out, recorder=rec, caller=getattr(request.state, "client", "unknown"),
                                     latency_ms=(time.perf_counter() - t0) * 1000,
                                     text="[booking details]" if booking_turn else text,   # names, numbers, codes
                                     model_version="vita-agent-v3-" + str(out.get("engine") or "rules"))
                rec.record(DECISIONS, record)
            except Exception:
                log.exception("telemetry record failed")
        if session and req.channel != "whatsapp":
            out["patient_session"] = session                       # the website signs the patient in with it
        if req.channel == "whatsapp":
            out["whatsapp"] = whatsapp.render(out)
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

    # ------------------------------------------------------------------ learning: patients and channels
    def runtime(request: Request):
        rt = getattr(request.app.state, "vita_runtime", None)
        if rt is None:
            raise HTTPException(status_code=503, detail="Vita learning store not available")
        return rt

    @router.post("/feedback", summary="Thumbs up / down on a conversation (feeds Vita's learning)")
    def feedback(body: FeedbackIn, request: Request, _client=Depends(guard)) -> dict:
        if not runtime(request).traces.feedback(body.session_id, body.rating, body.comment):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"ok": True}

    @router.delete("/patient-memory", summary="Forget everything Vita remembers about this patient")
    def forget(body: ForgetIn, request: Request, _client=Depends(guard)) -> dict:
        from ...application.booking import token_mobile
        trusted = "verified_mobile" in getattr(_client, "scopes", ())
        mobile = (body.verified_mobile if trusted and body.verified_mobile else None) or \
            (token_mobile(body.patient_token) if body.patient_token else None)
        if not mobile:
            raise HTTPException(status_code=422, detail="patient_token, or verified_mobile with that scope, required")
        return {"forgotten": runtime(request).memory.forget(mobile)}

    # ------------------------------------------------------------------ admin: control, review, examples, gate
    if admin_guard is not None:
        @router.get("/admin/control", summary="Vita's mode: agent (LangGraph + LLM) or rules, per channel")
        def get_control(request: Request, _client=Depends(admin_guard)) -> dict:
            rt = runtime(request)
            return {**rt.control(), "llm": rt.has_llm, "breakerOpen": rt.breaker.open}

        @router.put("/admin/control", summary="Switch Vita between agent and rules (kill switch), per channel")
        def put_control(body: ControlIn, request: Request, _client=Depends(admin_guard)) -> dict:
            return runtime(request).set_control(body.model_dump())

        @router.get("/admin/conversations", summary="Recent conversations, by outcome (booked, emergency, up, down)")
        def conversations(request: Request, outcome: str | None = None, limit: int = 50,
                          _client=Depends(admin_guard)) -> dict:
            return {"conversations": runtime(request).traces.conversations(outcome, min(limit, 200))}

        @router.get("/admin/conversations/{session_id}", summary="One conversation: every turn's understanding, "
                                                                 "decision, tools, checks and reply")
        def conversation(request: Request, session_id: str = Path(...), _client=Depends(admin_guard)) -> dict:
            rt = runtime(request)
            turns = rt.traces.turns(session_id)
            if not turns:
                raise HTTPException(status_code=404, detail="Conversation not found")
            return {"sessionId": session_id, "turns": turns, "state": _flow(request).state(session_id)}

        @router.get("/admin/examples", summary="Draft and active examples, and the promotion log")
        def examples(request: Request, _client=Depends(admin_guard)) -> dict:
            ex = runtime(request).examples
            return {"version": ex.version(), "draft": ex.drafts(), "active": ex.active()}

        @router.post("/admin/examples", summary="Approve a turn as an example, or correct its reply / specialist "
                                                "(goes to drafts; live only after the promotion gate)")
        def add_example(body: ExampleIn, request: Request, _client=Depends(admin_guard)) -> dict:
            rt = runtime(request)
            patient, reply, language = body.patient, body.reply, None
            if body.session_id is not None and body.turn is not None:
                turns = rt.traces.turns(body.session_id)
                if not 0 <= body.turn < len(turns):
                    raise HTTPException(status_code=404, detail="No such turn")
                t = turns[body.turn]
                patient, language = patient or t.get("patient"), t.get("language")
                reply = reply if reply is not None else t.get("reply")
            if not patient:
                raise HTTPException(status_code=422, detail="session_id + turn, or patient, required")
            corrected = bool(body.reply or body.specialist)
            return rt.examples.add_draft(patient=patient, reply=reply, specialist=body.specialist,
                                         language=language, note=body.note,
                                         source="corrected" if corrected else "approved",
                                         session_id=body.session_id, turn=body.turn,
                                         by=getattr(_client, "name", None))

        @router.delete("/admin/examples/{example_id}", summary="Remove a draft or active example")
        def delete_example(request: Request, example_id: str = Path(..., max_length=32),
                           _client=Depends(admin_guard)) -> dict:
            return {"deleted": runtime(request).examples.remove(example_id)}

        @router.post("/admin/examples/promote", summary="Run the scenario suite with and without the drafts; "
                                                        "make them live only if the score is not worse")
        def promote(request: Request, _client=Depends(admin_guard)) -> dict:
            from ...vita_graph.evals import promotion_gate
            return promotion_gate(request.app, runtime(request))

    return router
