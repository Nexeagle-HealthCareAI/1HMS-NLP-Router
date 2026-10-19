"""/v1/route-symptom (+ the unversioned /route-symptom alias), /v1/taxonomy, /model-info.

The alias keeps every field existing callers read (specialtyIds, usedDefault, method, confidence,
modelVersion, specialties, emergency, decisive, noMatch, matchScore); new fields are additive.
Logs carry request id, decision, specialty ids, scores and latency - never the patient's text.
"""
from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query as Q, Request
from pydantic import BaseModel, Field, field_validator

from ...domain.models import RouteDecision
from ...domain.taxonomy import SPECIALTY_NAMES

log = logging.getLogger("hms_router.route_symptom")


class RouteContext(BaseModel):
    ageGroup: Literal["child", "adult"] | None = Field(None, description="If known; also the answer to a clarify of kind 'age'")


class RouteSymptomRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Symptom text in Hinglish / Hindi / English")
    topK: int = Field(5, ge=1, le=10, description="How many ranked specialties to return")
    context: RouteContext | None = None

    @field_validator("query")
    @classmethod
    def _clean(cls, v: str) -> str:
        if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in v):
            raise ValueError("control characters are not allowed")
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class EvidenceOut(BaseModel):
    id: str
    text: str
    similarity: float


class SpecialtyOut(BaseModel):
    specialtyId: str
    name: str
    confidence: float = Field(..., description="Fused retrieval + classifier score, 0-1; sums to 1 over all specialties")
    evidence: list[EvidenceOut] | None = Field(None, description="Labelled sentences behind this score (include=evidence)")


class EmergencyOut(BaseModel):
    flag: bool
    reasons: list[str]
    source: str | None = Field(None, description="rules | model | rules+model")
    score: float = Field(..., description="Emergency-model probability, 0-1")
    recommendedSpecialtyIds: list[str] = Field(default_factory=list)


class ClarifyOut(BaseModel):
    kind: Literal["specialty", "age"] = Field(..., description="age: resend with context.ageGroup = the chosen optionId")
    question: str
    options: list[str]
    optionIds: list[str]


class RouteSymptomResponse(BaseModel):
    specialties: list[SpecialtyOut]
    emergency: EmergencyOut
    decision: Literal["answer", "clarify", "abstain"]
    clarify: ClarifyOut | None
    reason: str | None
    method: str | None
    decisive: bool
    noMatch: bool
    matchScore: float
    specialtyIds: list[str] = Field(..., description="1-3 ids when decisive, else the top-5 hedge; empty on noMatch")
    usedDefault: bool = False
    confidence: float | None
    modelVersion: str | None
    requestId: str | None


def to_response(d: RouteDecision, include_evidence: bool, request_id: str | None) -> RouteSymptomResponse:
    return RouteSymptomResponse(
        specialties=[SpecialtyOut(specialtyId=c.specialtyId, name=c.name, confidence=c.confidence,
                                  evidence=[EvidenceOut(id=e.id, text=e.text, similarity=e.similarity)
                                            for e in c.evidence] if include_evidence else None)
                     for c in d.candidates],
        emergency=EmergencyOut(**vars(d.emergency)),
        decision=d.decision.value, clarify=ClarifyOut(**vars(d.clarify)) if d.clarify else None,
        reason=d.reason, method=None if d.noMatch else d.method, decisive=d.decisive, noMatch=d.noMatch,
        matchScore=d.matchScore, specialtyIds=d.specialtyIds,
        confidence=d.candidates[0].confidence if d.candidates else None,
        modelVersion=d.modelVersion, requestId=request_id)


def build_router(guard) -> APIRouter:
    router = APIRouter()

    def service(request: Request):
        svc = getattr(request.app.state, "route", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Specialty router not initialised yet")
        return svc

    def handle(req: RouteSymptomRequest, request: Request, include: str | None) -> RouteSymptomResponse:
        svc = service(request)
        t0 = time.perf_counter()
        d = svc.route(req.query, req.topK, age_group=req.context.ageGroup if req.context else None)
        rid = getattr(request.state, "request_id", None)
        log.info("route rid=%s client=%s decision=%s ids=%s conf=%s emergency=%s method=%s ms=%.1f%s",
                 rid, getattr(request.state, "client", "-"), d.decision.value, d.specialtyIds,
                 d.candidates[0].confidence if d.candidates else None, d.emergency.flag, d.method,
                 (time.perf_counter() - t0) * 1000, (" notes=" + "|".join(d.notes)) if d.notes else "")
        return to_response(d, include == "evidence", rid)

    @router.post("/v1/route-symptom", response_model=RouteSymptomResponse, tags=["Specialty"],
                 summary="Top-k specialties, emergency flag, and answer / clarify / abstain (stateless)")
    def route_v1(req: RouteSymptomRequest, request: Request,
                 include: str | None = Q(None, pattern="^evidence$", description="'evidence' to return the sentences"),
                 _client=Depends(guard)) -> RouteSymptomResponse:
        """Check `emergency.flag` first, even when `noMatch` is true. `decision`: answer (act on
        specialties[0]) | clarify (ask `clarify.question`, or show the hedge) | abstain (suggest a GP)."""
        return handle(req, request, include)

    @router.post("/route-symptom", response_model=RouteSymptomResponse, tags=["Specialty"],
                 summary="Alias of /v1/route-symptom for existing callers (Doctor Dekho)")
    def route_alias(req: RouteSymptomRequest, request: Request,
                    include: str | None = Q(None, pattern="^evidence$"), _client=Depends(guard)) -> RouteSymptomResponse:
        return handle(req, request, include)

    @router.get("/v1/taxonomy", tags=["Specialty"], summary="The specialtyIds this service can return")
    def taxonomy(_client=Depends(guard)) -> dict:
        return {"specialties": [{"specialtyId": k, "name": v} for k, v in SPECIALTY_NAMES.items()]}

    @router.get("/model-info", tags=["Specialty"], summary="Model version, training stats and routing policy")
    def model_info(request: Request, _client=Depends(guard)) -> dict:
        svc = service(request)
        return {**getattr(request.app.state, "route_meta", {}), "policy": svc.policy.as_dict(),
                "reasoner": type(svc.reasoner).__name__}

    return router
