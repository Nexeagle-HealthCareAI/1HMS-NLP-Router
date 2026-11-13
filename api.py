"""HTTP entrypoint for the HMS NLP Router.  Run:  uvicorn api:app --port 5005

Endpoints (built in hms_router/adapters/http/):
    POST /chat, /feedback; GET|DELETE /session/{id}   conversational triage (unchanged behaviour; scope "chat")
    POST /v1/route-symptom  (+ /route-symptom alias)  stateless top-k specialties + emergency flag (scope "route")
    GET  /v1/taxonomy, /model-info                    specialty ids; model version + routing policy (scope "route")
    GET  /health, /ready                              liveness; readiness (model loaded, Redis reachable) - no auth

Configuration is environment-only; see hms_router/adapters/settings.py for every variable.
Local development needs none of it (no auth, in-memory Redis, LLM off). Production requires
HMS_ENV=production, HMS_API_KEYS and REDIS_URL, and refuses to start without them.
"""
from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from hms_router.adapters.http import create_app
from hms_router.adapters.http.security import ApiKeyAuth, Guard
from hms_router.adapters.settings import ApiClient
from hms_router.domain.models import RouteDecision


class _LazyLimiter:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def check(self, client: ApiClient, scope: str, ip: str | None = None) -> None:
        limiter = getattr(self.app.state, "limiter", None)
        if limiter is not None:
            limiter.check(client, scope, ip)


class RouteRequest(BaseModel):
    query: str = Field(max_length=1000)


class RouteResponse(BaseModel):
    specialtyIds: list[str]
    noMatch: bool
    method: str | None
    confidence: float | None
    modelVersion: str | None
    raw: dict


def _legacy_response(decision: RouteDecision) -> RouteResponse:
    return RouteResponse(
        specialtyIds=decision.specialtyIds,
        noMatch=decision.noMatch,
        method=None if decision.noMatch else decision.method,
        confidence=decision.matchScore,
        modelVersion=decision.modelVersion,
        raw=decision.to_dict(),
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

app = create_app()
legacy_route_guard = Guard(ApiKeyAuth(app.state.settings), _LazyLimiter(app), "route")


@app.post("/legacy/route-symptom", response_model=RouteResponse, tags=["Specialty"],
          summary="Legacy-compatible specialty routing response")
def route_symptom_legacy(
    req: RouteRequest,
    request: Request,
    _client=Depends(legacy_route_guard),
) -> RouteResponse:
    service = getattr(request.app.state, "route", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Specialty router not initialised yet")
    decision = service.route(req.query.strip())
    return _legacy_response(decision)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=5005, reload=True)
