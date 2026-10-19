"""/health (liveness, unchanged shape) and /ready (readiness: model loaded, Redis reachable). No auth."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


def _redis_ok(request: Request) -> bool:
    try:
        request.app.state.redis.ping()
        return True
    except Exception:
        return False


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health", tags=["System"], summary="Liveness check")
    def health(request: Request):
        """200 while the process is up; pings Redis to report connectivity."""
        return {"status": "ok", "redis": "ok" if _redis_ok(request) else "unreachable"}

    @router.get("/ready", tags=["System"], summary="Readiness: can this instance serve traffic?")
    def ready(request: Request):
        checks = {"redis": _redis_ok(request),
                  "chat": getattr(request.app.state, "chat", None) is not None,
                  "routeModel": getattr(request.app.state, "route", None) is not None}
        body = {"ready": all(checks.values()), "checks": checks,
                "modelVersion": getattr(request.app.state, "route_meta", {}).get("modelVersion")}
        return JSONResponse(body, status_code=200 if body["ready"] else 503)

    return router
