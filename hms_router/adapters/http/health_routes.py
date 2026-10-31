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
        rt = getattr(request.app.state, "vita_runtime", None)
        llm = getattr(request.app.state, "vita_llm", None) or {}
        vita = {"llm": llm.get("provider"), "model": llm.get("model"),
                "mode": rt.mode("web") if rt is not None else None,
                "breakerOpen": bool(rt.breaker.open) if rt is not None and rt.has_llm else None}
        return {"status": "ok", "redis": "ok" if _redis_ok(request) else "unreachable", "vita": vita}

    @router.get("/ready", tags=["System"], summary="Readiness: can this instance serve traffic?")
    def ready(request: Request):
        checks = {"redis": _redis_ok(request),
                  "vita": getattr(request.app.state, "vita_pipeline", None) is not None,
                  "routeModel": getattr(request.app.state, "route", None) is not None}
        body = {"ready": all(checks.values()), "checks": checks,
                "modelVersion": getattr(request.app.state, "route_meta", {}).get("modelVersion")}
        return JSONResponse(body, status_code=200 if body["ready"] else 503)

    return router
