"""/admin/rebuild and /admin/telemetry - used by the CMS "AI Agents" dashboard (API key scope "admin").

POST /admin/rebuild starts the learning-loop rebuild (application/retraining.py) in the background and
returns at once; GET /admin/rebuild reports its progress and last result.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse


def build_router(guard) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["Admin"])

    def retrainer(request: Request):
        r = getattr(request.app.state, "retrainer", None)
        if r is None:
            raise HTTPException(status_code=503, detail="Service still starting")
        return r

    @router.post("/rebuild", summary="Rebuild the route model with CMS-approved examples; promote only if not worse")
    def rebuild(request: Request, _client=Depends(guard)):
        r = retrainer(request)
        started = r.start("manual")
        return JSONResponse(status_code=202 if started else 409,
                            content={"started": started, "status": r.status})

    @router.get("/rebuild", summary="Progress / result of the last rebuild")
    def rebuild_status(request: Request, _client=Depends(guard)) -> dict:
        return retrainer(request).status

    @router.get("/telemetry", summary="Decision telemetry queue: pending, sent, dropped, last error; doctor lookups")
    def telemetry(request: Request, _client=Depends(guard)) -> dict:
        rec = getattr(request.app.state, "telemetry", None)
        out = rec.snapshot() if rec else {"enabled": False}
        directory = getattr(request.app.state, "doctors", None)
        out["doctors"] = {"enabled": bool(directory and directory.enabled), **(directory.stats if directory else {})}
        return out

    return router
