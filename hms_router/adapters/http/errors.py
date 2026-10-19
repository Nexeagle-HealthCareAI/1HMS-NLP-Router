"""RFC 9457 problem+json for every error, so callers parse one error shape everywhere."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("hms_router.http")

_TITLES = {400: "Bad request", 401: "Unauthorized", 403: "Forbidden", 404: "Not found", 413: "Payload too large",
           422: "Invalid request", 429: "Too many requests", 500: "Internal error", 503: "Service unavailable"}


def problem(status: int, detail: str, request: Request | None = None, headers: dict | None = None,
            errors: list | None = None) -> JSONResponse:
    body = {"type": "about:blank", "title": _TITLES.get(status, "Error"), "status": status, "detail": detail}
    if request is not None:
        body["instance"] = request.url.path
        rid = getattr(request.state, "request_id", None)
        if rid:
            body["requestId"] = rid
    if errors:
        body["errors"] = errors
    return JSONResponse(body, status_code=status, headers=headers, media_type="application/problem+json")


def install(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else _TITLES.get(exc.status_code, "Error")
        return problem(exc.status_code, detail, request, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errs = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
        return problem(422, "Request body failed validation", request, errors=errs)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return problem(500, "An unexpected error occurred. Please try again.", request)
