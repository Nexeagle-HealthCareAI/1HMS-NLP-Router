"""API-key authentication, per-client rate limiting, request ids and a body-size cap.

Keys: only SHA-256 digests are configured (HMS_API_KEYS); the raw key is compared in constant time.
Rate limits: fixed one-minute windows per client in Redis, so they hold across workers and behind Caddy
(per-IP limits would see every request as coming from localhost there).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from dataclasses import dataclass

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..settings import ApiClient, ServiceSettings

log = logging.getLogger("hms_router.security")

ANONYMOUS = ApiClient("anonymous-dev", frozenset({"route", "chat"}), "")


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ApiKeyAuth:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        if not settings.auth_enabled:
            log.warning("HMS_API_KEYS not set - authentication is OFF (development only)")

    def authenticate(self, request: Request, scope: str) -> ApiClient:
        if not self.settings.auth_enabled:
            return ANONYMOUS
        raw = request.headers.get("x-api-key", "")
        if not raw:
            raise HTTPException(401, "Missing X-Api-Key header", headers={"WWW-Authenticate": "ApiKey"})
        digest = hash_key(raw)
        client = next((c for c in self.settings.api_clients if hmac.compare_digest(c.key_sha256, digest)), None)
        if client is None:
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "ApiKey"})
        if scope not in client.scopes:
            raise HTTPException(403, f"API key not allowed to use '{scope}'")
        return client


class RateLimiter:
    def __init__(self, redis_client, limits: dict[str, int]):
        self.r, self.limits = redis_client, limits

    def check(self, client: ApiClient, scope: str, ip: str | None = None) -> None:
        limit = self.limits.get(scope)
        if not limit:
            return
        who = client.name if client is not ANONYMOUS else f"ip:{ip or 'unknown'}"
        window = int(time.time() // 60)
        key = f"rl:{scope}:{who}:{window}"
        try:
            pipe = self.r.pipeline()
            pipe.incr(key)
            pipe.expire(key, 70)
            count = int(pipe.execute()[0])
        except Exception:                      # Redis down: fail open for availability, but say so loudly
            log.exception("rate limiter unavailable")
            return
        if count > limit:
            retry = 60 - int(time.time()) % 60
            raise HTTPException(429, f"Rate limit of {limit}/minute exceeded", headers={"Retry-After": str(retry)})


@dataclass(eq=False)          # hashable by identity: FastAPI caches dependencies by key
class Guard:
    """FastAPI dependency: authenticate + rate-limit for one scope. Stores the client on request.state."""
    auth: ApiKeyAuth
    limiter: RateLimiter
    scope: str

    def __call__(self, request: Request) -> ApiClient:
        client = self.auth.authenticate(request, self.scope)
        self.limiter.check(client, self.scope, request.client.host if request.client else None)
        request.state.client = client.name
        return client


class RequestContextMiddleware(BaseHTTPMiddleware):
    """X-Request-Id in/out, and a hard cap on request body size (checked on Content-Length and stream)."""

    def __init__(self, app, max_body_bytes: int):
        super().__init__(app)
        self.max_body = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        rid = rid if rid and len(rid) <= 64 and rid.replace("-", "").isalnum() else uuid.uuid4().hex
        request.state.request_id = rid
        from .errors import problem
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > self.max_body:
            resp = problem(413, f"Body larger than {self.max_body} bytes", request)
        else:
            body = await request.body()
            if len(body) > self.max_body:
                resp = problem(413, f"Body larger than {self.max_body} bytes", request)
            else:
                resp = await call_next(request)
        resp.headers["X-Request-Id"] = rid
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp
