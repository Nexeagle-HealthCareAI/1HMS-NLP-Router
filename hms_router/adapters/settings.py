"""Service settings from the environment - the one place env vars are read for the HTTP service.

    HMS_ENV                 development (default) | production
    HMS_API_KEYS            comma-separated  client:scope1|scope2:sha256hex  (make one: python scripts/make_api_key.py)
                            scopes: route (route-symptom, taxonomy, model-info), chat (chat, feedback, session)
                            empty -> auth OFF; refused when HMS_ENV=production
    HMS_RATE_LIMIT_ROUTE    requests per minute per client on /route-symptom (default 120)
    HMS_RATE_LIMIT_CHAT     requests per minute per client on /chat and /feedback (default 30)
    HMS_CORS_ORIGINS        comma-separated browser origins; empty (default) = no CORS headers
    HMS_MAX_BODY_BYTES      request body cap (default 16384)
    HMS_ROUTE_MODEL         model file path (default hms_router/data/route_model.joblib; built on first start)
    HMS_LLM_PROVIDER        empty (default, reasoner OFF) | groq | anthropic | openai
    HMS_LLM_MODEL           provider model name; HMS_LLM_TIMEOUT seconds (default 2.0)
    REDIS_URL               required when HMS_ENV=production (sessions, rate limits)
    HMS_CMS_API_URL         CMS API base incl. version, e.g. https://cms-api.example.com/api/v1 - enables the
                            AI Agents dashboard feed (decision telemetry) and training on CMS-approved examples
    HMS_CMS_SERVICE_KEY     the CMS's ServiceAuth:SymptomRouterServiceKey (sent as X-Service-Key)
    HMS_TELEMETRY_HASH_KEY  key for the query hashes in telemetry (default: the service key)
    HMS_REBUILD_ON_START    1 (default when the CMS is configured) | 0 - rebuild once after start-up so a new
                            deployment includes examples approved since the image was built
    API key scope "admin"   POST/GET /admin/rebuild and GET /admin/telemetry (give it to the CMS's key)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ApiClient:
    name: str
    scopes: frozenset[str]
    key_sha256: str


def _parse_keys(raw: str) -> tuple[ApiClient, ...]:
    clients = []
    for item in filter(None, (x.strip() for x in raw.split(","))):
        parts = item.split(":")
        if len(parts) != 3 or len(parts[2]) != 64:
            raise ValueError("HMS_API_KEYS entries must be client:scope1|scope2:<64-hex sha256>")
        name, scopes, digest = parts
        clients.append(ApiClient(name, frozenset(s for s in scopes.split("|") if s), digest.lower()))
    return tuple(clients)


@dataclass(frozen=True)
class ServiceSettings:
    env: str = "development"
    api_clients: tuple[ApiClient, ...] = ()
    rate_limit_route: int = 120
    rate_limit_chat: int = 30
    cors_origins: tuple[str, ...] = ()
    max_body_bytes: int = 16_384
    route_model_path: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_timeout: float = 2.0
    redis_url: str | None = None
    cms_api_url: str | None = None
    cms_service_key: str | None = None
    telemetry_hash_key: str | None = None
    rebuild_on_start: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def cms_configured(self) -> bool:
        return bool(self.cms_api_url and self.cms_service_key)

    @property
    def production(self) -> bool:
        return self.env == "production"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_clients)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "ServiceSettings":
        e = os.environ if env is None else env
        s = cls(
            env=(e.get("HMS_ENV") or "development").strip().lower(),
            api_clients=_parse_keys(e.get("HMS_API_KEYS", "")),
            rate_limit_route=int(e.get("HMS_RATE_LIMIT_ROUTE", "120")),
            rate_limit_chat=int(e.get("HMS_RATE_LIMIT_CHAT", "30")),
            cors_origins=tuple(o.strip() for o in e.get("HMS_CORS_ORIGINS", "").split(",") if o.strip()),
            max_body_bytes=int(e.get("HMS_MAX_BODY_BYTES", "16384")),
            route_model_path=e.get("HMS_ROUTE_MODEL") or None,
            llm_provider=(e.get("HMS_LLM_PROVIDER") or "").strip().lower() or None,
            llm_model=e.get("HMS_LLM_MODEL") or None,
            llm_timeout=float(e.get("HMS_LLM_TIMEOUT", "2.0")),
            redis_url=e.get("REDIS_URL") or None,
            cms_api_url=(e.get("HMS_CMS_API_URL") or "").strip() or None,
            cms_service_key=(e.get("HMS_CMS_SERVICE_KEY") or "").strip() or None,
            telemetry_hash_key=(e.get("HMS_TELEMETRY_HASH_KEY") or "").strip() or None,
            rebuild_on_start=(e.get("HMS_REBUILD_ON_START") or "1").strip() not in ("0", "false", "no"),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if self.production and not self.api_clients:
            raise RuntimeError("HMS_ENV=production requires HMS_API_KEYS (auth cannot be off in production)")
        if self.production and not self.redis_url:
            raise RuntimeError("HMS_ENV=production requires REDIS_URL (no in-memory fallback in production)")
        if self.production and "*" in self.cors_origins:
            raise RuntimeError("HMS_CORS_ORIGINS='*' is not allowed in production")
        if self.rate_limit_route < 1 or self.rate_limit_chat < 1:
            raise RuntimeError("rate limits must be >= 1 per minute")
