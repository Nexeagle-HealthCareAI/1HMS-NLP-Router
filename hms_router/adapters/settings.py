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
    VITA_LLM_PROVIDER       empty (Vita on its rules engine) | groq | anthropic | openai - Vita's conversation
                            agent on LangGraph (hms_router/vita_graph); groq needs GROQ_API_KEY
    VITA_LLM_MODEL          default openai/gpt-oss-120b on groq; VITA_LLM_TIMEOUT seconds per call (default 6)
    REDIS_URL               required when HMS_ENV=production (sessions, rate limits)
    HMS_CMS_API_URL         CMS API base incl. version, e.g. https://cms-api.example.com/api/v1 - enables the
                            AI Agents dashboard feed (decision telemetry) and training on CMS-approved examples
    HMS_CMS_SERVICE_KEY     the CMS's ServiceAuth:SymptomRouterServiceKey (sent as X-Service-Key)
    HMS_TELEMETRY_HASH_KEY  key for the query hashes in telemetry (default: the service key)
    HMS_REBUILD_ON_START    1 (default when the CMS is configured) | 0 - rebuild once after start-up so a new
                            deployment includes examples approved since the image was built
    HMS_DOCTORS_MCP_URL     1HMSAPI's MCP endpoint, e.g. https://1hms-api.nexeagle.com/mcp - with the key, a
                            confident /chat recommendation also lists bookable doctors (hms_router/doctors.py)
    HMS_DOCTORS_MCP_KEY     this router's key in 1HMSAPI's Mcp:ServiceKeys (sent as X-Service-Key)
                            (HMS_MCP_URL / HMS_MCP_KEY are accepted as the same settings)
    HMS_DOCTORS_LIMIT       doctors per reply, 1-10 (default 3); HMS_DOCTORS_TIMEOUT seconds (default 2.0)
    HMS_PUBLIC_API_URL      1HMS public API base for Vita booking (default: HMS_DOCTORS_MCP_URL without /mcp)
    HMS_PUBLIC_API_KEY      1HMS public API key (X-Api-Key, the same kind Doctor Dekho uses); unset = booking OFF
    HMS_PROXY_FORWARDING_SECRET  1HMSAPI Internal:ProxyForwardingSecret: lets 1HMS rate-limit by the visitor's
                            IP (sent by the channel as client_ip) instead of the router's one IP
    HMS_BOOKING_TIMEOUT     seconds per 1HMS booking call (default 8)
    API key scope "admin"   POST/GET /admin/rebuild and GET /admin/telemetry (give it to the CMS's key)
    API key scope "verified_mobile"  the channel already verified the patient's number (the WhatsApp bot): its
                            verified_mobile is trusted for booking without an OTP
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
    vita_llm_provider: str | None = None
    vita_llm_model: str | None = None
    vita_llm_timeout: float = 6.0
    redis_url: str | None = None
    cms_api_url: str | None = None
    cms_service_key: str | None = None
    telemetry_hash_key: str | None = None
    rebuild_on_start: bool = True
    doctors_mcp_url: str | None = None
    doctors_mcp_key: str | None = None
    doctors_limit: int = 3
    doctors_timeout: float = 2.0
    public_api_url: str | None = None
    public_api_key: str | None = None
    proxy_forwarding_secret: str | None = None
    booking_timeout: float = 8.0
    extra: dict = field(default_factory=dict)

    @property
    def cms_configured(self) -> bool:
        return bool(self.cms_api_url and self.cms_service_key)

    @property
    def doctors_configured(self) -> bool:
        return bool(self.doctors_mcp_url and self.doctors_mcp_key)

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
            vita_llm_provider=(e.get("VITA_LLM_PROVIDER") or "").strip().lower() or None,
            vita_llm_model=(e.get("VITA_LLM_MODEL") or "").strip() or None,
            vita_llm_timeout=float(e.get("VITA_LLM_TIMEOUT", "6.0")),
            redis_url=e.get("REDIS_URL") or None,
            cms_api_url=(e.get("HMS_CMS_API_URL") or "").strip() or None,
            cms_service_key=(e.get("HMS_CMS_SERVICE_KEY") or "").strip() or None,
            telemetry_hash_key=(e.get("HMS_TELEMETRY_HASH_KEY") or "").strip() or None,
            rebuild_on_start=(e.get("HMS_REBUILD_ON_START") or "1").strip() not in ("0", "false", "no"),
            doctors_mcp_url=(e.get("HMS_DOCTORS_MCP_URL") or e.get("HMS_MCP_URL") or "").strip() or None,
            doctors_mcp_key=(e.get("HMS_DOCTORS_MCP_KEY") or e.get("HMS_MCP_KEY") or "").strip() or None,
            doctors_limit=int(e.get("HMS_DOCTORS_LIMIT") or "3"),
            doctors_timeout=float(e.get("HMS_DOCTORS_TIMEOUT") or "2.0"),
            public_api_url=(e.get("HMS_PUBLIC_API_URL") or "").strip() or None,
            public_api_key=(e.get("HMS_PUBLIC_API_KEY") or "").strip() or None,
            proxy_forwarding_secret=(e.get("HMS_PROXY_FORWARDING_SECRET") or "").strip() or None,
            booking_timeout=float(e.get("HMS_BOOKING_TIMEOUT") or "8"),
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
        if not 1 <= self.doctors_limit <= 10:
            raise RuntimeError("HMS_DOCTORS_LIMIT must be 1-10")
