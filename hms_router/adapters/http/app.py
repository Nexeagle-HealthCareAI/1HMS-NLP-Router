"""FastAPI app factory: wires settings, Redis, services, security and routers. `api.py` calls create_app().

Each router owns its endpoints; this file owns only composition, so adding an endpoint never means
editing another endpoint's code.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ...agent import HMSRouterAgent
from ...application.chat import ChatService
from ...application.container import build_route_symptom_service
from ...application.retraining import Retrainer
from ...doctors import build_directory
from ...rag.model_bundle import MODEL_PATH, ModelBundle
from ...telemetry import build_recorder
from ..llm.factory import build_llm_completion
from ..settings import ServiceSettings
from . import admin_routes, chat_routes, errors, health_routes, route_routes
from .security import ApiKeyAuth, Guard, RateLimiter, RequestContextMiddleware

log = logging.getLogger("hms_router.api")


def _build_redis(settings: ServiceSettings):
    if settings.redis_url:
        import redis
        log.info("Connecting to Redis")
        return redis.Redis.from_url(settings.redis_url, decode_responses=True)
    import fakeredis
    log.warning("REDIS_URL not set - using fakeredis (in-memory, dev only)")
    return fakeredis.FakeRedis(decode_responses=True)


class _LazyLimiter:
    """The limiter needs Redis, which exists only after startup; routes are built before that."""

    def __init__(self, app: FastAPI):
        self.app = app

    def check(self, client, scope, ip=None):
        limiter = getattr(self.app.state, "limiter", None)
        if limiter is not None:
            limiter.check(client, scope, ip)


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.redis = _build_redis(settings)
        app.state.limiter = RateLimiter(app.state.redis, {"route": settings.rate_limit_route,
                                                          "chat": settings.rate_limit_chat})
        app.state.chat = ChatService(HMSRouterAgent(app.state.redis))
        log.info("HMSRouterAgent ready")
        app.state.telemetry = build_recorder(settings.cms_api_url, settings.cms_service_key, settings.telemetry_hash_key)
        app.state.telemetry.start()
        app.state.doctors = build_directory(settings.doctors_mcp_url, settings.doctors_mcp_key,
                                            limit=settings.doctors_limit, timeout=settings.doctors_timeout)
        llm_complete = build_llm_completion(settings)

        def promote(bundle: ModelBundle) -> None:
            bundle.save(settings.route_model_path or MODEL_PATH)
            app.state.route = build_route_symptom_service(bundle, llm_complete=llm_complete)
            app.state.route_meta = bundle.meta

        app.state.retrainer = Retrainer(
            build_service=lambda bundle, with_llm: build_route_symptom_service(
                bundle, llm_complete=llm_complete if with_llm else None),
            get_current=lambda: (getattr(app.state, "route", None), getattr(app.state, "route_meta", {}) or {}),
            promote=promote, recorder=app.state.telemetry,
            cms_url=settings.cms_api_url, cms_key=settings.cms_service_key)
        try:
            bundle = ModelBundle.load_or_build(settings.route_model_path)
            app.state.route = build_route_symptom_service(bundle, llm_complete=llm_complete)
            app.state.route_meta = bundle.meta
            log.info("RouteSymptomService ready (model %s, reasoner %s)", bundle.meta.get("modelVersion"),
                     type(app.state.route.reasoner).__name__)
        except Exception:                      # /chat keeps serving; /ready reports the missing model
            log.exception("route-symptom model failed to load")
            app.state.route, app.state.route_meta = None, {}
        if settings.cms_configured and settings.rebuild_on_start:
            app.state.retrainer.start("deploy")
        yield
        log.info("shutting down")
        app.state.telemetry.stop()

    app = FastAPI(
        title="HMS NLP Router API",
        description="Hinglish symptom text -> specialist routing -> emergency flagging. "
                    "**Triage support only - not a diagnosis.**",
        version="2.0.0", lifespan=lifespan,
    )
    errors.install(app)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_body_bytes)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST", "DELETE"], allow_headers=["X-Api-Key", "Content-Type",
                                                                                   "X-Request-Id"])
    auth = ApiKeyAuth(settings)
    limiter = _LazyLimiter(app)
    app.include_router(health_routes.build_router())
    app.include_router(chat_routes.build_router(Guard(auth, limiter, "chat")))
    app.include_router(route_routes.build_router(Guard(auth, limiter, "route")))
    app.include_router(admin_routes.build_router(Guard(auth, limiter, "admin")))
    app.state.settings = settings
    return app
