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

from hms_router.adapters.http import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=5005, reload=True)
