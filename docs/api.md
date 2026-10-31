# HTTP API

One FastAPI service: `api.py` → `hms_router.adapters.http.create_app()` (routers in `hms_router/adapters/http/`).
The old `api_legacy/` app and `nlp_brain` are gone; this page replaces their description.

| Where | Address |
|---|---|
| Local | `uvicorn api:app --reload --port 5005` (no auth, in-memory Redis, LLMs off) |
| Dev VM | `http://151.185.45.77:5003` (container `nlp-router-dev`, `--network host`, `HMS_PORT=5003`) |
| Prod VM | `https://nlp-api.nexeagle.com` (Caddy → `localhost:5003`, container `nlp-router-prod`) |

The image defaults to port 5005; the VMs run it on 5003 (`deploy-router.yml`), the port every caller uses.
Interactive docs: `/docs` on any instance.

## Auth, limits, errors

- **API key** in `X-Api-Key`. Only SHA-256 digests are configured, in `HMS_API_KEYS`
  (`client:scope1|scope2:sha256hex`, comma-separated). Make one with
  `python scripts/make_api_key.py <client> <scope>...`. With no keys configured, auth is off (development only;
  production refuses to start).
- **Scopes:** `route` (specialty routing and the search bar), `chat` (Vita), `admin` (rebuild, telemetry, Vita's control),
  `verified_mobile` (a channel that already verified the patient's number, e.g. WhatsApp; lets Vita skip the OTP).
- **Rate limits** per client per minute, in Redis: `route` 120 (`HMS_RATE_LIMIT_ROUTE`), `chat` 30
  (`HMS_RATE_LIMIT_CHAT`). Over the limit → `429`.
- **Body cap** 16 KB (`HMS_MAX_BODY_BYTES`) → `413`. Every response carries `X-Request-Id`.
- **Errors** are RFC 9457 `application/problem+json`: `{type, title, status, detail, instance}` (+ `errors` on 422).

## Endpoints

| Method and path | Scope | What it does |
|---|---|---|
| `POST /v2/agent/chat` | chat | Vita, the patient agent: find a doctor, book, emergencies, visit help. See [vita_doctor_search.md](vita_doctor_search.md) and [vita_whatsapp.md](vita_whatsapp.md) |
| `GET /v2/agent/session/{id}`, `DELETE /v2/agent/session/{id}` | chat | What Vita remembers / forget it |
| `POST /v2/agent/feedback` | chat | 👍/👎 on a conversation (Vita's learning) |
| `DELETE /v2/agent/patient-memory` | chat | Forget a returning patient (token, or verified mobile with that scope) |
| `POST /v2/agent/search-intent` | route | One search-bar query → what Vita reads: specialty or symptoms (with route-symptom's decision), doctor name, degree, hospital, city, near me, emergency. See [vita_graph.md](vita_graph.md#the-search-bar) |
| `GET/PUT /v2/agent/admin/control` | admin | Vita's mode, per channel: `agent` (the model labels, code picks the move), `planner` (the model plans with tools around the patient's needs) or `off` (code steps and the fallback only) |
| `GET /v2/agent/admin/conversations`, `GET /v2/agent/admin/conversations/{id}` | admin | Review conversations by outcome; one conversation turn by turn |
| `GET/POST /v2/agent/admin/examples`, `DELETE /v2/agent/admin/examples/{id}`, `POST /v2/agent/admin/examples/promote` | admin | Approve or correct turns as examples; make them live through the scenario-suite gate. See [vita_graph.md](vita_graph.md) |
| `POST /v1/route-symptom` (alias `POST /route-symptom`) | route | Symptom text → ranked specialties + emergency flag. See [route_symptom.md](route_symptom.md) |
| `POST /legacy/route-symptom` | route | Same engine, the old `{specialtyIds, noMatch, method, confidence, modelVersion, raw}` shape |
| `GET /v1/taxonomy` | route | The 30 specialty ids and names |
| `GET /model-info` | route | Version and validation metrics of the serving model |
| `POST /chat` | chat | **Deprecated** alias of `POST /v2/agent/chat` (Vita answers). The old triage agent, `/feedback` and `/session/{id}` were removed on 2026-10-31 |
| `POST /admin/rebuild` | admin | Start a rebuild with CMS-approved examples; promoted only if not worse. `202` started, `409` already running |
| `GET /admin/rebuild` | admin | Progress and last result (`state` idle / running / done / failed, `promoted`, `verdict`, `validation`) |
| `GET /admin/telemetry` | admin | Decision telemetry queue and doctor-lookup stats |
| `GET /health` | none | Liveness, and whether Vita's LLM is on: `vita: {llm, model, mode, breakerOpen}` |
| `GET /ready` | none | Readiness: model loaded and Redis reachable (`503` until then) |

### `POST /v1/route-symptom`

```json
{"query": "daant mein bahut dard hai", "topK": 5, "context": {"ageGroup": "adult"}}
```

`query` 1–1000 characters (no control characters); `topK` 1–10 (default 5); `context.ageGroup` `child` | `adult`,
also the answer to a clarify of kind `age`. Add `?include=evidence` to get the labelled sentences behind each score.

```json
{
  "decision": "answer",
  "specialties": [{"specialtyId": "dentistry", "name": "Dentistry", "confidence": 0.98, "evidence": null}],
  "emergency": {"flag": false, "reasons": [], "source": null, "score": 0.01, "recommendedSpecialtyIds": []},
  "clarify": null, "reason": "…", "method": "deterministic", "decisive": true, "noMatch": false,
  "matchScore": 0.71, "specialtyIds": ["dentistry"], "usedDefault": false, "confidence": 0.98,
  "modelVersion": "2026-10-30-133204-fe4967a5", "requestId": "…"
}
```

Read `emergency.flag` first, even when `noMatch` is true. `decision`: `answer` (act on `specialties[0]`),
`clarify` (ask `clarify.question`, offer `clarify.options`, or show the hedge list), `abstain` (nothing medical
recognised: suggest a General Physician).

### `POST /v2/agent/chat`

```json
{"text": "mujhe 3 din se bukhar hai", "session_id": null, "city": "Purnia", "channel": "web"}
```

Optional: `lat`/`lng`, `hospital_id`, `select_doctor_id`, `page`, `ui_language`, `channel` (`web` | `whatsapp`),
`reply_id` (a tapped WhatsApp row), `client_ip`, `patient_token`, `verified_mobile`, `visit_appointment_id`,
`new_search`, `remember` (the patient agreed that Vita may remember who they book for and past doctors). The reply has `kind` (`follow_up`, `emergency`, `recommendation`, `doctor_list`, `doctor_selected`,
`info`, `handoff`, booking and visit kinds), `message`, `clarify` (tap options), `doctors`, `card`,
`language`, `engine` (`step`, `agent`, `agent+step`, `agent+template`, `planner`, `planner+step`, `planner+template`, `fallback`; see [vita_graph.md](vita_graph.md)) and, for WhatsApp, `whatsapp`
(ready-to-send messages).

## Changing the API

Add the route to the router module of its concern in `hms_router/adapters/http/` (business logic belongs in
`hms_router/application/`), give it a scope through `Guard`, and add a test with FastAPI's `TestClient`
(`tests/test_http_api.py`, `tests/test_api.py`). Contract changes for existing callers (Doctor Dekho, the CMS,
the WhatsApp bot) must be additive.
