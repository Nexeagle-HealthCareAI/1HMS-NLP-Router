# api/ — the FastAPI layer

HTTP contract + orchestration **only**. This layer loads one
`nlp_brain.SymptomClassifier` at startup and translates HTTP requests into
`classifier.predict()` calls — no TF-IDF/gibberish/coverage-gate logic is
reimplemented here. If you find yourself writing ML logic in this
directory, it belongs in `nlp_brain/` instead (see
[docs/nlp_brain.md](nlp_brain.md)).

Entry point: `api.main:app` — run it with `uvicorn api.main:app`.

## Module map

| File | Responsibility | Touch it when... |
|---|---|---|
| `main.py` | Creates the `FastAPI` app, wires the `lifespan` (loads the classifier once at startup), attaches the rate limiter | Changing how/when the model is loaded, or adding another cross-cutting concern (auth, CORS, etc. — see the security note in `docs/testing.md`'s caveats). |
| `routes.py` | The three endpoints; translates `PredictionResult` → `RouteResponse` | Adding a new endpoint, or changing what `/route-symptom` returns. |
| `schemas.py` | `RouteRequest`/`RouteResponse` pydantic models, `MAX_QUERY_LENGTH` | Changing the request/response contract NexEagleWebsite and `voice/` depend on. |
| `rate_limit.py` | The shared `slowapi.Limiter` instance | Changing the rate limit (currently 30/min per IP, set in `routes.py`'s `@limiter.limit("30/minute")` decorator) or the key function (currently raw client IP — see the caveat in `main.py`'s comment about reverse proxies). |

## Endpoints

### `GET /health`
Liveness/readiness probe. `ready: false` only in the narrow window before
startup finishes loading the classifier. What `deploy-nlp.yml`'s
health-check step polls after every deploy.

### `GET /model-info`
Returns `model_meta.json` as-is (model version, last retrain time,
validation metrics). Read fresh on every call, not cached — so it always
reflects the latest promoted model even without a restart.

### `POST /route-symptom`
```json
// request
{"query": "Dant mein bahut dard hai, thanda garam nahi sehta"}
```
```json
// response
{
  "specialtyIds": ["dentistry"],
  "noMatch": false,
  "method": "classifier",
  "confidence": 0.65,
  "modelVersion": "2026-08-04-083503",
  "raw": {
    "specialist": "Dentist",
    "matchRatio": 0.65,
    "closestKnownExample": "Daant mein bahut dard hai...",
    "flaggedGibberish": false,
    "message": null
  }
}
```
`specialtyIds` has at most one entry (mapped from the Brain's internal
label via `specialty_mapping.LABEL_TO_NEXEAGLE_SPECIALTY_ID`). `noMatch:
true` with empty `specialtyIds` means gibberish or "too dissimilar from
anything trained on" — there is deliberately no default-specialist
fallback. Rate-limited to 30 requests/minute per client IP; a `429` means
the caller needs to back off, not retry immediately.

## Common recipes

**Run locally against the committed model:**
```bash
uvicorn api.main:app --reload --port 5003
```

**Add a new endpoint:** add the route function to `routes.py`, add its
response shape to `schemas.py` if it returns structured data, then add a
test to `tests/test_api.py` using the `api_client` fixture (see
[docs/testing.md](testing.md) — it's wired to a small fixture classifier,
not the production bundle, so it stays fast).

**Change the rate limit:** edit the `@limiter.limit("30/minute")` string
in `routes.py`. `tests/test_api.py::TestRateLimit` asserts the exact
current number — update that test in the same change.

**Test against a specific model version:** point `MODEL_BUNDLE_PATH`
elsewhere (it's derived from `nlp_brain.MODEL_OUT` in `main.py`) — most
often easier to just swap `symptom_specialist_classifier.joblib` on disk
and restart.
