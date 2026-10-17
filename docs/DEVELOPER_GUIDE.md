# EasyHMS NLP Symptom Router — Developer Guide

**Purpose of this document:** a single, complete reference to this repository —
what it does, how it's built, every API contract, the ML internals, and how
it's deployed — written for a developer joining the project with zero prior
context. Sections are self-contained and ordered so they can be read straight
through or split apart (e.g. into slides): overview → architecture → NLP
brain → API → voice/speech → data pipeline → testing → deployment →
production status.

> Diagram note (for anyone converting this into slides/a visual doc): each
> major section names the diagram it implies (architecture layers, request
> flow, CI/CD pipeline stages, deployment topology). A consolidated list is
> at the very end under **"Diagrams worth drawing."**

---

## 1. What this repository does

**One sentence:** given a free-text symptom description in Hinglish
(Hindi-English code-mixed text — the way people actually type/speak in
India), it returns which kind of doctor/specialist to route the patient to.

**Example:**
```
Input:  "pet mein dard hai aur sar bhi dukh raha hai"
        (stomach pain AND a headache — two problems in one sentence)
Output: specialtyIds: ["gastroenterology", "general-physician"]
```

It exists to power symptom-based doctor search for **Doctor Dekho /
NexEagleWebsite**. A patient describes what's wrong in their own words
(typed or spoken); this service figures out which specialist(s) to show
them, instead of the patient having to already know the right medical
category.

### Core design decisions, and why

| Decision | Why |
|---|---|
| Splits multi-symptom sentences into segments and classifies each independently | One sentence often names more than one problem ("pet mein dard **aur** sar bhi dukh raha hai"). Classifying the whole sentence as one blob would only ever produce one specialist, silently dropping the second complaint. |
| Surfaces a close runner-up specialist alongside the top pick (up to 3 total) | Forcing a single confident-looking answer on a genuinely ambiguous query is worse than admitting there are two reasonable answers. |
| Refuses to guess on gibberish or low-similarity input ("No matches found.") | There is deliberately **no default-specialist fallback**. A wrong guess sends a patient to the wrong doctor; an honest "we're not sure" lets the frontend ask a follow-up question instead. |
| Three independently-deployable layers (Voice / API / NLP Brain) | A microphone-capture layer and an ML layer have almost nothing in common operationally (one needs PortAudio, one needs scikit-learn) — bundling them would mean every deploy drags dependencies neither part actually needs, and neither layer can be developed/tested/scaled independently. |

---

## 2. Architecture — the four packages

```
voice/  --HTTP-->  api/  -->  nlp_brain/
              \              /
               \            /
                >  speech/ <
```
*(Diagram opportunity: render this as three stacked boxes — Voice, API, NLP
Brain — connected top-to-bottom by arrows labeled "HTTP" and "function call"
respectively, with a fourth box, Speech, drawn to the side with arrows
pointing INTO both Voice and API, not between them.)*

| Layer | Package | Owns | Never imports |
|---|---|---|---|
| **NLP Brain** | `nlp_brain/` | All ML logic: data loading, feature extraction, model training/selection, gibberish detection, the cosine-similarity coverage gate, multi-symptom segmentation, candidate ranking, prediction. | FastAPI, HTTP, microphones — zero knowledge that it's being called over a network at all. |
| **API** | `api/` | HTTP contract + orchestration only: loads one trained model at startup, turns requests into `classifier.predict()` calls, maps internal labels to NexEagleWebsite's specialty slugs, rate limiting. | No ML logic — if you catch yourself writing TF-IDF/classification code inside `api/`, it belongs in `nlp_brain/` instead. |
| **Voice** | `voice/` | Live microphone capture → HTTP call into `api/`. Can run on a completely different machine (one with a mic, none of the ML dependencies). | `nlp_brain` — the *only* way this layer reaches the Brain is through the FastAPI HTTP contract, never a direct import. |
| **Speech** (shared utility, not a "layer") | `speech/` | Audio → Roman-script text: STT transcription + Devanagari→Roman transliteration. Used by **both** `voice/` (live mic) and `api/` (`POST /route-symptom-audio`, for uploaded audio from a browser/mobile client that can't run this Python package). | Depends on neither `voice/` nor `api/` — exists as its own package specifically so neither layer has to depend on the other just to share this logic. |

**The rule that keeps this from rotting:** each layer only ever calls the one
"below" it in the diagram. `nlp_brain` doesn't know `api` exists; `api`
doesn't know `voice` exists. This is enforced by convention/code review, not
a lint rule — when adding code, ask "which layer does this belong to" before
"where's convenient to put it."

### Why this specific split (history)

Earlier versions of this codebase had the voice-capture code import the ML
internals directly, or re-implement the entire train/predict pipeline
inline, bypassing the API. That meant: no independent deployability, no
shared rate limiting or input validation on requests coming from voice vs.
NexEagleWebsite, and multiple copies of prediction logic to keep in sync
every time it changed — which is exactly how a real bug shipped once (the
joblib model-bundle schema was hand-built as a dict literal in two different
places; one included a field the other didn't, and nothing caught the
mismatch until it crashed in production). `SymptomClassifier.save()`/`load()`
are now the *only* place that schema is defined, and `SymptomClassifier` is
the only implementation of the predict pipeline, for the same reason.

---

## 3. The NLP Brain (`nlp_brain/`) — deep dive

This is the ML core. **Zero FastAPI/voice/HTTP imports** — it only knows how
to turn text into a prediction.

### 3.1 Public API

```python
from nlp_brain import SymptomClassifier, PredictionResult, MODEL_OUT, MATCH_THRESHOLD, NO_MATCH_MESSAGE, clean_text, is_gibberish
```

| Method | When to use it |
|---|---|
| `SymptomClassifier.train(data_path)` | Training a new model from a CSV. Returns `(classifier, held_out_metrics)`. |
| `SymptomClassifier.load(path)` | Loading an already-trained bundle — what `api/main.py`'s startup does. |
| `classifier.save(path)` | Persisting a trained classifier — the **only** place the joblib bundle's schema is defined. |
| `classifier.predict(text)` | Getting a prediction. Returns a `PredictionResult`. |

```python
classifier = SymptomClassifier.load()
result = classifier.predict("dant mein bahut dard hai")

if result.specialist:
    print(result.specialist, result.match_ratio)
elif result.flagged_gibberish:
    print("nonsense input")
else:
    print("didn't recognize this — closest known:", result.closest_known_example)
```

### 3.2 The prediction pipeline, step by step

*(Diagram opportunity: this is the single most valuable flowchart in the
whole doc — a linear pipeline with a branch/exit at each gate.)*

```
Raw input text
      │
      ▼
1. clean_text()            — normalize whitespace/case, strip noise
      │
      ▼
2. split_segments()        — is this really 1 symptom or several?
      │                        ("pet mein dard aur sar dukh raha hai"
      │                         → ["pet mein dard", "sar dukh raha hai"])
      ▼
   FOR EACH SEGMENT:
      │
      ▼
3. is_gibberish() check    ── YES ──▶ flag this segment, no prediction
      │ NO
      ▼
4. TF-IDF vectorize          (char n-grams + word n-grams, via
      │                       build_feature_union())
      ▼
5. best_match() coverage gate — cosine similarity vs. every known
      │                          training example
      │
      ├── below MATCH_THRESHOLD (0.25) ──▶ "No matches found." (no guess)
      │
      ▼ above threshold
6. Classify                — best of LinearSVC / LogisticRegression /
      │                       ComplementNB (chosen at TRAINING time by
      │                       cross-validation, not per-request)
      ▼
7. build_candidates()      — expand top pick into a shortlist: include
      │                       a runner-up if it's within CANDIDATE_MARGIN
      │                       (0.12) of the top score
      ▼
   [per-segment result: specialist, candidates[], match_ratio, ...]

ALL SEGMENTS DONE
      │
      ▼
8. merge_candidates()      — combine every segment's candidates, dedupe,
                              re-cap at MAX_CANDIDATES (3) total
      │
      ▼
Final PredictionResult
```

### 3.3 Module map

| File | Responsibility | Touch it when... |
|---|---|---|
| `config.py` | `DATA_PATH`, `MODEL_OUT`, `MATCH_THRESHOLD` (0.25), `CANDIDATE_MARGIN` (0.12), `MAX_CANDIDATES` (3) | Changing the dataset path, or recalibrating the coverage-gate threshold — **read the big comment above `MATCH_THRESHOLD` first**, it documents exactly how that number was picked and why it can't be picked from `validation_set.csv`. |
| `text_utils.py` | `clean_text()`, `is_gibberish()` — pure functions, no model | Adding a new gibberish-detection rule. |
| `features.py` | `build_feature_union()` — the TF-IDF feature extractor | Changing n-gram ranges, adding a new feature type. |
| `matching.py` | `normalize_matrix()`, `best_match()` — the cosine-similarity coverage gate | Changing HOW similarity is computed. **Do not** call `cosine_similarity()` against a raw, non-pre-normalized corpus matrix inside a per-request path — that regressed `/route-symptom` to ~50ms/request once (see §3.5). |
| `candidates.py` | `ranked_labels()`, `build_candidates()`, `merge_candidates()` | Changing how many candidates get surfaced, or how a close runner-up is decided. |
| `segmentation.py` | `split_segments()` | Adding/removing a recognized separator word, or the "don't over-split a short fragment" guard. |
| `training.py` | `load_data()`, `evaluate_candidates()` | Adding a new candidate classifier algorithm to the training-time comparison. |
| `classifier.py` | `SymptomClassifier`, `PredictionResult` | Changing what a prediction returns, or the joblib bundle schema. |
| `cli.py` | `python -m nlp_brain.cli [train\|predict "<text>"]` | Script/REPL access without the API. |

### 3.4 Model selection

At **training time** (not per-request), three algorithms are cross-validated
against each other — LinearSVC, LogisticRegression, ComplementNB — and
whichever scores best is what actually gets saved into the joblib bundle.
LinearSVC has no `predict_proba`; when it wins, candidate ranking falls back
to a softmax of its `decision_function` output instead.

### 3.5 Performance: the coverage-gate regression (why `matching.py` looks the way it does)

`SymptomClassifier.__init__` precomputes and caches a **normalized** copy of
the entire training corpus matrix once, at load time. The reason this
matters: an earlier version re-normalized that matrix on **every** call to
`predict()`, which made `/route-symptom` take ~50ms server-side. Precomputing
it once brought that down to ~15ms. `tests/test_performance.py` is a
regression guard specifically so this can't silently come back.

### 3.6 Gibberish detection

Dependency-free regex/heuristic check (`text_utils.is_gibberish()`) — no
external "gibberish-detector" package, no `big.model` file. Runs **before**
the coverage-gate check, per segment.

### 3.7 Multi-candidate ranking — the bug this session fixed

`predict()` used to merge multiple segments' candidate lists with unbounded
dedup logic that didn't re-apply `MAX_CANDIDATES` after combining — a real
multi-symptom query could come back with 5 candidates (cap should be 3),
including an irrelevant specialist. Fixed by extracting a single
`merge_candidates(segment_candidates, MAX_CANDIDATES)` function
(`nlp_brain/candidates.py`) that both `predict()` and its tests go through —
verified live: the same query that used to return 5 now correctly returns
exactly 3. **This fix is on `develop` and is one of the things not yet on
`main`/Prod — see §9.**

---

## 4. The API layer (`api/`)

Entry point: `api.main:app` — run with `uvicorn api.main:app`.

### 4.1 Module map

| File | Responsibility |
|---|---|
| `main.py` | Creates the FastAPI app, `lifespan` (loads the classifier once at startup), attaches the rate limiter. |
| `routes.py` | The four endpoints. |
| `schemas.py` | Request/response Pydantic models, `MAX_QUERY_LENGTH` (1000 chars), `MAX_AUDIO_BYTES` (10MB). |
| `rate_limit.py` | Shared `slowapi.Limiter` instance, keyed by client IP. |

### 4.2 Endpoints — full contract

#### `GET /health`
Liveness/readiness probe.
```json
{"status": "ok", "ready": true}
```
`ready: false` only in the narrow window before startup finishes loading the
classifier.

#### `GET /model-info`
Read fresh from `model_meta.json` on every call (not cached at startup) —
always reflects the latest promoted model even without a restart.
```json
{
  "modelVersion": "2026-08-04-083503",
  "lastRetrainedAt": "2026-08-04T08:35:03.464205+00:00",
  "trainingRowCount": 15486,
  "validationRowCount": 1518,
  "validationMetrics": {
    "top1Accuracy": 0.9717,
    "noMatchRate": 0.0007,
    "confidentlyWrongRate": 0.0277
  }
}
```

#### `POST /route-symptom` — the main endpoint
Rate limit: **30 requests/minute per client IP**.

Request:
```json
{"query": "Dant mein bahut dard hai, thanda garam nahi sehta"}
```
(max 1000 characters — longer is rejected before it reaches the model)

Response:
```json
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

| Field | Meaning |
|---|---|
| `specialtyIds` | Ordered most-confident-first, mapped from internal labels to NexEagleWebsite slugs (`specialty_mapping.py`). Normally 1 entry, up to `MAX_CANDIDATES` (3) when a close runner-up exists. Empty array when `noMatch: true`. |
| `noMatch` | `true` means: gibberish, OR too dissimilar from anything trained on. **There is no default-specialist fallback** — the caller must handle this case explicitly (e.g. ask a follow-up question) rather than assume a specialist is always returned. |
| `confidence` | The top pick's cosine-similarity match ratio (0-1). `null` when `noMatch`. |
| `raw` | Debug-oriented internal detail — `raw.specialist` is the internal label (pre-mapping), `raw.closestKnownExample` shows what training example the input matched closest to, useful for understanding *why* a prediction (or rejection) happened. |

A `429` response means the caller hit the rate limit — back off, don't retry immediately.

#### `POST /route-symptom-audio` — voice counterpart
Rate limit: **10 requests/minute per client IP** (tighter — audio decoding +
speech recognition is real work on top of classification).

```
POST /route-symptom-audio
Content-Type: multipart/form-data

audio: <recording, any format ffmpeg can decode — webm/opus, m4a, wav, ...>
```
Max upload size: 10MB.

Response — same shape as `/route-symptom`, plus `transcript`:
```json
{
  "transcript": "Dant mein bahut dard hai",
  "specialtyIds": ["dentistry"],
  "noMatch": false,
  "method": "classifier",
  "confidence": 0.65,
  "modelVersion": "2026-08-04-083503",
  "raw": { "...": "same shape as /route-symptom" }
}
```
`transcript` is shown to the user so they can see/correct what the server
heard before trusting the routing decision. It's `null` (with `noMatch:
true`) when nothing intelligible could be transcribed — **not** an error
response, the same "honest no-answer" philosophy as unclear text input.

**Error responses:**

| Status | Meaning | Client should... |
|---|---|---|
| `400` | Uploaded bytes aren't decodable as audio at all | Fix the recording format |
| `413` | Audio over 10MB | Compress / shorten the recording |
| `502` | Speech recognition service itself failed (network, quota) | Retry later |
| `503` | Transcription unavailable on this server (e.g. `ffmpeg` missing) | Not the client's fault — an operational problem |

Intended for a browser (`MediaRecorder`) or mobile client that records its
own audio — `voice/`'s own CLI uses a live microphone directly and doesn't
need this endpoint (see §5).

### 4.3 Who calls this API

- **NexEagleWebsite / Doctor Dekho** → `POST /route-symptom` (text) and/or
  `POST /route-symptom-audio` (browser-recorded voice) for routing decisions.
- **`voice/`'s own CLI** → `POST /route-symptom` only, via
  `voice.api_client.SymptomRouterClient` (never `/route-symptom-audio` — it
  transcribes locally with a live mic and sends text).
- Any monitoring/dashboard → `GET /health`, `GET /model-info`.

---

## 5. Voice (`voice/`) and Speech (`speech/`)

### 5.1 `voice/` — microphone capture + API client

Entry point: `python -m voice.cli`. Needs `requirements-voice.txt` (not part
of the deployed API image — `PyAudio` is a live-microphone-only dependency
that has no reason to be dragged into a server container).

```
Microphone → speech_to_text.listen_from_microphone() → sr.AudioData
           → speech.transcribe_audio_data() → Roman-script text
           → is_stop_command() check (exit phrase?)
           → api_client.SymptomRouterClient.route_symptom(text)
           → POST /route-symptom on the API layer
           → print result to the user
```

| File | Responsibility |
|---|---|
| `config.py` | `API_BASE_URL` (env-overridable, default `http://127.0.0.1:5003`), `SPEECH_LANGUAGE`, `STOP_PHRASES` |
| `speech_to_text.py` | `listen_from_microphone()` — mic capture only, NOT transcription itself |
| `api_client.py` | `SymptomRouterClient` — the **only** path this layer has to the NLP Brain |
| `cli.py` | The mic → transcribe → API → print interactive loop |

Point at a specific API instance:
```bash
export NLP_API_BASE_URL=http://151.185.45.77:5003   # Dev
```

**Can a user see what they're saying transcribed?** Yes — the transcript is
surfaced back to them (both in the CLI's printed output and, for a web
client, via the `/route-symptom-audio` response's `transcript` field) before
the routing result, specifically so they can catch a mis-transcription
before trusting the specialist recommendation.

### 5.2 `speech/` — shared transcription + transliteration

Used by both `voice/` (live mic) and `api/` (`POST /route-symptom-audio`).
Lives in its own package specifically so neither of those two has to import
the other just to share it.

| File | Responsibility |
|---|---|
| `transliteration.py` | `devanagari_to_roman()` — normalizes Devanagari-script input to match the Roman-script Hinglish the model was trained on |
| `transcription.py` | `transcribe_audio_data()`, `transcribe_audio_file()`, the `TranscriptionError` hierarchy |

**Three failure modes** (why they're distinguished — see §4.2's error table):

| Exception | Meaning |
|---|---|
| `AudioUnintelligible` | Valid audio, nothing recognizable said |
| `AudioFormatError` | Bytes aren't decodable as audio at all |
| `SpeechServiceError` | The recognition service itself failed |

**Why `ffmpeg` is a hard dependency:** `SpeechRecognition` only reads
WAV/AIFF/FLAC natively; a browser's `MediaRecorder` typically produces
webm/opus. `pydub` normalizes everything to WAV first by shelling out to
`ffmpeg` — even already-WAV input, there's no format-sniffing shortcut. The
Docker image installs `ffmpeg` via `apt-get` specifically for this. If
`ffmpeg` is missing, this surfaces as a `503` at the API layer (an
operational problem, not the caller's fault).

---

## 6. Data pipeline & retraining (`data_pipeline/`)

Not part of the deployed API image, not one of the three runtime layers —
this is offline/batch tooling that produces the artifacts `api/` and
`nlp_brain/` consume at runtime.

### 6.1 `retrain_pipeline.py` — the active pipeline

Runs nightly via `.github/workflows/retrain.yml` (and on-demand via CMS's
"Retrain now" button / `workflow_dispatch`):

```
1. Fetch current CMS-editable training set + production feedback
        (from CMSAPI)
2. Merge + dedupe with existing training data
3. Train a CANDIDATE model via nlp_brain
4. Evaluate candidate against validation_set.csv (frozen)
5. Compare candidate's metrics vs. the CURRENTLY-DEPLOYED model's
        recorded metrics (model_meta.json)
6a. If candidate is not worse by more than TOLERANCE (0.01) on
        accuracy or wrong-rate → PROMOTE:
        overwrite Hinglish_Symptoms_V28.csv + model_meta.json +
        symptom_specialist_classifier.joblib, push to develop
        → triggers deploy-nlp.yml automatically
6b. Otherwise → REJECT: nothing is written, nothing is deployed
```

**Feedback weighting** (`feedback_to_rows()`): corrections (model suggested
X, patient actually booked Y) are weighted 2x; silent accepts (suggested X,
booked X) 0.5x; CMS-curated training examples are the 1x baseline.
Corrections are unambiguous signal; silent accepts are weaker (the patient
might just not have looked elsewhere).

**Known caveat:** `validation_set.csv` has ~97% row overlap with the
training data, which inflates its absolute accuracy numbers. It's still
useful as a **regression** gate (candidate vs. baseline on the same biased
yardstick is apples-to-apples) but not as an absolute quality measure.

### 6.2 Current operational issue (as of this document)

The nightly retrain workflow has been failing: `cms-api.nexeagle.com`'s
training-examples endpoint returns HTTP 404 (the base domain itself
responds, so it's not a total outage — just that specific route). This is
diagnosed as a **CMSAPI-side issue, not fixable from this repository** —
flagged here so a new developer doesn't spend time debugging retrain code
that isn't the actual problem.

---

## 7. Testing

**Mandatory after any code change** — enforced two ways:
1. **CI gate**: every push/PR runs the full suite; a failure blocks the
   Docker build/deploy entirely.
2. **Pre-commit hook** (`.githooks/pre-commit`, opt in via `git config
   core.hooksPath .githooks`): fast local feedback, bypassable with
   `--no-verify` (the CI gate is what actually can't be skipped).

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                          # everything
pytest -m "not integration"     # fast subset (pre-commit hook)
pytest -k gibberish              # anything matching "gibberish"
```

| Test file | Covers |
|---|---|
| `test_text_utils.py` | `clean_text()`, `is_gibberish()` |
| `test_matching.py` | Coverage gate, small hand-built matrices |
| `test_classifier.py` | `SymptomClassifier` train/predict/save/load + integration tests against the real bundle |
| `test_api.py` | Every endpoint via `TestClient`, including size/rate limits |
| `test_voice.py` | `is_stop_command()`, `SymptomRouterClient` (mocked HTTP) |
| `test_speech.py` | Transliteration, transcription (recognizer mocked) |
| `test_retrain_pipeline.py` | Pure merge/promotion decision logic |
| `test_performance.py` | Latency regression guard (see §3.5) |

Key fixture: `tiny_classifier` (`tests/conftest.py`) — a REAL
`SymptomClassifier` trained on a small synthetic ~32-row dataset, fast and
independent of the production dataset. Most tests use this, not the real
bundle.

**Deliberately not covered by automated tests:** live microphone capture,
real `ffmpeg` conversion (one self-skipping test exercises it if available),
actually calling Google's speech API (always mocked), live CMSAPI calls,
Docker/deployment itself (verified by watching the CI run, not a test).

---

## 8. Deployment & infrastructure

### 8.1 Docker image

One `Dockerfile` at repo root builds the deployed API image:
```
FROM python:3.12-slim
+ ffmpeg (apt-get)                    — for /route-symptom-audio
+ requirements.txt
+ nlp_brain/  api/  speech/            — the three deployed layers
+ specialty_mapping.py, model_meta.json, symptom_specialist_classifier.joblib
CMD uvicorn api.main:app --host 0.0.0.0 --port 5003
```
**`voice/` is never in this image** — it's a client package, not a server
component. **`location_API/` is never in this image either** — it's a
completely separate, independently-deployed service (own Dockerfile, own
CI/CD, own port). See §10.

### 8.2 CI/CD pipeline (`.github/workflows/deploy-nlp.yml`)

*(Diagram opportunity: 4-stage horizontal pipeline.)*

```
Stage 1: Build         pip install + import smoke test + full pytest suite
              │  (a failure here stops everything below)
              ▼
Stage 2: Docker         build image, push to GHCR
              │
              ▼
Stage 3: Deploy → Dev    (only on `develop` branch)
              │           SSH to Dev VM, docker pull + run, health check
              ▼
Stage 4: Deploy → Prod   (only on `main` branch)
                          SSH to Prod VM, docker pull + run, health check
```
Same pattern powers `.github/workflows/deploy-location-api.yml` for the
separate location_API service, and `retrain.yml` for nightly retraining
(which, on a successful promotion, pushes to `develop` and re-triggers this
same pipeline — promotion = a normal redeploy).

### 8.3 Where things actually run

*(Diagram opportunity: deployment topology — two VM boxes, each containing
multiple labeled service boxes, with the API layer's requests flowing in
from NexEagleWebsite.)*

| Environment | VM | NLP Router | Domain (HTTPS) |
|---|---|---|---|
| **Dev** | `151.185.45.77` | `http://151.185.45.77:5003` (raw IP:port only — no HTTPS domain set up for the NLP router on Dev) | — |
| **Prod** | `151.185.45.67` | `http://151.185.45.67:5003` | `https://nlp-api.nexeagle.com` ✅ live |

Both VMs run `docker run --network host`, matching the convention every
other EasyHMS backend service on these VMs uses. **Both VMs are shared,
multi-tenant machines** — the Dev VM alone also hosts (at time of writing):
`easyhms-api`, `cms-api`/`cms-web`, `nexmarket-api`/`nexmarket-web`,
`1rad-api`, a WhatsApp-webhook service, `nexeagle-website`, SQL Server,
two Redis instances, and a native (non-Docker, systemd-managed) Caddy
reverse proxy that terminates HTTPS for several of the above. This matters
operationally: **assume nothing about a shared VM without checking first**
— see the loc-dev.nexeagle.com incident in §10.2 for exactly why.

### 8.4 HTTPS / domain mapping — how it actually works

Both `nlp-api.nexeagle.com` (Prod) and `loc-dev.nexeagle.com` (Dev, for
location_API) are handled by **Caddy**, but the setup differs meaningfully
between the two VMs:

- **Prod**: Caddy already runs there (a Docker container, bind-mounted
  Caddyfile at `/opt/caddy/Caddyfile`, `--network host`), originally set up
  for `www.nexeagle.com`. Adding a new domain there is just: append a site
  block to the existing Caddyfile, `docker exec caddy caddy reload`. Caddy
  handles the Let's Encrypt certificate automatically. Documented step by
  step in [OPS-HTTPS-SETUP.md](OPS-HTTPS-SETUP.md).
- **Dev**: Caddy runs there too, but as a **native (non-Docker, systemd)
  process** — not a container, so `docker ps` never shows it. It's a
  *shared* reverse proxy for multiple Dev-VM services, not something
  scoped to this repo. Adding a domain here means finding that native
  process's actual config file (via its running command line, not assuming
  a path) and appending to *that*, then `systemctl reload caddy`.

**The NLP router itself has no Dev-side HTTPS domain today** — Dev
deliberately stayed on raw `http://151.185.45.77:5003` (a pattern shared
with `NexEagleWebsite`'s own Dev environment). Only `nlp-api.nexeagle.com`
(Prod) and `loc-dev.nexeagle.com` (Dev, location_API only) exist as HTTPS
domains right now.

---

## 9. Current status & what's pending for Prod

**As of this document, `develop` is 10 commits ahead of `main`** — nothing
in this list has been deployed to Prod yet. A PR/merge from `develop` →
`main` is what promotes it.

| # | Commit (develop) | What it does | Prod impact if not yet merged |
|---|---|---|---|
| 1 | `332672f` | Adds location_API + location_brain | location_API doesn't exist on Prod at all — confirmed live: `http://151.185.45.67:5004/health` currently times out. |
| 2 | `1b252d7` | Repo cleanup | — |
| 3 | `e3e510d` | Fixes broken NLP router Docker build, separates location_API into its own deploy, **fixes the MAX_CANDIDATES bug** (§3.7) | **Prod's `/route-symptom` may still exhibit the candidate-cap bug** — a multi-symptom query could return more than 3 candidates. |
| 4 | `0c09413` | Adds `GET/POST /locate` (smart unified location search) | `/locate` doesn't exist on Prod (location_API isn't deployed there at all yet). |
| 5 | `89618cc` | Fixes a CI path-filter gap | CI-only, no runtime impact |
| 6 | `c7bb59e`–`26b25eb` | Sets up `loc-dev.nexeagle.com` → location_API on Dev | Dev-only ops tooling; **Prod has no equivalent domain set up** for location_API (would need its own Caddy site block on Prod's existing Caddy — the Prod runbook pattern from §8.4 applies directly, this just hasn't been done because location_API isn't on Prod yet). |
| 7 | `64682e8` | Makes `/search` an advanced, full-coverage endpoint | N/A — location_API isn't on Prod. |

**Verified live right now:**
- ✅ Prod NLP router IS running and healthy (`https://nlp-api.nexeagle.com/health` → 200), but on **pre-reorg code** — specifically missing the MAX_CANDIDATES fix.
- ❌ Prod location_API is **not deployed** (port 5004 closed on the Prod VM).
- ⚠️ The nightly retrain pipeline has been failing due to an external CMSAPI 404 (§6.2) — unrelated to any of the above, but also still open.

**To promote everything to Prod:** merge `develop` → `main` (this repo's
`main` branch triggers Stage 4 of both `deploy-nlp.yml` and
`deploy-location-api.yml`). After that, location_API would be live on Prod
at `http://151.185.45.67:5004` but **without** an HTTPS domain until someone
explicitly sets one up on Prod's Caddy (a much simpler version of §10's Dev
story, since Prod's Caddy is already a known, documented, Docker-based
setup — see [OPS-HTTPS-SETUP.md](OPS-HTTPS-SETUP.md)).

---

## 10. location_API — the sibling service

A separate, independently-deployed service in the same repo
(`location_API/`) — Indian city/pincode/coordinate lookup, exposed to other
applications. **Not consumed by anything in `nlp_brain`/`api`/`voice`/
`speech`** — it lives in this repo for convenience (shared VMs, shared CI
patterns) but is architecturally unrelated to the symptom router. Its own
[README](../location_API/README.md) is the full reference; summarized here
for completeness since this guide covers "the whole repo":

- Own Dockerfile, own CI/CD (`deploy-location-api.yml`), own port (**5004**
  vs. the NLP router's 5003), same Dev/Prod VMs.
- Same SOLID layering pattern as `nlp_brain`: `repositories/` (one class per
  CSV) → `services/` → `finder.py` (facade) → `main.py` (FastAPI, HTTP only).
- Five endpoints: `GET /health`, `GET/POST /locate` (smart unified search —
  pincode/coordinates/free-text auto-detected), `GET/POST /find-pincode`,
  `GET/POST /coordinates`, `GET /search` (autocomplete, upgraded to full
  multi-dataset coverage — see its README for the current response shape).
- Currently Dev-only in practice: deployed and reachable at
  `https://loc-dev.nexeagle.com` (§8.4) and `http://151.185.45.77:5004`; not
  yet on Prod (§9).

### 10.1 Why it's not just part of the NLP router's API layer

It's a genuinely different product surface — meant to be called by other
applications independently of symptom routing — so it gets its own image,
port, and deploy lifecycle rather than being bundled into `api/`'s Docker
image (which would mean every location-data change forces an NLP-router
redeploy and vice versa).

### 10.2 A cautionary tale from setting up `loc-dev.nexeagle.com`

Worth including here because it's a real lesson about the shared Dev VM
(§8.3): the first attempt to map `loc-dev.nexeagle.com` to location_API
assumed no reverse proxy existed on the Dev VM (external probes on ports
80/443 both timed out) and started a brand-new Dockerized Caddy container.
That container immediately crash-looped — because a **native, non-Docker**
Caddy process was already running on that VM the whole time (serving other
Dev-VM services), and the new container fought it for an internal admin
port. Fixed by finding the real native process and appending to *its*
config instead of assuming "nothing responds externally" meant "nothing is
running." **Takeaway for future infra changes on either VM: verify what's
actually running (`docker ps -a` AND native/systemd processes) before
assuming a clean slate — these are shared, multi-tenant boxes with other
teams' services on them.**

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Hinglish | Hindi-English code-mixed text, written in Roman script (e.g. "pet mein dard hai") — the input language this model is trained on. |
| Coverage gate | The cosine-similarity check (`MATCH_THRESHOLD`) that decides whether the input is similar enough to something previously seen to trust a prediction at all. |
| Candidate / MAX_CANDIDATES | A specialist worth surfacing alongside the top pick, when its confidence is close enough (`CANDIDATE_MARGIN`) — capped at 3 total. |
| `specialtyId` slug | NexEagleWebsite's external identifier for a specialist type (e.g. `"cardiology"`) — distinct from this repo's internal training labels (e.g. `"Cardiologist (Heart)"`), mapped via `specialty_mapping.py`. |
| Promotion | A retrain that passed the regression gate and got written as the new live model (`data_pipeline/retrain_pipeline.py`). |
| GHCR | GitHub Container Registry — where built Docker images are pushed before deployment. |

---

## Diagrams worth drawing

If turning this into a slide deck / visual document, these are the specific
diagrams called out inline above, gathered in one place:

1. **Architecture layers** (§2) — Voice → API → NLP Brain stack, with Speech
   as a shared side-dependency feeding into both Voice and API (not between
   them).
2. **Prediction pipeline flowchart** (§3.2) — the single richest diagram in
   this document: raw text in, through clean → segment → per-segment
   (gibberish check → TF-IDF → coverage gate → classify → candidates) →
   merge/re-cap → final result out. Include the two "exit early" branches
   (gibberish, below-threshold) as distinct dead-ends, not just annotations.
3. **CI/CD pipeline** (§8.2) — 4-stage horizontal flow: Build → Docker →
   Deploy Dev → Deploy Prod, with the branch gating (`develop` vs. `main`)
   labeled on the last two stages.
4. **Deployment topology** (§8.3) — two VM boxes (Dev/Prod), each containing
   labeled service boxes (NLP router, location_API, and — for Dev — the
   other unrelated tenants like cms-api/nexmarket/etc. to make the "shared
   VM" point visually, not just in text), with the HTTPS domains routing in
   from outside.
5. **Request flow, text query** — NexEagleWebsite/Doctor Dekho →
   `POST /route-symptom` → API layer → NLP Brain → response back, as a
   simple left-to-right sequence diagram.
6. **Request flow, voice query** — two variants worth showing side by side:
   (a) `voice/` CLI: mic → local transcription → `POST /route-symptom` (text
   only); (b) browser client: mic → `POST /route-symptom-audio` (raw audio)
   → server-side transcription → routing, both converging on the same NLP
   Brain call.
