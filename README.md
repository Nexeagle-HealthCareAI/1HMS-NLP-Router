# EasyHMS NLP Symptom Router

Hinglish (Hindi-English code-mixed) symptom → specialist classifier for the Doctor
Dekho / NexEagleWebsite doctor search. Given a free-text query like *"pet mein dard
hai aur sar bhi dukh raha hai"* (which names two problems — stomach pain AND a
headache), splits it into per-symptom segments (`nlp_brain/segmentation.py`) and
routes each independently via TF-IDF (char+word n-gram) features and the best of
LinearSVC / LogisticRegression / ComplementNB (picked by cross-validation at
training time), merging the results into an ordered, deduped specialist list —
one query can come back with more than one recommended specialist, both because
of multi-symptom segmentation and because a single segment's own close runner-up
gets surfaced too (`nlp_brain/candidates.py`) rather than forcing an overconfident
single pick. Before trusting a prediction, each segment is checked for gibberish
(a dependency-free heuristic) and must have high enough cosine similarity to a
known training example — genuinely unclear or nonsense input gets "No matches
found." instead of a guess.

## Architecture: three independent layers + one shared utility

```
voice/  --HTTP-->  api/  -->  nlp_brain/
              \              /
               \            /
                >  speech/ <
```

- **`nlp_brain/`** — the NLP "Brain". All ML logic: data loading, feature
  extraction, model selection/CV, the gibberish + cosine-similarity coverage
  gate, and prediction. `nlp_brain.SymptomClassifier` is its one public
  entry point (`load()` / `save()` / `predict()` / `train()`) — it owns the
  joblib bundle schema and the predict pipeline, so neither is duplicated
  elsewhere. Zero FastAPI/voice/HTTP imports.
- **`api/`** — the FastAPI layer. HTTP contract + orchestration only: loads
  one `SymptomClassifier` at startup, translates requests into
  `classifier.predict()` calls, maps results to NexEagleWebsite's
  `specialtyId` slugs, and handles cross-cutting HTTP concerns (rate
  limiting). No ML logic lives here. Also exposes `POST /route-symptom-audio`
  for clients that record their own audio (browser, mobile app) and need
  server-side transcription — see `speech/` below.
- **`voice/`** — the Voice-to-Text layer: live microphone capture, followed
  by an HTTP call into `api/` via `voice.api_client.SymptomRouterClient`. It
  never imports `nlp_brain` directly — the only way it reaches the Brain is
  through the FastAPI contract, so it can run on a different machine (one
  with a mic, none of the ML dependencies) and be developed/tested
  independently.
- **`speech/`** — shared transcription + Devanagari→Roman transliteration,
  used by both `voice/` (live mic audio) and `api/` (uploaded audio). Exists
  as its own package specifically so neither `voice/` nor `api/` has to
  depend on the other just to share this logic.

Each of `voice/`/`api/`/`nlp_brain/` is only ever a caller of the one
"below" it — `nlp_brain` doesn't know `api` exists, and `api` doesn't know
`voice` exists. `speech/` depends on none of the three and is depended on
by two of them.

## Contents

- `nlp_brain/` — see above. Run `python -m nlp_brain.cli train` to train and
  save `symptom_specialist_classifier.joblib`, `python -m nlp_brain.cli
  predict "<text>"` for a single query, or `python -m nlp_brain.cli` (no
  args) for an interactive prompt — all against the trained bundle, no API
  needed.
- `api/` — see above. Entry point is `api.main:app`.
- `voice/` — see above. Entry point is `python -m voice.cli` (needs
  `requirements-voice.txt` installed and the API already running — set
  `NLP_API_BASE_URL` if it's not on `http://127.0.0.1:5003`).
- `speech/` — see above. Not runnable standalone; a library used by `voice/`
  and `api/`.
- `symptom_specialist_classifier.joblib` — the trained pipeline (feature
  extractors + model + label list + raw training texts + their TF-IDF
  vectors) that `api/` loads at startup. Produced offline by
  `nlp_brain.SymptomClassifier.train()` or `data_pipeline/retrain_pipeline.py`.
- `model_meta.json` — model version / last-retrained time / validation
  metrics, served as-is by `GET /model-info`. Written only by
  `data_pipeline/retrain_pipeline.py` on a successful promotion.
- `Hinglish_Symptoms_V28.csv` — the current training dataset (`text,
  specialist, type` columns; 31 specialists, ~15.5k rows).
- `Hinglish_Symptoms_Reference.csv` / `Hinglish_Symptoms_Reference_v2.csv` —
  earlier dataset versions, still read by `data_pipeline/generate_dataset.py`
  and `data_pipeline/build_validation_set.py` respectively; not used at
  runtime or by the active retrain pipeline.
- `data_pipeline/` — fetches CMS-editable training data + production
  feedback, retrains via `nlp_brain`, evaluates against a frozen validation
  set, and promotes (rewrites the V28 CSV + `model_meta.json` + the joblib
  bundle) only if it doesn't regress vs. the currently-promoted model. Not
  needed at runtime (excluded from the Docker image).
- `tests/` — the test suite (mandatory after any code change — see
  [Testing](#testing) below and [docs/testing.md](docs/testing.md)).
- `docs/` — one page per layer, with a module map and common recipes for
  each: [nlp_brain.md](docs/nlp_brain.md), [api.md](docs/api.md),
  [voice.md](docs/voice.md), [speech.md](docs/speech.md),
  [data_pipeline.md](docs/data_pipeline.md), [testing.md](docs/testing.md).
  New to the repo? Start with
  [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) instead — one
  comprehensive top-to-bottom reference (architecture, full API contracts,
  the prediction pipeline, deployment/HTTPS setup, and current Prod status)
  that ties all of the above together.
  Ops runbooks (manual, VM-side steps — not part of the CI/CD pipeline):
  [OPS-HTTPS-SETUP.md](docs/OPS-HTTPS-SETUP.md).

## API

- `GET /health` → `{"status": "ok", "ready": true}`
- `GET /model-info` → contents of `model_meta.json`.
- `POST /route-symptom` `{"query": "<Hinglish text>"}` →
  ```json
  {
    "specialtyIds": ["cardiology"],
    "noMatch": false,
    "method": "classifier",
    "confidence": 0.93,
    "modelVersion": "2026-08-04-083503",
    "raw": {
      "specialist": "Cardiologist (Heart)",
      "matchRatio": 0.93,
      "closestKnownExample": "...",
      "flaggedGibberish": false,
      "message": null
    }
  }
  ```
  `specialtyIds` is ordered most-confident-first (mapped via
  `LABEL_TO_NEXEAGLE_SPECIALTY_ID` in `specialty_mapping.py`) and normally
  has one entry, but may have up to `MAX_CANDIDATES` (3) when a runner-up
  specialist's confidence is within `CANDIDATE_MARGIN` of the top pick —
  see `nlp_brain/candidates.py` — rather than forcing an overconfident
  single guess on a genuinely ambiguous query. `noMatch: true` (with an
  empty `specialtyIds`) means the input was flagged as gibberish, or didn't
  sufficiently overlap with any known training example — there is no
  default-specialist fallback. Rate-limited to 30 requests/minute per IP.
- `POST /route-symptom-audio` — same as above, but takes an uploaded audio
  recording (`multipart/form-data`, field name `audio`, any format `ffmpeg`
  can decode) instead of text, transcribes + transliterates it server-side,
  and returns the same shape plus a `transcript` field. For a browser
  (`MediaRecorder`) or mobile client, not for Python code with a live mic —
  see [docs/api.md](docs/api.md) and [docs/speech.md](docs/speech.md) for
  the full contract, error modes, and size/rate limits (tighter than the
  text endpoint: 10MB / 10 requests-per-minute).

## Local development

```bash
pip install -r requirements.txt
python -m nlp_brain.cli train      # trains and saves symptom_specialist_classifier.joblib
uvicorn api.main:app --reload --port 5003
```

To also run the voice client against it:

```bash
pip install -r requirements-voice.txt
python -m voice.cli
```

## Testing

Tests are mandatory after any code change, enforced by a CI gate (a failing
test blocks the Docker build/deploy) and a pre-commit hook for fast local
feedback. Full details, fixtures, and how to add a new test:
[docs/testing.md](docs/testing.md). Quick start:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                            # full suite
pytest -m "not integration"       # fast subset (what the pre-commit hook runs)

git config core.hooksPath .githooks   # one-time, activates the local pre-commit hook
```

## Deployment

`.github/workflows/deploy-nlp.yml` builds a Docker image (containing
`nlp_brain/` + `api/` + `speech/`, plus the `ffmpeg` system package `speech/`
needs for audio format conversion — `voice/` is a separate client, never
part of the server image), pushes it to GHCR, and deploys to the same
dev/prod VMs the
rest of EasyHMS runs on — `develop` branch → Dev VM (`151.185.45.77:5003`),
`main` branch → Prod VM (`151.185.45.67:5003`), both via `docker run
--network host` matching the other backend services' convention.

`.github/workflows/retrain.yml` runs nightly (and on demand, via
workflow_dispatch) to retrain against the latest CMS data + production
feedback, and auto-promotes the result — which redeploys automatically via
`deploy-nlp.yml` — only if it doesn't regress against the currently-deployed
model's recorded validation metrics.
