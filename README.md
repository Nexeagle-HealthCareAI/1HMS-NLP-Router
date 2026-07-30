# EasyHMS NLP Symptom Router

Hinglish (Hindi-English code-mixed) symptom → specialist classifier for the Doctor
Dekho / NexEagleWebsite doctor search. Given a free-text query like *"pet mein dard
hai aur sar bhi dukh raha hai"*, routes it to a specialist via TF-IDF (char+word
n-gram) features and the best of LinearSVC / LogisticRegression / ComplementNB
(picked by cross-validation at training time). Before trusting a prediction, the
query is checked for gibberish (character-bigram model) and must have enough
word-level overlap with a known training example — genuinely unclear or nonsense
input gets "No matches found." instead of a guess.

## Contents

- `Model_1_revised.py` — the core router: data loading, training (feature
  extraction + model selection/CV), the gibberish + word-overlap coverage gate,
  and `predict()`. Run standalone (`python Model_1_revised.py`) to train, save
  the joblib bundle, and drop into an interactive prompt.
- `app.py` — FastAPI wrapper. Loads the pre-trained
  `symptom_specialist_classifier.joblib` bundle once at process startup and keeps
  it in memory — it does **not** train from CSV at boot. The bundle (and
  `model_meta.json`) are produced offline, by `Model_1_revised.main()` or by
  `data_pipeline/retrain_pipeline.py`.
- `big.model` — pretrained character-bigram model for the `gibberish-detector`
  package (trained from a general-English corpus); used for the gibberish
  pre-check. Regenerate with `gibberish-detector train <corpus.txt> > big.model`
  if it needs retraining on different text.
- `symptom_specialist_classifier.joblib` — the trained pipeline (feature
  extractors + model + label list + raw training texts) that `app.py` loads at
  startup.
- `Hinglish_Symptoms_Reference_V3.csv` — the current training dataset (`text,
  specialist, type` columns).
- `Hinglish_Symptoms_Reference.csv` / `Hinglish_Symptoms_Reference_v2.csv` —
  earlier dataset versions, kept for reference/audit only; not used at runtime.
- `data_pipeline/` — fetches CMS-editable training data + production feedback,
  retrains, evaluates against a frozen validation set, and promotes (rewrites the
  V3 CSV + `model_meta.json` + the joblib bundle) only if it doesn't regress vs.
  the currently-promoted model. Not needed at runtime (excluded from the Docker
  image).

## API

- `GET /health` → `{"status": "ok", "ready": true}`
- `GET /model-info` → contents of `model_meta.json` (model version, last retrain
  time, validation metrics).
- `POST /route-symptom` `{"query": "<Hinglish text>"}` →
  ```json
  {
    "specialtyIds": ["cardiology"],
    "noMatch": false,
    "method": "classifier",
    "confidence": 0.93,
    "modelVersion": "2026-07-22-baseline",
    "raw": {
      "specialist": "Cardiologist (Heart)",
      "matchRatio": 0.93,
      "closestKnownExample": "...",
      "flaggedGibberish": false,
      "message": null
    }
  }
  ```
  `specialtyIds` contains at most one entry (mapped to NexEagleWebsite's own
  `specialtyId` slugs via `LABEL_TO_NEXEAGLE_SPECIALTY_ID` in
  `specialty_mapping.py`). `noMatch: true` (with an empty `specialtyIds`) means the
  input was flagged as gibberish, or didn't sufficiently overlap with any known
  training example — there is no default-specialist fallback.

## Local development

```bash
pip install -r requirements.txt
python Model_1_revised.py   # trains and saves symptom_specialist_classifier.joblib
uvicorn app:app --reload --port 5003
```

## Deployment

`.github/workflows/deploy-nlp.yml` builds a Docker image, pushes it to GHCR, and
deploys to the same dev/prod VMs the rest of EasyHMS runs on — `develop` branch →
Dev VM (`151.185.45.77:5003`), `main` branch → Prod VM (`151.185.45.67:5003`), both
via `docker run --network host` matching the other backend services' convention.

`.github/workflows/retrain.yml` runs nightly (and on demand, via workflow_dispatch)
to retrain against the latest CMS data + production feedback, and auto-promotes
the result — which redeploys automatically via `deploy-nlp.yml` — only if it
doesn't regress against the currently-deployed model's recorded validation metrics.
