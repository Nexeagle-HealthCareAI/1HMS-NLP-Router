# EasyHMS NLP Symptom Router

Hinglish (Hindi-English code-mixed) symptom → specialist classifier for the Doctor
Dekho / NexEagleWebsite doctor search. Given a free-text query like *"pet mein dard
hai aur sar bhi dukh raha hai"*, splits multi-symptom sentences into segments and
routes each to a specialist via TF-IDF (char+word n-gram) + Logistic Regression, with
a nearest-known-phrase cosine-similarity search as fallback and a confidence-gated
default (General Physician) for genuinely unclear input.

## Contents

- `Model_1_Doctor_Dekho.py` — the core router: data loading, training, segmentation,
  classification, confidence gating, and a CLI (`--demo` / `--query` / interactive).
- `app.py` — FastAPI wrapper. Trains once at process startup from
  `Hinglish_Symptoms_Reference_v2.csv` and keeps the model in memory.
- `Hinglish_Symptoms_Reference_v2.csv` — the training dataset: 32 specialist classes
  aligned to EasyHMS's own `dbo.MedicalSpecialities.PatientFacingCategory` taxonomy
  (plus Dentist/Veterinarian, kept as-is). ~2500 rows.
- `Hinglish_Symptoms_Reference.csv` — the original 17-class dataset, kept for
  reference/audit only; not used at runtime.
- `data_pipeline/` — the seed-phrase bank + augmentation pipeline that produced the
  v2 dataset from the legacy one. Not needed at runtime (excluded from the Docker
  image); re-run this if the dataset needs expanding.

## API

- `GET /health` → `{"status": "ok", "ready": true}`
- `POST /route-symptom` `{"query": "<Hinglish text>"}` →
  ```json
  {
    "specialtyIds": ["cardiology"],
    "usedDefault": false,
    "raw": { "specialists": [...], "segments": [...] }
  }
  ```
  `specialtyIds` are already mapped to NexEagleWebsite's own `specialtyId` slugs
  (see `LABEL_TO_NEXEAGLE_SPECIALTY_ID` in `app.py`) and deduplicated. This service
  intentionally trains on the full 32-class taxonomy (not the standalone router's
  merged 29-class one) since NexEagleWebsite's own specialty list already
  distinguishes medical/surgical siblings (e.g. neurology vs. neurosurgery) — the
  existing top-k candidate mechanism surfaces both when genuinely ambiguous instead
  of collapsing them.

## Local development

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 5003
```

## Deployment

`.github/workflows/deploy-nlp.yml` builds a Docker image, pushes it to GHCR, and
deploys to the same dev/prod VMs the rest of EasyHMS runs on — `develop` branch →
Dev VM (`151.185.45.77:5003`), `main` branch → Prod VM (`151.185.45.67:5003`), both
via `docker run --network host` matching the other backend services' convention.
