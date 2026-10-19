"""The labelled corpus the RAG pipeline retrieves from and trains on.

Sources (all map onto domain.taxonomy specialtyIds):
  Datasets/combined_symptom_dataset.csv   6.3k Hinglish sentences, with triage labels  -> id "ds:<id>"
  Datasets/Hinglish_Symptoms_V28.csv      15.5k rows of the old nlp_brain training set -> id "v28:<row>"
  hms_router/data/specialty_seed_en.csv   short English phrases (needs clinician review) -> id "seed:<row>"
  CMS SymptomTrainingExamples (optional)  rows an admin added or approved in the CMS "AI Agents" review
                                          queue, fetched at build time (load_cms_examples)  -> id "cms:<guid>"

data_pipeline/validation_set.csv is the held-out test set: its sentences are removed from training
(load_validation_texts / ModelBundle.train) so the accuracy the CMS shows is measured on sentences the
model never saw.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..domain.taxonomy import DATASET_SPECIALTY_TO_ID, LEGACY_LABEL_TO_ID, SPECIALTY_NAMES
from ..textnorm import preprocess

_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = _ROOT / "Datasets" / "combined_symptom_dataset.csv"
LEGACY_DATASET_PATH = _ROOT / "Datasets" / "Hinglish_Symptoms_V28.csv"
SEED_PATH = _ROOT / "hms_router" / "data" / "specialty_seed_en.csv"
SEED_WEIGHT = 5.0                          # one seed phrase stands in for a handful of real sentences
CMS_WEIGHT = 3.0                           # a human-approved correction counts for a few dataset sentences
VALIDATION_PATH = _ROOT / "data_pipeline" / "validation_set.csv"


@dataclass(frozen=True)
class LabelledExample:
    id: str
    text: str                              # as written (shown back as evidence)
    normalized: str                        # preprocess(text) - what is indexed
    specialtyId: str
    emergency: bool | None                 # None = this source has no triage label
    weight: float
    source: str                            # dataset | legacy | seed | cms


def load_examples(dataset_path: Path | None = DATASET_PATH, legacy_path: Path | None = LEGACY_DATASET_PATH,
                  seed_path: Path | None = SEED_PATH, extra: list[LabelledExample] | None = None,
                  holdout: set[str] | None = None) -> list[LabelledExample]:
    """All labelled examples, deduplicated on (normalized text, specialty); first occurrence wins.
    `extra` (e.g. CMS rows) is appended last; `holdout` normalized texts are left out."""
    rows: list[LabelledExample] = []
    if dataset_path and Path(dataset_path).exists():
        with open(dataset_path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                sid = DATASET_SPECIALTY_TO_ID.get(r["specialty"].strip())
                if sid:
                    text = r["symptoms_hinglish"].strip()
                    rows.append(LabelledExample(f"ds:{r['id'].strip()}", text, preprocess(text), sid,
                                                r["triage"].strip() == "Emergency" or sid == "emergencymedicine",
                                                1.0, "dataset"))
    if legacy_path and Path(legacy_path).exists():
        with open(legacy_path, newline="", encoding="utf-8-sig") as f:
            for i, r in enumerate(csv.DictReader(f), start=2):
                sid = LEGACY_LABEL_TO_ID.get(r["specialist"].strip())
                if sid:
                    text = r["text"].strip()
                    rows.append(LabelledExample(f"v28:{i}", text, preprocess(text), sid, None, 1.0, "legacy"))
    if seed_path and Path(seed_path).exists():
        with open(seed_path, newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f), start=2):
                if r["specialtyId"] in SPECIALTY_NAMES:
                    text = r["text"].strip()
                    rows.append(LabelledExample(f"seed:{i}", text, preprocess(text), r["specialtyId"], None,
                                                SEED_WEIGHT, "seed"))
    rows.extend(extra or [])
    holdout = holdout or set()
    seen, out = set(), []
    for r in rows:
        key = (r.normalized, r.specialtyId)
        if r.normalized and key not in seen and r.normalized not in holdout:
            seen.add(key)
            out.append(r)
    return out


def load_validation(path: Path | None = VALIDATION_PATH) -> list[tuple[str, str]]:
    """(text, specialtyId) pairs of the held-out validation set; rows with unmapped labels are skipped."""
    out: list[tuple[str, str]] = []
    if path and Path(path).exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                sid = LEGACY_LABEL_TO_ID.get((r.get("specialist") or "").strip())
                text = (r.get("text") or "").strip()
                if sid and text:
                    out.append((text, sid))
    return out


def load_validation_texts(path: Path | None = VALIDATION_PATH) -> set[str]:
    return {preprocess(t) for t, _ in load_validation(path)}


def load_cms_examples(base_url: str, service_key: str, *, timeout: float = 15.0,
                      page_size: int = 500, max_pages: int = 200) -> list[LabelledExample]:
    """Active rows of the CMS SymptomTrainingExamples table (GET {base}/symptom-router/training-examples/service).
    Labels are the CMS specialist names (LEGACY_LABEL_TO_ID); rows with an unknown label are skipped.
    Raises on HTTP errors so a rebuild can report "CMS unreachable" instead of silently training without them."""
    import httpx
    out: list[LabelledExample] = []
    base = base_url.rstrip("/")
    for page in range(1, max_pages + 1):
        r = httpx.get(f"{base}/symptom-router/training-examples/service", params={"page": page, "limit": page_size},
                      headers={"X-Service-Key": service_key}, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        for row in body.get("data") or []:
            sid = LEGACY_LABEL_TO_ID.get((row.get("specialist") or "").strip())
            text = (row.get("text") or "").strip()
            if sid and text:
                out.append(LabelledExample(f"cms:{row.get('id')}", text, preprocess(text), sid, None, CMS_WEIGHT, "cms"))
        if page >= int((body.get("pagination") or {}).get("totalPages") or 0):
            break
    return out
