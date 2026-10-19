"""The labelled corpus the RAG pipeline retrieves from and trains on.

Sources (all map onto domain.taxonomy specialtyIds):
  Datasets/combined_symptom_dataset.csv   6.3k Hinglish sentences, with triage labels  -> id "ds:<id>"
  Datasets/Hinglish_Symptoms_V28.csv      15.5k rows of the old nlp_brain training set -> id "v28:<row>"
  hms_router/data/specialty_seed_en.csv   short English phrases (needs clinician review) -> id "seed:<row>"
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


@dataclass(frozen=True)
class LabelledExample:
    id: str
    text: str                              # as written (shown back as evidence)
    normalized: str                        # preprocess(text) - what is indexed
    specialtyId: str
    emergency: bool | None                 # None = this source has no triage label
    weight: float
    source: str                            # dataset | legacy | seed


def load_examples(dataset_path: Path | None = DATASET_PATH, legacy_path: Path | None = LEGACY_DATASET_PATH,
                  seed_path: Path | None = SEED_PATH) -> list[LabelledExample]:
    """All labelled examples, deduplicated on (normalized text, specialty); first occurrence wins."""
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
    seen, out = set(), []
    for r in rows:
        key = (r.normalized, r.specialtyId)
        if r.normalized and key not in seen:
            seen.add(key)
            out.append(r)
    return out
