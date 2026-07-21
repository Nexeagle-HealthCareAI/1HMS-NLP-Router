"""
FastAPI wrapper around Model_1_Doctor_Dekho's Hinglish symptom -> specialist router.

Trains once at process startup from the bundled dataset and keeps the vectorizer,
classifier, and search index in memory for the process lifetime — no per-request
retraining, no external calls, no secrets required.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from Model_1_Doctor_Dekho import (
    DATA_PATH,
    build_search_index,
    classify_sentence,
    load_data,
    train_classifier,
)
from specialty_mapping import LABEL_TO_NEXEAGLE_SPECIALTY_ID

# Written only by data_pipeline/retrain_pipeline.py on a successful promotion — the live
# service just reads it, never writes it.
MODEL_META_PATH = Path(__file__).parent / "model_meta.json"

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    texts, labels = load_data(DATA_PATH, apply_output_merges=False)
    vectorizer, clf = train_classifier(texts, labels)
    index_matrix, index_texts, index_labels = build_search_index(texts, labels, vectorizer)
    _state.update(
        vectorizer=vectorizer, clf=clf,
        index_matrix=index_matrix, index_texts=index_texts, index_labels=index_labels,
    )
    yield
    _state.clear()


app = FastAPI(title="EasyHMS NLP Symptom Router", lifespan=lifespan)


class RouteRequest(BaseModel):
    query: str


class RouteResponse(BaseModel):
    specialtyIds: list[str]
    usedDefault: bool
    raw: dict


@app.get("/health")
def health():
    return {"status": "ok", "ready": bool(_state)}


@app.get("/model-info")
def model_info():
    try:
        with open(MODEL_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"modelVersion": "unknown", "lastRetrainedAt": None, "validationMetrics": None}


@app.post("/route-symptom", response_model=RouteResponse)
def route_symptom(req: RouteRequest):
    query = (req.query or "").strip()
    if not query:
        return RouteResponse(specialtyIds=[], usedDefault=True, raw={"specialists": [], "segments": []})

    specialists, per_segment = classify_sentence(
        query, _state["vectorizer"], _state["clf"],
        _state["index_matrix"], _state["index_texts"], _state["index_labels"],
    )

    specialty_ids: list[str] = []
    for label in specialists:
        slug = LABEL_TO_NEXEAGLE_SPECIALTY_ID.get(label)
        if slug and slug not in specialty_ids:
            specialty_ids.append(slug)

    used_default = any(r["method"] == "default (low confidence)" for r in per_segment)

    return RouteResponse(
        specialtyIds=specialty_ids,
        usedDefault=used_default,
        raw={
            "specialists": specialists,
            "segments": [
                {k: v for k, v in r.items() if k not in ("classifier_top", "search_top")}
                for r in per_segment
            ],
        },
    )
