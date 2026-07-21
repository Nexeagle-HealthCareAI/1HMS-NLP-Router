"""
FastAPI wrapper around Model_1_Doctor_Dekho's Hinglish symptom -> specialist router.

Trains once at process startup from the bundled dataset and keeps the vectorizer,
classifier, and search index in memory for the process lifetime — no per-request
retraining, no external calls, no secrets required.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from Model_1_Doctor_Dekho import (
    DATA_PATH,
    build_search_index,
    classify_sentence,
    load_data,
    train_classifier,
)

# Maps our internal 32-class taxonomy to NexEagleWebsite's own `specialtyId` slugs
# (src/data/patient.ts `specialties` array). NexEagleWebsite already distinguishes the
# medical-vs-surgical siblings (neurology/neurosurgery, cardiology/cardiothoracicsurgery)
# that the standalone router's MODEL_OUTPUT_MERGES collapses for single-label accuracy —
# this service trains WITHOUT that collapse (see lifespan() below) and lets the existing
# top-k candidate mechanism (classify_segment's build_candidates, margin 0.12) surface
# both sibling specialties when genuinely ambiguous, instead of losing the distinction.
LABEL_TO_NEXEAGLE_SPECIALTY_ID = {
    "General Physician": "general",
    "Paediatrician": "pediatrics",
    "Cardiologist (Heart)": "cardiology",
    "Dermatologist (Skin)": "dermatology",
    "Orthopaedic Surgeon (Bone)": "orthopedics",
    "Gynaecologist": "gynecology",
    "Dentist": "dentistry",
    "ENT Specialist": "ent",
    "Ophthalmologist (Eye)": "ophthalmology",
    "Neurologist": "neurology",
    "Psychiatrist": "psychiatry",
    "Urologist": "urology",
    "Gastroenterologist": "gastroenterology",
    "Endocrinologist (Hormones/Diabetes)": "endocrinology",
    "Pulmonologist (Chest/Lungs)": "pulmonology",
    "Nephrologist (Kidney)": "nephrology",
    "Oncologist (Cancer)": "oncology",
    "Rheumatologist": "rheumatology",
    "Physiotherapist / Rehab": "physiotherapy",
    "General Surgeon": "generalsurgery",
    "Neurosurgeon": "neurosurgery",
    "Plastic Surgeon": "plasticsurgery",
    "Vascular Surgeon": "vascularsurgery",
    "Cardiothoracic Surgeon": "cardiothoracicsurgery",
    "Anaesthesiologist": "anesthesiology",
    "Radiologist": "radiology",
    "Pathologist": "pathology",
    "Emergency Medicine Specialist": "emergencymedicine",
    "Geriatrician": "geriatrics",
    "Sports Medicine Specialist": "sportsmedicine",
    # No distinct "surgical GI" id on NexEagleWebsite — its own "generalsurgery" blurb
    # ("Hernia, gallbladder & general operations") is the closest real bucket.
    "GI/Surgical Gastroenterologist": "generalsurgery",
    # Not a human-medicine category on a doctor-booking site — no target, dropped.
    "Veterinarian": None,
}

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
