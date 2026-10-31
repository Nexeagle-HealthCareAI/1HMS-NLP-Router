"""One search-bar query -> what Vita understands from it (POST /v2/agent/search-intent).

Doctor Dekho's search bar used to read queries with its own Groq prompt and the route-symptom engine, and understood
only symptoms and a city. It now asks Vita, so the bar, the chat and WhatsApp read people the same way: the same
spelling-tolerant reader (doctor_intent.parse: specialty, doctor name, degree, hospital, place, near me, lady doctor,
available today), the same city and name lists from 1HMS, the same emergency check, and the same specialty engine for
symptoms. One call, no conversation, nothing stored.
"""
from __future__ import annotations

import re

from ..domain.taxonomy import SPECIALTY_NAMES
from ..safety.denial import strip_denials
from ..textnorm import preprocess

# doctors.specialty_key() keys -> NexEagle specialtyIds (domain/taxonomy.py), the ids Doctor Dekho filters by
KEY_TO_ID = {
    "anaesthesiology": "anesthesiology", "breast-surgery": "generalsurgery", "cardiology": "cardiology",
    "cardiothoracic-surgery": "cardiothoracicsurgery", "dentistry": "dentistry", "dermatology": "dermatology",
    "emergency-medicine": "emergencymedicine", "endocrinology": "endocrinology", "ent": "ent",
    "gastroenterology": "gastroenterology", "general-medicine": "general", "general-surgery": "generalsurgery",
    "geriatrics": "geriatrics", "gi-surgery": "generalsurgery", "gynaecology": "gynecology", "nephrology": "nephrology",
    "neurology": "neurology", "neurosurgery": "neurosurgery", "oncology": "oncology", "ophthalmology": "ophthalmology",
    "orthopaedics": "orthopedics", "paediatrics": "pediatrics", "pathology": "pathology",
    "physiotherapy": "physiotherapy", "plastic-surgery": "plasticsurgery", "psychiatry": "psychiatry",
    "pulmonology": "pulmonology", "radiology": "radiology", "rheumatology": "rheumatology",
    "sports-medicine": "sportsmedicine", "urology": "urology", "vascular-surgery": "vascularsurgery",
}
assert set(KEY_TO_ID.values()) <= set(SPECIALTY_NAMES)


def understand_query(query: str, steps, route_service=None, *, city: str | None = None,
                     age_group: str | None = None) -> dict:
    """steps: application.doctor_chat.DoctorChat (its reader, emergency check and directory). route_service: the
    route-symptom service (symptoms -> specialty), or None while it loads. age_group: "child" | "adult", the answer to
    a clarify of kind "age". For symptoms the reply also carries route-symptom's own fields (decision, decisive,
    clarify, reason, usedDefault), so a caller of /route-symptom reads it unchanged."""
    ask = steps.read_ask(query)
    signs = steps.emergency_signs(query)
    known, cities = steps.directory.index() if steps.directory.enabled else ([], None)
    from ..doctor_intent import search_text
    corrected = search_text(query, known_names=known, cities=cities, medical_terms=steps.terms)

    ids: list[str] = []
    method, confidence, model = "vita", None, None
    decision_out = {"decision": None, "decisive": False, "clarify": None, "reason": None, "usedDefault": False}
    recommended: list[str] = []
    if ask.specialty_key and KEY_TO_ID.get(ask.specialty_key):
        ids = [KEY_TO_ID[ask.specialty_key]]
        confidence = 1.0
    symptomish = ask.symptomish and not (ask.specialty_key or ask.name_query or ask.degree or ask.hospital_query)
    nothing_else = not (ask.specialty_key or ask.name_query or ask.degree or ask.hospital_query or ask.place or ask.near
                        or ask.gender)
    if not ids and (symptomish or nothing_else) and route_service is not None:          # an emergency too: the list stays
        text = strip_denials(preprocess(query))[0] or preprocess(query)
        decision = route_service.route(text, top_k=5, age_group=age_group)
        recommended = [i for i in decision.emergency.recommendedSpecialtyIds if i in SPECIALTY_NAMES]
        if not decision.noMatch:
            ids = [i for i in decision.specialtyIds if i in SPECIALTY_NAMES][:5]
            confidence = round(decision.candidates[0].confidence, 4) if decision.candidates else None
            method, model = "vita+route-symptom", decision.modelVersion
        cl = decision.clarify
        decision_out = {"decision": decision.decision.value, "decisive": decision.decisive,
                        "clarify": {"kind": cl.kind, "question": cl.question, "options": list(cl.options),
                                    "optionIds": list(cl.optionIds)} if cl else None,
                        "reason": decision.reason, "usedDefault": bool(decision.noMatch)}

    place = None
    if ask.place:
        place = (cities.resolve(ask.place) if cities else None) or ask.place.title()
    name = steps.directory.fix_name(ask.name_query) if ask.name_query else None
    kind = ("emergency" if signs is not None else "name" if name else "specialty" if ask.specialty_key
            else "degree" if ask.degree else "hospital" if ask.hospital_query else "symptom" if ids
            else "place" if place else "near" if ask.near else "none")
    return {
        "specialtyId": ids[0] if ids else None, "specialtyIds": ids,
        "city": place or ("NEAR_ME" if ask.near else None),
        "near": bool(ask.near), "radiusKm": ask.radius_km,
        "doctorName": name, "degree": ask.degree, "hospital": ask.hospital_query, "gender": ask.gender,
        "availableToday": bool(ask.available_today),
        "keywords": [w for w in re.findall(r"[a-z]+", corrected) if len(w) >= 3][:10],
        "corrected": corrected, "kind": kind,
        "emergency": {"flag": signs is not None, "reasons": [s.capitalize() for s in signs or []],
                      "recommendedSpecialtyIds": recommended},
        "method": method, "confidence": confidence, "modelVersion": model, **decision_out,
    }
