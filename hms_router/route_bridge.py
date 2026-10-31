"""Vita's specialist choice comes from the /v1/route-symptom engine (RouteSymptomService), not the older
SpecialistRetriever. Probe 2026-10-30: the retriever sent fever to a Gastroenterologist, toothache to a
Dermatologist, pregnancy with swollen feet to an Orthopedist, and a red eye, missed periods, anxiety and high sugar
to a General Physician with "can't tell"; route-symptom chose General Physician, Dentistry, Gynaecology,
Ophthalmology, Gynaecology, Psychiatry and Endocrinology.

Vita keeps its own conversation (follow-up questions, wording, doctor lists); only the decision changes:
  * the whole conversation so far (the patient's own messages) is routed, with the age group when the patient says
    it ("2 saal ka beta", "baby", "8 year old") so a child goes to Paediatrics;
  * ANSWER -> top specialty is PRIMARY, the next one SECONDARY when it has some support;
  * CLARIFY (a close call) -> the same when the top specialty has at least CLARIFY_MIN confidence, otherwise
    low confidence (Vita says it can't tell and suggests a General Physician first, as before);
  * ABSTAIN -> low confidence;
  * an emergency, or the engine not being ready or failing -> the caller keeps the older retriever's result.
Emergency Medicine, Pathology, Radiology and Anaesthesiology are never offered as a booking specialty here; Oncology
and the surgical subspecialties are never the second option, and Paediatrics only for a child.
Known gap: route-symptom sends Hindi-script fever (बुखार) to Cardiology; the LLM understanding step addresses it.
"""
from __future__ import annotations

import re

from .doctors import SPECIALIST_TITLES, specialty_key
from .specialist_rag import FALLBACKS, Recommendation, RoutingResult
from .textnorm import preprocess

SOURCE = "route-symptom"
CLARIFY_MIN = 0.30              # a close call still names the top specialty when it has this much support
SECONDARY_MIN = 0.25            # below this the second option is a General Physician instead
# never offered as the second option: alarming next to a common complaint ("Oncologist" for loose motions)
NOT_SECONDARY = {"oncology", "neurosurgery", "cardiothoracicsurgery", "vascularsurgery", "plasticsurgery"}
NOT_BOOKABLE = {"emergencymedicine", "pathology", "radiology", "anesthesiology"}
HISTORY_TURNS = 6

_CHILD = re.compile(r"\b(baby|babies|infant|newborn|new born|toddler|child|children|kid|kids|bach+h?[aeiy]+|bachch?[aeiy]+|"
                    r"bacch?[aeiy]+|shishu|navjaat|navjat|nawjat|munna|munni|my son|my daughter)\b")
_YEARS = re.compile(r"\b(\d{1,3})\s*(?:-\s*)?(?:saal|sal|saala|saalaa|year|years|yr|yrs|varsh|varsha|baras)\b")
_MONTHS_OLD = re.compile(r"\b\d{1,2}\s*(?:mahine|mahina|mahinon|month|months)\s*(?:ka|ki|ke|kaa|kee|old)\b")
_CHILD_RELATION = re.compile(r"\b(beta|bete|beti|betaa|betii|son|daughter)\b")


def age_group_of(text: str) -> str | None:
    """'child' / 'adult' when the patient says so, else None (route-symptom then decides without it)."""
    t = preprocess(text or "")
    years = [int(n) for n in _YEARS.findall(t)]
    if years:
        y = min(years) if _CHILD_RELATION.search(t) or _CHILD.search(t) else years[0]
        return "child" if y < 14 else "adult" if y >= 18 else None
    if _MONTHS_OLD.search(t) or _CHILD.search(t):
        return "child"
    return None


def title(name: str) -> str:
    """'Dentistry' -> 'Dentist', 'Paediatrics' -> 'Paediatrician': the names doctor search and replies use."""
    key = specialty_key(name)
    return SPECIALIST_TITLES.get(key, name) if key else name


def conversation_text(history: list[dict]) -> str:
    texts = [(m.get("raw") or m.get("normalized") or "").strip() for m in history][-HISTORY_TURNS:]
    return " . ".join(t for t in texts if t)


def to_routing(decision, symptoms: list[str], query: str, age_group: str | None = None) -> RoutingResult:
    """A RouteDecision in the shape OutputGenerator.build expects (always three outputs)."""
    kind = getattr(decision.decision, "value", decision.decision)
    cands = [c for c in decision.candidates if c.specialtyId not in NOT_BOOKABLE and c.confidence > 0]
    low = (kind == "abstain" or not cands or (kind == "clarify" and cands[0].confidence < CLARIFY_MIN))
    recs: list[Recommendation] = []
    if not low:
        top = cands[0]
        why = "route-symptom " + kind + (f" ({decision.reason})" if decision.reason else "")
        recs.append(Recommendation(title(top.name), round(top.confidence, 3), list(symptoms), why, top.specialtyId, SOURCE))
        nxt = next((c for c in cands[1:] if c.confidence >= SECONDARY_MIN and title(c.name) != recs[0].specialist
                    and c.specialtyId not in NOT_SECONDARY and (c.specialtyId != "pediatrics" or age_group == "child")),
                   None)
        if nxt is not None:
            recs.append(Recommendation(title(nxt.name), round(nxt.confidence, 3), list(symptoms), why, nxt.specialtyId,
                                       SOURCE))
    for name in FALLBACKS:                                                 # always 3 outputs
        if len(recs) >= 3:
            break
        if all(r.specialist != name for r in recs):
            recs.append(Recommendation(name, 0.0, [], "Fallback when no better match", None, "fallback"))
    return RoutingResult(recs[:3], False, low, query)
