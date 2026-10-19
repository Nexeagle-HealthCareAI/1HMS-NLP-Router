"""Application services exposed as agent tools.

`route_symptom_payload` is framework-free (plain dict in, dict out); `as_langchain_tools` wraps it for
LangChain/LangGraph agents. Tools only read - nothing here books, writes or calls out on its own.
"""
from __future__ import annotations

from ..application.route_symptom import RouteSymptomService


def route_symptom_payload(service: RouteSymptomService, text: str, top_k: int = 3,
                          age_group: str | None = None) -> dict:
    d = service.route(text, top_k, age_group=age_group)
    return {
        "decision": d.decision.value,
        "specialties": [{"specialtyId": c.specialtyId, "name": c.name, "confidence": c.confidence,
                         "evidenceIds": [e.id for e in c.evidence]} for c in d.candidates],
        "emergency": {"flag": d.emergency.flag, "reasons": d.emergency.reasons,
                      "recommendedSpecialtyIds": d.emergency.recommendedSpecialtyIds},
        "clarify": None if d.clarify is None else {"kind": d.clarify.kind, "question": d.clarify.question,
                                                     "options": d.clarify.options, "optionIds": d.clarify.optionIds},
        "reason": d.reason,
    }


def as_langchain_tools(service: RouteSymptomService) -> list:
    from langchain_core.tools import StructuredTool

    def route_symptom(text: str, top_k: int = 3) -> dict:
        """Route a patient's symptom description to medical specialties. Returns ranked specialtyIds with
        confidences, an emergency flag with reasons, and either an answer, a clarifying question, or an
        abstain decision. Never diagnoses."""
        return route_symptom_payload(service, text, top_k)

    return [StructuredTool.from_function(route_symptom, name="route_symptom")]
