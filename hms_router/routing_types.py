"""The shape a specialty decision takes inside the conversational agent: three outputs (the doc's "3-output rule").
Filled from the /v1/route-symptom engine by route_bridge.to_routing, or by route_bridge.fallback_routing when that
engine is not available. OutputGenerator.build turns it into the reply card."""
from __future__ import annotations

from dataclasses import asdict, dataclass

EMERGENCY_SPECIALIST = "Emergency Medicine"
FALLBACKS = ["General Physician", "Telemedicine consult", "OPD triage desk"]     # pad to 3 outputs


@dataclass
class Recommendation:
    specialist: str
    confidence: float
    matched_symptoms: list[str]
    rationale: str
    cluster_id: str | None         # the specialtyId it came from (feedback refers to it)
    source: str                    # route-symptom | emergency | fallback


@dataclass
class RoutingResult:
    recommendations: list[Recommendation]
    emergency: bool
    low_confidence: bool
    query: str

    def to_dict(self) -> dict:
        return asdict(self)
