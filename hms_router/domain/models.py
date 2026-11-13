"""Domain objects for specialty routing. Pure data: no I/O, no frameworks.

Every layer above (rag, safety, reasoning, application, adapters) speaks in these types, so a new
retriever, detector or reasoner only has to produce/consume them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ANSWER = "answer"        # confident enough to act on
    CLARIFY = "clarify"      # close call: ask one question (callers may also show the hedge list)
    ABSTAIN = "abstain"      # nothing medical recognised; say so instead of guessing


AGE_GROUPS = ("child", "adult")


@dataclass
class Query:
    """One routing request. `features` is a cache slot the rag layer fills (opaque to the domain)."""
    raw: str
    normalized: str
    language: str = "unknown"            # english | hinglish | hindi | unknown
    age_group: str | None = None         # child | adult | None (unknown) - from the caller or a clarify answer
    features: Any = None


@dataclass(frozen=True)
class Evidence:
    """A labelled example sentence retrieved for a query - the grounding for every suggestion."""
    id: str
    text: str
    specialtyId: str
    similarity: float


@dataclass
class Candidate:
    specialtyId: str
    name: str
    confidence: float                      # fused score, 0-1, sums to 1 over all specialties
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class EmergencyAssessment:
    flag: bool
    reasons: list[str] = field(default_factory=list)
    source: str | None = None              # rules | model | rules+model
    score: float = 0.0                     # emergency-model probability
    recommendedSpecialtyIds: list[str] = field(default_factory=list)


@dataclass
class Clarification:
    question: str
    options: list[str]                     # what to show the patient
    optionIds: list[str]                   # specialtyIds (kind "specialty") or ageGroup values (kind "age")
    kind: str = "specialty"                # specialty | age


@dataclass
class ReasonerResult:
    """What a reasoner proposes. Must be checked by the Verifier before use."""
    choices: list[str]                     # specialtyIds, best first
    citedEvidenceIds: list[str]
    reason: str


@dataclass
class RouteDecision:
    query: str
    decision: Decision
    candidates: list[Candidate]
    specialtyIds: list[str]                # back-compat shortlist for existing callers
    decisive: bool
    noMatch: bool
    matchScore: float
    emergency: EmergencyAssessment
    clarify: Clarification | None
    reason: str | None
    method: str                            # deterministic | deterministic+llm | none
    modelVersion: str | None
    notes: list[str] = field(default_factory=list)   # e.g. "llm_rejected: ..." for audit logs

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d
