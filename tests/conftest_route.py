"""Shared fixtures for the /route-symptom suites: one real model bundle per test session (trained once,
~40 s, cached at hms_router/data/route_model.joblib) and small fakes for flow tests."""
from __future__ import annotations

import pytest

from hms_router.domain.models import EmergencyAssessment, Evidence, Query


@pytest.fixture(scope="session")
def bundle():
    from hms_router.rag.model_bundle import ModelBundle
    return ModelBundle.load_or_build()


@pytest.fixture(scope="session")
def service(bundle):
    from hms_router.application import build_route_symptom_service
    return build_route_symptom_service(bundle)


class FakeRetriever:
    def __init__(self, evidence: list[Evidence]):
        self.evidence, self.calls = evidence, 0

    def retrieve(self, query: Query, k: int) -> list[Evidence]:
        self.calls += 1
        return self.evidence[:k]


class FakeScorer:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score(self, query: Query) -> dict[str, float]:
        return dict(self.scores)


class FakeDetector:
    def __init__(self, flag: bool = False):
        self.flag, self.calls = flag, 0

    def detect(self, query: Query) -> EmergencyAssessment:
        self.calls += 1
        return EmergencyAssessment(flag=self.flag, reasons=["test"] if self.flag else [],
                                   source="rules" if self.flag else None,
                                   recommendedSpecialtyIds=["emergencymedicine"] if self.flag else [])


def ev(i: str, sid: str, sim: float, text: str = "dant dard") -> Evidence:
    return Evidence(i, text, sid, sim)
