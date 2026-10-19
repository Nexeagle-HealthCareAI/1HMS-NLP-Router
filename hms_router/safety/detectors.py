"""Emergency detectors. Deterministic, run before any model or LLM, and never skipped.

    RuleDetector   red-flag phrases (red_flags.csv) on the text + canonical symptoms from the
                   spelling/NER stages /chat already uses (read-only reuse of severity.emergency_check)
    ModelDetector  an emergency classifier's probability >= threshold (the score function is injected,
                   so this module has no ML dependency)
    AnyOf          flags if any detector flags - detectors can only ADD urgency, never remove it
"""
from __future__ import annotations

import logging
from typing import Callable, Protocol

from ..domain.models import EmergencyAssessment, Query
from ..severity import EMERGENCY, emergency_check
from .rules import Rule, recommended_specialties

log = logging.getLogger("hms_router.safety")


class EmergencyDetector(Protocol):
    def detect(self, query: Query) -> EmergencyAssessment: ...


class RuleDetector:
    def __init__(self, rules: list[Rule], extractor: Callable[[str], dict] | None = None):
        """extractor(normalized_text) -> {"symptoms": [...], "negated_symptoms": [...], "corrected": str}."""
        if not rules:
            raise ValueError("RuleDetector needs at least one rule")
        self.rules = rules
        self.extractor = extractor

    def detect(self, query: Query) -> EmergencyAssessment:
        state, texts = {"symptoms": [], "negated_symptoms": []}, [query.normalized]
        if self.extractor is not None:
            try:
                ex = self.extractor(query.normalized)
                state = {"symptoms": ex.get("symptoms", []), "negated_symptoms": ex.get("negated_symptoms", [])}
                if ex.get("corrected") and ex["corrected"] not in texts:
                    texts.append(ex["corrected"])
            except Exception:                       # extraction failing must never switch the safety net off
                log.exception("symptom extraction failed; red-flag phrases still checked on raw text")
        hits = [h for h in emergency_check(state, texts, self.rules) if h["flag"] == EMERGENCY]
        if not hits:
            return EmergencyAssessment(flag=False)
        return EmergencyAssessment(flag=True, reasons=list(dict.fromkeys(h["english"] for h in hits)),
                                   source="rules",
                                   recommendedSpecialtyIds=recommended_specialties([h["canonical"] for h in hits]))


class ModelDetector:
    def __init__(self, score: Callable[[Query], float], threshold: float):
        self.score, self.threshold = score, threshold

    def detect(self, query: Query) -> EmergencyAssessment:
        try:
            s = float(self.score(query))
        except Exception:
            log.exception("emergency model failed; rules still apply")
            return EmergencyAssessment(flag=False)
        if s >= self.threshold:
            return EmergencyAssessment(flag=True, reasons=["Similar to emergency cases in the training data"],
                                       source="model", score=round(s, 4),
                                       recommendedSpecialtyIds=["emergencymedicine"])
        return EmergencyAssessment(flag=False, score=round(s, 4))


class AnyOf:
    def __init__(self, detectors: list[EmergencyDetector]):
        if not detectors:
            raise ValueError("AnyOf needs at least one detector")
        self.detectors = detectors

    def detect(self, query: Query) -> EmergencyAssessment:
        results = [d.detect(query) for d in self.detectors]
        flagged = [r for r in results if r.flag]
        score = max((r.score for r in results), default=0.0)
        if not flagged:
            return EmergencyAssessment(flag=False, score=score)
        sources = sorted({r.source for r in flagged if r.source}, key=lambda s: ("rules", "model").index(s)
                         if s in ("rules", "model") else 9)
        reasons, recs = [], []
        for r in flagged:                            # rule reasons first: they are specific
            reasons += [x for x in r.reasons if x not in reasons]
            recs += [x for x in r.recommendedSpecialtyIds if x not in recs]
        if any(r.source == "rules" for r in flagged):
            reasons = [x for x in reasons if x != "Similar to emergency cases in the training data"] or reasons
        return EmergencyAssessment(flag=True, reasons=reasons, source="+".join(sources), score=score,
                                   recommendedSpecialtyIds=recs)
