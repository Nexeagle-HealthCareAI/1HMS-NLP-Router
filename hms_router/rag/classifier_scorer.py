"""Specialty scores from a trained probabilistic classifier (logistic regression on the same features)."""
from __future__ import annotations

from ..domain.models import Query
from .encoder import TextEncoder


class ClassifierScorer:
    def __init__(self, encoder: TextEncoder, classifier):
        self.encoder, self.classifier = encoder, classifier
        self.classes = [str(c) for c in classifier.classes_]

    def score(self, query: Query) -> dict[str, float]:
        probs = self.classifier.predict_proba(self.encoder.encode(query))[0]
        return {c: float(p) for c, p in zip(self.classes, probs)}


class EmergencyModelScore:
    """Callable for safety.ModelDetector: probability that the query is an emergency."""

    def __init__(self, encoder: TextEncoder, model):
        self.encoder, self.model = encoder, model
        self._pos = list(model.classes_).index(True)

    def __call__(self, query) -> float:
        return float(self.model.predict_proba(self.encoder.encode(query))[0][self._pos])
