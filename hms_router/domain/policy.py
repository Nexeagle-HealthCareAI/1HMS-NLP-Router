"""Routing policy: the thresholds and the pure decisions made with them.

Tuned on held-out data and Doctor Dekho query set 1 only (scripts/eval_route_symptom.py --sweep).
Re-run that script after retraining or editing data before changing any number here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingPolicy:
    min_match: float = 0.20            # below: nothing in training resembles the query -> abstain
    low_match: float = 0.33            # between min_match and this ...
    low_match_min_conf: float = 0.30   # ... also require this top confidence, else abstain
    no_signal_min_match: float = 0.60  # no health word in the query: abstain unless the corpus match is this strong
    decisive_min_conf: float = 0.60    # top fused confidence needed to answer without asking
    shortlist_ratio: float = 0.5       # decisive shortlist keeps specialties within 50% of the top one
    max_shortlist: int = 3
    hedge_size: int = 5                # what non-decisive callers get (Doctor Dekho then asks Groq)
    retrieve_k: int = 25               # labelled sentences retrieved per query
    evidence_per_candidate: int = 3
    fusion_weight_retrieval: float = 0.5
    emergency_min_score: float = 0.60  # emergency-model probability that raises the flag on its own
    child_first_pediatrics: bool = True   # a child whose top-3 includes paediatrics starts there (clinical rule)

    def apply_age_rule(self, ranked: list[tuple[str, float]], age_group: str | None) -> tuple[list, bool]:
        """Returns (ranking, rule_applied). Order changes; confidences do not."""
        if not (self.child_first_pediatrics and age_group == "child"):
            return ranked, False
        ids = [s for s, _ in ranked[:3]]
        if "pediatrics" not in ids:
            return ranked, False
        if ids[0] == "pediatrics":            # already first: the rule still settles it, so a parent who
            return ranked, True               # answered "child" is not asked a second question
        ped = next(x for x in ranked if x[0] == "pediatrics")
        return [ped] + [x for x in ranked if x[0] != "pediatrics"], True

    def is_no_match(self, match: float, top_conf: float) -> bool:
        return match < self.min_match or (match < self.low_match and top_conf < self.low_match_min_conf)

    def is_off_topic(self, has_medical_signal: bool, match: float, emergency: bool) -> bool:
        """"cricket score today": no health word, only a weak resemblance, no emergency -> not ours to route."""
        return not has_medical_signal and not emergency and match < self.no_signal_min_match

    def is_decisive(self, top_conf: float) -> bool:
        return top_conf >= self.decisive_min_conf

    def shortlist(self, ranked: list[tuple[str, float]], decisive: bool) -> list[str]:
        """Back-compat `specialtyIds`: Doctor Dekho trusts 1-3 ids and asks Groq when given more."""
        if not ranked:
            return []
        if not decisive:
            return [s for s, _ in ranked[: self.hedge_size]]
        top = ranked[0][1]
        return [s for s, p in ranked[: self.max_shortlist] if p >= top * self.shortlist_ratio]

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
