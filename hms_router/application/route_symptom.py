"""Use case: one symptom text in, a grounded RouteDecision out. Stateless; no HTTP, Redis or ML imports.

Flow (the order is the safety design - see docs/route_symptom.md):
  1 normalize            textnorm.preprocess (Devanagari -> Roman, persistence fix)
  2 safety               EmergencyDetector - always, before anything else, never skipped
  3 retrieve + score     Retriever (labelled sentences) + Scorers, fused into one ranking
  4 coverage gate        nothing similar in the corpus, or no health word at all -> ABSTAIN
  5 decisive?            yes -> ANSWER (method deterministic)
  6 reasoner (optional)  proposes a choice among candidates; Verifier must accept it -> ANSWER (+llm)
  7 otherwise            CLARIFY: one question + the hedge list
Every collaborator is injected (constructor), so each can be swapped or faked in tests.
"""
from __future__ import annotations

import logging
from typing import Callable

from ..domain.models import AGE_GROUPS, Candidate, Decision, EmergencyAssessment, Query, RouteDecision
from ..domain.policy import RoutingPolicy
from ..domain.taxonomy import specialty_name
from ..reasoning.clarify import ClarifyingQuestions
from ..reasoning.explain import Explainer
from ..reasoning.protocols import NoReasoner, Reasoner
from ..reasoning.verifier import Verifier
from ..rag.example_retriever import EvidenceVoteScorer
from ..rag.fusion import WeightedFusion
from ..rag.protocols import Retriever, Scorer
from ..safety.detectors import EmergencyDetector
from ..textnorm import has_letters, preprocess

log = logging.getLogger("hms_router.route_symptom")

MAX_TOP_K = 10


class RouteSymptomService:
    def __init__(self, *, retriever: Retriever, scorer: Scorer, detector: EmergencyDetector,
                 policy: RoutingPolicy | None = None, reasoner: Reasoner | None = None,
                 verifier: Verifier | None = None, explainer: Explainer | None = None,
                 clarifier: ClarifyingQuestions | None = None,
                 language_of: Callable[[str], str] | None = None, model_version: str | None = None,
                 medical_signal: Callable[[str], bool] | None = None):
        """medical_signal(normalized) -> does the text contain a health word? None = assume yes."""
        self.retriever, self.scorer, self.detector = retriever, scorer, detector
        self.policy = policy or RoutingPolicy()
        self.voter = EvidenceVoteScorer(retriever, k=self.policy.retrieve_k)
        self.fusion = WeightedFusion([self.policy.fusion_weight_retrieval, 1 - self.policy.fusion_weight_retrieval])
        self.reasoner = reasoner or NoReasoner()
        self.verifier = verifier or Verifier()
        self.explainer = explainer or Explainer()
        self.clarifier = clarifier or ClarifyingQuestions()
        self.language_of = language_of or (lambda _t: "unknown")
        self.model_version = model_version
        self.medical_signal = medical_signal

    # ------------------------------------------------------------------ public
    def route(self, text: str, top_k: int = 5, age_group: str | None = None) -> RouteDecision:
        """age_group: "child" | "adult" | None - from the caller, or the answer to an age question."""
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        if age_group is not None and age_group not in AGE_GROUPS:
            raise ValueError(f"age_group must be one of {AGE_GROUPS}")
        query = Query(raw=text, normalized=preprocess(text), age_group=age_group)
        query.language = self._language(query.normalized)

        emergency = self.detector.detect(query)                              # 2: always first
        if not has_letters(query.normalized):
            return self._abstain(query, emergency, 0.0)

        evidence = self.retriever.retrieve(query, self.policy.retrieve_k)    # 3
        scores = self.scorer.score(query)
        if age_group == "adult":                                             # paediatrics cannot apply
            evidence = [e for e in evidence if e.specialtyId != "pediatrics"]
            scores = {s: p for s, p in scores.items() if s != "pediatrics"}
            total = sum(scores.values())
            scores = {s: p / total for s, p in scores.items()} if total else {}
        ranked = self.fusion.fuse([self.voter.score_evidence(evidence), scores])
        match = evidence[0].similarity if evidence else 0.0
        top_conf = ranked[0][1] if ranked else 0.0

        if not ranked or self.policy.is_no_match(match, top_conf):          # 4
            return self._abstain(query, emergency, match)
        if self.policy.is_off_topic(self._has_signal(query.normalized), match, emergency.flag):
            return self._abstain(query, emergency, match, notes=["no health terms in the query"])

        notes: list[str] = []
        ranked, age_rule = self.policy.apply_age_rule(ranked, age_group)
        if age_rule:
            notes.append("policy: child -> paediatrics first")
        candidates = self._candidates(ranked, evidence, top_k)
        decisive = age_rule or self.policy.is_decisive(top_conf)
        if decisive:                                                         # 5
            return self._answer(query, candidates, ranked, emergency, match, "deterministic", notes)

        proposal = self.reasoner.decide(query, candidates)                   # 6
        if proposal is not None:
            verdict = self.verifier.check(proposal, candidates)
            if verdict.ok:
                chosen = self._reorder(candidates, proposal.choices)
                ranked_ids = [c.specialtyId for c in chosen]
                return self._answer(query, chosen, [(s, 0.0) for s in ranked_ids], emergency, match,
                                    "deterministic+llm", notes, reason=proposal.reason, shortlist=ranked_ids[:3])
            notes.append("llm_rejected: " + "; ".join(verdict.errors))
            log.warning("reasoner proposal rejected: %s", verdict.errors)

        return RouteDecision(                                                # 7
            query=text, decision=Decision.CLARIFY, candidates=candidates,
            specialtyIds=self.policy.shortlist(ranked, decisive=False), decisive=False, noMatch=False,
            matchScore=round(match, 4), emergency=emergency, clarify=self.clarifier.pick(query, candidates),
            reason=self.explainer.reason(query, candidates[0]), method="deterministic",
            modelVersion=self.model_version, notes=notes)

    # ------------------------------------------------------------------ helpers
    def _language(self, normalized: str) -> str:
        try:
            return self.language_of(normalized)
        except Exception:
            return "unknown"

    def _has_signal(self, normalized: str) -> bool:
        if self.medical_signal is None:
            return True
        try:
            return self.medical_signal(normalized)
        except Exception:                                   # a broken check must not hide a real query
            log.exception("medical signal check failed; treating the query as medical")
            return True

    def _candidates(self, ranked, evidence, top_k) -> list[Candidate]:
        per = self.policy.evidence_per_candidate
        out = []
        for sid, score in ranked[:top_k]:
            ev = [e for e in evidence if e.specialtyId == sid][:per]
            out.append(Candidate(sid, specialty_name(sid), round(float(score), 4), ev))
        return out

    @staticmethod
    def _reorder(candidates: list[Candidate], choices: list[str]) -> list[Candidate]:
        by_id = {c.specialtyId: c for c in candidates}
        first = [by_id[s] for s in choices if s in by_id]
        return first + [c for c in candidates if c.specialtyId not in choices]

    def _answer(self, query, candidates, ranked, emergency, match, method, notes, reason=None, shortlist=None):
        return RouteDecision(
            query=query.raw, decision=Decision.ANSWER, candidates=candidates,
            specialtyIds=shortlist or self.policy.shortlist(ranked, decisive=True), decisive=True, noMatch=False,
            matchScore=round(match, 4), emergency=emergency, clarify=None,
            reason=reason or self.explainer.reason(query, candidates[0] if candidates else None),
            method=method, modelVersion=self.model_version, notes=notes)

    def _abstain(self, query: Query, emergency: EmergencyAssessment, match: float,
                 notes: list[str] | None = None) -> RouteDecision:
        reason = ("Emergency signs found; please follow the emergency advice." if emergency.flag else
                  "I could not match this to a specialty. A General Physician can assess it first.")
        return RouteDecision(
            query=query.raw, decision=Decision.ABSTAIN, candidates=[], specialtyIds=[], decisive=False,
            noMatch=True, matchScore=round(match, 4), emergency=emergency, clarify=None, reason=reason,
            method="none", modelVersion=self.model_version, notes=notes or [])
