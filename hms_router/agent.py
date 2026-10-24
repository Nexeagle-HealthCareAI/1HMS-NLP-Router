"""The agent: one call per user message, all layers of the architecture in order.

  L1  ingest            Session -> language -> script normalisation                  (pipeline.ingest)
  L2  correct + extract Spelling repair -> keyword tags -> NER -> symptom object     (pipeline, keywords)
  L3  accumulate        Redis multi-turn buffer, merged "latest statement wins"      (accumulator)
  L4  SAFETY NET        hardcoded emergency check, on the merged state + raw text    (severity.emergency_check)
      follow-up?        not enough info and no emergency -> ask the next question
      severity          rules (+ optional ML that can only RAISE the score)          (severity)
      route             RAG: embed -> vector store -> rerank -> specialists          (specialist_rag)
  L5  respond           3 specialist outputs, or the Emergency Response Card         (output)
      feedback          Step 8 via agent.feedback / LearningLoop

Design rules from the doc's IMPORTANT box, enforced here:
  * The safety net always runs and does not depend on Steps 2-3 succeeding: if spelling/NER raise, the agent
    falls back to an empty symptom object and the safety net still scans the raw text.
  * An emergency is never delayed by follow-up questions.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable

from .accumulator import AccumulatedState, SymptomAccumulator
from .config import Settings
from .keywords import KeywordTag, KeywordTagger
from .learning import LearningLoop
from .lexicon import Lexicon
from .output import OutputGenerator, ResponseCard
from .pipeline import HMSRouterPipeline
from .severity import ALERT, EMERGENCY, SeverityClassifier, SeverityResult, band, emergency_check
from .specialist_rag import SpecialistRetriever

log = logging.getLogger("hms_router")

FOLLOW_UP, EMERGENCY_KIND, RECOMMENDATION = "follow_up", "emergency", "recommendation"


@dataclass
class AgentReply:
    session_id: str
    turn: int
    kind: str                                  # follow_up | emergency | recommendation
    message: str                               # ready-to-send text, in the user's language
    question: str | None = None
    card: ResponseCard | None = None
    state: AccumulatedState | None = None
    severity: SeverityResult | None = None
    keywords: list[KeywordTag] = field(default_factory=list)
    language: str = "english"
    degraded: list[str] = field(default_factory=list)      # components that failed this turn (safety net still ran)

    def to_dict(self) -> dict:
        return asdict(self)


class HMSRouterAgent:
    def __init__(self, redis_client, settings: Settings | None = None, lexicon: Lexicon | None = None,
                 retriever: SpecialistRetriever | None = None, classifier: SeverityClassifier | None = None,
                 emergency_rules: list | None = None, dataset_path: str | None = None,
                 emergency_detector: "Callable[[], object | None] | None" = None):
        """Create the agent.

        Args:
            dataset_path: Optional override path for combined_symptom_dataset.csv.
                          By default, the RAG corpus is built from the project
                          dataset.
            emergency_detector: returns the SAME emergency detector /v1/route-symptom uses
                          (rules + model + denial handling), or None while it is not built yet. A callable,
                          not the detector itself, so a model rebuild that swaps the route service is picked
                          up on the next message. When it returns a detector, it alone decides whether a turn
                          is an emergency; the older severity.emergency_check is only the fallback.
        """
        self.emergency_detector = emergency_detector
        self.r = redis_client
        self.cfg = settings or Settings()
        self.pipeline = HMSRouterPipeline(redis_client, lexicon, self.cfg)
        self.accumulator = SymptomAccumulator(redis_client, self.cfg.session_ttl_seconds,
                                              self.cfg.followup_max_turns, self.cfg.max_buffer)
        self.retriever = retriever or SpecialistRetriever(top_k=self.cfg.rag_top_k,
                                                          min_confidence=self.cfg.rag_min_confidence,
                                                          dataset_path=dataset_path)
        self.classifier = classifier or SeverityClassifier(extra_rules=emergency_rules)
        self.tagger = KeywordTagger(self.retriever.corpus)
        self.output = OutputGenerator()
        self.learning = LearningLoop(redis_client, self.pipeline, self.retriever, self.cfg)
        self.feedback = self.learning.feedback

    # ------------------------------------------------------------------ main entry
    def handle(self, text: str, session_id: str | None = None) -> AgentReply:
        ing = self.pipeline.ingest(text, session_id)                              # Layer 1
        lang = ing.language.language
        degraded: list[str] = []

        cor, obj, tags = None, None, []
        try:
            cor = self.pipeline.correct(ing)                                      # Layer 2: spelling
            tags = self.tagger.tag(cor)
            obj = self.pipeline.extract(ing, cor).to_dict()                       # Layer 2: NER -> symptom object
        except Exception:                                                         # never let NLP failure hide an emergency
            log.exception("NLP stage failed; continuing with the safety net on raw text")
            degraded.append("nlp")
        if obj is None:         # keep whatever spelling/lexicon matching already found (no negation info -> fail safe)
            obj = {"symptoms": list(cor.symptoms) if cor else [], "negated_symptoms": [],
                   "body_parts": list(cor.body_parts) if cor else [], "body_part": None, "duration": None,
                   "intensity": None, "language_mix": lang, "turn": ing.turn, "raw_input": ing.raw_text}
        obj["script"] = ing.language.script

        texts = [m["normalized"] for m in self.pipeline.sessions.history(ing.session_id)]
        if cor is not None and cor.corrected_text not in texts:
            texts.append(cor.corrected_text)                                      # spelling-repaired text catches typos

        shared = self._shared_emergency(ing.raw_text, texts)                      # None = shared detector unavailable

        def safety(state: AccumulatedState) -> bool:                              # Layer 4 - ahead of everything else
            if shared is not None:
                return shared.flag
            hits = emergency_check(state.to_dict(), texts, self.classifier.rules)
            return any(h["flag"] == EMERGENCY for h in hits)

        state = self.accumulator.add(ing.session_id, obj, user_text=ing.normalized_text, turn=ing.turn,
                                     emergency=safety)                            # Layer 3
        sev = self.classifier.classify(state.to_dict(), texts)                    # safety net + rules (+ ML)
        if shared is not None:
            sev = self._apply_shared_emergency(sev, shared)

        base = dict(session_id=ing.session_id, turn=ing.turn, state=state, severity=sev, keywords=tags,
                    language=lang, degraded=degraded)
        if not state.ready and not sev.emergency:                                 # need more information
            return AgentReply(kind=FOLLOW_UP, message=state.next_question or "", question=state.next_question, **base)

        routing = self.retriever.route(state.to_dict(), sev.to_dict())            # Layer 4: RAG
        card = self.output.build(state.to_dict(), sev.to_dict(), routing)         # Layer 5
        kind = EMERGENCY_KIND if card.kind == "emergency" else RECOMMENDATION
        return AgentReply(kind=kind, message=self.output.render(card, lang), card=card, **base)

    # ------------------------------------------------------------------ shared emergency detector
    def _shared_emergency(self, raw_text: str, session_texts: list[str]):
        """The route-symptom detector's verdict on this message and on the whole conversation so far
        (a chest-pain turn followed by a sweating turn is judged together). None when the detector is not
        available or fails, so the caller falls back to the older check instead of losing the safety net."""
        if self.emergency_detector is None:
            return None
        try:
            detector = self.emergency_detector()
            if detector is None:
                return None
            from .domain.models import EmergencyAssessment, Query
            from .textnorm import preprocess

            candidates = [raw_text]
            joined = " . ".join(t for t in session_texts if t)
            if joined and joined != preprocess(raw_text):
                candidates.append(joined)
            flagged = [a for a in (detector.detect(Query(raw=t, normalized=preprocess(t))) for t in candidates)
                       if a.flag]
            if not flagged:
                return EmergencyAssessment(flag=False)
            reasons: list[str] = []
            recs: list[str] = []
            for a in flagged:
                reasons += [r for r in a.reasons if r not in reasons]
                recs += [r for r in a.recommendedSpecialtyIds if r not in recs]
            return EmergencyAssessment(flag=True, reasons=reasons, source=flagged[0].source,
                                       score=max(a.score for a in flagged), recommendedSpecialtyIds=recs)
        except Exception:                                                         # never lose the safety net
            log.exception("shared emergency detector failed; using the built-in safety net")
            return None

    @staticmethod
    def _apply_shared_emergency(sev: SeverityResult, shared) -> SeverityResult:
        """Make the severity result agree with the shared detector: raise it to EMERGENCY when the detector
        flags, and drop an emergency raised only by the older phrase list (e.g. "saans lene mein koi dikkat
        nahi") when the detector, which understands denials, does not."""
        if shared.flag:
            if sev.emergency:
                return sev
            level, dispo = band(max(sev.score, 90))
            hits = [{"english": r, "canonical": r.lower(), "flag": EMERGENCY, "source": "shared_detector"}
                    for r in shared.reasons]
            return SeverityResult(
                score=max(sev.score, 90), level=level, disposition=dispo, emergency=True, alert=ALERT,
                triggered_by="shared_detector", emergency_hits=hits, red_flags=sev.red_flags,
                rule_score=sev.rule_score, ml_score=sev.ml_score,
                factors=sev.factors + ["emergency detector: " + ", ".join(shared.reasons)])
        if sev.emergency and sev.triggered_by == "safety_net":
            score = max(sev.rule_score, sev.ml_score or 0, 70 if sev.red_flags else 0)
            level, dispo = band(score)
            if level == "EMERGENCY":                                              # other evidence still says emergency
                return sev
            return SeverityResult(
                score=score, level=level, disposition=dispo, emergency=False, alert=None,
                triggered_by="rules" if score else "none", emergency_hits=[], red_flags=sev.red_flags,
                rule_score=sev.rule_score, ml_score=sev.ml_score,
                factors=sev.factors + ["emergency phrase overruled: the emergency detector found a denial"])
        return sev
