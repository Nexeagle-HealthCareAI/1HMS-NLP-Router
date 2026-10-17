"""Layer 4 - Specialist Routing Engine  (flowchart: Symptom-Specialist Matcher → Emergency Flag Check
→ Generate 3 Specialist Outputs  |  EMERGENCY ALERT → Route to ER / Ambulance).

Flowchart implemented here
──────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  LAYER 4 — SPECIALIST ROUTING ENGINE                         │
  │                                                              │
  │  [Layer 2 RetrievalResult]                                   │
  │          │                                                   │
  │          ▼                                                   │
  │  ┌───────────────────────────┐                               │
  │  │  Symptom-Specialist       │  top-K clusters from vector   │
  │  │  Matcher                  │  store, reranked by symptom   │
  │  │  (Top-3 candidates)       │  overlap & body-part match    │
  │  └─────────────┬─────────────┘                               │
  │                │                                             │
  │                ▼                                             │
  │  ┌─────────────────────────┐                                 │
  │  │  Emergency Flag Check   │  severity.emergency_check()     │
  │  │  (diamond decision)     │  + ML severity score            │
  │  └──────┬──────────────────┘                                 │
  │         │                                                    │
  │    ┌────┴──────┐                                             │
  │    │ NO        │ YES                                         │
  │    ▼           ▼                                             │
  │  Generate 3   EMERGENCY ALERT                                │
  │  Specialists  Route → ER / Ambulance                         │
  │  Primary      (ResponseCard kind=emergency)                  │
  │  Secondary                                                   │
  │  Alternative                                                 │
  └──────────────────────────────────────────────────────────────┘

Public API
──────────
  RoutingEngine(retrieval_layer, severity_classifier)
      .route(state, texts) → RoutingDecision

  RoutingDecision
      .emergency         bool
      .routing           RoutingResult    (from Layer 2 RAG)
      .severity          SeverityResult   (from Step 5)
      .card              ResponseCard     (Layer 5 output)
      .message           str              rendered text for the patient
      .to_dict()
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .output import OutputGenerator, ResponseCard
from .retrieval import RetrievalLayer, RetrievalResult
from .severity import EMERGENCY, SeverityClassifier, SeverityResult, emergency_check

log = logging.getLogger("hms_router.routing_engine")


# =============================================================================
# Result dataclass
# =============================================================================
@dataclass
class RoutingDecision:
    """Complete output of the Layer 4 Routing Engine for one request.

    Attributes
    ----------
    emergency       True when any emergency flag was raised (safety net OR ML score).
    routing         The RAG RoutingResult from Layer 2 (recommendations list).
    severity        Full SeverityResult from Step 5.
    card            The ResponseCard built by the OutputGenerator (Layer 5).
    message         Patient-facing text rendered in the detected language.
    session_id      Echoed from the input state.
    turn            Current conversation turn number.
    routing_ms      Wall-clock time for the full routing step (ms).
    """
    emergency: bool
    routing: Any          # specialist_rag.RoutingResult
    severity: SeverityResult
    card: ResponseCard
    message: str
    session_id: str
    turn: int
    language: str = "english"
    routing_ms: float = 0.0
    degraded: list[str] = field(default_factory=list)  # components that soft-failed

    def to_dict(self) -> dict:
        d = asdict(self)
        d["routing"] = self.routing.to_dict()
        d["severity"] = self.severity.to_dict()
        d["card"] = self.card.to_dict()
        return d


# =============================================================================
# Emergency gate  (the diamond in the flowchart)
# =============================================================================
class EmergencyGate:
    """The 'Emergency Flag Check' diamond node in the flowchart.

    Combines two independent signals:
      1. Hardcoded safety net (severity.emergency_check) — scans symptom state
         and raw text for known emergency phrases. Always runs first.
      2. ML-based severity score (SeverityClassifier.classify) — can raise the
         final score into EMERGENCY band but cannot lower it.

    Design rule: emergency is never delayed or suppressed by any other logic.
    """

    def __init__(self, classifier: SeverityClassifier):
        self.classifier = classifier

    def check(
        self,
        state: dict,
        texts: list[str],
        rules: list | None = None,
    ) -> tuple[bool, SeverityResult]:
        """Run the full emergency check pipeline.

        Args:
            state: AccumulatedState.to_dict()
            texts: All normalized user messages in the session (for phrase scan).
            rules: Optional extra emergency rules (loaded from emergency_keywords.csv).

        Returns:
            (is_emergency, severity_result)
        """
        # Step 1: hardcoded safety net — runs before any ML
        hits = emergency_check(state, texts, rules or self.classifier.rules)
        safety_net_fired = any(h["flag"] == EMERGENCY for h in hits)

        # Step 2: rule-based + optional ML score
        sev = self.classifier.classify(state, texts)

        # Safety net always wins: if it fired, lock to emergency regardless of score
        if safety_net_fired and not sev.emergency:
            log.warning(
                "Safety net overrides ML: emergency_check fired but classifier said %s",
                sev.level,
            )
            # Patch sev to reflect the override
            object.__setattr__(sev, "emergency", True) if hasattr(sev, "__setattr__") else None
            # Use dataclass replacement if frozen
            try:
                from dataclasses import replace
                sev = replace(sev, emergency=True, level="EMERGENCY",
                              alert="Safety net emergency override",
                              triggered_by=["safety_net_override"])
            except TypeError:
                pass  # SeverityResult not frozen — direct attribute set
            try:
                sev.emergency = True
                sev.level = "EMERGENCY"
            except AttributeError:
                pass

        return sev.emergency or safety_net_fired, sev


# =============================================================================
# Symptom-Specialist Matcher  (top node of Layer 4)
# =============================================================================
class SymptomSpecialistMatcher:
    """Wraps the RetrievalLayer's vector search + rerank to produce top-3 candidates.

    This is the 'Symptom-Specialist Matcher / Top-3 specialist candidates' box
    in the flowchart.  It delegates entirely to VectorRetriever.retrieve().
    """

    def __init__(self, retrieval_layer: RetrievalLayer):
        self.layer = retrieval_layer

    def match(self, state: dict, severity: dict | None = None) -> RetrievalResult:
        """Run Layer 2 retrieval and return a RetrievalResult with top-3 candidates."""
        return self.layer.retrieve(state, severity)


# =============================================================================
# Routing Engine  (full Layer 4 implementation)
# =============================================================================
class RoutingEngine:
    """Layer 4 — Specialist Routing Engine.

    Executes the flowchart in order:
      1. Symptom-Specialist Matcher (top-3 via Layer 2 RAG)
      2. Emergency Flag Check (safety net + ML severity)
      3a. Emergency path  → EMERGENCY ALERT → ResponseCard(kind=emergency)
      3b. Normal path     → 3 specialist outputs (Primary / Secondary / Alternative)

    Args
    ----
    retrieval_layer   : RetrievalLayer — the Layer 2 facade (session + vector + LTM).
    severity_classifier: SeverityClassifier — Step 5 rule + optional ML scorer.
    extra_rules       : Additional emergency rules from emergency_keywords.csv.
    """

    def __init__(
        self,
        retrieval_layer: RetrievalLayer,
        severity_classifier: SeverityClassifier | None = None,
        extra_rules: list | None = None,
    ):
        self.retrieval = retrieval_layer
        self.classifier = severity_classifier or SeverityClassifier(extra_rules=extra_rules)
        self.gate = EmergencyGate(self.classifier)
        self.matcher = SymptomSpecialistMatcher(retrieval_layer)
        self.output_gen = OutputGenerator()

    # ------------------------------------------------------------------ main entry
    def route(
        self,
        state: dict,
        texts: list[str] | None = None,
        language: str = "english",
        degraded: list[str] | None = None,
    ) -> RoutingDecision:
        """Run the full Layer 4 flowchart for one patient turn.

        Args:
            state:    AccumulatedState.to_dict() — merged symptom state from Layer 3.
            texts:    All normalized user messages in the session (for safety net phrase scan).
                      If None, falls back to session cache via RetrievalLayer.
            language: Detected patient language (hinglish / hindi / english) for output rendering.
            degraded: List of component names that soft-failed upstream (passed through to result).

        Returns:
            RoutingDecision with the full routing result, severity, response card, and message.
        """
        t0 = time.perf_counter()
        session_id = state.get("session_id", "")
        turn = state.get("turn", 0)
        _degraded = list(degraded or [])

        # ── Step 1: Symptom-Specialist Matcher ────────────────────────────────
        # Pull session context + run RAG (embed → ANN → rerank → top-3)
        try:
            retrieval_result = self.matcher.match(state)
        except Exception:
            log.exception("VectorRetriever failed; continuing with empty candidates")
            _degraded.append("vector_retrieval")
            from .specialist_rag import RoutingResult
            retrieval_result_routing = RoutingResult([], False, True, "")
            retrieval_result = type("_R", (), {"routing": retrieval_result_routing,
                                               "session_summary": {},
                                               "ltm_health": {}})()

        routing = getattr(retrieval_result, "routing", retrieval_result)

        # ── Step 2: Emergency Flag Check (diamond) ────────────────────────────
        # Texts for phrase scan: passed in, or pulled from session cache
        if texts is None:
            texts = self.retrieval.session.all_normalized_texts(session_id)

        is_emergency, sev = self.gate.check(state, texts)

        # ── Step 3: Build response (two branches of the diamond) ──────────────
        try:
            card = self.output_gen.build(state, sev.to_dict(), routing)
        except Exception:
            log.exception("OutputGenerator failed; building minimal fallback card")
            _degraded.append("output_generator")
            card = self._fallback_card(sev, is_emergency)

        try:
            message = self.output_gen.render(card, language)
        except Exception:
            log.exception("OutputGenerator.render failed")
            _degraded.append("output_render")
            message = card.summary or "Please seek medical advice."

        routing_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info(
            "route: session=%s turn=%d emergency=%s level=%s specialists=%s ms=%.1f degraded=%s",
            session_id, turn, is_emergency, sev.level,
            [r.specialist for r in routing.recommendations[:3]],
            routing_ms, _degraded or None,
        )

        return RoutingDecision(
            emergency=is_emergency,
            routing=routing,
            severity=sev,
            card=card,
            message=message,
            session_id=session_id,
            turn=turn,
            language=language,
            routing_ms=routing_ms,
            degraded=_degraded,
        )

    # ------------------------------------------------------------------ fallback
    @staticmethod
    def _fallback_card(sev: SeverityResult, is_emergency: bool) -> ResponseCard:
        """Minimal ResponseCard when OutputGenerator itself fails."""
        from .specialist_rag import EMERGENCY_SPECIALIST, FALLBACKS
        from .output import SpecialistOutput, ResponseCard, GENERAL_WATCH_FOR
        if is_emergency:
            return ResponseCard(
                kind="emergency",
                summary="Emergency symptoms detected",
                severity_level="EMERGENCY",
                severity_score=100,
                disposition="Go to ER / call 112 immediately",
                outputs=[SpecialistOutput("EMERGENCY", EMERGENCY_SPECIALIST, 1.0,
                                         "Emergency flag raised", [])],
                watch_for=[],
                alert=sev.alert or "EMERGENCY — seek immediate medical attention",
            )
        return ResponseCard(
            kind="recommendation",
            summary="Symptom assessment incomplete",
            severity_level=sev.level,
            severity_score=sev.score,
            disposition=sev.disposition,
            outputs=[SpecialistOutput("PRIMARY", FALLBACKS[0], 0.0,
                                     "Fallback — please describe your symptoms in more detail", [])],
            watch_for=[GENERAL_WATCH_FOR],
            low_confidence=True,
        )

    # ------------------------------------------------------------------ info
    def corpus_info(self) -> dict:
        """Summary of the vector DB corpus (useful for health checks)."""
        return {
            "corpus_size": self.retrieval.vector.corpus_size,
            "top_k": self.retrieval.vector._retriever.top_k,
            "min_confidence": self.retrieval.vector._retriever.min_confidence,
        }
