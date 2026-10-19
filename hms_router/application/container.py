"""Composition root: the one place that knows which concrete class fills each Protocol.

Swap an implementation here (or in a test) - nothing else changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..domain.policy import RoutingPolicy
from ..rag.classifier_scorer import ClassifierScorer, EmergencyModelScore
from ..rag.encoder import TextEncoder
from ..rag.example_retriever import ExampleRetriever
from ..rag.model_bundle import ModelBundle
from ..rag.vocabulary import MedicalVocabulary
from ..reasoning.llm_reasoner import LlmReasoner
from ..reasoning.protocols import NoReasoner
from ..safety.denial import DenialAware
from ..safety.detectors import AnyOf, ModelDetector, RuleDetector
from ..safety.rules import RED_FLAGS_PATH, load_red_flags
from .route_symptom import RouteSymptomService


def _nlp_helpers() -> tuple[Callable[[str], dict], Callable[[str], str]]:
    """Read-only reuse of /chat's spelling + NER + language stages (no Redis, no state)."""
    from ..language import LanguageDetector
    from ..lexicon import Lexicon
    from ..ner import SymptomExtractor
    from ..spelling import SpellingCorrector

    lex = Lexicon.from_csv()
    corrector, extractor, lang = SpellingCorrector(lex), SymptomExtractor(), LanguageDetector(lex)

    def extract(normalized: str) -> dict:
        cor = corrector.correct(normalized)
        obj = extractor.extract(cor, normalized, "unknown", "", 0)
        return {"symptoms": obj.symptoms, "negated_symptoms": obj.negated_symptoms, "corrected": cor.corrected_text}

    def language_of(normalized: str) -> str:
        return lang.detect(normalized).language

    return extract, language_of


def build_route_symptom_service(bundle: ModelBundle | None = None, *, model_path: str | Path | None = None,
                                policy: RoutingPolicy | None = None,
                                llm_complete: Callable[[str, str], str] | None = None,
                                red_flags_path: str | Path = RED_FLAGS_PATH) -> RouteSymptomService:
    bundle = bundle or ModelBundle.load_or_build(model_path)
    policy = policy or RoutingPolicy()
    encoder = TextEncoder(bundle.vectorizer)
    extract, language_of = _nlp_helpers()
    detector = DenialAware(AnyOf([
        RuleDetector(load_red_flags(red_flags_path), extract),
        ModelDetector(EmergencyModelScore(encoder, bundle.emergency_model), policy.emergency_min_score),
    ]))
    return RouteSymptomService(
        retriever=ExampleRetriever(encoder, bundle.index, bundle.examples),
        scorer=ClassifierScorer(encoder, bundle.classifier),
        detector=detector, policy=policy,
        reasoner=LlmReasoner(llm_complete) if llm_complete else NoReasoner(),
        language_of=language_of, model_version=bundle.meta.get("modelVersion"),
        medical_signal=MedicalVocabulary.from_data().has_signal)
