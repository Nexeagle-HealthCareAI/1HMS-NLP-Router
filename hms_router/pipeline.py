"""Glue: Step 1 (ingestion) + Step 2 (normalization & spelling) + Step 3 (NER -> symptom object)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .language import LanguageDetector, LanguageResult
from .lexicon import Lexicon
from .script import normalize_script
from .session import SessionManager
from .ner import SymptomExtractor, SymptomObject
from .spelling import CorrectionResult, RedisReviewQueue, SpellingCorrector


@dataclass
class IngestionResult:
    session_id: str
    is_new_session: bool
    turn: int
    raw_text: str
    language: LanguageResult
    normalized_text: str
    combined_context: str        # all normalized messages in this session, oldest first


@dataclass
class PipelineResult:
    ingestion: IngestionResult
    correction: CorrectionResult
    symptom_object: SymptomObject


class HMSRouterPipeline:
    def __init__(self, redis_client, lexicon: Lexicon | None = None, settings: Settings | None = None,
                 extractor: SymptomExtractor | None = None):
        self.cfg = settings or Settings()
        self.lex = lexicon or Lexicon.from_csv()
        self.sessions = SessionManager(redis_client, self.cfg.session_ttl_seconds)
        self.detector = LanguageDetector(self.lex, self.cfg.fasttext_model_path)
        self.extractor = extractor or SymptomExtractor()
        self.corrector = SpellingCorrector(self.lex, self.cfg, RedisReviewQueue(redis_client, self.cfg.review_queue_key))

    # -------- Step 1
    def ingest(self, text: str, session_id: str | None = None) -> IngestionResult:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty message")
        text = text[: self.cfg.max_input_chars]

        sid, is_new = self.sessions.open(session_id)             # active session? append : init new (TTL 30 min)
        lang = self.detector.detect(text)                        # fasttext / marker heuristic
        normalized = normalize_script(text)                      # Devanagari -> lowercase Roman
        turn = self.sessions.append(sid, {"raw": text, "normalized": normalized, "language": lang.language})
        self.sessions.set_context(sid, language=lang.language, script=lang.script)
        combined = " | ".join(m["normalized"] for m in self.sessions.history(sid))
        return IngestionResult(sid, is_new, turn, text, lang, normalized, combined)

    # -------- Step 2 / Step 3 as separate stages (the agent runs the safety net between them and the rest)
    def correct(self, ing: IngestionResult) -> CorrectionResult:
        return self.corrector.correct(ing.normalized_text)

    def extract(self, ing: IngestionResult, cor: CorrectionResult) -> SymptomObject:
        return self.extractor.extract(cor, ing.raw_text, ing.language.language, ing.session_id, ing.turn)

    def reload_lexicon(self, lexicon: Lexicon) -> None:
        """Step 8: swap in a grown lexicon (rebuilds the spelling indexes; sessions are untouched).
        Bug #6 fix: also propagate the new lexicon to any extractor backends that hold a
        lexicon reference (e.g. BertNER), so their phrase_index stays current."""
        self.lex = lexicon
        self.detector = LanguageDetector(lexicon, self.cfg.fasttext_model_path)
        self.corrector = SpellingCorrector(lexicon, self.cfg, self.corrector.unknown_sink)
        # Update any NER backends that hold a lexicon (BertNER stores self.lex)
        for backend in getattr(self.extractor, "backends", []):
            if hasattr(backend, "lex"):
                backend.lex = lexicon

    # -------- Steps 1 + 2 + 3
    def process(self, text: str, session_id: str | None = None) -> PipelineResult:
        ing = self.ingest(text, session_id)
        cor = self.correct(ing)
        obj = self.extract(ing, cor)                                                                      # Step 3
        return PipelineResult(ing, cor, obj)
