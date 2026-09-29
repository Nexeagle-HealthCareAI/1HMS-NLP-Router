"""Groq-backed counterpart to classifier.py's SymptomClassifier.

Same overall shape as the sklearn pipeline -- gibberish check -> segment ->
classify -> merge candidates -- and returns the identical PredictionResult
(imported from .classifier, not redefined here, so callers that already
depend on that shape don't need to change). The only thing that's actually
different is what "classify" means: instead of a coverage-gate cosine-
similarity check against a training corpus (matching.py) followed by a
trained sklearn model's .predict_proba() (candidates.ranked_labels()), each
segment goes to a Groq-hosted LLM (groq_client.classify_segment()), which
returns its own ranked list and self-reported confidence directly -- no
training corpus, no trained model artifact, no matching.py/features.py/
training.py involved at all.

candidates.merge_candidates() IS reused unchanged below, since it's generic
over any list of per-segment candidate lists and doesn't care how they were
produced. candidates.build_candidates() and candidates.ranked_labels() are
NOT reused -- they're written specifically around a fitted sklearn model
and a fixed CANDIDATE_MARGIN cutoff; this pipeline's candidate-shortlisting
rule (always top GROQ_MAX_CANDIDATES, with DEFAULT_SPECIALIST guaranteed to
appear) is different enough that it's implemented directly below instead.
"""
import os
from typing import List, Optional

from groq import Groq

from .candidates import merge_candidates
from .classifier import PredictionResult
from .config import (
    DEFAULT_SPECIALIST,
    GROQ_CONFIDENCE_THRESHOLD,
    GROQ_MAX_CANDIDATES,
    GROQ_MAX_RETRIES,
    GROQ_MODEL,
)
from .groq_client import classify_segment
from .segmentation import split_segments
from .text_utils import clean_text, is_gibberish


class GroqSymptomClassifier:
    """Classifies Hinglish symptom text into one of a fixed set of
    specialists using a Groq-hosted LLM, in place of a trained sklearn
    model. No .train()/.save()/.load() here -- there's no model artifact to
    fit or persist, just a fixed specialist list. Construct with that list
    directly, or via `from_training_csv` to reuse the label set already
    present in an existing training CSV (e.g. config.DATA_PATH) without
    doing any actual training."""

    def __init__(
        self,
        specialists: List[str],
        api_key: Optional[str] = None,
        model: str = GROQ_MODEL,
        confidence_threshold: float = GROQ_CONFIDENCE_THRESHOLD,
        max_candidates: int = GROQ_MAX_CANDIDATES,
        max_retries: int = GROQ_MAX_RETRIES,
        default_specialist: str = DEFAULT_SPECIALIST,
    ):
        if not specialists:
            raise ValueError("`specialists` must be a non-empty list of allowed labels.")

        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.specialists = specialists
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_candidates = max_candidates
        self.max_retries = max_retries
        # Guaranteed to appear in every non-gibberish result -- see
        # config.DEFAULT_SPECIALIST for the reasoning.
        self.default_specialist = default_specialist

    @classmethod
    def from_training_csv(cls, data_path: str, text_col: str = "text",
                           label_col: str = "specialist", **kwargs) -> "GroqSymptomClassifier":
        """Builds the specialist list from an existing training CSV (e.g.
        the same one training.load_data() reads for the sklearn pipeline),
        without doing any actual training -- just reads the unique labels."""
        import pandas as pd
        df = pd.read_csv(data_path)
        specialists = sorted(df[label_col].dropna().unique().tolist())
        return cls(specialists=specialists, **kwargs)

    # ------------------------------------------------------------------
    # Public API -- mirrors SymptomClassifier.predict()
    # ------------------------------------------------------------------
    def predict(self, text: str) -> PredictionResult:
        """Same segmentation/merge structure as SymptomClassifier.predict():
        a multi-symptom sentence is split into segments, each classified
        independently, then merged into one deduped, capped shortlist. See
        that method's docstring for why merging is needed even though each
        segment's own candidate list is already capped."""
        cleaned = clean_text(text)
        segments = split_segments(cleaned)
        segment_results = [self._predict_segment(seg) for seg in segments]

        candidates = merge_candidates(
            [r.candidates for r in segment_results], self.max_candidates
        )

        source_by_label = {}
        for result in segment_results:
            for label in result.candidates:
                if label in candidates and label not in source_by_label:
                    source_by_label[label] = result

        if not candidates:
            first = segment_results[0]
            return PredictionResult(
                None, [], first.match_ratio, first.closest_known_example,
                first.flagged_gibberish, True,
            )

        primary = source_by_label[candidates[0]]
        return PredictionResult(
            candidates[0], candidates, primary.match_ratio,
            primary.closest_known_example, False, False,
        )

    # ------------------------------------------------------------------
    # Per-segment pipeline -- mirrors SymptomClassifier._predict_segment()
    # ------------------------------------------------------------------
    def _predict_segment(self, segment: str) -> PredictionResult:
        """Two differences from the sklearn version: (1) there's no
        coverage-gate cosine-similarity check, since there's no training
        corpus to check coverage against -- the LLM's own confidence plays
        that role instead; (2) any non-gibberish segment always gets at
        least DEFAULT_SPECIALIST, rather than ever reporting no_match, since
        an LLM with a fixed specialist list can always make some guess."""
        if len(segment) < 3:
            return PredictionResult(None, [], None, None, False, True)

        if is_gibberish(segment):
            return PredictionResult(None, [], None, None, True, True)

        result = classify_segment(
            self.client, segment, self.specialists, self.model, self.max_retries,
        )

        if result is None:
            # API call failed outright -- fall back to the safe default
            # rather than reporting no_match for a perfectly valid query.
            return PredictionResult(
                self.default_specialist, [self.default_specialist],
                None, None, False, False,
            )

        confidence = result["confidence"]
        if confidence < self.confidence_threshold or not result["ranked_labels"]:
            return PredictionResult(
                self.default_specialist, [self.default_specialist],
                confidence, None, False, False,
            )

        candidates = self._build_candidates(result["ranked_labels"])
        return PredictionResult(candidates[0], candidates, confidence, None, False, False)

    # ------------------------------------------------------------------
    # Candidate shortlist -- always the top N by score, default guaranteed
    # ------------------------------------------------------------------
    def _build_candidates(self, ranked_labels) -> List[str]:
        """Returns up to `max_candidates` specialists ranked by score.
        Unlike candidates.build_candidates(), this is NOT filtered by a
        margin from the top score -- it always returns the top N Groq
        provided, however far apart their scores are."""
        ranked_sorted = sorted(ranked_labels, key=lambda pair: pair[1], reverse=True)
        candidates = [label for label, _ in ranked_sorted[: self.max_candidates]]
        return self._ensure_default_included(candidates)

    def _ensure_default_included(self, candidates: List[str]) -> List[str]:
        """Guarantees `default_specialist` is in the final shortlist for any
        non-gibberish input. If already present, the list is untouched;
        otherwise it's added, dropping the lowest-ranked existing entry
        (not truncating from the end, which would cut off the newly-added
        default itself) to stay within `max_candidates`."""
        if self.default_specialist in candidates:
            return candidates
        if len(candidates) < self.max_candidates:
            return candidates + [self.default_specialist]
        return candidates[: self.max_candidates - 1] + [self.default_specialist]
