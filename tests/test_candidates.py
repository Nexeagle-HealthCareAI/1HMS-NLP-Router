"""Unit tests for nlp_brain.candidates -- expanding a single top pick into a
close-margin shortlist of runner-up specialists, and merging multiple
segments' shortlists into one final, capped result.

test_merging_two_capped_segment_lists_can_still_need_recapping (in
TestMergeCandidates) pins down a real bug: a 3-symptom query once returned
5 candidates (specialtyIds all the way out to the API response) despite
MAX_CANDIDATES=3, because classifier.predict() concatenated each segment's
already-capped list without re-applying the cap to the merged result. See
nlp_brain.classifier.SymptomClassifier.predict()'s docstring.
"""
import numpy as np
import pytest

from nlp_brain.candidates import build_candidates, merge_candidates, ranked_labels


class _StubProbaModel:
    """Minimal stand-in for a fitted LogisticRegression/ComplementNB: has
    predict_proba and classes_, nothing else ranked_labels() needs."""
    def __init__(self, classes, proba_row):
        self.classes_ = np.array(classes)
        self._proba_row = np.array(proba_row)

    def predict_proba(self, feats):
        return np.array([self._proba_row])


class _StubDecisionFunctionModel:
    """Minimal stand-in for a fitted LinearSVC: no predict_proba, only
    decision_function + classes_."""
    def __init__(self, classes, scores_row):
        self.classes_ = np.array(classes)
        self._scores_row = np.array(scores_row)

    def decision_function(self, feats):
        return np.array([self._scores_row])


class TestRankedLabels:
    def test_uses_predict_proba_when_available_and_sorts_descending(self):
        model = _StubProbaModel(
            classes=["Cardiologist", "Dentist", "Dermatologist"],
            proba_row=[0.2, 0.7, 0.1],
        )
        ranked = ranked_labels(model, feats=None)

        assert [label for label, _ in ranked] == ["Dentist", "Cardiologist", "Dermatologist"]
        assert ranked[0][1] == pytest.approx(0.7)

    def test_falls_back_to_softmax_of_decision_function_when_no_predict_proba(self):
        # LinearSVC has no predict_proba -- ranked_labels() must still produce
        # a comparable, sorted, sums-to-1 distribution from decision_function.
        model = _StubDecisionFunctionModel(
            classes=["Cardiologist", "Dentist"],
            scores_row=[5.0, 1.0],
        )
        ranked = ranked_labels(model, feats=None)

        assert [label for label, _ in ranked] == ["Cardiologist", "Dentist"]
        assert sum(score for _, score in ranked) == pytest.approx(1.0)
        assert 0.0 <= ranked[1][1] <= ranked[0][1] <= 1.0


class TestBuildCandidates:
    def test_always_includes_at_least_the_top_pick(self):
        ranked = [("Cardiologist", 0.9), ("Dentist", 0.05), ("Dermatologist", 0.05)]
        candidates = build_candidates(ranked, margin=0.0, max_candidates=3)
        assert candidates == ["Cardiologist"]

    def test_includes_runner_ups_within_margin(self):
        ranked = [("Cardiologist", 0.50), ("Dentist", 0.45), ("Dermatologist", 0.05)]
        candidates = build_candidates(ranked, margin=0.12, max_candidates=3)
        assert candidates == ["Cardiologist", "Dentist"]

    def test_excludes_runner_ups_outside_margin(self):
        ranked = [("Cardiologist", 0.80), ("Dentist", 0.15), ("Dermatologist", 0.05)]
        candidates = build_candidates(ranked, margin=0.12, max_candidates=3)
        assert candidates == ["Cardiologist"]

    def test_respects_max_candidates_even_when_more_qualify(self):
        ranked = [
            ("Cardiologist", 0.30), ("Dentist", 0.28), ("Dermatologist", 0.26),
            ("Paediatrician", 0.24), ("Neurologist", 0.22),
        ]
        candidates = build_candidates(ranked, margin=0.50, max_candidates=3)
        assert candidates == ["Cardiologist", "Dentist", "Dermatologist"]

    def test_margin_is_measured_from_the_top_score_not_chained_between_neighbors(self):
        # Dermatologist is 0.13 below Cardiologist (outside a 0.12 margin) even
        # though it's only 0.06 below Dentist -- margin must compare against the
        # TOP score, not accumulate step by step down the ranking.
        ranked = [("Cardiologist", 0.50), ("Dentist", 0.44), ("Dermatologist", 0.37)]
        candidates = build_candidates(ranked, margin=0.12, max_candidates=3)
        assert candidates == ["Cardiologist", "Dentist"]


class TestMergeCandidates:
    def test_single_segment_passes_through_unchanged(self):
        assert merge_candidates([["Dentist", "Cardiologist"]], max_candidates=3) == ["Dentist", "Cardiologist"]

    def test_merges_two_segments_in_first_mention_order(self):
        result = merge_candidates([["Dentist"], ["Cardiologist"]], max_candidates=3)
        assert result == ["Dentist", "Cardiologist"]

    def test_dedupes_a_label_shared_across_segments(self):
        result = merge_candidates([["Dentist", "Cardiologist"], ["Cardiologist", "Dermatologist"]], max_candidates=5)
        assert result == ["Dentist", "Cardiologist", "Dermatologist"]

    def test_merging_two_capped_segment_lists_can_still_need_recapping(self):
        # Each segment individually respects max_candidates=3 -- but two of
        # them combined (6 unique labels) must still come back capped at 3.
        # This is the exact shape of the real bug: a 3-symptom query merged
        # three already-capped segment lists into 5 unique candidates.
        segment_a = ["Neurologist", "Cardiologist", "Dermatologist"]
        segment_b = ["Gastroenterologist", "GI/Surgical Gastroenterologist", "Dentist"]

        result = merge_candidates([segment_a, segment_b], max_candidates=3)

        assert len(result) == 3
        assert result == ["Neurologist", "Cardiologist", "Dermatologist"]

    def test_empty_segment_lists_yield_no_candidates(self):
        assert merge_candidates([[], []], max_candidates=3) == []

    def test_a_segment_with_no_match_contributes_nothing(self):
        # _predict_segment() returns candidates=[] for a gibberish/no-match
        # segment -- merge_candidates() must simply skip it, not error.
        result = merge_candidates([["Dentist"], []], max_candidates=3)
        assert result == ["Dentist"]
