"""Tests for data_pipeline/retrain_pipeline.py's pure logic -- the parts
that decide WHAT gets trained on and WHETHER a candidate model is safe to
promote. Deliberately does not exercise fetch_live() (needs a real CMSAPI)
or main() end-to-end (needs live/fixture data plus a full retrain) -- those
are exercised by actually running the nightly workflow. This file is where
you add a test if you're changing the merge/dedupe/regression-check rules.
"""
import pytest

from data_pipeline.retrain_pipeline import (
    TOLERANCE,
    WEIGHT_ACCEPTED,
    WEIGHT_BASE,
    WEIGHT_CORRECTION,
    evaluate,
    feedback_to_rows,
    is_regression,
    merge_rows,
)


class TestFeedbackToRows:
    def test_correction_uses_the_actually_booked_specialty_at_double_weight(self):
        feedback = [{
            "query": "dant mein dard hai", "hasBooking": True, "wasCorrection": True,
            "predictedSpecialtyId": "cardiology", "actualBookedSpecialtyId": "dentistry",
        }]
        rows = feedback_to_rows(feedback)
        assert rows == [("dant mein dard hai", "Dentist", "Production Feedback - Correction", WEIGHT_CORRECTION)]

    def test_silent_accept_uses_the_predicted_specialty_at_half_weight(self):
        feedback = [{
            "query": "dant mein dard hai", "hasBooking": True, "wasCorrection": False,
            "predictedSpecialtyId": "dentistry",
        }]
        rows = feedback_to_rows(feedback)
        assert rows == [("dant mein dard hai", "Dentist", "Production Feedback - Accepted", WEIGHT_ACCEPTED)]

    def test_rows_with_no_booking_carry_no_signal_and_are_dropped(self):
        feedback = [{"query": "dant mein dard hai", "hasBooking": False}]
        assert feedback_to_rows(feedback) == []

    def test_rows_with_an_unmappable_specialty_id_are_dropped(self):
        feedback = [{
            "query": "dant mein dard hai", "hasBooking": True, "wasCorrection": False,
            "predictedSpecialtyId": "not-a-real-specialty",
        }]
        assert feedback_to_rows(feedback) == []

    def test_rows_with_an_empty_query_are_dropped(self):
        feedback = [{
            "query": "  ", "hasBooking": True, "wasCorrection": False,
            "predictedSpecialtyId": "dentistry",
        }]
        assert feedback_to_rows(feedback) == []


class TestMergeRows:
    def test_training_examples_and_feedback_both_appear(self):
        training = [{"text": "dant mein dard hai", "specialist": "Dentist", "type": "CMS"}]
        feedback_rows = [("sar mein dard hai", "Neurologist", "Production Feedback - Accepted", WEIGHT_ACCEPTED)]

        texts, labels, types, weights = merge_rows(training, feedback_rows)

        assert texts == ["dant mein dard hai", "sar mein dard hai"]
        assert labels == ["Dentist", "Neurologist"]
        assert weights == [WEIGHT_BASE, WEIGHT_ACCEPTED]

    def test_training_examples_win_ties_over_feedback(self):
        # Same normalized text + specialist from both sources -- the
        # DB-curated training example should be kept, not the feedback copy
        # (which would carry the wrong weight/type for a curated row).
        training = [{"text": "Dant Mein Dard Hai", "specialist": "Dentist", "type": "CMS"}]
        feedback_rows = [("dant mein dard hai", "Dentist", "Production Feedback - Accepted", WEIGHT_ACCEPTED)]

        texts, labels, types, weights = merge_rows(training, feedback_rows)

        assert len(texts) == 1
        assert weights == [WEIGHT_BASE]
        assert types == ["CMS"]

    def test_different_specialists_for_the_same_text_are_both_kept(self):
        training = [{"text": "dard hai", "specialist": "Dentist", "type": "CMS"}]
        feedback_rows = [("dard hai", "Neurologist", "Production Feedback - Accepted", WEIGHT_ACCEPTED)]

        texts, labels, types, weights = merge_rows(training, feedback_rows)

        assert len(texts) == 2
        assert set(labels) == {"Dentist", "Neurologist"}


class TestIsRegression:
    def test_no_baseline_is_never_a_regression(self):
        assert is_regression({"top1Accuracy": 0.5, "confidentlyWrongRate": 0.5}, None) is False

    def test_baseline_with_no_validation_metrics_is_never_a_regression(self):
        assert is_regression({"top1Accuracy": 0.5, "confidentlyWrongRate": 0.5}, {}) is False

    def test_baseline_with_an_incompatible_schema_is_never_a_regression(self):
        baseline = {"validationMetrics": {"topKAccuracy": 0.9}}  # old, pre-migration schema
        assert is_regression({"top1Accuracy": 0.5, "confidentlyWrongRate": 0.5}, baseline) is False

    def test_lower_accuracy_beyond_tolerance_is_a_regression(self):
        baseline = {"validationMetrics": {"top1Accuracy": 0.90, "confidentlyWrongRate": 0.05}}
        candidate = {"top1Accuracy": 0.90 - TOLERANCE - 0.001, "confidentlyWrongRate": 0.05}
        assert is_regression(candidate, baseline) is True

    def test_accuracy_drop_within_tolerance_is_not_a_regression(self):
        baseline = {"validationMetrics": {"top1Accuracy": 0.90, "confidentlyWrongRate": 0.05}}
        candidate = {"top1Accuracy": 0.90 - TOLERANCE / 2, "confidentlyWrongRate": 0.05}
        assert is_regression(candidate, baseline) is False

    def test_higher_wrong_rate_beyond_tolerance_is_a_regression(self):
        baseline = {"validationMetrics": {"top1Accuracy": 0.90, "confidentlyWrongRate": 0.05}}
        candidate = {"top1Accuracy": 0.90, "confidentlyWrongRate": 0.05 + TOLERANCE + 0.001}
        assert is_regression(candidate, baseline) is True

    def test_improvement_on_both_metrics_is_not_a_regression(self):
        baseline = {"validationMetrics": {"top1Accuracy": 0.90, "confidentlyWrongRate": 0.05}}
        candidate = {"top1Accuracy": 0.95, "confidentlyWrongRate": 0.02}
        assert is_regression(candidate, baseline) is False


class TestEvaluate:
    def test_returns_metrics_in_the_expected_shape(self, tiny_classifier, sample_training_rows):
        val_texts = [row[0] for row in sample_training_rows[:8]]
        val_labels = [row[1] for row in sample_training_rows[:8]]

        metrics = evaluate(
            tiny_classifier.features, tiny_classifier.model,
            tiny_classifier.texts, tiny_classifier.texts_matrix,
            val_texts, val_labels,
        )

        assert set(metrics.keys()) == {"top1Accuracy", "noMatchRate", "confidentlyWrongRate"}
        for value in metrics.values():
            assert 0.0 <= value <= 1.0
