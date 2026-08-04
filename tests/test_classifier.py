"""Tests for nlp_brain.classifier.SymptomClassifier -- the NLP Brain's one
public entry point. If you're adding a new field to the joblib bundle or
changing predict()'s decision logic, this is the file to extend.
"""
import joblib
import pytest

from nlp_brain.classifier import PredictionResult, SymptomClassifier
from tests.conftest import requires_real_bundle, REAL_MODEL_PATH


class TestTrain:
    def test_produces_a_working_classifier(self, tiny_classifier):
        assert tiny_classifier.model is not None
        assert tiny_classifier.features is not None
        assert set(tiny_classifier.classes) == {
            "Cardiologist", "Dentist", "Dermatologist", "Paediatrician",
        }

    def test_returns_held_out_metrics(self, sample_training_csv):
        _, metrics = SymptomClassifier.train(sample_training_csv, verbose=False)
        assert "heldOutAccuracy" in metrics
        assert "heldOutMacroF1" in metrics
        assert 0.0 <= metrics["heldOutAccuracy"] <= 1.0

    def test_raises_a_clear_error_for_a_missing_required_column(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("text,specialist\nfoo,Dentist\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing expected column"):
            SymptomClassifier.train(str(bad_csv), verbose=False)


class TestPredict:
    def test_routes_a_clear_query_to_the_right_specialist(self, tiny_classifier):
        result = tiny_classifier.predict("dant mein bahut dard ho raha hai kaafi dino se")
        assert result.specialist == "Dentist"
        assert result.no_match is False
        assert result.flagged_gibberish is False

    def test_gibberish_input_is_rejected_before_reaching_the_classifier(self, tiny_classifier):
        result = tiny_classifier.predict("zxcvbnmlkjhgfdsaqwerty")
        assert result == PredictionResult(None, [], None, None, True, True)

    def test_too_short_input_is_rejected_without_being_called_gibberish(self, tiny_classifier):
        # Distinguishes "nothing to work with" from "flagged as nonsense" --
        # api/routes.py surfaces these differently (raw.flaggedGibberish).
        result = tiny_classifier.predict("ok")
        assert result.no_match is True
        assert result.flagged_gibberish is False

    def test_empty_string_is_rejected(self, tiny_classifier):
        result = tiny_classifier.predict("")
        assert result.no_match is True

    def test_off_topic_but_coherent_text_is_rejected_as_no_match(self, tiny_classifier):
        # Nothing in the training vocabulary resembles this -- the coverage
        # gate should catch it even though is_gibberish() wouldn't (it's
        # made of real English words, just not medical ones).
        result = tiny_classifier.predict("what is the weather forecast for tomorrow")
        assert result.no_match is True
        assert result.flagged_gibberish is False

    def test_match_ratio_is_populated_on_a_confident_prediction(self, tiny_classifier):
        result = tiny_classifier.predict("bachche ko bukhar hai aur khaana nahi kha raha")
        assert result.match_ratio is not None
        assert 0.0 <= result.match_ratio <= 1.0
        assert result.closest_known_example is not None

    def test_candidates_starts_with_the_specialist_and_is_bounded(self, tiny_classifier):
        from nlp_brain.config import MAX_CANDIDATES

        result = tiny_classifier.predict("dant mein bahut dard ho raha hai kaafi dino se")
        assert result.candidates[0] == result.specialist
        assert len(result.candidates) <= MAX_CANDIDATES
        assert len(result.candidates) == len(set(result.candidates))  # no duplicates

    def test_no_match_and_gibberish_predictions_have_no_candidates(self, tiny_classifier):
        assert tiny_classifier.predict("zxcvbnmlkjhgfdsaqwerty").candidates == []
        assert tiny_classifier.predict("what is the weather forecast for tomorrow").candidates == []
        assert tiny_classifier.predict("").candidates == []


class TestSaveLoadRoundTrip:
    def test_save_then_load_preserves_prediction_behavior(self, tiny_classifier, tmp_path):
        path = str(tmp_path / "roundtrip.joblib")
        tiny_classifier.save(path)

        reloaded = SymptomClassifier.load(path)
        query = "skin par laal daane ho gaye hain aur khujli ho rahi hai"

        assert reloaded.predict(query) == tiny_classifier.predict(query)

    def test_saved_bundle_has_the_expected_schema(self, tiny_classifier, tmp_path):
        # Pins down the bundle's on-disk shape -- data_pipeline/retrain_pipeline.py
        # builds one of these by hand (not via .train()), so this is what
        # keeps that construction honest. If this test needs to change, the
        # retrain pipeline's SymptomClassifier(...) call needs to change too.
        path = str(tmp_path / "schema_check.joblib")
        tiny_classifier.save(path)
        bundle = joblib.load(path)
        assert set(bundle.keys()) == {"features", "model", "model_name", "classes", "texts", "texts_matrix"}


@requires_real_bundle
@pytest.mark.integration
class TestRealBundleIntegration:
    """Sanity-checks the ACTUAL currently-committed model artifact, not a
    test fixture. Runs by default (the bundle is committed to the repo, so
    it's normally present) but is slower -- loads a ~35MB bundle -- so the
    pre-commit hook's fast local loop excludes it via `-m "not integration"`.
    Also self-skips if the bundle/dataset files aren't present."""

    @pytest.fixture(scope="class")
    def real_classifier(self):
        return SymptomClassifier.load(str(REAL_MODEL_PATH))

    def test_loads_without_error(self, real_classifier):
        assert real_classifier.model is not None
        assert len(real_classifier.classes) > 0

    def test_sample_queries_all_predict_a_specialist(self, real_classifier):
        from nlp_brain.config import SAMPLE_QUERIES
        for query in SAMPLE_QUERIES:
            result = real_classifier.predict(query)
            assert result.specialist is not None, f"expected a match for {query!r}, got no_match"

    def test_gibberish_samples_are_all_rejected(self, real_classifier):
        from nlp_brain.config import GIBBERISH_SAMPLE_QUERIES
        for query in GIBBERISH_SAMPLE_QUERIES:
            result = real_classifier.predict(query)
            assert result.no_match is True, f"expected no_match for {query!r}, got {result.specialist}"
