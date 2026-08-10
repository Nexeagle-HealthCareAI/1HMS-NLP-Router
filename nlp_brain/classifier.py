"""The NLP Brain's single public entry point.

Owns two things that used to be duplicated across files (and broke when the
duplicates drifted out of sync):
  1. The joblib bundle schema (save()/load()).
  2. The gibberish-check -> coverage-gate -> classify pipeline (predict()) --
     previously reimplemented a second time by the API layer.
"""
from dataclasses import dataclass
from typing import Optional

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .candidates import build_candidates, merge_candidates, ranked_labels
from .config import CANDIDATE_MARGIN, DATA_PATH, MATCH_THRESHOLD, MAX_CANDIDATES, MODEL_OUT, RANDOM_STATE
from .features import build_feature_union
from .matching import best_match, normalize_matrix
from .segmentation import split_segments
from .text_utils import clean_text, is_gibberish
from .training import evaluate_candidates, load_data


@dataclass
class PredictionResult:
    specialist: Optional[str]
    # Ordered, deduped shortlist of specialists worth showing -- candidates[0]
    # is always `specialist` when it's set. Has more than one entry only when
    # a runner-up's ranked-confidence score is within CANDIDATE_MARGIN of the
    # top pick (see nlp_brain.candidates.build_candidates); [] when no_match.
    candidates: list
    match_ratio: Optional[float]
    closest_known_example: Optional[str]
    flagged_gibberish: bool
    no_match: bool


class SymptomClassifier:
    """The trained pipeline: feature extractor + classifier + the coverage-
    gate's reference corpus, bundled together. Construct one via `.train()`
    (from a CSV) or `.load()` (from a saved bundle) -- calling `__init__`
    directly is for those two classmethods and tests/conftest.py's
    `tiny_classifier` fixture; application code should not build one field
    by field."""

    def __init__(self, features, model, model_name, classes, texts, texts_matrix):
        self.features = features
        self.model = model
        self.model_name = model_name
        self.classes = classes
        self.texts = texts
        self.texts_matrix = texts_matrix
        # Precomputed once here (NOT persisted in the joblib bundle -- cheap
        # to rebuild on load, and storing it would nearly double the bundle's
        # size) so every predict() call only has to normalize a single query
        # row instead of re-normalizing the whole corpus matrix every time.
        # See matching.normalize_matrix() for why this matters.
        self._texts_matrix_normalized = normalize_matrix(texts_matrix)

    @classmethod
    def load(cls, path: str = MODEL_OUT) -> "SymptomClassifier":
        """Loads a previously-saved bundle. Use this at process startup
        (see api/main.py's lifespan()) or in a one-off script/notebook that
        needs predictions without retraining -- NOT per-request; loading
        deserializes the whole corpus matrix from disk and re-normalizes it
        (see __init__), which takes real time (roughly 0.2-2s depending on
        disk cache) that a request handler shouldn't pay."""
        bundle = joblib.load(path)
        return cls(**bundle)

    def save(self, path: str = MODEL_OUT) -> None:
        """Writes this classifier's bundle to disk in the schema `.load()`
        expects. Call this after `.train()` produces a classifier you want
        to keep (nlp_brain/cli.py's `train` command and
        data_pipeline/retrain_pipeline.py's promotion step both do this) --
        this is the ONLY place that schema is defined, specifically so nlp_brain/cli.py
        and retrain_pipeline.py can't drift out of sync with each other (they used
        to each build the bundle dict by hand)."""
        joblib.dump({
            "features": self.features,
            "model": self.model,
            "model_name": self.model_name,
            "classes": self.classes,
            "texts": self.texts,
            "texts_matrix": self.texts_matrix,
        }, path)

    @classmethod
    def train(cls, data_path: str = DATA_PATH, verbose: bool = True) -> tuple:
        """Trains a fresh classifier from a training CSV. Returns
        (classifier, held_out_metrics). held_out_metrics come from a genuine
        80/20 split of `data_path` itself -- NOT
        data_pipeline/validation_set.csv, which has ~97% row overlap with
        the training data and isn't a trustworthy generalization signal."""
        df = load_data(data_path)
        if verbose:
            print(f"Final dataset: {len(df)} rows, {df['specialist'].nunique()} specialists\n")

        X = df["text"].values
        y = df["specialist"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        features = build_feature_union()
        X_train_feats = features.fit_transform(X_train)
        X_test_feats = features.transform(X_test)

        if verbose:
            print("Cross-validating candidate models on training data:")
        best_name, best_clf = evaluate_candidates(X_train_feats, y_train)

        best_clf.fit(X_train_feats, y_train)
        y_pred = best_clf.predict(X_test_feats)

        metrics = {
            "heldOutAccuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "heldOutMacroF1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        }
        if verbose:
            print("\n=== Held-out test set performance ===")
            print(f"Accuracy:  {metrics['heldOutAccuracy']:.4f}")
            print(f"Macro-F1:  {metrics['heldOutMacroF1']:.4f}\n")
            print(classification_report(y_test, y_pred, zero_division=0))

        # Refit feature extractor + model on ALL data for the deployed pipeline.
        final_features = build_feature_union()
        X_all_feats = final_features.fit_transform(X)
        final_clf = type(best_clf)(**best_clf.get_params())
        final_clf.fit(X_all_feats, y)

        classifier = cls(
            features=final_features,
            model=final_clf,
            model_name=best_name,
            classes=sorted(df["specialist"].unique().tolist()),
            texts=df["text"].tolist(),
            texts_matrix=X_all_feats,
        )
        return classifier, metrics

    def predict(self, text: str, threshold: float = MATCH_THRESHOLD) -> PredictionResult:
        """Predict specialist(s) for a piece of Hinglish symptom text.

        A sentence naming more than one problem ("pet mein dard hai aur sar
        bhi dukh raha hai") is split into segments (segmentation.split_segments)
        and each is run through _predict_segment() independently; their
        candidate lists are merged (deduped, in first-mention order) into the
        final result, then re-capped at MAX_CANDIDATES -- each segment's own
        list already respects the cap (see build_candidates()), but merging
        N segments' capped lists can still exceed it (e.g. two 3-candidate
        segments could combine to 6 before this trims it back down). A plain
        single-symptom sentence is just the one-segment case of the same
        path -- nothing here special-cases it.
        """
        cleaned = clean_text(text)
        segments = split_segments(cleaned)
        segment_results = [self._predict_segment(seg, threshold) for seg in segments]

        candidates = merge_candidates([r.candidates for r in segment_results], MAX_CANDIDATES)

        source_by_label: dict = {}
        for result in segment_results:
            for label in result.candidates:
                if label in candidates and label not in source_by_label:
                    source_by_label[label] = result

        if not candidates:
            # No segment cleared the gibberish/coverage bar -- report using the
            # first segment's diagnostics (match_ratio/closest_known_example/
            # flagged_gibberish) so callers still have something to show/debug,
            # same as the single-segment path always did.
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

    def _predict_segment(self, segment: str, threshold: float) -> PredictionResult:
        """The single-segment pipeline predict() used to run directly on the
        whole query -- now run once per segment and merged by predict().
        `segment` is already clean_text()-passed (split_segments() operates
        on already-cleaned input), so no re-cleaning here.

        Two checks run before the classifier's prediction is trusted:
          1. Gibberish check (text_utils.is_gibberish) -- catches
             keyboard-mash input like "dfgskjbnskfjdn".
          2. Coverage check (matching.best_match): cosine similarity between
             the segment's TF-IDF vector and every known training example's --
             if the closest known example isn't similar enough (below
             `threshold`), the classifier's guess isn't trusted.
        """
        if len(segment) < 3:
            return PredictionResult(None, [], None, None, False, True)

        if is_gibberish(segment):
            return PredictionResult(None, [], None, None, True, True)

        ratio, closest = best_match(segment, self.features, self._texts_matrix_normalized, self.texts)
        if ratio < threshold:
            return PredictionResult(None, [], ratio, closest, False, True)

        feats = self.features.transform([segment])
        ranked = ranked_labels(self.model, feats)
        candidates = build_candidates(ranked, CANDIDATE_MARGIN, MAX_CANDIDATES)
        return PredictionResult(candidates[0], candidates, ratio, closest, False, False)
