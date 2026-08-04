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

from .config import DATA_PATH, MATCH_THRESHOLD, MODEL_OUT, RANDOM_STATE
from .features import build_feature_union
from .matching import best_match
from .text_utils import clean_text, is_gibberish
from .training import evaluate_candidates, load_data


@dataclass
class PredictionResult:
    specialist: Optional[str]
    match_ratio: Optional[float]
    closest_known_example: Optional[str]
    flagged_gibberish: bool
    no_match: bool


class SymptomClassifier:
    def __init__(self, features, model, model_name, classes, texts, texts_matrix):
        self.features = features
        self.model = model
        self.model_name = model_name
        self.classes = classes
        self.texts = texts
        self.texts_matrix = texts_matrix

    @classmethod
    def load(cls, path: str = MODEL_OUT) -> "SymptomClassifier":
        bundle = joblib.load(path)
        return cls(**bundle)

    def save(self, path: str = MODEL_OUT) -> None:
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
        """Predict a specialist for a piece of Hinglish symptom text.

        Two checks run before the classifier's prediction is trusted:
          1. Gibberish check (text_utils.is_gibberish) -- catches
             keyboard-mash input like "dfgskjbnskfjdn".
          2. Coverage check (matching.best_match): cosine similarity between
             the input's TF-IDF vector and every known training example's --
             if the closest known example isn't similar enough (below
             `threshold`), the classifier's guess isn't trusted.
        """
        cleaned = clean_text(text)

        if len(cleaned) < 3:
            return PredictionResult(None, None, None, False, True)

        if is_gibberish(cleaned):
            return PredictionResult(None, None, None, True, True)

        ratio, closest = best_match(cleaned, self.features, self.texts_matrix, self.texts)
        if ratio < threshold:
            return PredictionResult(None, ratio, closest, False, True)

        feats = self.features.transform([cleaned])
        pred = self.model.predict(feats)[0]
        return PredictionResult(pred, ratio, closest, False, False)
