"""
Hinglish Symptom -> Specialist Classifier
==========================================
Trains a text classifier that maps a Hinglish (Hindi-English code-mixed,
Roman script) symptom description to the medical specialist it should be
routed to.

Dataset: Hinglish_Symptoms_Reference_v3_combined.csv
Columns: text, specialist, type
test
Approach
--------
- Char-ngram TF-IDF (robust to the spelling variations present in the data,
  e.g. "mein"/"me", "doctor"/"daktar"/"docter") combined with word-ngram
  TF-IDF (captures medical keywords/terms).
- Compares Linear SVM vs Logistic Regression vs Complement Naive Bayes
  via stratified cross-validation, picks the best, then reports held-out
  test performance.
- Saves the trained pipeline + label list to disk for reuse.
"""

import re
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score, f1_score

DATA_PATH = "Hinglish_Symptoms_Reference_V4_.csv"
MODEL_OUT = "symptom_specialist_classifier.joblib"
RANDOM_STATE = 42

# Sample queries to sanity-check the trained model on (edit/add your own).
SAMPLE_QUERIES = [
    "Dil mein bahut zor se dard ho raha hai aur saans phool rahi hai",
    "Bachche ko bukhar hai aur khaana nahi kha raha",
    "Skin par laal daane ho gaye hain aur khujli ho rahi hai",
    "Dant mein bahut dard hai, thanda garam nahi sehta",
]


class TextSelector(BaseEstimator, TransformerMixin):
    """Pass a text column straight through a FeatureUnion step."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def clean_text(s: str) -> str:
    """Light normalization: collapse whitespace, strip. Case and punctuation
    are left mostly intact -- they carry signal here (e.g. 'BP', 'CT scan')."""
    return " ".join(str(s).split())


def is_gibberish(text: str) -> bool:
    """Heuristic check for gibberish / nonsensical input.

    Flags text that is empty, too short, has almost no vowels, contains a
    long run of repeated characters, or has a long run of consonants with
    no vowel in between -- all signs of random keyboard mashing rather than
    an actual (Hinglish) symptom description. This is intentionally a
    simple, dependency-free heuristic rather than a model, so it runs
    instantly before we bother the classifier."""
    text = clean_text(text)

    if len(text) < 3:
        return True

    letters_only = re.sub(r"[^a-zA-Z]", "", text)
    if len(letters_only) < 3:
        return True

    # Almost no vowels at all -> unlikely to be real words.
    vowels = sum(1 for c in letters_only.lower() if c in "aeiou")
    vowel_ratio = vowels / len(letters_only)
    if vowel_ratio < 0.15:
        return True

    # Same character repeated 4+ times in a row (e.g. "aaaaa", "asdfff").
    if re.search(r"(.)\1{3,}", text):
        return True

    # A run of 6+ consonants with no vowel in between (e.g. "qwrtplk").
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", text.lower()):
        return True

    if len(text.split()) == 0:
        return True

    return False


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].map(clean_text)

    before = len(df)
    df = df.drop_duplicates(subset=["text", "specialist"]).reset_index(drop=True)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate/empty rows ({before} -> {after}).")

    # Drop classes with too few examples to stratify/split (need >= 2 for
    # a stratified split, and ideally more for CV folds).
    counts = df["specialist"].value_counts()
    too_small = counts[counts < 2].index.tolist()
    if too_small:
        print(f"Dropping classes with <2 examples: {too_small}")
        df = df[~df["specialist"].isin(too_small)].reset_index(drop=True)

    return df


def build_feature_union() -> FeatureUnion:
    """Combine word-level and char-level TF-IDF. Char n-grams give
    robustness to the spelling variations in this dataset; word n-grams
    capture medical terms and multi-word keywords."""
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )),
    ])


def evaluate_candidates(X_train_feats, y_train):
    """5-fold stratified CV over a few candidate classifiers; returns the
    name of the best one by mean macro-F1."""
    candidates = {
        "LinearSVC": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=5.0, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "ComplementNB": ComplementNB(),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, clf in candidates.items():
        scores = cross_val_score(
            clf, X_train_feats, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        results[name] = scores
        print(f"{name:20s}  macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    best_name = max(results, key=lambda k: results[k].mean())
    print(f"\nBest model: {best_name}")
    return best_name, candidates[best_name]


def main():
    df = load_data(DATA_PATH)
    print(f"Final dataset: {len(df)} rows, {df['specialist'].nunique()} specialists\n")

    X = df["text"].values
    y = df["specialist"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    features = build_feature_union()
    X_train_feats = features.fit_transform(X_train)
    X_test_feats = features.transform(X_test)

    print("Cross-validating candidate models on training data:")
    best_name, best_clf = evaluate_candidates(X_train_feats, y_train)

    # Refit best model on full training set, evaluate on held-out test set
    best_clf.fit(X_train_feats, y_train)
    y_pred = best_clf.predict(X_test_feats)

    print("\n=== Held-out test set performance ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro-F1:  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit feature extractor + model on ALL data for the final deployed pipeline
    final_features = build_feature_union()
    X_all_feats = final_features.fit_transform(X)
    final_clf = type(best_clf)(**best_clf.get_params())
    final_clf.fit(X_all_feats, y)

    pipeline_bundle = {
        "features": final_features,
        "model": final_clf,
        "model_name": best_name,
        "classes": sorted(df["specialist"].unique().tolist()),
    }
    joblib.dump(pipeline_bundle, MODEL_OUT)
    print(f"\nSaved trained pipeline to {MODEL_OUT}")

    print("\n=== Sample query predictions ===")
    for query in SAMPLE_QUERIES:
        if is_gibberish(query):
            print(f"{query!r:70s} -> Wrong input. Try again")
            continue
        feats = final_features.transform([clean_text(query)])
        pred = final_clf.predict(feats)[0]
        print(f"{query!r:70s} -> {pred}")


def predict(text: str, bundle_path: str = MODEL_OUT):
    """Convenience function to load the saved model and predict a specialist
    for a new piece of Hinglish symptom text. Returns None (and prints a
    message) for gibberish input instead of generating a prediction."""
    if is_gibberish(text):
        print("Wrong input. Try again")
        return None

    bundle = joblib.load(bundle_path)
    feats = bundle["features"].transform([clean_text(text)])
    pred = bundle["model"].predict(feats)[0]
    return pred


def interactive_test(bundle_path: str = MODEL_OUT):
    """Prompt the user for symptom text and print the predicted specialist,
    one query at a time, until they type 'quit'."""
    print("\n=== Try your own queries (type 'quit' to exit) ===")
    while True:
        text = input("\nEnter symptom text: ").strip()
        if text.lower() in ("quit", "exit", ""):
            break
        if is_gibberish(text):
            print("Wrong input. Try again")
            continue
        print(f"Predicted specialist: {predict(text, bundle_path)}")

if __name__ == "__main__":
    main()
    interactive_test()
    