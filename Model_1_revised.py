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
- Saves the trained pipeline + label list + raw training texts to disk for
  reuse.
- Before trusting the classifier's prediction, a lightweight "coverage"
  check compares the user's input against every known training text using
  word-level overlap (word order doesn't matter). If at least 90% of the
  words in the user's input also appear in some known training example
  (per-word similarity, so minor spelling variation is still tolerated),
  we trust the classifier's guess. Otherwise we don't trust it and ask
  the user for more information.
"""

import sys
import os
import difflib
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

# --- gibberish-detector setup -----------------------------------------
# Uses the `gibberish-detector` PyPI package (pip install gibberish-detector).
# GIBBERISH_MODEL_PATH points at 'big.model', a pretrained character-bigram
# model shipped with that package (trained on a large general-English
# corpus) -- place it in the same folder as this notebook. No training
# step is required to use it as-is.
#
# NOTE: 'big.model' is English-only. Since our queries are Hinglish
# (Hindi words in Roman script mixed with English), it will sometimes be
# a bit too aggressive or too lenient on Hindi words that don't look like
# English (see the "Do you need more data?" discussion after this cell).
# If you find it misfiring often, you can train a Hinglish-specific model
# on this project's own training text instead:
#     gibberish-detector train hinglish_corpus.txt > hinglish.model
# and point GIBBERISH_MODEL_PATH at that file instead.
GIBBERISH_MODEL_PATH = "big.model"

try:
    from gibberish_detector import detector as _gibberish_detector_module
    if os.path.exists(GIBBERISH_MODEL_PATH):
        GIBBERISH_DETECTOR = _gibberish_detector_module.create_from_model(GIBBERISH_MODEL_PATH)
    else:
        print(f"Warning: gibberish model '{GIBBERISH_MODEL_PATH}' not found in the "
              f"working directory -- gibberish pre-check will be skipped.")
        GIBBERISH_DETECTOR = None
except ImportError:
    print("Warning: package 'gibberish_detector' is not installed "
          "(pip install gibberish-detector) -- gibberish pre-check will be skipped.")
    GIBBERISH_DETECTOR = None
# ------------------------------------------------------------------------

# Message returned for input that is either gibberish or doesn't
# sufficiently match any known training example.
NO_MATCH_MESSAGE = "No matches found."

DATA_PATH = "Hinglish_Symptoms_Reference_V3.csv"
MODEL_OUT = "symptom_specialist_classifier.joblib"
RANDOM_STATE = 42

# Minimum fraction (0-1) of the words in a user's input that must be
# found (word order doesn't matter) in at least one known training
# example before we trust the model's prediction. Below this, we ask the
# user for more information instead of guessing.
MATCH_THRESHOLD = 0.9

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


# ---------------------------------------------------------------------------
# Word-level match (order-independent)
# ---------------------------------------------------------------------------
def tokenize(s: str) -> list:
    """Lowercase, whitespace-split into words (light punctuation stripped)."""
    s = clean_text(s).lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return [w for w in s.split() if w]


def word_similarity(w1: str, w2: str) -> float:
    """Similarity (0-1) between two individual words, tolerant of minor
    spelling variation (e.g. 'dard'/'darrd'), via difflib's ratio."""
    if not w1 or not w2:
        return 0.0
    return difflib.SequenceMatcher(None, w1, w2).ratio()


def word_match_ratio(a: str, b: str, word_match_threshold: float = 0.85) -> float:
    """Fraction (0-1) of the words in `a` that have a close match somewhere
    in `b`, ignoring word order entirely. Each word in `a` is matched
    against its single best-matching word in `b` (a word in `b` can be
    reused to match more than one word in `a`). Two words are considered a
    match if their `word_similarity` is >= `word_match_threshold`, which
    absorbs small spelling differences without requiring an exact match.
    """
    a_words = tokenize(a)
    b_words = tokenize(b)
    if not a_words or not b_words:
        return 0.0

    matched = 0
    for wa in a_words:
        best = max(word_similarity(wa, wb) for wb in b_words)
        if best >= word_match_threshold:
            matched += 1
    return matched / len(a_words)


def best_match(text: str, corpus_texts) -> tuple:
    """Compare `text` against every string in `corpus_texts` and return
    (best_ratio, best_matching_text) for the closest one found, where
    `best_ratio` is the word_match_ratio (word order doesn't matter)."""
    best_ratio = 0.0
    best_text = None
    for candidate in corpus_texts:
        ratio = word_match_ratio(text, candidate)
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = candidate
    return best_ratio, best_text


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
        # Raw training texts, kept so predict()/interactive_test() can run
        # the order-preserving letter-match coverage check against them.
        "texts": df["text"].tolist(),
    }
    joblib.dump(pipeline_bundle, MODEL_OUT)
    print(f"\nSaved trained pipeline to {MODEL_OUT}")

    print("\n=== Sample query predictions ===")
    for query in SAMPLE_QUERIES:
        result = predict(query, bundle=pipeline_bundle)
        print(f"{query!r:70s} -> {result}")

    # A couple of nonsense/gibberish queries, to confirm the fallback works.
    GIBBERISH_SAMPLE_QUERIES = ["jgjhkj", "asfgtqwafazfyiur", "zxcvbnmlkjhgfdsaqwerty"]
    print("\n=== Gibberish sample query checks (should all say 'Wrong input. Search again.') ===")
    for query in GIBBERISH_SAMPLE_QUERIES:
        result = predict(query, bundle=pipeline_bundle)
        print(f"{query!r:70s} -> {result}")


def predict(text: str, bundle_path: str = MODEL_OUT, bundle: dict = None,
            threshold: float = MATCH_THRESHOLD, verbose: bool = False):
    """Predict a specialist for a new piece of Hinglish symptom text.

    Two layers of "trust this input?" checks run before the classifier's
    prediction is returned:
      1. Gibberish check (gibberish-detector library, if its model file is
         available) -- catches keyboard-mash input like "dfgskjbnskfjdn".
      2. Word-level match ratio (see `word_match_ratio`) against every
         known training example -- word order does not matter, only
         whether the words themselves are present. If fewer than
         `threshold` (default 0.9, i.e. 90%) of the input's words are
         found in ANY training example, the classifier's guess isn't
         trusted.

    If either check fails, this returns NO_MATCH_MESSAGE
    ("No matches found.") instead of a specialist name.

    `bundle` can be passed directly (e.g. from within main(), where it's
    already in memory) to avoid re-loading from disk on every call.
    """
    if bundle is None:
        bundle = joblib.load(bundle_path)

    cleaned = clean_text(text)

    # Reject empty/too-short input outright.
    if len(cleaned) < 3:
        if verbose:
            print("Input too short to evaluate.")
        return NO_MATCH_MESSAGE

    # Gibberish pre-check: if the gibberish-detector model is available and
    # flags the input as gibberish, don't bother with the word-match check
    # or the classifier -- go straight to the fallback message.
    if GIBBERISH_DETECTOR is not None and GIBBERISH_DETECTOR.is_gibberish(cleaned):
        if verbose:
            print("Flagged as gibberish by gibberish-detector.")
        return NO_MATCH_MESSAGE

    ratio, closest = best_match(cleaned, bundle["texts"])
    if verbose:
        print(f"Best match ratio: {ratio:.2f} (closest known example: {closest!r})")

    if ratio < threshold:
        return NO_MATCH_MESSAGE

    feats = bundle["features"].transform([cleaned])
    pred = bundle["model"].predict(feats)[0]
    return pred


def interactive_test(bundle_path: str = MODEL_OUT):
    """Prompt the user for symptom text and print the predicted specialist,
    one query at a time, until they type 'quit'."""
    bundle = joblib.load(bundle_path)
    print("\n=== Try your own queries (type 'quit' to exit) ===")
    while True:
        text = input("\nEnter symptom text: ").strip()
        if text.lower() in ("quit", "exit", ""):
            break
        result = predict(text, bundle=bundle)
        if result == NO_MATCH_MESSAGE:
            print(result)
        else:
            print(f"Predicted specialist: {result}")


if __name__ == "__main__":
    main()
    interactive_test()
