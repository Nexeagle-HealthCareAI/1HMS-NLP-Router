#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Hinglish Symptom -> Specialist Classifier
==========================================
Trains a text classifier that maps a Hinglish (Hindi-English code-mixed,
Roman script) symptom description to the medical specialist it should be
routed to.

Dataset: Hinglish_Symptoms_Reference_V4_.csv
Columns: text, specialist, type

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
  check compares the user's input against every known training text via
  cosine similarity of their TF-IDF vectors (the same features the
  classifier itself uses). If the closest known training example isn't
  similar enough (>= MATCH_THRESHOLD), we don't trust the classifier's
  guess -- interactive_test() then asks the user to describe their
  symptom in more detail and retries, rather than silently returning a
  low-confidence label.
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
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "Hinglish_Symptoms_Reference_V4_.csv"
MODEL_OUT = "symptom_specialist_classifier.joblib"
RANDOM_STATE = 42

# Minimum cosine similarity (0-1) between a user's input and the closest known
# training example (both as TF-IDF vectors) before we trust the model's
# prediction. Below this, we ask the user for more information instead of
# guessing.
#
# NOTE on calibration: data_pipeline/validation_set.csv can't be used to pick
# this -- ~97% of its rows are exact-text duplicates of rows already in the
# training CSV, so every query scores a trivial 1.0 against it (see the
# validation-set/training-set overlap note in data_pipeline/README or ask
# before relying on validationMetrics for anything threshold-related). This
# value was instead picked from realistic hand-written Hinglish symptom
# queries NOT present in the training data (which scored ~0.29-0.87) vs.
# gibberish/keyboard-mash strings (~0.0-0.19) -- 0.25 clears all the former
# with margin while rejecting the latter. It does NOT reliably reject
# coherent but off-topic Hinglish text (e.g. "aaj cricket match kab hai"
# scored ~0.26) -- cosine similarity over TF-IDF is a lexical/character
# overlap signal, not a semantic one, so some shared function words
# ("mein", "hai", "kab") are enough to clear this bar. Tightening it to
# filter those out would also start rejecting real symptom queries (the
# lowest-scoring genuine one seen was ~0.29) -- a real fix needs a semantic
# signal, not just a higher threshold.
MATCH_THRESHOLD = 0.25

NO_MATCH_MESSAGE = "No matches found."

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

    # Almost no vowels at all -> unlikely to be real words.vowels = sum(1 for c in letters_only.lower() if c in "aeiouy")
    vowels = sum(1 for c in letters_only.lower() if c in "aeiouy")
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
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
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
# Coverage check: cosine similarity against every known training example
# ---------------------------------------------------------------------------
def best_match(cleaned_text: str, features, texts_matrix, texts) -> tuple:
    """Compare `cleaned_text` against every row of `texts_matrix` (the
    training corpus's TF-IDF vectors, precomputed once at training time) via
    cosine similarity, and return (best_score, best_matching_text) for the
    closest one found. `features` is the same fitted FeatureUnion the
    classifier uses, so this reuses the classifier's own notion of
    similarity instead of a separate word-overlap heuristic -- a single
    vectorized matrix multiply instead of an O(corpus size) per-word scan,
    so it stays fast even against a multi-thousand-row corpus."""
    query_vec = features.transform([cleaned_text])
    sims = cosine_similarity(query_vec, texts_matrix)[0]
    idx = int(sims.argmax())
    return float(sims[idx]), texts[idx]


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
        # Raw training texts + their precomputed TF-IDF vectors, kept so
        # predict()/interactive_test() can run the cosine-similarity
        # coverage check against them without re-vectorizing the whole
        # corpus on every call.
        "texts": df["text"].tolist(),
        "texts_matrix": X_all_feats,
    }
    joblib.dump(pipeline_bundle, MODEL_OUT)
    print(f"\nSaved trained pipeline to {MODEL_OUT}")

    print("\n=== Sample query predictions ===")
    for query in SAMPLE_QUERIES:
        result = predict(query, bundle=pipeline_bundle, verbose=True)
        print(f"{query!r:70s} -> {result if result else NO_MATCH_MESSAGE}")


def predict(text: str, bundle_path: str = MODEL_OUT, bundle: dict = None,
            threshold: float = MATCH_THRESHOLD, verbose: bool = False):
    """Predict the specialist for a piece of Hinglish symptom text, or
    return None if the input isn't trustworthy enough to classify.

    Two checks run before the classifier's prediction is returned:
      1. Gibberish check (`is_gibberish`) -- catches keyboard-mash input
         like "dfgskjbnskfjdn".
      2. Coverage check (`best_match`): cosine similarity between the
         input's TF-IDF vector and every known training example's -- if
         the closest known example isn't similar enough (below
         `threshold`), the classifier's guess isn't trusted.

    If either check fails, this returns None instead of a specialist name,
    so callers (e.g. interactive_test()) can decide how to react -- such as
    asking the user for more detail and retrying.

    `bundle` can be passed directly (e.g. from within main(), where it's
    already in memory) to avoid re-loading from disk on every call.
    """
    if bundle is None:
        bundle = joblib.load(bundle_path)

    cleaned = clean_text(text)

    if is_gibberish(cleaned):
        if verbose:
            print("Flagged as gibberish.")
        return None

    ratio, closest = best_match(cleaned, bundle["features"], bundle["texts_matrix"], bundle["texts"])
    if verbose:
        print(f"Best match similarity: {ratio:.2f} (closest known example: {closest!r})")

    if ratio < threshold:
        return None

    feats = bundle["features"].transform([cleaned])
    pred = bundle["model"].predict(feats)[0]
    return pred


#def interactive_test(bundle_path: str = MODEL_OUT):
    """Prompt the user for symptom text and print the predicted specialist.

    If the input is gibberish or too dissimilar from anything the model was
    trained on, this doesn't just give up -- it tells the user why, asks
    them to describe their symptom in more detail, appends that detail to
    what they've said so far, and retries. This repeats until either a
    confident prediction is made or the user quits.

    print("Loading symptom-specialist classifier...")
    bundle = joblib.load(bundle_path)

    print("\n=== Try your own queries (type 'quit' to exit) ===")
    while True:
        text = input("\nEnter symptom text: ").strip()
        if text.lower() in ("quit", "exit", ""):
            break

        result = predict(text, bundle=bundle, verbose=True)

        while result is None:
            print("Sorry, I couldn't confidently match that to a specialist.")
            more = input("Could you describe your symptoms in a bit more detail? ").strip()
            if more.lower() in ("quit", "exit", ""):
                result = "quit"
                break
            text = f"{text} {more}".strip()
            result = predict(text, bundle=bundle, verbose=True)

        if result == "quit":
            break
        if result:
            print(f"Predicted specialist: {result}") """
    
def interactive_test(bundle_path: str = MODEL_OUT):
    print("Loading symptom-specialist classifier...")
    bundle = joblib.load(bundle_path)

    print("\n=== Try your own queries (type 'quit' to exit) ===")
    while True:
        text = input("\nEnter symptom text: ").strip()
        if text.lower() in ("quit", "exit", ""):
            break

        result = predict(text, bundle=bundle, verbose=True)

        while result is None:
            if is_gibberish(clean_text(text)):
                print("Sorry, that didn't look like a symptom description.")
                more = input("Could you describe your symptoms in your own words? ").strip()
                if more.lower() in ("quit", "exit", ""):
                    result = "quit"
                    break
                # The old text was gibberish and carries no real signal -
                # start over with the fresh input instead of dragging the
                # gibberish along and re-triggering the same check forever.
                text = more
            else:
                print("Sorry, I couldn't confidently match that to a specialist.")
                more = input("Could you describe your symptoms in a bit more detail? ").strip()
                if more.lower() in ("quit", "exit", ""):
                    result = "quit"
                    break
                # Legitimate text, just under-detailed - append rather than discard.
                text = f"{text} {more}".strip()

            result = predict(text, bundle=bundle, verbose=True)

        if result == "quit":
            break
        if result:
            print(f"Predicted specialist: {result}")


if __name__ == "__main__":
    main()
    interactive_test()


# In[ ]:




