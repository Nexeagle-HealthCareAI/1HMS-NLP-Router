"""Training pipeline: turns a labeled CSV into a fitted feature extractor +
classifier. No FastAPI/voice concerns -- this only knows about data in,
model out."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE
from .text_utils import clean_text

REQUIRED_COLUMNS = {"text", "specialist", "type"}


def load_data(path: str) -> pd.DataFrame:
    """Load and clean a training CSV -- the first step of
    SymptomClassifier.train(). Call this directly (rather than .train())
    if you just want to inspect/validate a candidate dataset (row counts,
    class balance, duplicates) before committing to a full training run.
    Uses the tolerant Python parser (vs. the default C engine) and skips
    unparseable rows, since some dataset revisions have had the odd
    unescaped comma inside a free-text field."""
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected column(s): {missing}")

    df["text"] = df["text"].map(clean_text)
    df["specialist"] = df["specialist"].map(lambda s: str(s).strip())

    before = len(df)
    df = df.drop_duplicates(subset=["text", "specialist"]).reset_index(drop=True)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate/empty rows ({before} -> {after}).")

    # Drop classes with too few examples to stratify/split (need >= 2 for a
    # stratified split, and ideally more for CV folds).
    counts = df["specialist"].value_counts()
    too_small = counts[counts < 2].index.tolist()
    if too_small:
        print(f"Dropping classes with <2 examples: {too_small}")
        df = df[~df["specialist"].isin(too_small)].reset_index(drop=True)

    return df


def evaluate_candidates(X_train_feats, y_train):
    """Stratified CV over a few candidate classifiers; returns the name and
    an unfitted instance of the best one by mean macro-F1. Called by
    SymptomClassifier.train() -- reach for this directly only if you're
    experimenting with a NEW candidate model (add it to the `candidates`
    dict below) and want to compare it against the existing ones on
    already-vectorized features, without running a full train(). Fold
    count is capped by the smallest class's example count, so this doesn't
    blow up on a dataset with a thin class."""
    min_class_count = pd.Series(y_train).value_counts().min()
    n_splits = max(2, min(5, min_class_count))

    candidates = {
        "LinearSVC": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=5.0, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "ComplementNB": ComplementNB(),
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
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
