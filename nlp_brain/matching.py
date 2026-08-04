"""Coverage check: is a query similar enough to something the model actually
learned from, or should we admit uncertainty instead of guessing?"""
from sklearn.preprocessing import normalize


def normalize_matrix(matrix):
    """L2-normalize a TF-IDF matrix once, up front.

    Kept separate from best_match() deliberately: normalizing here is a
    one-time cost paid once at load/training time. Calling
    sklearn.metrics.pairwise.cosine_similarity() per-request instead (the
    previous approach) re-normalizes BOTH of its arguments internally on
    every single call -- including the entire multi-thousand-row training
    corpus matrix, which never changes between requests. Profiling showed
    that repeated re-normalization was the dominant cost of every
    /route-symptom call (~50ms out of ~54ms total) once the training corpus
    grew past a few thousand rows."""
    return normalize(matrix, copy=True)


def best_match(cleaned_text: str, features, texts_matrix_normalized, texts) -> tuple:
    """Compare `cleaned_text` against every row of `texts_matrix_normalized`
    (the corpus's TF-IDF vectors, already L2-normalized via
    normalize_matrix() -- NOT the raw texts_matrix) via cosine similarity,
    and return (best_score, best_matching_text) for the closest one found.
    Only the query itself is normalized per call; the corpus side already
    is, once."""
    query_vec = normalize(features.transform([cleaned_text]), copy=False)
    sims = (texts_matrix_normalized @ query_vec.T).toarray().ravel()
    idx = int(sims.argmax())
    return float(sims[idx]), texts[idx]
