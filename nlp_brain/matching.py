"""Coverage check: is a query similar enough to something the model actually
learned from, or should we admit uncertainty instead of guessing?"""
from sklearn.metrics.pairwise import cosine_similarity


def best_match(cleaned_text: str, features, texts_matrix, texts) -> tuple:
    """Compare `cleaned_text` against every row of `texts_matrix` (the
    training corpus's TF-IDF vectors, precomputed once at training time) via
    cosine similarity, and return (best_score, best_matching_text) for the
    closest one found. `features` is the same fitted FeatureUnion the
    classifier uses, so this reuses the classifier's own notion of
    similarity -- a single vectorized matrix multiply instead of a per-word
    scan, so it stays fast even against a multi-thousand-row corpus."""
    query_vec = features.transform([cleaned_text])
    sims = cosine_similarity(query_vec, texts_matrix)[0]
    idx = int(sims.argmax())
    return float(sims[idx]), texts[idx]
