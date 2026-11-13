"""Text -> sparse TF-IDF vector (character 2-5 grams + word 1-2 grams), shared by the retriever, the
classifier scorer and the emergency model so a query is encoded once per request."""
from __future__ import annotations

import numpy as np

from ..domain.models import Query
from ..textnorm import STOPWORDS


def content_text(normalized: str) -> str:
    kept = [w for w in normalized.split() if w not in STOPWORDS]
    return " ".join(kept) or normalized


def build_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union
    return make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
    )


def l2_normalize_rows(X):
    from scipy.sparse import diags
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return diags(1.0 / norms) @ X


class TextEncoder:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def encode(self, query: Query):
        """Raw (un-normalized) TF-IDF row, cached on the query - the classifiers were trained on these."""
        if query.features is None:
            query.features = self.vectorizer.transform([content_text(query.normalized)])
        return query.features

    def encode_unit(self, query: Query):
        X = self.encode(query)
        n = float(np.sqrt(X.multiply(X).sum()))
        return (X / n) if n else X
