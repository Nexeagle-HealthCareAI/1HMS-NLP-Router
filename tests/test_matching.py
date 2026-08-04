"""Unit tests for nlp_brain.matching -- the cosine-similarity coverage gate.

See tests/test_performance.py for the regression guard that pins down the
actual perf characteristics this module is responsible for (it caught a
real bug once: sklearn.metrics.pairwise.cosine_similarity() re-normalizes
its FULL corpus-matrix argument on every call, which made /route-symptom
take ~50ms per request before normalize_matrix() fixed it).
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nlp_brain.matching import best_match, normalize_matrix


@pytest.fixture
def toy_matrix():
    """4 rows, already-obvious relationships: row 0 and row 1 point the
    same direction (just different magnitude), row 2 is orthogonal to
    both, row 3 is the zero vector (degenerate case)."""
    return csr_matrix(np.array([
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]))


class TestNormalizeMatrix:
    def test_rows_become_unit_length(self, toy_matrix):
        normalized = normalize_matrix(toy_matrix)
        row_norms = np.sqrt(normalized.multiply(normalized).sum(axis=1)).A1
        assert row_norms[0] == pytest.approx(1.0)
        assert row_norms[1] == pytest.approx(1.0)
        assert row_norms[2] == pytest.approx(1.0)

    def test_zero_row_stays_zero_no_divide_by_zero_error(self, toy_matrix):
        # Row 3 is all zeros -- normalizing it must not raise or produce NaN.
        normalized = normalize_matrix(toy_matrix)
        assert normalized[3].nnz == 0

    def test_does_not_mutate_the_original_matrix(self, toy_matrix):
        original_data = toy_matrix.data.copy()
        normalize_matrix(toy_matrix)
        assert (toy_matrix.data == original_data).all()


class _StubFeatures:
    """Minimal stand-in for a fitted FeatureUnion: turns a raw string
    directly into the sparse row vector a real test wants, without needing
    an actual TfidfVectorizer in these pure best_match() tests."""
    def __init__(self, vector_by_text: dict):
        self._vector_by_text = vector_by_text

    def transform(self, texts):
        return csr_matrix(np.array([self._vector_by_text[texts[0]]]))


class TestBestMatch:
    def test_finds_the_closest_row_by_cosine_similarity(self, toy_matrix):
        texts_matrix_normalized = normalize_matrix(toy_matrix)
        texts = ["a", "b (same direction as a)", "c (orthogonal)", "d (zero vector)"]
        features = _StubFeatures({"query": [1.0, 0.0, 0.0]})

        ratio, closest = best_match("query", features, texts_matrix_normalized, texts)

        # Query points exactly like rows 0 and 1 -- either is a valid "closest"
        # match at similarity 1.0 (both are correct; argmax picks the first).
        assert ratio == pytest.approx(1.0)
        assert closest in ("a", "b (same direction as a)")

    def test_orthogonal_query_scores_zero_similarity(self, toy_matrix):
        texts_matrix_normalized = normalize_matrix(toy_matrix)
        texts = ["a", "b", "c", "d"]
        # Orthogonal to rows 0/1, same direction as row 2.
        features = _StubFeatures({"query": [0.0, 1.0, 0.0]})

        ratio, closest = best_match("query", features, texts_matrix_normalized, texts)

        assert ratio == pytest.approx(1.0)
        assert closest == "c"
