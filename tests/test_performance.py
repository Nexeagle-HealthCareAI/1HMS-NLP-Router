"""Latency regression guards.

Context: nlp_brain.matching.best_match() used to call
sklearn.metrics.pairwise.cosine_similarity() directly against the raw,
un-normalized training corpus matrix. That function re-normalizes BOTH of
its arguments internally on every call -- including the entire
multi-thousand-row corpus, which never changes between requests. On the
production dataset that made /route-symptom take ~50ms/request server-side
and turned even 8 concurrent requests into 300-450ms each. Precomputing the
normalized matrix once (nlp_brain.matching.normalize_matrix(), cached on
SymptomClassifier at load time) dropped that to ~15ms/request.

test_predict_latency_stays_well_under_the_pre_fix_baseline is the guard
against silently regressing back to the old behavior -- e.g. someone
"simplifying" predict() by calling cosine_similarity() directly again. It
is NOT a strict production SLA and deliberately uses a generous multiple of
the fixed measurement to stay stable across slower CI runners.
"""
import time

import pytest

from tests.conftest import requires_real_bundle, REAL_MODEL_PATH


class TestFixturePredictLatency:
    """Fast, always-on sanity check using the small test fixture -- catches
    an accidentally-quadratic change (e.g. re-normalizing per call again)
    even though the tiny corpus here is too small to reproduce the actual
    ~50ms regression the integration test below guards against."""

    def test_predict_is_fast_on_a_small_corpus(self, tiny_classifier):
        query = "dant mein bahut dard ho raha hai"
        tiny_classifier.predict(query)  # warm up (first call pays import/JIT-ish costs)

        start = time.perf_counter()
        for _ in range(20):
            tiny_classifier.predict(query)
        elapsed_ms = (time.perf_counter() - start) / 20 * 1000

        assert elapsed_ms < 50, (
            f"predict() took {elapsed_ms:.1f}ms/call on a ~30-row fixture corpus -- "
            f"something is doing far more work than it should for this input size."
        )


@requires_real_bundle
@pytest.mark.integration
class TestRealBundlePredictLatency:
    """Runs by default (needs the actual production bundle, ~35MB /
    ~15,000 rows, to be meaningful -- and it's committed to the repo, so
    it's normally present). The pre-commit hook's fast local loop excludes
    it via `-m "not integration"`; self-skips if the bundle is missing."""

    @pytest.fixture(scope="class")
    def real_classifier(self):
        from nlp_brain.classifier import SymptomClassifier
        return SymptomClassifier.load(str(REAL_MODEL_PATH))

    def test_predict_latency_stays_well_under_the_pre_fix_baseline(self, real_classifier):
        query = "Dil mein bahut zor se dard ho raha hai aur saans phool rahi hai"
        real_classifier.predict(query)  # warm up

        start = time.perf_counter()
        for _ in range(20):
            real_classifier.predict(query)
        elapsed_ms = (time.perf_counter() - start) / 20 * 1000

        # Measured ~15ms/call after the fix, ~54ms/call before it, on the
        # same production bundle. 100ms leaves headroom for slower CI
        # hardware while still catching a real regression back toward (or
        # past) the pre-fix number.
        assert elapsed_ms < 100, (
            f"predict() took {elapsed_ms:.1f}ms/call against the real bundle -- "
            f"expected roughly 15ms based on the post-fix measurement. This likely means "
            f"the coverage-gate is re-normalizing the full corpus matrix per request again "
            f"(see nlp_brain.matching.normalize_matrix)."
        )
