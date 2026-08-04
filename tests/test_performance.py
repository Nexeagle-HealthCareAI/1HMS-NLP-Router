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
the fixed measurement to stay stable across slower CI runners. It
deliberately uses a SINGLE-symptom query (see nlp_brain.segmentation) --
per-segment cost scaling with segment count is expected/correct behavior
(each segment independently runs the full gibberish/coverage-gate/classify
pipeline), not a regression; test_multi_segment_predict_scales_roughly_
linearly_with_segment_count below is the guard for THAT, separately.
"""
import time

import pytest

from nlp_brain.segmentation import split_segments
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
        # Single-segment on purpose -- see the class-level "context" docstring.
        query = "sar mein dard hai"
        assert split_segments(query) == [query]  # guards the isolation this test relies on
        real_classifier.predict(query)  # warm up

        start = time.perf_counter()
        for _ in range(20):
            real_classifier.predict(query)
        elapsed_ms = (time.perf_counter() - start) / 20 * 1000

        # Measured ~15ms/call right after the fix (25df289, ~small-thousands-row
        # corpus), ~50-60ms/call as of 2026-08-04 on today's larger corpus
        # (15,486 rows) -- that growth is organic (more training rows = more
        # work per cosine-similarity coverage-gate call) and expected, not a
        # regression. 250ms keeps meaningful headroom below the OLD pathological
        # regime this guards against (~300-450ms/call under concurrency from
        # re-normalizing the full corpus per request; ~9.8s/call from the O(N^2)
        # difflib version before that) while still catching a real regression
        # back toward either of those, and leaving room for further organic
        # corpus growth + slower CI hardware.
        assert elapsed_ms < 250, (
            f"predict() took {elapsed_ms:.1f}ms/call against the real bundle -- "
            f"expected roughly 50-60ms based on the current corpus size. This likely means "
            f"the coverage-gate is re-normalizing the full corpus matrix per request again "
            f"(see nlp_brain.matching.normalize_matrix), or the corpus has grown enough that "
            f"this threshold needs revisiting on its own merits."
        )

    def test_multi_segment_predict_scales_roughly_linearly_with_segment_count(self, real_classifier):
        """A 2-segment query legitimately runs the full gibberish/coverage-gate/
        classify pipeline twice (SymptomClassifier._predict_segment, once per
        segment) -- roughly 2x a single-segment call is expected and fine. This
        guards against it becoming WORSE than roughly linear (e.g. an accidental
        quadratic re-scan across segments), not against the expected 2x itself.
        """
        single = "sar mein dard hai"
        multi = "sar mein dard hai aur pet mein dard ho raha hai kaafi der se"
        assert len(split_segments(multi)) == 2  # guards the premise of this test

        real_classifier.predict(single)  # warm up
        real_classifier.predict(multi)

        start = time.perf_counter()
        for _ in range(20):
            real_classifier.predict(single)
        single_ms = (time.perf_counter() - start) / 20 * 1000

        start = time.perf_counter()
        for _ in range(20):
            real_classifier.predict(multi)
        multi_ms = (time.perf_counter() - start) / 20 * 1000

        assert multi_ms < single_ms * 3.5, (
            f"2-segment predict() took {multi_ms:.1f}ms vs {single_ms:.1f}ms for 1 segment "
            f"({multi_ms / single_ms:.1f}x) -- expected roughly 2x. This suggests segments "
            f"aren't being processed independently/linearly (see SymptomClassifier.predict())."
        )
