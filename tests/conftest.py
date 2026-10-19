"""Fail fast with a readable message when required corpus data is missing."""
from pathlib import Path

import pytest
from hms_router.specialist_rag import _default_corpus

_ROOT = Path(__file__).resolve().parent.parent
_REQUIRED = [
    _ROOT / "hms_router" / "data" / "hinglish_lexicon.csv",
    _ROOT / "hms_router" / "data" / "specialist_mapping.csv",
    _ROOT / "Datasets" / "combined_symptom_dataset.csv",
]


# Tests for the retired nlp_brain classifier and its speech/retrain tooling. Their code was deleted from
# develop on 2026-10-17 (commit e90f92f) or needs ffmpeg/pydub that hms_router does not use. Kept on disk
# for reference; collected again only if nlp_brain is restored.
collect_ignore = [
    "test_api.py", "test_candidates.py", "test_classifier.py", "test_matching.py", "test_performance.py",
    "test_segmentation.py", "test_text_utils.py", "test_retrain_pipeline.py", "test_speech.py",
]
collect_ignore_glob = ["location_API/*"]          # location_API has its own requirements and workflow


def pytest_sessionstart(session):
    missing = [str(p.relative_to(_ROOT)) for p in _REQUIRED if not p.exists()]
    if missing:
        pytest.exit("Missing required data files: " + ", ".join(missing), returncode=2)


@pytest.fixture(autouse=True)
def _fresh_corpus_cache():
    _default_corpus.cache_clear()
    yield
    _default_corpus.cache_clear()

# /route-symptom fixtures (bundle, service) and fakes live in tests/conftest_route.py
from tests.conftest_route import bundle, service  # noqa: E402,F401
