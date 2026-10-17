"""Fail fast with a readable message if the data files are missing, instead of 90 confusing tracebacks."""
from pathlib import Path

import pytest
#import pytest
from hms_router.specialist_rag import _default_corpus


_ROOT = Path(__file__).resolve().parent.parent
_REQUIRED = [
    _ROOT / "hms_router" / "data" / "hinglish_lexicon.csv",
    _ROOT / "hms_router" / "data" / "spelling_variants.csv",
    _ROOT / "hms_router" / "data" / "specialist_mapping.csv",
    _ROOT / "Datasets" / "combined_symptom_dataset.csv",
]


def pytest_sessionstart(session):
    missing = [str(p.relative_to(_ROOT)) for p in _REQUIRED if not p.exists()]
    if missing:
        pytest.exit("Missing required data files: " + ", ".join(missing), returncode=2)
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_REQUIRED = [
    _ROOT / "hms_router" / "data" / "hinglish_lexicon.csv",
    _ROOT / "hms_router" / "data" / "spelling_variants.csv",
    _ROOT / "hms_router" / "data" / "specialist_mapping.csv",
    _ROOT / "Datasets" / "combined_symptom_dataset.csv",
]


def pytest_sessionstart(session):
    missing = [str(p.relative_to(_ROOT)) for p in _REQUIRED if not p.exists()]
    if missing:
        pytest.exit("Missing required data files: " + ", ".join(missing), returncode=2)


@pytest.fixture(autouse=True)
def _fresh_corpus_cache():
    _default_corpus.cache_clear()
    yield
    _default_corpus.cache_clear()