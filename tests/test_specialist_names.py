import pytest

from hms_router.dataset_builder import canonical_specialist, load_corpus_from_dataset
from hms_router.specialist_rag import _default_corpus, load_corpus


@pytest.mark.parametrize("raw,expected", [
    ("Neurology", "Neurologist"), ("Dermatology & Venereology", "Dermatologist"),
    ("  cardiology ", "Cardiologist"), ("General Medicine", "General Physician"),
    ("Breast Surgery", "Breast clinic"), ("Dental", "Dental"),            # unknown -> unchanged
])
def test_canonical_specialist(raw, expected):
    assert canonical_specialist(raw) == expected


def test_dataset_clusters_use_the_single_vocabulary():
    names = {c.specialist for c in load_corpus_from_dataset()}
    assert not names & {"Neurology", "Cardiology", "Dermatology", "Gastroenterology", "Orthopaedics"}


def test_default_corpus_is_cached_but_callers_get_a_copy():
    a, b = load_corpus(), load_corpus()
    assert a == b and a is not b
    a.clear()
    assert load_corpus()                      # mutating one result never corrupts the cache
    assert _default_corpus.cache_info().hits >= 1
