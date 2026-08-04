"""Shared fixtures for the whole test suite.

Most tests train a small, fast, in-memory classifier (`tiny_classifier`)
instead of loading the real multi-thousand-row production bundle -- this
keeps the everyday test run (the one the pre-commit hook and CI run on
every change) fast and independent of whatever the production dataset looks
like on a given day. Tests that specifically need to exercise the real,
currently-deployed bundle are marked `@pytest.mark.integration` (see
pytest.ini) and skip themselves automatically if the real files aren't
present -- run them explicitly with `pytest -m integration`.
"""
import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

REAL_MODEL_PATH = REPO_ROOT / "symptom_specialist_classifier.joblib"
REAL_DATA_PATH = REPO_ROOT / "Hinglish_Symptoms_V28.csv"

# A small, clearly-separable dataset covering 4 specialists. Real
# nlp_brain.SymptomClassifier.train() code runs against this -- same
# feature extraction, model selection, and coverage-gate logic as
# production, just on ~30 rows instead of ~15,000 so tests run in a
# fraction of a second. Each specialist's rows share distinctive keywords
# repeated across multiple rows (TfidfVectorizer's min_df=2 drops words
# that appear only once, so a word needs to show up at least twice to be
# usable signal at all).
SAMPLE_TRAINING_ROWS = [
    # Cardiologist -- heart/chest keywords
    ("dil mein bahut dard ho raha hai", "Cardiologist", "seed"),
    ("chest mein dard aur ghabrahat mehsoos ho rahi hai", "Cardiologist", "seed"),
    ("dil ki dhadkan bahut tez chal rahi hai", "Cardiologist", "seed"),
    ("heart mein takleef ho rahi hai subah se", "Cardiologist", "seed"),
    ("seene mein dard hai aur saans phool rahi hai", "Cardiologist", "seed"),
    ("dil mein jakadan si mehsoos ho rahi hai", "Cardiologist", "seed"),
    ("chest mein bhari pan hai kaafi der se", "Cardiologist", "seed"),
    ("dil ki dhadkan aniyamit lag rahi hai", "Cardiologist", "seed"),
    # Dentist -- teeth/mouth keywords
    ("dant mein bahut dard ho raha hai", "Dentist", "seed"),
    ("daant mein takleef hai thanda garam nahi sehta", "Dentist", "seed"),
    ("teeth mein pain hai kaafi dino se", "Dentist", "seed"),
    ("muh mein dant ka dard hai bahut zyada", "Dentist", "seed"),
    ("dant dukh raha hai khana khane mein takleef", "Dentist", "seed"),
    ("daant mein keeda lag gaya hai dard ho raha hai", "Dentist", "seed"),
    ("dant ki jad mein sujan aur dard hai", "Dentist", "seed"),
    ("teeth sensitive ho gaye hain thanda lagta hai", "Dentist", "seed"),
    # Dermatologist -- skin keywords
    ("skin par laal daane ho gaye hain", "Dermatologist", "seed"),
    ("khujli ho rahi hai skin par kaafi dino se", "Dermatologist", "seed"),
    ("charmrog ho gaya hai poore sharir par", "Dermatologist", "seed"),
    ("skin mein rashes ho gaye hain aur khujli hai", "Dermatologist", "seed"),
    ("twacha par laal nishan hain kaafi der se", "Dermatologist", "seed"),
    ("skin par daane aur khujli dono ho rahi hai", "Dermatologist", "seed"),
    ("charmrog ki wajah se skin laal ho gayi hai", "Dermatologist", "seed"),
    ("twacha mein jalan aur khujli mehsoos ho rahi hai", "Dermatologist", "seed"),
    # Paediatrician -- children keywords
    ("bachche ko bukhar hai aur khaana nahi kha raha", "Paediatrician", "seed"),
    ("baccha bimar hai bukhar ke saath khansi bhi hai", "Paediatrician", "seed"),
    ("bachche ko khansi hai kaafi dino se", "Paediatrician", "seed"),
    ("baby ko fever hai subah se kam nahi ho raha", "Paediatrician", "seed"),
    ("bachche ka pet dard kar raha hai aur bukhar hai", "Paediatrician", "seed"),
    ("baccha khaana nahi kha raha aur bukhar hai", "Paediatrician", "seed"),
    ("bachche ko ulti aur bukhar dono ho rahe hain", "Paediatrician", "seed"),
    ("baby bahut ro raha hai aur bukhar hai use", "Paediatrician", "seed"),
]


@pytest.fixture(scope="session")
def sample_training_rows():
    return SAMPLE_TRAINING_ROWS


@pytest.fixture(scope="session")
def sample_training_csv(tmp_path_factory, sample_training_rows):
    """Writes SAMPLE_TRAINING_ROWS to a real CSV file, matching the
    text,specialist,type schema nlp_brain.training.load_data() expects."""
    csv_path = tmp_path_factory.mktemp("data") / "sample_training.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "specialist", "type"])
        writer.writerows(sample_training_rows)
    return str(csv_path)


@pytest.fixture(scope="session")
def tiny_classifier(sample_training_csv):
    """A real SymptomClassifier (same train() code path as production),
    trained on the small in-memory dataset above. Session-scoped: training
    it is the slowest part of the whole suite (still well under a second),
    so every test that needs a working classifier reuses this one instance
    instead of retraining per-test."""
    from nlp_brain.classifier import SymptomClassifier

    classifier, _ = SymptomClassifier.train(sample_training_csv, verbose=False)
    return classifier


@pytest.fixture
def api_client(monkeypatch, tiny_classifier):
    """A FastAPI TestClient for api.main:app, wired to `tiny_classifier`
    instead of loading the real production bundle from disk -- keeps API
    tests fast and independent of whatever's currently trained/deployed.
    Resets the rate limiter's in-memory counters before and after each
    test so one test's requests can't affect another's rate-limit budget.
    """
    import api.main as main_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        main_module.SymptomClassifier, "load",
        classmethod(lambda cls, path=None: tiny_classifier),
    )
    main_module.limiter.reset()
    with TestClient(main_module.app) as client:
        yield client
    main_module.limiter.reset()


def real_bundle_available() -> bool:
    return REAL_MODEL_PATH.exists() and REAL_DATA_PATH.exists()


requires_real_bundle = pytest.mark.skipif(
    not real_bundle_available(),
    reason="symptom_specialist_classifier.joblib / Hinglish_Symptoms_V28.csv not present -- "
           "run `python -m nlp_brain.cli train` first, or skip integration tests.",
)
