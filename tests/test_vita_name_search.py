"""A doctor's name typed wrong ("dr avinsahh" for Dr. Avinash Kumar): Vita offers the listed names that look or sound like
it ("did you mean?"), and "similar naam se dhundo" / "spelling galat hai" after a miss searches more loosely."""
import pytest

from hms_router.search_spelling import similar_names
from tests import test_vita_doctor_chat as fake
from tests.test_vita_graph import client, vita  # noqa: F401

NAMES = ["Dr. Avinash Kumar", "Dr. Abhinav Singh", "Dr. Anita Sen", "Dr. Neha Sharma", "Dr. Vinay Rao", "Dr. Priya Bose"]


@pytest.mark.parametrize("typed, expect", [
    ("avinsahh", ["Dr. Avinash Kumar"]), ("abinash", ["Dr. Avinash Kumar"]), ("avinas", ["Dr. Avinash Kumar"]),
    ("anitha", ["Dr. Anita Sen"]), ("neha sarmaa", ["Dr. Neha Sharma"]), ("vinai", ["Dr. Vinay Rao"]),
    ("xyzq", []), ("rakesh", [])])
def test_names_that_look_or_sound_alike(typed, expect):
    assert similar_names(typed, NAMES) == expect


@pytest.fixture
def avinash(client, monkeypatch):
    doc = fake._doc(8, "Dr. Avinash Kumar", "General Physician", "MBBS, MD", fake.H3, 12, 400, "M")
    monkeypatch.setattr(fake, "DOCTORS", fake.DOCTORS + [doc])
    before = client.app.state.doctors
    client.app.state.doctors = fake.make_directory(fake.Fake1Hms())
    yield doc
    client.app.state.doctors = before


def names(out):
    return [d["name"] for d in (out.get("doctors") or {}).get("items") or []]


def test_a_misspelt_name_finds_the_one_it_sounds_like(client, avinash):
    out = vita(client, "dr avinsahh ko dekhna hai")
    assert out["kind"] == "doctor_list" and names(out) == ["Dr. Avinash Kumar"]


def test_when_1hms_misses_vita_offers_names_that_sound_alike(client, avinash, monkeypatch):
    d = client.app.state.doctors
    monkeypatch.setattr(d, "fix_name", lambda q: q)                 # as if the spelling fix didn't catch it
    real = d.search
    monkeypatch.setattr(d, "search", lambda **a: real(**a) if a.get("query") in (None, "Dr. Avinash Kumar")
                        else {"doctors": [], "total": 0, "specialtyId": None, "args": a})
    out = vita(client, "dr avinsahh ko dekhna hai")
    assert names(out) == ["Dr. Avinash Kumar"]
    assert "exact doctor nahi mile" in out["message"] and "dhoondh rahe the" in out["message"]


def test_the_right_spelling_is_found_in_another_city(client, avinash):
    out = vita(client, "Dr Avinash Kumar")                       # Purnia, while the chat is in Kolkata
    assert names(out) == ["Dr. Avinash Kumar"] and "exact" not in out["message"]


def test_similar_names_after_a_miss(client, avinash):
    miss = vita(client, "dr zzyzx ko dikhana hai")
    assert names(miss) == []
    out = vita(client, "search karo similar name se", miss["session_id"])
    assert out["engine"] == "step" and "jaisa naam kisi doctor ka nahi mila" in out["message"]


@pytest.mark.parametrize("follow", ["search karo similar name se", "spelling galat hai", "milte julte naam dikhao",
                                    "similar doctors by name"])
def test_after_a_name_search_the_follow_up_is_understood(client, avinash, follow):
    miss = vita(client, "dr avnsh kmr")
    out = vita(client, follow, miss["session_id"])
    assert out["kind"] == "doctor_list" and "Dr. Avinash Kumar" in names(out)


def test_a_likeness_word_about_health_is_not_a_name_retry(client, avinash):
    miss = vita(client, "dr zzyzx")
    out = vita(client, "bukhar jaisa lag raha hai", miss["session_id"])
    assert out.get("engine") != "step" or out["kind"] != "doctor_list"
