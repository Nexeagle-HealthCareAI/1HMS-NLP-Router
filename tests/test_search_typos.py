"""Doctor search with typing mistakes (2026-10-30). Each case below failed before this change: misspelled specialty
and doctor words ("cardiolgist", "haddi ka doktor") got "What's bothering you?", "kunal ghose" was read as the
symptom gas, "skin doctor kolkata" searched for a doctor named Kolkata, a misspelled city found nobody (1HMS matches
cities exactly), "chest pian" missed the chest-pain emergency, "tooth ache" was read as acne and "eye redness" as
fatigue."""
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from hms_router.doctor_intent import fix_search_words, parse
from hms_router.search_spelling import CityIndex, NameIndex, WordIndex
from hms_router.severity import _close
from tests.test_vita_doctor_chat import DOCTORS, Fake1Hms, make_directory

NAMES = [d["name"] for d in DOCTORS]
CITIES = CityIndex(["Kolkata", "Purnia", "Siliguri", "Salt Lake", "Bengaluru"])


# ------------------------------------------------------------------------------------------- search words
@pytest.mark.parametrize("typed,fixed", [
    ("cardiolgist", "cardiologist"), ("kardiologist", "cardiologist"), ("dermotologist", "dermatologist"),
    ("gynacologist", "gynaecologist"), ("gynecolojist", "gynecologist"), ("pediatrision", "pediatrician"),
    ("peadiatrician", "pediatrician"), ("neurolgist", "neurologist"), ("gastroentrologist", "gastroenterologist"),
    ("opthalmologist", "ophthalmologist"), ("ortopedic", "orthopedic"), ("urolojist", "urologist"),
    ("dentest", "dentist"), ("dnb dermatolgy", "dnb dermatology"),
    ("haddi ka doktor", "haddi ka doctor"), ("mujhe doctr chahiye", "mujhe doctor chahiye"),
    ("bone specilist", "bone specialist"), ("skn doctor", "skin doctor"), ("hart ka doctor", "heart ka doctor"),
    ("kidny doctor", "kidney doctor"), ("nak kan gala doctor", "naak kaan gala doctor"),
    ("mbs doctor", "mbbs doctor"), ("mbbbs doctor", "mbbs doctor"),
])
def test_search_words_are_repaired(typed, fixed):
    assert fix_search_words(typed) == fixed


@pytest.mark.parametrize("text", [
    "radiologist", "hematologist", "psychologist", "physical weakness", "done doctor", "pet dard", "sar dard",
    "donor chahiye", "special offer", "dosti", "achha", "nahi", "theek hai", "surgery doctor",
])
def test_ordinary_and_other_words_are_left_alone(text):
    assert fix_search_words(text) == text


@pytest.mark.parametrize("text,key", [
    ("cardiolgist", "cardiology"), ("dermatologst", "dermatology"), ("orthopadic doctor", "orthopaedics"),
    ("haddi ka doktor", "orthopaedics"), ("bone specilist", "orthopaedics"), ("skn doctor", "dermatology"),
    ("pediatrision", "paediatrics"), ("opthalmologist", "ophthalmology"), ("dm cardiolgy", "cardiology"),
])
def test_specialty_from_a_misspelling(text, key):
    assert parse(text).specialty_key == key


def test_listed_names_are_never_repaired():
    # a doctor called "Hart" stays Dr Hart even though "hart" is one letter from "heart"
    assert parse("dr hart", names=["Dr. Rohit Hart"], known_names=["Dr. Rohit Hart"]).name_query == "hart"


# ------------------------------------------------------------------------------------------- names
def test_name_words_by_spelling_and_sound():
    ix = NameIndex(NAMES + ["Dr. P. Chatterjee", "Dr. R. Mukherjee"])
    assert ix.fix("neha sarma") == "neha sharma"
    assert ix.fix("kunal ghose") == "kunal ghosh"
    assert ix.fix("chaterji") == "chatterjee"
    assert ix.fix("mukharji") == "mukherjee"
    assert ix.fix("bansal") == "bansal"                 # nobody by that name: kept as typed
    assert ix.word("nahi") is None                      # "nahi" is not "neha"


@pytest.mark.parametrize("text,name", [("kunal ghose", "kunal ghose"), ("neha shrma", "neha shrma"),
                                       ("anitta sen", "anitta sen"), ("S. Roy", "roy"), ("k ghosh", "ghosh")])
def test_bare_names_with_typos(text, name):
    assert parse(text, names=NAMES).name_query == name


# ------------------------------------------------------------------------------------------- cities
@pytest.mark.parametrize("typed,city", [
    ("kolkatta", "Kolkata"), ("Kolkata", "Kolkata"), ("calcutta", "Kolkata"), ("siligudi", "Siliguri"),
    ("purnea", "Purnia"), ("saltlake", "Salt Lake"), ("bangalore", "Bengaluru"), ("patna", None), ("kol", None),
])
def test_city_spellings(typed, city):
    assert CITIES.resolve(typed) == city


@pytest.mark.parametrize("text,place,key", [
    ("skin doctor kolkata", "kolkata", "dermatology"), ("cardiologist kolkatta", "kolkata", "cardiology"),
    ("calcutta ka cardiologist", "kolkata", "cardiology"), ("doctor in siligudi", "siliguri", None),
    ("haddi ka doktor purnea", "purnia", "orthopaedics"),
])
def test_a_city_anywhere_in_a_doctor_ask(text, place, key):
    a = parse(text, names=NAMES, known_names=NAMES, cities=CITIES)
    assert a.place == place and a.specialty_key == key and a.name_query is None


def test_a_city_in_a_symptom_message_is_not_a_search():
    assert parse("bukhar hai kolkata se", cities=CITIES).place is None


def test_word_index_refuses_a_close_call():
    ix = WordIndex({"sharma": "a", "verma": "b"})
    assert ix.closest("sarma") == "sharma"
    assert ix.closest("xarma") is None                  # one letter from both: no guess


# ------------------------------------------------------------------------------------------- emergency and symptoms
def test_swapped_letters_count_as_one_typo_in_emergency_phrases():
    assert _close("pian", "pain") and _close("cehst", "chest") and not _close("pina", "pain")


# ------------------------------------------------------------------------------------------- end to end
@pytest.fixture
def chat():
    server = Fake1Hms()
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(server)

        def send(text, city="Kolkata"):
            before = len(server.calls)
            out = c.post("/v2/agent/chat", json={"text": text, "city": city}).json()
            return out, [a for n, a in server.calls[before:] if n == "search_doctors"]
        yield send


def names_in(out):
    return [d["name"] for d in (out.get("doctors") or {}).get("items") or []]


@pytest.mark.parametrize("text,who", [
    ("cardiolgist", "Dr. Neha Sharma"), ("haddi ka doktor", "Dr. Kunal Ghosh"), ("skn doctor", "Dr. Anita Sen"),
    ("dermotologist", "Dr. Priya Bose"), ("dnb dermatolgy", "Dr. Priya Bose"), ("dm cardiolgy", "Dr. Neha Sharma"),
])
def test_misspelled_specialty_lists_doctors(chat, text, who):
    out, _ = chat(text)
    assert out["kind"] == "doctor_list" and who in names_in(out)


@pytest.mark.parametrize("text,who,sent_as", [
    ("kunal ghose", "Dr. Kunal Ghosh", "kunal ghosh"), ("dr neha sarma", "Dr. Neha Sharma", "neha sharma"),
    ("ghosh doctor", "Dr. Kunal Ghosh", "ghosh"),
])
def test_misspelled_names_find_the_doctor(chat, text, who, sent_as):
    out, sent = chat(text)
    assert out["kind"] == "doctor_list" and who in names_in(out) and sent[0]["query"] == sent_as
    assert sent_as in out["message"]


@pytest.mark.parametrize("text", ["cardiologist kolkatta", "cardiologist in calcutta", "skin doctor kolkata"])
def test_misspelled_or_old_city_names(chat, text):
    out, sent = chat(text, city=None)
    assert out["kind"] == "doctor_list" and names_in(out) and sent[0]["city"] == "Kolkata"


@pytest.mark.parametrize("text,said", [
    ("chest pian", "emergency"), ("tooth ache", "toothache"), ("back ache", "back pain"), ("eye redness", "red eye"),
    ("coughing blood", "coughing up blood"), ("hedache since 2 days", "headache"), ("stomuch pain", "stomach"),
])
def test_misspelled_symptoms(chat, text, said):
    out, _ = chat(text)
    assert said in (out["kind"] + " " + out["message"]).lower()


def test_achha_is_not_ache(chat):
    out, _ = chat("achha")
    assert "pain" not in out["message"].lower()


def test_no_is_not_a_doctor_called_neha(chat):
    out, sent = chat("nahi")
    assert out["kind"] != "doctor_list" and not sent
