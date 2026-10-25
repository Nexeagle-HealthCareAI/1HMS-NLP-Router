"""The patient changes what they want mid-way (a new specialty, a symptom, other doctors, another place, another person)
and Vita follows instead of holding on to the first flow."""
import pytest

from hms_router.application.booking import _clean_name
from hms_router.doctor_intent import parse
from tests.test_vita_booking import api, choose_anita, jwt, labels, say  # noqa: F401
from tests.test_vita_doctor_chat import chat, client, hms  # noqa: F401


def at_name(client):
    sel = choose_anita(client)
    day = say(client, sel, labels(sel)[0])
    return say(client, day, "Mere liye")


def specialist(out):
    return (out.get("doctors") or {}).get("specialist")


# ============================================================================ parser
@pytest.mark.parametrize("text, key", [
    ("dil ka", "cardiology"), ("dil wala", "cardiology"), ("nahi, haddi ka", "orthopaedics"), ("aankh ka", "ophthalmology"),
    ("eye wala", "ophthalmology"), ("skin", "dermatology"), ("nahi, mujhe bachche ke liye doctor chahiye", "paediatrics")])
def test_a_short_answer_names_the_specialty(text, key):
    assert parse(text).specialty_key == key


@pytest.mark.parametrize("text", ["dil ki dhadkan tez hai", "pet mein", "kamar mein dard", "mere liye", "Rina Kumari"])
def test_a_body_part_in_a_sentence_is_not_a_specialty(text):
    assert parse(text).specialty_key is None


@pytest.mark.parametrize("text, place", [("Patna mein dikhao", "patna"), ("nahi, kolkata me", "kolkata"),
                                         ("ranchi mein bhi dikhao", "ranchi"), ("Salt Lake mein", "salt lake")])
def test_only_a_place_is_the_same_search_there(text, place):
    ask = parse(text)
    assert ask.place == place and ask.refine


@pytest.mark.parametrize("text", ["subah mein", "shaam me", "morning me", "agle hafte mein", "subah mein skin doctor"])
def test_a_time_is_not_a_place(text):
    assert parse(text).place is None


@pytest.mark.parametrize("text", ["dil ka doctor chahiye", "mujhe bukhar hai", "Dr Neha Sharma dikhao", "doosre doctor dikhao",
                                  "actually mummy ke liye", "kya ye theek hai?"])
def test_a_request_is_not_a_name(text):
    assert _clean_name(text) is None


@pytest.mark.parametrize("text, name", [("Rina Kumari", "Rina Kumari"), ("mera naam Sunita Devi hai", "Sunita Devi"),
                                        ("Mohammad Arif", "Mohammad Arif")])
def test_names_are_still_names(text, name):
    assert _clean_name(text) == name


# ============================================================================ after a list
def test_a_short_answer_after_a_list_changes_the_specialty(client, hms):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata")
    out = say(client, lst, "dil ka")
    assert out["kind"] == "doctor_list" and specialist(out) == "Cardiologist"


def test_a_place_after_a_list_searches_there(client, hms):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata")
    out = say(client, lst, "Patna mein dikhao")
    assert out["kind"] == "doctor_list" and specialist(out) == "Dermatologist" and "Patna" in out["message"]


def test_a_child_after_a_recommendation_shows_childrens_doctors(client, hms):
    out = chat(client, "mujhe bukhar hai", city="Kolkata")
    out = say(client, out, "3 din se")
    out = say(client, out, "nahi, mujhe bachche ke liye doctor chahiye")
    assert out["kind"] == "doctor_list" and "Paediatrician" in out["message"]


# ============================================================================ while booking
@pytest.mark.parametrize("text, kind, spec", [
    ("dil ka doctor chahiye", "doctor_list", "Cardiologist"),
    ("mujhe bukhar hai", "recommendation", None),
    ("Dr Neha Sharma dikhao", "doctor_list", None),
    ("Patna mein skin doctor", "doctor_list", "Dermatologist")])
def test_a_new_ask_instead_of_the_name_ends_the_booking(client, hms, api, text, kind, spec):
    out = say(client, at_name(client), text)
    assert out["kind"] == kind and "umar" not in out["message"]
    if spec:
        assert specialist(out) == spec
    nxt = say(client, out, "Rina Kumari")                                 # the booking is over: no "umar kitni hai?"
    assert (nxt.get("clarify") or {}).get("kind") != "book_age"
    assert not api.bookings()


@pytest.mark.parametrize("text", ["doosre doctor dikhao", "nahi, koi aur doctor dikhao"])
def test_other_doctors_while_booking_shows_the_list_again(client, hms, api, text):
    sel = choose_anita(client)
    out = say(client, sel, text)
    assert out["kind"] == "doctor_list" and len(out["doctors"]["items"]) == 2


def test_a_new_specialty_at_the_day_step(client, hms, api):
    sel = choose_anita(client)
    out = say(client, sel, "dil ka doctor chahiye")
    assert out["kind"] == "doctor_list" and specialist(out) == "Cardiologist"


def test_a_time_answer_stays_in_the_booking(client, hms, api):
    sel = choose_anita(client)
    out = say(client, sel, "subah mein")
    assert (out.get("clarify") or {}).get("kind", "").startswith("book_")


def test_a_real_name_is_still_taken(client, hms, api):
    out = say(client, at_name(client), "Sunita Devi")
    assert out["clarify"]["kind"] == "book_age" and "Sunita Devi" in out["message"]


def test_a_sentence_for_a_name_is_asked_again(client, hms, api):
    out = say(client, at_name(client), "naam kyun chahiye?")
    assert out["clarify"]["kind"] == "book_name"


def test_changing_who_it_is_for_asks_the_new_name(client, hms, api):
    name = at_name(client)
    out = say(client, name, "actually mummy ke liye")
    assert out["clarify"]["kind"] == "book_name" and out["message"] == "Patient ka poora naam?"
    for t in ("Asha Devi", "60", "Mahila", "9876543210", "123456"):
        out = say(client, out, t)
    assert out["clarify"]["kind"] == "book_confirm" and "Asha Devi (60, F)" in out["message"]
    out = say(client, out, "Haan, book karo")
    assert out["kind"] == "booked"
    (_, _, body, _), = api.bookings()
    assert body["patient"]["fullName"] == "Asha Devi" and body["patient"]["guardianRelation"] == "Mother"


def test_changing_to_a_known_family_member_fills_them_in(client, hms, api):
    token = jwt("9123456789")
    sel = choose_anita(client, patient_token=token)
    who = say(client, sel, labels(sel)[0], patient_token=token)
    confirm = say(client, who, "myself", patient_token=token)
    assert "Rina Kumari (32, F)" in confirm["message"]
    out = say(client, confirm, "nahi, mummy ke liye", patient_token=token)
    assert out["clarify"]["kind"] == "book_confirm" and "Asha Devi (60, F)" in out["message"]


def test_back_to_myself(client, hms, api):
    token = jwt("9123456789")
    sel = choose_anita(client, patient_token=token)
    who = say(client, sel, labels(sel)[0], patient_token=token)
    confirm = say(client, who, "Asha Devi (Mother)", patient_token=token)
    out = say(client, confirm, "nahi, mere liye", patient_token=token)
    assert out["clarify"]["kind"] == "book_confirm" and "Rina Kumari (32, F)" in out["message"]
