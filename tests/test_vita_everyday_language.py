"""Everyday patient language (catalog "Vita: what patients type and how Vita should reply", Oct 2026).

287 phrases in 14 categories (tests/fixtures/everyday_phrases.json) go through /v2/agent/chat. Before this work Vita gave a
sensible reply to 115 of them; the target is 90%, and every emergency. Then the specific behaviours: safety, typing
tidy-up, the spelling-repair guard, the new answers, repair words, and side questions while booking.
"""
import json
import re
from pathlib import Path

import pytest

from hms_router.application import everyday
from hms_router.doctor_intent import parse
from hms_router.domain.models import Query
from hms_router.safety.detectors import ModelDetector, says_nothing
from hms_router.textnorm import preprocess, tidy
from tests.test_vita_booking import api, choose_anita, labels  # noqa: F401
from tests.test_vita_doctor_chat import URL, chat, client, hms  # noqa: F401

PHRASES = json.loads((Path(__file__).parent / "fixtures" / "everyday_phrases.json").read_text(encoding="utf-8"))
GENERIC = re.compile(r"takleef ho rahi hai\? Thoda|What's bothering you|naam ke koi doctor|named '|Doctor Dekho par nahi mila")


def say(client, prev, text, **kw):
    return chat(client, text, session_id=prev["session_id"], **kw)


def handled(expect: str, out: dict) -> bool:
    kind, msg, clarify = out["kind"], out["message"], (out.get("clarify") or {}).get("kind")
    if expect == "greet":
        return clarify == "ask_kind" and bool(re.match(r"(Hello|Namaste)", msg))
    if expect == "about":
        return "Vita" in msg and "AI" in msg
    if expect == "list":
        return (kind == "doctor_list" and not GENERIC.search(msg)) or clarify == "location"
    if expect == "search":                      # a list, a location question, or "which kind?" keeping the filter
        return not GENERIC.search(msg) and kind in ("doctor_list", "follow_up", "info")
    if expect == "symptom":
        return bool((out.get("state") or {}).get("symptoms")) or kind in ("recommendation", "emergency", "doctor_list") \
            or clarify == "warning_signs"
    if expect == "emergency":
        return kind == "emergency" or clarify == "warning_signs"
    if expect == "human":
        return kind == "handoff"
    if expect == "advice":
        return bool(re.search(r"salah nahi|doctor hi|takleef batayein", msg)) or bool((out.get("state") or {}).get("symptoms"))
    return not GENERIC.search(msg) and kind not in ("emergency", "handoff")


def test_most_everyday_phrases_get_a_sensible_reply(client, hms):
    total, good, misses = 0, 0, []
    for cat, spec in PHRASES.items():
        for text in spec["phrases"]:
            out = client.post(URL, json={"text": text, "city": "Kolkata"}).json()
            total += 1
            if handled(spec["expect"], out):
                good += 1
            else:
                misses.append((cat, text, out["message"][:60]))
    assert good / total >= 0.9, f"{good}/{total}: {misses}"


def test_every_emergency_phrase_is_caught(client, hms):
    for text in PHRASES["F. Emergencies in everyday words"]["phrases"]:
        out = chat(client, text, city="Kolkata")
        assert out["kind"] == "emergency" or out["clarify"]["kind"] == "warning_signs", text


# ============================================================================ safety
@pytest.mark.parametrize("text", ["ek taraf ka haath kaam nahi kar raha", "papa ka ek haath kaam nahi kar raha",
                                  "delivery ka dard shuru ho gaya", "pregnancy mein paani gir gaya"])
def test_stroke_and_labour_are_emergencies(client, hms, text):
    assert chat(client, text, city="Kolkata")["kind"] == "emergency"


def test_one_side_weakness_denied_is_not(client, hms):
    assert chat(client, "ek taraf kamzori nahi hai, bas bukhar hai", city="Kolkata")["kind"] != "emergency"


def test_self_harm_names_tele_manas(client, hms):
    out = chat(client, "marna chahta hoon", city="Kolkata")
    assert out["kind"] == "emergency" and "Tele-MANAS 14416" in out["message"]
    assert "khud ko nuksaan pahunchane ke vichaar" in out["message"]


@pytest.mark.parametrize("text", ["nahi", "haan", "ok", "pata nahi"])
def test_a_bare_answer_is_never_an_emergency_for_the_model(text):
    q = Query(raw=text, normalized=preprocess(text))
    assert says_nothing(q) and not ModelDetector(lambda _q: 0.99, 0.6).detect(q).flag


def test_head_injury_asks_the_warning_signs_and_yes_is_an_emergency(client, hms):
    out = chat(client, "sar pe chot lagi", city="Kolkata")
    assert out["clarify"]["kind"] == "warning_signs" and "behoshi" in out["message"]
    yes = say(client, out, "haan", city="Kolkata")
    assert yes["kind"] == "emergency" and "sar par chot" in yes["message"]


def test_burn_without_warning_signs_gets_a_doctor(client, hms):
    out = chat(client, "ungli jal gayi chai se", city="Kolkata")
    assert out["clarify"]["kind"] == "warning_signs"
    no = say(client, out, "nahi", city="Kolkata")
    assert no["kind"] == "doctor_list" and no["message"].startswith("Theek hai. Phir bhi doctor ko dikhayein")


# ============================================================================ typing
@pytest.mark.parametrize("text, want", [("nhi pta", "nahi pata"), ("ky kru", "kya karu"), ("doc chaiye", "doctor chahiye"),
                                        ("bukharrr hai", "bukhar hai"), ("bhot dard h", "bahut dard hai"),
                                        ("\U0001f912 2 din se", "bukhar 2 din se"), ("k", "ok"),
                                        ("ek taraf ka haath kaam nahi kar raha", "ek taraf ka haath kaam nahi kar raha"),
                                        ("aankh se paani", "aankh se paani"), ("pet mai dard", "pet mai dard")])
def test_tidy(text, want):
    from hms_router.application.doctor_chat import _known_words
    assert tidy(text, _known_words()) == want


@pytest.mark.parametrize("text", ["kaunsi dawai lu", "kaunsa better hai", "bekaar ho tum", "ayushman card chalega"])
def test_everyday_words_are_not_repaired_into_symptoms(client, hms, text):
    out = chat(client, text, city="Kolkata")
    assert not (out.get("state") or {}).get("symptoms")


@pytest.mark.parametrize("text, symptom", [("zukam hai", "cold"), ("dast lag gaye", "diarrhoea"), ("acidity", "acidity"),
                                           ("kamzori", "weakness"), ("periods late hai", "delayed periods"),
                                           ("pet saaf nahi hota", "constipation"), ("safed pani", "white discharge"),
                                           ("BP high hai", "high blood pressure"), ("ghabrahat", "anxiety"),
                                           ("naak beh rahi hai", "runny nose"), ("haath sunn", "numbness")])
def test_everyday_symptom_words(client, hms, text, symptom):
    out = chat(client, text, city="Kolkata")
    assert symptom in (out.get("state") or {}).get("symptoms", []), out["message"]


@pytest.mark.parametrize("text, key", [("bachon ka doctor", "paediatrics"), ("ladies doctor", "gynaecology"),
                                       ("delivery wali doctor", "gynaecology"), ("BP ka doctor", "general-medicine"),
                                       ("pathri ka doctor", "urology"), ("piles doctor", "general-surgery"),
                                       ("baal ka doctor", "dermatology")])
def test_everyday_specialty_words(text, key):
    assert parse(text).specialty_key == key


def test_lady_doctor_is_still_any_woman_doctor():
    ask = parse("lady doctor chahiye")
    assert ask.gender == "female" and ask.specialty_key is None


def test_the_review_sheet_drives_the_words():
    from hms_router.doctor_intent import _load_everyday_specialties
    words = _load_everyday_specialties()
    assert words["pathri"] == "urology" and "sex" not in words           # "needs decision" rows are not used


# ============================================================================ answers
@pytest.mark.parametrize("text, words", [
    ("tum kaun ho", "AI assistant"), ("kya tum insaan ho", "insaan ya doctor nahi"),
    ("kya ye free hai", "free hai"), ("online consultation hota hai kya", "jald hi"),
    ("ayushman card chalega", "jald hi"), ("lab test book karna hai", "jald hi"),
    ("ambulance ka number", "108"), ("paracetamol le sakta hu", "salah nahi"),
    ("aaj mausam kaisa hai", "sirf doctor dhoondhne"), ("number kyun chahiye", "OTP"),
    ("Doctor Dekho kya hai", "free mein book"), ("shubh ratri", "Shubh ratri")])
def test_answers(client, hms, text, words):
    out = chat(client, text, city="Kolkata")
    assert words in out["message"], out["message"]


def test_a_medicine_question_with_a_symptom_says_so_then_triages(client, hms):
    out = chat(client, "bukhar hai, kaunsi dawai lu", city="Kolkata")
    assert out["message"].startswith("Main dawai ya dose ki salah nahi de sakti")
    assert "fever" in out["state"]["symptoms"]


def test_fees_of_the_chosen_doctor(client, hms, api):
    sel = choose_anita(client)
    out = say(client, sel, "fees kitni hai")
    assert "Rs 600" in out["message"] and "hospital mein" in out["message"]
    assert out["clarify"]["kind"] == "book_day"                           # the booking question again


def test_number_why_at_the_mobile_step(client, hms, api):
    sel = choose_anita(client)
    out = say(client, sel, labels(sel)[0])
    for t in ("Mere liye", "Rina Kumari", "32", "Mahila"):
        out = say(client, out, t)
    assert out["clarify"]["kind"] == "book_mobile"
    why = say(client, out, "number kyun chahiye?")
    assert "OTP" in why["message"] and why["clarify"]["kind"] == "book_mobile"
    done = say(client, why, "9876543210")
    assert done["clarify"]["kind"] == "book_otp"


def test_upset_twice_offers_a_person(client, hms):
    first = chat(client, "tum pagal ho", city="Kolkata")
    second = say(client, first, "bekaar ho tum", city="Kolkata")
    assert "Kisi insaan se baat karein" in [o["label"] for o in second["clarify"]["options"]]


# ============================================================================ repair words
def test_pick_from_the_list(client, hms):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata")
    first_id = lst["doctors"]["items"][0]["doctorId"]
    out = say(client, lst, "pehla wala", city="Kolkata")
    assert out.get("selected", {}).get("doctorId") == first_id or out.get("doctor", {}).get("doctorId") == first_id \
        or out["clarify"]["kind"].startswith("book")


def test_what_repeats_the_last_question(client, hms):
    q = chat(client, "pet mein dard hai", city="Kolkata")
    again = say(client, q, "samjha nahi", city="Kolkata")
    assert again["message"].startswith("Main dobara poochti hoon:") and q["message"] in again["message"]


def test_restart(client, hms):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata")
    out = say(client, lst, "shuru se", city="Kolkata")
    assert out["message"].startswith("Theek hai, shuru se karte hain") and out["clarify"]["kind"] == "ask_kind"
    more = say(client, out, "aur", city="Kolkata")
    assert more["kind"] != "doctor_list"                                  # the old list is gone


def test_which_is_better_offers_sorting(client, hms):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata")
    out = say(client, lst, "kaunsa better hai", city="Kolkata")
    assert out["clarify"]["kind"] == "sort" and "better nahi keh sakti" in out["message"]
    cheap = say(client, out, out["clarify"]["options"][0]["label"], city="Kolkata")
    assert cheap["kind"] == "doctor_list"


def test_a_filter_alone_is_kept_for_the_next_message(client, hms):
    out = chat(client, "sabse sasta doctor", city="Kolkata")
    assert out["message"].startswith("Theek hai, kam fees wale pehle dikhaungi.")
    lst = say(client, out, "skin ka doctor", city="Kolkata")
    fees = [d.get("discountedFee") or d.get("consultFee") for d in lst["doctors"]["items"]]
    assert fees == sorted(fees)


def test_typed_yes_to_a_booking_offer_books(client, hms, api):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata")
    offer = say(client, lst, "Dr Anita Sen kab milengi", city="Kolkata")
    assert offer["clarify"]["kind"] == "book_offer"
    out = say(client, offer, "haan", city="Kolkata")
    assert out["clarify"]["kind"].startswith("book_")


def test_detect_never_fires_on_a_doctor_or_symptom_word():
    for text in ("skin doctor", "bukhar hai", "dil ka doctor", "kal milenge kya"):
        assert everyday.detect(text) is None, text


def test_free_about_a_day_is_the_doctors_availability(client, hms, api):
    sel = choose_anita(client)
    out = say(client, sel, "kal free hain kya?")
    assert "booking free hai" not in out["message"] and out["clarify"]["kind"].startswith("book_")
