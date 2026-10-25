"""Two follow-ups (2026-10-25):

1. Vita speaks as female in Hinglish and Hindi ("dikhaungi", "kar sakti hoon"), like the Doctor Dekho site.
2. After choosing a doctor, "kal milenge kya?" is a question about that doctor's day, in every language, with booking off
   (the day's hours) and on (said first, then the booking goes on with that day, or the days the doctor is in).
"""
import datetime as dt
import pathlib
import re

import pytest

from hms_router.doctor_intent import availability_when, parse
from hms_router.textnorm import intent_text
from tests.test_vita_booking import api, choose_anita, labels, say  # noqa: F401
from tests.test_vita_doctor_chat import DOCTORS, chat, client, hms  # noqa: F401

ANITA = DOCTORS[2]["doctorId"]
ROOT = pathlib.Path(__file__).resolve().parents[1] / "hms_router"


def _ist_today():
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=330)).date()


def _next(weekday: int) -> str:
    """The next date (from today, inclusive) on that weekday, as the patient's 'monday' / 'sunday' means it."""
    t = _ist_today()
    return (t + dt.timedelta(days=(weekday - t.weekday()) % 7)).isoformat()


# ============================================================================ 1. Vita is female
MASCULINE_FIRST_PERSON = re.compile(
    r"\b(main|mai)\b[^\"'.!?]{0,60}\b(sakta|raha|karta|deta|leta|chuka|samjha|paaya|paya|[a-z]+unga)\b"
    r"|\b(samajh gaya|jod raha hoon|kar sakta hoon|bhejunga|dikhaunga|bata dunga|bhej paaya|pahunch paaya|kar paya)\b"
    r"|मैं(?!ने)[^\"'।]{0,60}(सकता|रहा हूँ|ऊँगा|ूँगा|पाया)|समझ गया|भेजूँगा|दिखाऊँगा|बता दूँगा")


@pytest.mark.parametrize("path", sorted(p for p in ROOT.rglob("*.py") if "application" in p.parts or p.name in ("voice.py", "graph.py")))
def test_vita_never_speaks_as_male(path):
    found = [m.group(0) for m in MASCULINE_FIRST_PERSON.finditer(path.read_text(encoding="utf-8"))]
    assert not found, f"{path.name}: {found}"


def test_vita_speaks_as_female_in_replies(client, hms, api):
    out = chat(client, "seene mein tez dard aur saans nahi aa rahi", city="Kolkata")          # emergency, no location
    assert "dikhaungi" in out["message"] and "bata dungi" in out["message"]
    sel = choose_anita(client)
    s = say(client, sel, labels(sel)[0])
    s = say(client, s, "Mere liye")
    s = say(client, s, "Rina Kumari")
    s = say(client, s, "32")
    s = say(client, s, "Mahila")
    assert "bhejungi" in s["message"]


def test_the_patients_decline_chip_has_no_gender_and_is_still_understood():
    for t in ("Nahi, bas jaanna tha", "नहीं, बस जानना था", "No, just checking"):
        assert parse(t).move == "just_looking"


# ============================================================================ 2. "kal milenge kya?"
@pytest.mark.parametrize("text, when", [
    ("kal milenge kya?", "+1"), ("parso milengi?", "+2"), ("monday ko milenge kya", "dow:0"), ("कल मिलेंगे क्या?", "+1"),
    ("Is she available tomorrow?", "+1"), ("is she in on sunday?", "dow:6"), ("kab milenge?", "next"),
    ("day after tomorrow available?", "+2"), ("kal", None), ("mere liye", None), ("skin ka doctor", None),
    ("dr sharma in kolkata", None)])
def test_which_day_is_asked(text, when):
    assert availability_when(intent_text(text)) == when


@pytest.mark.parametrize("text, lang_word", [("kal milenge kya?", "baithte"), ("कल मिलेंगे क्या?", "बैठते"),
                                             ("Is she available tomorrow?", "works")])
def test_booking_off_answers_the_chosen_doctors_day(client, hms, text, lang_word):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata")
    sel = chat(client, "Choose", session_id=lst["session_id"], select_doctor_id=ANITA)
    out = chat(client, text, session_id=sel["session_id"])
    tomorrow_works = (_ist_today() + dt.timedelta(days=1)).weekday() != 6
    assert out["kind"] == "info" and "Dr. Anita Sen" in out["message"]
    if tomorrow_works:
        assert lang_word in out["message"] and "09:00-13:00" in out["message"]
    assert "doctors" not in out                                           # answered, no new list


def test_booking_off_a_day_off_names_the_next_day(client, hms):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata")
    sel = chat(client, "Choose", session_id=lst["session_id"], select_doctor_id=ANITA)
    out = chat(client, "sunday ko milenge kya?", session_id=sel["session_id"])
    assert out["kind"] == "info" and "nahi" in out["message"].lower()


def test_booking_on_says_yes_then_goes_on_with_that_day(client, hms, api):
    sel = choose_anita(client)
    monday = _next(0)
    out = say(client, sel, "monday ko milenge kya?")
    assert out["message"].startswith("Haan, Dr. Anita Sen") and "Subah 09:00-13:00" in out["message"]
    assert out["clarify"]["kind"] == "book_who"
    for _ in ("Mere liye", "Rina Kumari", "32"):
        out = say(client, out, _)
    out = say(client, out, "Mahila")
    out = say(client, out, "9876543210")
    out = say(client, out, "123456")
    assert out["clarify"]["kind"] == "book_confirm" and out["booking"]["date"] == monday


def test_booking_on_a_day_off_offers_the_days_the_doctor_is_in(client, hms, api):
    sel = choose_anita(client)
    out = say(client, sel, "sunday ko milenge?")
    assert "nahi baithte" in out["message"] and out["clarify"]["kind"] == "book_day"
    assert all(o["id"] != _next(6) for o in out["clarify"]["options"])


def test_changing_the_day_at_the_summary_keeps_everything_else(client, hms, api):
    sel = choose_anita(client)
    out = say(client, sel, labels(sel)[0])
    for t in ("Mere liye", "Rina Kumari", "32", "Mahila", "9876543210", "123456"):
        out = say(client, out, t)
    assert out["clarify"]["kind"] == "book_confirm"
    first = out["booking"]["date"]
    other = next(o["label"] for o in sel["clarify"]["options"] if o["id"] not in (first, "cancel"))
    day_word = other.split(" ")[0].lower()                                # "tue" / "wed": a weekday the doctor is in
    out = say(client, out, f"{day_word} ko milenge kya?")
    assert out["message"].startswith("Haan,") and out["clarify"]["kind"] == "book_confirm"
    assert out["booking"]["date"] != first and out["booking"]["patientName"] == "Rina Kumari"
