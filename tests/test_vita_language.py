"""Vita mirrors the patient's language: English, Hinglish or Hindi script, for the whole conversation.

Three layers, tested separately:
  1. detection of one message (language.LanguageDetector)
  2. the conversation's language across turns (language.resolve, through the chat endpoint)
  3. replies, chips and questions written in that language (voice.py, doctor_chat)
"""
import re

import pytest

from hms_router.doctor_intent import parse
from hms_router.language import LanguageDetector, LanguageResult, resolve, ui_language
from hms_router.lexicon import Lexicon
from hms_router.textnorm import intent_text
from tests.test_vita_doctor_chat import chat, client, hms  # noqa: F401 (pytest fixtures)

DEV = re.compile(r"[ऀ-ॿ]")
HINGLISH_TELLS = re.compile(r"\b(aap|hain|karein|dabayein|mile|kya|nahi|batayein|chahiye|gaya)\b", re.I)


@pytest.fixture(scope="module")
def detector():
    return LanguageDetector(Lexicon.from_csv())


def is_hindi(text):
    return len(DEV.findall(text)) > 10


def is_hinglish(text):
    return not DEV.search(text) and bool(HINGLISH_TELLS.search(text))


def is_english(text):
    return not DEV.search(text) and not HINGLISH_TELLS.search(text)


# ============================================================================ 1. one message
@pytest.mark.parametrize("text, lang, strength", [
    ("I have had fever and headache for 3 days", "english", "strong"),
    ("give me Dr Anita Sen personal number", "english", "strong"),        # "me" is not Hindi here
    ("show me a lady doctor among these", "english", "strong"),
    ("I need a doctor for my mother, she has fever since kal", "english", "strong"),   # one Hindi word
    ("I have a fever since morning", "english", "strong"),
    ("hello I need a cardiologist", "english", "strong"),
    ("kal raat se sir dard ho raha hai", "hinglish", "strong"),
    ("mujhe fever hai", "hinglish", "strong"),
    ("pet me dard hai", "hinglish", "strong"),
    ("namaste ji", "hinglish", "strong"),
    ("inme se kam fees wala", "hinglish", "strong"),
    ("dr sharma kal milenge", "hinglish", "strong"),
    ("bas dekh raha tha", "hinglish", "strong"),
    ("skin ka doctor", "hinglish", "weak"),
    ("मुझे बुखार है", "hindi", "strong"),
    ("हाँ", "hindi", "strong"),                                          # script is always clear
    ("আমার জ্বর হয়েছে", "bangla", "strong"),
])
def test_detect_one_message(detector, text, lang, strength):
    r = detector.detect(text)
    assert (r.language, r.strength) == (lang, strength)


@pytest.mark.parametrize("text", ["Kolkata", "Dr Anita Sen", "3", "cardiologist", "ok", "MD medicine doctor"])
def test_neutral_messages_say_nothing(detector, text):
    assert detector.detect(text).strength == "none"


@pytest.mark.parametrize("text", ["thank you", "thank you so much", "ok thanks", "no thanks", "hello", "haan ji",
                                  "nahi", "yes"])
def test_courtesy_words_never_switch(detector, text):
    assert detector.detect(text).strength != "strong"


# ============================================================================ 2. the conversation
def test_resolve_rules():
    strong_en = LanguageResult("english", "latin", "markers", "strong")
    weak_hg = LanguageResult("hinglish", "latin", "markers", "weak")
    none = LanguageResult("unknown", "latin", "markers", "none")
    assert resolve(strong_en, "hinglish").language == "english"           # the patient switched: follow
    assert resolve(weak_hg, "english").language == "english"              # "haan" keeps English
    assert resolve(none, "hindi").language == "hindi"                     # "Kolkata" keeps Hindi
    assert resolve(none, None, "hinglish").language == "hinglish"         # first message neutral: the UI's
    assert resolve(none, None).language == "english"
    assert resolve(weak_hg, None).language == "hinglish"                  # first message: any evidence wins


@pytest.mark.parametrize("code, lang", [("en", "english"), ("hi", "hindi"), ("hinglish", "hinglish"),
                                        ("bn", "bangla"), ("fr", None), (None, None)])
def test_ui_language_codes(code, lang):
    assert ui_language(code) == lang


def test_neutral_reply_keeps_hinglish(client, hms):
    first = chat(client, "paas mein cardiologist")
    assert first["language"] == "hinglish" and first["clarify"]["kind"] == "location"
    assert [o["label"] for o in first["clarify"]["options"]][0] == "Location share karein"
    nxt = chat(client, "Kolkata", session_id=first["session_id"])
    assert nxt["language"] == "hinglish" and is_hinglish(nxt["message"])


def test_patient_switches_and_vita_follows(client, hms):
    en = chat(client, "I need a skin doctor", city="Kolkata")
    assert en["language"] == "english" and is_english(en["message"])
    hg = chat(client, "inme se kam fees wala", session_id=en["session_id"], city="Kolkata")
    assert hg["language"] == "hinglish" and "sabse kam fees pehle" in hg["message"]
    hi = chat(client, "इनमें से कोई महिला डॉक्टर", session_id=en["session_id"], city="Kolkata")
    assert hi["language"] == "hindi" and is_hindi(hi["message"])


def test_courtesy_reply_keeps_the_language(client, hms):
    first = chat(client, "mujhe skin ka doctor chahiye", city="Kolkata")
    thanks = chat(client, "thank you", session_id=first["session_id"])
    assert thanks["language"] == "hinglish" and is_hinglish(thanks["message"])


def test_ui_hint_only_for_a_neutral_first_message(client, hms):
    named = chat(client, "Dr Anita Sen", city="Kolkata", ui_language="hi")
    assert named["language"] == "hindi" and is_hindi(named["message"])
    worded = chat(client, "I need a skin doctor", city="Kolkata", ui_language="hi")
    assert worded["language"] == "english"                                 # the patient's words win


# ============================================================================ 3. replies in that language
def test_english_triage_is_english_throughout(client):
    q = chat(client, "I have had fever and headache for 3 days")
    assert q["kind"] == "follow_up" and q["message"].startswith("Got it: fever and headache.")
    assert is_english(q["message"])
    rec = chat(client, "yes, a little cough too", session_id=q["session_id"])
    assert rec["kind"] == "recommendation" and is_english(rec["message"])
    assert "GREEN" not in rec["message"] and "YELLOW" not in rec["message"]          # no printout card
    assert "not a diagnosis" in rec["message"] and "108" in rec["message"]


def test_hindi_triage_is_hindi_throughout(client):
    q = chat(client, "मुझे तीन दिन से बुखार और सिर दर्द है")
    assert q["language"] == "hindi" and q["message"].startswith("समझ गई")
    rec = chat(client, "हाँ, थोड़ी खाँसी भी है", session_id=q["session_id"])
    assert rec["kind"] == "recommendation" and is_hindi(rec["message"])
    assert "3 दिन से" in rec["message"] and "निदान नहीं" in rec["message"]


def test_hinglish_triage_reads_like_a_person(client):
    q = chat(client, "3 din se bukhar aur sar dard hai")
    assert q["message"] == "Samajh gayi, bukhar aur sar dard. Koi aur symptoms hain?"
    rec = chat(client, "haan thodi khasi bhi", session_id=q["session_id"])
    assert rec["message"].startswith("Samajh gayi: 3 din se bukhar, sar dard aur khansi.")
    assert "Gambhirta" not in rec["message"]


@pytest.mark.parametrize("text, lang", [("seene mein tez dard aur saans nahi aa rahi", "hinglish"),
                                        ("सीने में तेज़ दर्द और साँस नहीं आ रही", "hindi"),
                                        ("severe chest pain and I can't breathe", "english")])
def test_emergency_number_first_in_the_patients_language(client, hms, text, lang):
    r = chat(client, text, city="Kolkata")
    assert r["kind"] == "emergency" and r["language"] == lang and "doctors" not in r
    first = r["message"].splitlines()[0]
    assert "108" in first and "112" in first
    if lang != "english":
        assert "Call 108 (ambulance) or 112 now." in r["message"]        # English backup line


@pytest.mark.parametrize("text, kind, needle", [
    ("मुझे स्किन डॉक्टर चाहिए", "doctor_list", "त्वचा के डॉक्टर"),
    ("बस देख रहा था", "info", "कोई बात नहीं"),
    ("किसी इंसान से बात करनी है", "handoff", "सपोर्ट टीम"),
    ("नमस्ते", "follow_up", "मैं Vita हूँ"),
    ("डॉक्टर का नंबर दो", "info", "निजी नंबर"),
])
def test_hindi_script_asks_are_understood_and_answered_in_hindi(client, hms, text, kind, needle):
    r = chat(client, text, city="Kolkata")
    assert r["kind"] == kind and r["language"] == "hindi" and needle in r["message"]


def test_hindi_chips(client):
    r = chat(client, "नमस्ते")
    assert [o["label"] for o in r["clarify"]["options"]][:2] == ["लक्षण बताएँ", "स्पेशलिटी चुनें"]


def test_english_chips(client):
    r = chat(client, "hello")
    assert [o["label"] for o in r["clarify"]["options"]][:2] == ["Describe a symptom", "Choose a specialty"]


def test_english_refine_among_these(client, hms):
    first = chat(client, "I need a skin doctor", city="Kolkata")
    lady = chat(client, "show me a lady doctor among these", session_id=first["session_id"], city="Kolkata")
    assert lady["kind"] == "doctor_list" and lady["doctors"]["filters"].get("gender") == "female"
    assert is_english(lady["message"])


def test_private_number_in_english_is_english(client, hms):
    r = chat(client, "give me Dr Anita Sen personal number", city="Kolkata")
    assert r["language"] == "english" and r["message"].startswith("I can't share")


def test_bangla_is_recognised_and_answered_politely(client):
    r = chat(client, "আমার জ্বর হয়েছে")
    assert r["language"] == "bangla" and r["kind"] == "follow_up"
    assert "বাংলায়" in r["message"] and "Hindi ya English" in r["message"] and "108" in r["message"]


# ============================================================================ Hindi script for the doctor rules
@pytest.mark.parametrize("text, expect", [
    ("मुझे स्किन डॉक्टर चाहिए", "mujhe skin doctor chahiye"),
    ("इनमें से कोई महिला डॉक्टर", "inme se koi mahila doctor"),
    ("पास में दिल का डॉक्टर", "paas mein dil ka doctor"),
    ("और डॉक्टर दिखाओ", "aur doctor dikhao"),
    ("सदर अस्पताल पूर्णिया के डॉक्टर", "sadar hospital purnia ke doctor"),
    ("skin doctor chahiye", "skin doctor chahiye"),                       # Roman text is untouched
])
def test_intent_text(text, expect):
    assert intent_text(text) == expect


@pytest.mark.parametrize("text, field, value", [
    ("कोलकाता में स्किन डॉक्टर", "place", "kolkata"),
    ("kolkata mein skin doctor", "place", "kolkata"),
    ("purnia ke paas cardiologist", "place", "purnia"),
    ("आज उपलब्ध बच्चों का डॉक्टर", "available_today", True),
    ("इनमें से सबसे सस्ता वाला", "sort", "fee"),
    ("डॉ अनीता सेन कल मिलेंगी?", "when", "+1"),
])
def test_hindi_and_place_first_asks(text, field, value):
    assert getattr(parse(text), field) == value


def test_place_first_never_steals_a_symptom():
    assert parse("kamar mein dard hai").place is None


def test_mahila_rog_is_gynaecology_not_a_lady_filter():
    a = parse("mahila rog doctor")
    assert a.specialty_key == "gynaecology" and a.gender is None
