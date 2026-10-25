"""Seen in a real chat: "heyyy" was not a greeting, "neend nhi a raha" was not heard as a symptom, and Vita asked the
same question twice. "so nahi pa raha" (can't sleep) was even flagged as an emergency by the model."""
import pytest

from hms_router.doctor_intent import parse
from hms_router.domain.models import Query
from hms_router.safety.detectors import ModelDetector, sleep_only
from hms_router.textnorm import preprocess
from tests.test_vita_doctor_chat import chat, client, hms  # noqa: F401


def say(client, prev, text, **kw):
    return chat(client, text, session_id=prev["session_id"], **kw)


def q(text):
    return Query(raw=text, normalized=preprocess(text))


@pytest.mark.parametrize("text", ["heyyy", "heyy", "helloo", "hlo", "hy", "hii there", "hey vita", "namastey", "good morning",
                                  "kya haal hai", "how are you"])
def test_greetings(text):
    assert parse(text).move == "greeting"


def test_a_greeting_with_a_symptom_is_a_symptom():
    from hms_router.application.doctor_chat import medical_terms
    ask = parse("hello mujhe bukhar hai", medical_terms=medical_terms)
    assert ask.move is None and ask.symptomish


@pytest.mark.parametrize("text", ["neend nhi a raha", "neend nahi aati", "mujhe raat bhar neend nahi aati", "so nahi pa raha",
                                  "raat ko so nahi paata", "insomnia", "I can't sleep"])
def test_cannot_sleep_is_heard(client, hms, text):
    out = chat(client, text, city="Kolkata")
    assert out["kind"] in ("follow_up", "recommendation")
    assert out["state"]["symptoms"] == ["insomnia"]
    assert "takleef ho rahi hai" not in out["message"] and "What's bothering you" not in out["message"]


def test_denying_it_is_still_a_denial(client, hms):
    out = chat(client, "insomnia nahi hai, bukhar hai", city="Kolkata")
    assert out["state"]["symptoms"] == ["fever"]


def test_the_users_chat(client, hms):
    hi = chat(client, "heyyy", city="Kolkata")
    assert hi["clarify"]["kind"] == "ask_kind"
    out = say(client, hi, "neend nhi a raha", city="Kolkata")
    assert out["state"]["symptoms"] == ["insomnia"] and out["language"] == "hinglish"


def test_a_greeting_does_not_fix_the_language(client, hms):
    hi = chat(client, "helloo", city="Kolkata")
    out = say(client, hi, "so nahi pa raha", city="Kolkata")
    assert out["language"] == "hinglish" and out["kind"] != "emergency"


@pytest.mark.parametrize("text", ["so nahi pa raha", "neend nahi aa rahi 2 hafte se", "I can't sleep at night"])
def test_cannot_sleep_alone_is_never_an_emergency_for_the_model(text):
    assert sleep_only(q(text))
    assert not ModelDetector(lambda _q: 0.99, 0.6).detect(q(text)).flag


@pytest.mark.parametrize("text", ["saans nahi aa rahi", "so nahi pa raha aur saans phool rahi hai", "bol nahi pa raha",
                                  "neend nahi aa rahi aur seene mein dard"])
def test_anything_more_keeps_the_model(text):
    assert not sleep_only(q(text))
    assert ModelDetector(lambda _q: 0.99, 0.6).detect(q(text)).flag


def test_real_emergencies_still_are(client, hms):
    assert chat(client, "saans nahi aa rahi", city="Kolkata")["kind"] == "emergency"
    assert chat(client, "neend nahi aa rahi aur seene mein tez dard", city="Kolkata")["kind"] == "emergency"


def test_never_the_same_question_twice(client, hms):
    first = chat(client, "kuch bhi", city="Kolkata")
    second = say(client, first, "asdf", city="Kolkata")
    assert second["message"] != first["message"]
    assert second["message"].startswith("Maaf kijiye, main samajh nahi paayi")
    assert "Symptom batayein" in [o["label"] for o in second["clarify"]["options"]]
    third = say(client, second, "qwerty", city="Kolkata")
    assert "Kisi insaan se baat karein" in [o["label"] for o in third["clarify"]["options"]]
