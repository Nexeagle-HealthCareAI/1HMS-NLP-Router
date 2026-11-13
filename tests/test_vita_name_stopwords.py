"""Seen on Doctor Dekho: "doctor bataoo" got "'bataoo' naam ke koi doctor nahi mile". Asking words, also stretched or
doubled ("bataoo", "dikhaoo", "chahiyeee", "acchha"), are never a doctor's name."""
import pytest

from hms_router.doctor_intent import parse
from tests.test_vita_doctor_chat import chat, client, hms  # noqa: F401


@pytest.mark.parametrize("text", ["doctor bataoo", "doctor btao", "dr bataoo na", "doctor dikhaoo", "doctor suggest karo",
                                  "koi acchha doctor bataiye", "doctor chahiyeee", "doctor dhoondho"])
def test_asking_words_are_not_names(text):
    assert parse(text).name_query is None


@pytest.mark.parametrize("text, name", [("dr anita sen", "anita sen"), ("dr sharma ji", "sharma"), ("dr kar", "kar"),
                                        ("doctor Anna Thomas", "anna thomas"), ("dr neha sharma dikhao", "neha sharma")])
def test_names_are_still_names(text, name):
    assert parse(text).name_query == name


def test_doctor_bataoo_asks_what_kind(client, hms):
    out = chat(client, "doctor bataoo", city="Kolkata")
    assert "naam ke koi doctor nahi" not in out["message"]
    assert out["clarify"]["kind"] == "ask_kind" and out["language"] == "hinglish"
