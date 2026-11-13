"""Warning-sign questions before "call 108" (2026-10-30): a minor burn went straight to 108 because the emergency model
alone flagged it, and pregnancy + swelling got no warning at all. A red-flag rule, self-harm, or a warning sign said in
the first message still means 108 at once."""
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from tests.test_vita_doctor_chat import Fake1Hms, make_directory


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        yield c


def vita(client, text, sid=None):
    r = client.post("/v2/agent/chat", json={"text": text, "session_id": sid, "city": "Kolkata"})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("first,kind", [
    ("haath jal gaya garam tel se", "burn"), ("garam chai gir gayi pair pe", "burn"), ("ungli jal gayi", "burn"),
    ("main 6 mahine pregnant hoon, pair mein sujan hai", "pregnancy"),
    ("I am 7 months pregnant and my feet are swollen", "pregnancy"),
    ("मैं 7 महीने से प्रेग्नेंट हूँ, पैरों में सूजन है", "pregnancy"),
    ("sar pe chot lagi", "head"),
])
def test_asks_the_warning_signs_first(client, first, kind):
    out = vita(client, first)
    assert out["kind"] == "follow_up" and out["clarify"]["kind"] == "warning_signs"
    assert "108" not in out["message"]


@pytest.mark.parametrize("first", ["haath jal gaya garam tel se", "main 6 mahine pregnant hoon, pair mein sujan hai",
                                   "sar pe chot lagi"])
def test_yes_is_an_emergency(client, first):
    sid = vita(client, first)["session_id"]
    out = vita(client, "haan", sid)
    assert out["kind"] == "emergency" and "108" in out["message"]


def test_burn_no_is_a_doctor_today(client):
    sid = vita(client, "haath jal gaya garam tel se")["session_id"]
    out = vita(client, "nahi", sid)
    assert out["kind"] == "doctor_list" and "doctor ko dikhayein" in out["message"]


def test_pregnancy_no_is_a_gynaecologist_within_a_day(client):
    sid = vita(client, "main 6 mahine pregnant hoon, pair mein sujan hai")["session_id"]
    out = vita(client, "nahi", sid)
    assert "gynaecologist" in out["message"].lower() and "108" in out["message"]     # when to call stays in the reply
    assert out["doctors"]["filters"].get("specialtyId") or out["doctors"]["fallbackUsed"]


@pytest.mark.parametrize("first", ["bijli ka current lag ke haath jal gaya", "chehra jal gaya", "acid se haath jal gaya",
                                   "pregnant hoon, pair mein sujan aur dhundhla dikh raha hai",
                                   "sar pe chot lagi aur behosh ho gaya", "sar pe chot lagi aur ulti ho rahi hai",
                                   "seene mein dard aur pasina aa raha hai"])
def test_a_sign_in_the_first_message_is_108_at_once(client, first):
    out = vita(client, first)
    assert out["kind"] == "emergency" and "108" in out["message"]


