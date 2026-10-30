"""Vita's specialist comes from the route-symptom engine (route_bridge.py). Probe 2026-10-30 cases."""
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from hms_router.route_bridge import age_group_of
from tests.test_vita_doctor_chat import Fake1Hms, make_directory


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        yield c


def _until_advice(client, first):
    sid, out = None, None
    for msg in [first, "nahi", "3 din se", "nahi"]:
        out = client.post("/v2/agent/chat", json={"text": msg, "session_id": sid, "city": "Kolkata"}).json()
        sid = out["session_id"]
        if out["kind"] != "follow_up":
            break
    return out


@pytest.mark.parametrize("first,primary", [
    ("mujhe 3 din se bukhar hai", "General Physician"),             # was Gastroenterologist
    ("I have fever since 3 days", "General Physician"),
    ("daant mein bahut dard hai", "Dentist"),                       # was Dermatologist
    ("I have a toothache", "Dentist"),
    ("mere 2 saal ke bete ko tez bukhar hai", "Paediatrician"),
    # pregnant + swelling (was Orthopedist) now gets the warning-sign question first: test_vita_warning_signs.py
    ("aankh laal hai aur paani aa raha hai", "Eye Specialist"),
    ("periods 2 mahine se nahi aaye", "Gynaecologist"),
    ("bahut ghabrahat hoti hai raat ko neend nahi aati", "Psychiatrist"),
    ("sugar badha hua hai 300", "Endocrinologist"),
    ("peshab mein jalan", "Urologist"),
    ("skin pe khujli aur daane", "Dermatologist"),
])
def test_vita_primary_specialist(client, first, primary):
    out = _until_advice(client, first)
    assert out["kind"] == "recommendation"
    roles = {o["role"]: o["specialist"] for o in out["card"]["outputs"]}
    assert roles["PRIMARY"] == primary and not out["card"]["low_confidence"]


@pytest.mark.parametrize("first", ["pet mein dard aur dast", "sar dard, ulti aur chakkar aa rahe hain", "mujhe 3 din se bukhar hai"])
def test_no_alarming_second_option(client, first):
    roles = {o["role"]: o["specialist"] for o in _until_advice(client, first)["card"]["outputs"]}
    assert roles["SECONDARY"] not in ("Oncologist", "Neurosurgeon", "Paediatrician")


@pytest.mark.parametrize("text,group", [
    ("mere 2 saal ke bete ko bukhar", "child"), ("baby ko dast", "child"), ("8 year old with cough", "child"),
    ("6 mahine ka baccha ro raha hai", "child"), ("meri beti 25 saal ki hai, sar dard", "adult"),
    ("main 6 mahine pregnant hoon", None), ("mujhe bukhar hai", None), ("मेरे बच्चे को बुखार है", "child"),
])
def test_age_group(text, group):
    assert age_group_of(text) == group


def test_chat_endpoint_unchanged(client):
    """/chat keeps the older retriever (route_service is only given to Vita)."""
    assert client.app.state.chat.agent.route_service is None
    assert client.app.state.vita_agent.route_service is not None
