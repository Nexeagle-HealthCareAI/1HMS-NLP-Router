"""Doctor search, phase 1 (2026-10-30): by symptoms, name, degree, specialty and nearby. Each case below failed before
this change (see the commit): names were searched only in the patient's city and only after "Dr", initials and dotted
degrees were not read, "-ology" specialty words were not specialties, a city with no match never looked elsewhere,
a place alone was not a search, and only the first 1HMS category of a specialty was searched."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from hms_router.doctor_intent import parse
from hms_router.doctors import DoctorDirectory, McpClient, specialty_key
from tests.test_vita_doctor_chat import DOCTORS, Fake1Hms, make_directory

NAMES = [d["name"] for d in DOCTORS]


# ------------------------------------------------------------------------------------------- reading the ask
@pytest.mark.parametrize("text,name", [
    ("Dr Neha Sharma", "neha sharma"), ("neha sharma", "neha sharma"), ("anita sen ji", "anita sen"),
    ("Dr. S. Roy", "roy"), ("dr a k sen", "sen"), ("Dr Neha Sharma ka appointment chahiye", "neha sharma"),
    ("डॉ नेहा शर्मा", "neha sharma"), ("dr nehaa sharmaa", "nehaa sharmaa"),
])
def test_names(text, name):
    assert parse(text, names=NAMES).name_query == name


@pytest.mark.parametrize("text", ["bukhar hai", "pet dard", "mera naam rahul hai", "kal aana", "bas"])
def test_ordinary_messages_are_not_names(text):
    assert parse(text, names=NAMES).name_query is None


@pytest.mark.parametrize("text,degree", [
    ("MD doctor", "MD"), ("M.D. doctor", "MD"), ("m d doctor", "MD"), ("mbbs md doctor", "MD"),
    ("MBBS, MD (Medicine) doctor", "MD"), ("DM cardiology doctor", "DM"), ("koi MS wala surgeon", "MS"),
    ("BDS dentist", "BDS"), ("dnb doctor", "DNB"), ("MCh doctor", "MCh"),
])
def test_degrees_most_specific_wins(text, degree):
    assert parse(text).degree == degree


@pytest.mark.parametrize("text,key", [
    ("cardiology doctor", "cardiology"), ("dermatology", "dermatology"), ("gynaecology", "gynaecology"),
    ("DM cardiology doctor", "cardiology"), ("neurology", "neurology"), ("orthopedics", "orthopaedics"),
    ("skin doctor", "dermatology"), ("haddi ka doctor", "orthopaedics"), ("बच्चों के डॉक्टर", "paediatrics"),
    ("chest specialist", "pulmonology"), ("medicine doctor", "general-medicine"),
])
def test_specialty_words(text, key):
    assert parse(text).specialty_key == key


@pytest.mark.parametrize("text,place", [("kishanganj ke doctor", "kishanganj"), ("purnia wale doctors", "purnia")])
def test_a_place_alone(text, place):
    a = parse(text)
    assert a.place == place and not a.direct


@pytest.mark.parametrize("dept,key", [
    ("Medicine", "general-medicine"), ("Ortho", "orthopaedics"), ("Orthopaedic Surgeon (Bone)", "orthopaedics"),
    ("Chest", "pulmonology"), ("TB & Chest", "pulmonology"), ("Heart", "cardiology"), ("Surgery", "general-surgery"),
    ("E.N.T", "ent"), ("Obs & Gynae", "gynaecology"), ("OPD", "general-medicine"), ("Family Medicine", "general-medicine"),
    ("Resident Medical Officer", None), ("Accident", None), ("Accident & Emergency", "emergency-medicine"),
    ("Dental", "dentistry"), ("Neurosurgery", "neurosurgery"), ("Sports Medicine", "sports-medicine"),
])
def test_1hms_department_names(dept, key):
    assert specialty_key(dept) == key


def test_every_category_of_a_specialty_is_searched():
    calls = []

    def server(request):
        msg = json.loads(request.content)
        if msg.get("method") != "tools/call":
            if "id" not in msg:
                return httpx.Response(202)
            result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "f", "version": "1"}}
        else:
            name, args = msg["params"]["name"], msg["params"].get("arguments") or {}
            calls.append((name, args))
            data = {"specialties": [{"specialtyId": "Gynaecology", "doctorCount": 2},
                                    {"specialtyId": "Obs & Gynae", "doctorCount": 1},
                                    {"specialtyId": "Medicine", "doctorCount": 4}]} \
                if name == "list_specialties" else {"doctors": [], "total": 0}
            result = {"structuredContent": data}
        body = "event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "id": msg.get("id"), "result": result}) + "\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    d = DoctorDirectory(McpClient("http://api/mcp", "k", transport=httpx.MockTransport(server)))
    d.search(specialty_key="gynaecology")
    d.search(specialty_key="general-medicine")
    sent = [a["specialtyIds"] for n, a in calls if n == "search_doctors"]
    assert sent == [["Gynaecology", "Obs & Gynae"], ["Medicine"]]


# ------------------------------------------------------------------------------------------- end to end
@pytest.fixture
def chat():
    server = Fake1Hms()
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(server)

        def send(text, city="Kolkata", sid=None):
            before = len(server.calls)
            out = c.post("/v2/agent/chat", json={"text": text, "city": city, "session_id": sid}).json()
            sent = [a for n, a in server.calls[before:] if n == "search_doctors"]
            return out, sent
        yield send


def names_in(out):
    return [d["name"] for d in (out.get("doctors") or {}).get("items") or []]


def test_name_in_another_city(chat):
    out, sent = chat("Dr Neha Sharma", city="Purnia")
    assert names_in(out) == ["Dr. Neha Sharma"] and "city" not in sent[0]


@pytest.mark.parametrize("text,who", [("neha sharma", "Dr. Neha Sharma"), ("Dr. S. Roy", "Dr. S. Roy")])
def test_name_without_dr_or_with_an_initial(chat, text, who):
    out, _ = chat(text)
    assert out["kind"] == "doctor_list" and who in names_in(out)


def test_dotted_degree(chat):
    out, sent = chat("M.D. doctor")
    assert out["kind"] == "doctor_list" and sent[0]["degree"] == "MD" and names_in(out)


def test_degree_and_specialty_together(chat):
    out, sent = chat("DM cardiology doctor")
    assert sent[0]["degree"] == "DM" and sent[0]["specialtyIds"] == ["Cardiologist (Heart)"]
    assert names_in(out) == ["Dr. Neha Sharma"]


@pytest.mark.parametrize("text,specialty", [("cardiology doctor", "Cardiologist (Heart)"),
                                            ("dermatology", "Dermatologist (Skin)")])
def test_ology_words(chat, text, specialty):
    out, sent = chat(text)
    assert out["kind"] == "doctor_list" and sent[0]["specialtyIds"] == [specialty] and names_in(out)


def test_nobody_in_the_city_looks_in_other_towns_and_says_so(chat):
    out, sent = chat("skin doctor", city="Purnia")
    assert sent[0]["city"] == "Purnia" and "city" not in sent[1]
    assert set(names_in(out)) <= {"Dr. Anita Sen", "Dr. Priya Bose"} and names_in(out)
    assert "Purnia" in out["message"] and "other towns" in out["message"]
    assert not out["doctors"]["fallbackUsed"]                     # specialists elsewhere before a General Physician


def test_a_place_alone_then_the_kind_of_doctor(chat):
    first, sent = chat("kolkata ke doctor", city="Purnia")
    assert first["kind"] == "follow_up" and "Kolkata" in first["message"] and not sent
    out, sent = chat("skin wale", city="Purnia", sid=first["session_id"])
    assert sent[0]["city"].lower() == "kolkata" and names_in(out)


def test_symptoms_still_go_to_triage(chat):
    out, _ = chat("mujhe 3 din se bukhar hai")
    assert out["kind"] in ("follow_up", "recommendation")


