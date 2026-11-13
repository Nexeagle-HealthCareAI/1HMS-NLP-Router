"""POST /v2/agent/search-intent: Doctor Dekho's search bar reads queries with Vita's own reader (2026-10-31)."""
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from hms_router.application.search_intent import KEY_TO_ID
from hms_router.doctors import specialty_key
from hms_router.domain.taxonomy import SPECIALTY_NAMES
from tests.test_vita_doctor_chat import Fake1Hms, make_directory


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        yield c


def intent(client, q, **kw):
    r = client.post("/v2/agent/search-intent", json={"query": q, **kw})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("q,kind,ids,extra", [
    ("cardiolgist kolkatta", "specialty", ["cardiology"], {"city": "Kolkata"}),
    ("lady skin doctor", "specialty", ["dermatology"], {"gender": "female"}),
    ("dr neha sarma", "name", [], {"doctorName": "neha sharma"}),
    ("mbbs md doctor near me", "degree", [], {"degree": "MD", "city": "NEAR_ME", "near": True}),
    ("Sadar hospital purnia", "hospital", [], {}),
    ("pet mein dard hai", "symptom", ["gastroenterology"], {"method": "vita+route-symptom"}),
    ("daant dard", "symptom", ["dentistry"], {}),
    ("bachche ko bukhar", "symptom", ["pediatrics"], {}),
    ("hello", "none", [], {"specialtyId": None}),
])
def test_what_the_bar_understands(client, q, kind, ids, extra):
    out = intent(client, q)
    assert out["kind"] == kind and out["specialtyIds"][:len(ids)] == ids
    for k, v in extra.items():
        assert out[k] == v, (k, out[k])
    assert out["specialtyId"] == (out["specialtyIds"][0] if out["specialtyIds"] else None)


def test_an_emergency_is_flagged_and_still_has_a_list(client):
    out = intent(client, "chest pain")
    assert out["kind"] == "emergency" and out["emergency"]["flag"] and out["emergency"]["reasons"] == ["Chest pain"]
    assert out["emergency"]["recommendedSpecialtyIds"][0] == "emergencymedicine" and out["specialtyIds"]


def test_query_is_required(client):
    assert client.post("/v2/agent/search-intent", json={"query": ""}).status_code == 422


def test_every_specialty_key_has_a_site_id():
    assert set(KEY_TO_ID.values()) <= set(SPECIALTY_NAMES)
    for word, site in (("Cardiologist", "cardiology"), ("Orthopaedic Surgeon", "orthopedics"), ("Paediatrician", "pediatrics"),
                       ("General Physician", "general"), ("Gynaecologist", "gynecology")):
        assert KEY_TO_ID[specialty_key(word)] == site


def test_symptoms_carry_route_symptoms_own_fields(client):
    out = intent(client, "pet mein dard hai")
    assert out["decision"] in ("answer", "clarify") and "reason" in out and "usedDefault" in out
    assert "recommendedSpecialtyIds" in out["emergency"]


def test_the_age_answer_is_passed_on(client):
    child = intent(client, "bukhar hai", context={"ageGroup": "child"})
    assert child["specialtyIds"][0] == "pediatrics"


def test_old_chat_path_is_vita(client):
    out = client.post("/chat", json={"text": "cardiologist", "city": "Kolkata"}).json()
    assert out["kind"] == "doctor_list" and out["doctors"]["items"]
    assert client.post("/chat", json={"text": "seene mein dard aur pasina"}).json()["kind"] == "emergency"
