"""Regression tests for the 2026-10-22 safety fixes (Symptom Router & Doctor Dekho analysis, issues 3, 4, 5, 7
and test-plan check H-03):

* /chat uses the same emergency detector as /v1/route-symptom (it used to miss self-harm, seizure, stroke ...)
* an emergency is answered directly, never as a clarifying question
* head injury with vomiting or fainting is an emergency
* ageGroup=child settles on Paediatrics when it is in the top 3, instead of asking again
* the LLM reasoner's explanation may not name a symptom the patient did not say
"""
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from hms_router.domain.models import Candidate, Evidence, ReasonerResult
from hms_router.domain.policy import RoutingPolicy
from hms_router.reasoning.verifier import Verifier

EMERGENCIES = [
    "seene mein tez dard aur saans nahi aa rahi",
    "seene mein dard kam nahi ho raha",
    "mujhe marne ka mann kar raha hai",
    "bachche ko daura pad raha hai jhatke aa rahe hain",
    "pregnancy mein bleeding ho rahi hai",
    "galti se zeher pi liya",
    "achanak chehra tedha ho gaya aur haath kamzor",
    "behosh ho gaye hain hosh nahi aa raha",
    "bahut khoon beh raha hai ruk nahi raha",
    "chest pain and sweating since 20 minutes",
    "can't breathe properly lips turning blue",
    "sir pe chot lagi aur ulti ho rahi hai",
]
ROUTINE = [
    "saans lene mein koi dikkat nahi",
    "pet mein dard",
    "2 din se bukhar",
    "aankh laal hai",
    "kamar dard",
    "skin pe daane",
    "dant mein dard",
    "chest pain nahi hai bas khansi hai",
    "periods late hain",
    "neend nahi aati",
]


@pytest.fixture(scope="module")
def client(bundle):
    with TestClient(create_app(ServiceSettings.from_env({}))) as c:
        yield c


# ----------------------------------------------------------------------------- one detector everywhere
@pytest.mark.parametrize("text", EMERGENCIES)
def test_vita_flags_every_emergency_route_symptom_flags(client, text):
    assert client.post("/v1/route-symptom", json={"query": text}).json()["emergency"]["flag"] is True
    reply = client.post("/v2/agent/chat", json={"text": text}).json()
    assert reply["kind"] == "emergency", reply.get("message")


@pytest.mark.parametrize("text", ROUTINE)
def test_vita_raises_no_false_alarm_on_routine_text(client, text):
    assert client.post("/v2/agent/chat", json={"text": text}).json()["kind"] != "emergency"


def test_vita_emergency_card_carries_the_detector_reasons(client):
    card = client.post("/v2/agent/chat", json={"text": "mujhe marne ka mann kar raha hai"}).json()["card"]
    assert card["kind"] == "emergency"
    assert any("self-harm" in r.lower() or "suicid" in r.lower() for r in card["emergency_reasons"])


# ----------------------------------------------------------------------------- emergency answered directly
@pytest.mark.parametrize("text", EMERGENCIES)
def test_route_symptom_answers_an_emergency_without_a_question(client, text):
    r = client.post("/v1/route-symptom", json={"query": text}).json()
    assert r["decision"] == "answer"
    assert r["clarify"] is None
    assert r["specialtyIds"][0] == "emergencymedicine"
    assert r["method"] == "emergency"
    assert r["decisive"] is True


def test_head_injury_routes_to_emergency_then_neurosurgery(client):
    r = client.post("/v1/route-symptom", json={"query": "sir pe chot lagi aur ulti ho rahi hai"}).json()
    assert r["emergency"]["flag"] is True
    assert "Head injury with vomiting" in r["emergency"]["reasons"]
    assert r["specialtyIds"][:2] == ["emergencymedicine", "neurosurgery"]


def test_routine_answer_is_unchanged(client):
    r = client.post("/v1/route-symptom", json={"query": "pet mein dard"}).json()
    assert r["decision"] == "answer" and r["specialtyIds"] == ["gastroenterology"] and r["method"] == "deterministic"


# ----------------------------------------------------------------------------- child rule
def test_child_settles_on_paediatrics_when_already_first():
    ranked = [("pediatrics", 0.55), ("general", 0.36), ("ent", 0.06)]
    out, applied = RoutingPolicy().apply_age_rule(ranked, "child")
    assert applied is True and out[0][0] == "pediatrics"


def test_child_does_not_force_paediatrics_outside_the_top_three():
    ranked = [("cardiology", 0.5), ("general", 0.3), ("pulmonology", 0.1), ("pediatrics", 0.05)]
    out, applied = RoutingPolicy().apply_age_rule(ranked, "child")
    assert applied is False and out == ranked


def test_age_answer_child_gives_one_answer_not_a_second_question(client):
    first = client.post("/v1/route-symptom", json={"query": "2 din se bukhar aur gale mein dard hai"}).json()
    assert first["decision"] == "clarify" and first["clarify"]["kind"] == "age"
    second = client.post("/v1/route-symptom", json={"query": "2 din se bukhar aur gale mein dard hai",
                                                     "context": {"ageGroup": "child"}}).json()
    assert second["decision"] == "answer"
    assert second["specialtyIds"][0] == "pediatrics"


# ----------------------------------------------------------------------------- H-03: no invented symptoms
def _cands():
    ev = Evidence(id="e1", text="pet mein dard", specialtyId="gastroenterology", similarity=1.0)
    return [Candidate("gastroenterology", "Gastroenterology", 0.6, [ev]),
            Candidate("cardiology", "Cardiology", 0.2, [])]


def _verifier():
    from hms_router.rag.vocabulary import MedicalVocabulary
    return Verifier(medical_terms=MedicalVocabulary.from_data().terms_in)


def _result(reason):
    return ReasonerResult(choices=["gastroenterology"], citedEvidenceIds=["e1"], reason=reason)


def test_verifier_rejects_a_reason_with_a_symptom_the_patient_did_not_say():
    verdict = _verifier().check(_result("Suggested because of chest pain and fever."), _cands(),
                                patient_text="pet mein dard")
    assert not verdict.ok
    assert any("did not say" in e for e in verdict.errors)


def test_verifier_accepts_the_patients_own_words_and_the_specialty_name():
    verdict = _verifier().check(_result("Gastroenterology fits because of the pet dard you described."), _cands(),
                                patient_text="pet mein dard")
    assert verdict.ok, verdict.errors


def test_verifier_tolerates_the_patients_spelling():
    verdict = _verifier().check(_result("Your stomach pain matches Gastroenterology cases."), _cands(),
                                patient_text="stomach pian since morning")
    assert verdict.ok, verdict.errors


def test_verifier_without_vocabulary_keeps_the_old_behaviour():
    verdict = Verifier().check(_result("Suggested because of chest pain."), _cands(), patient_text="pet mein dard")
    assert verdict.ok
