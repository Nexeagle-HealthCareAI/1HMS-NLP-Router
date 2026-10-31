"""After Vita has said "call 108": the next messages are judged on their own (a stomach ache is not the earlier
breathlessness again), "ambulance aa gayi" / "ab theek hoon" / "108 nahi lag raha" get a fitting answer, the number is
repeated once before anything else, and the model's internal reason is never shown as a sign."""
import pytest

from hms_router.vita_graph.engine import VitaRuntime
from tests.test_vita_doctor_chat import chat, client, hms  # noqa: F401
from tests.test_vita_graph import ScriptedLLM, say as reply, u


def scripted(client, *steps):
    a = client.app.state
    a.vita_runtime = VitaRuntime(ScriptedLLM(*steps), a.redis, route_service=lambda: a.route)


def say(client, prev, text):
    return chat(client, text, session_id=prev["session_id"], city="Kolkata")


@pytest.fixture
def breathless(client, hms):
    out = chat(client, "saans mai bahut dikkat ho rahi hai", city="Kolkata")
    assert out["kind"] == "emergency"
    assert out["card"]["emergency_reasons"] == ["Breathlessness"] and out["card"]["emergency_signs"] == ["saans phoolna"]
    return out


def test_the_models_reason_is_not_a_sign(client, hms):
    out = chat(client, "emergency", city="Kolkata")
    assert out["kind"] == "emergency"
    assert "training data" not in out["message"] and not out["card"]["emergency_reasons"]
    assert "Signs:" not in out["message"]


def test_a_new_symptom_is_not_the_old_emergency(client, breathless):
    scripted(client, u("symptoms", "stomach pain", enough=True), reply("Pet dard ke liye ye doctor sahi rahenge."),
             u("symptoms", "stomach pain and vomiting", enough=True), reply("Ye doctor sahi rahenge."))
    out = say(client, breathless, "pet mai dard")
    assert out["kind"] == "recommendation" and out["card"]["outputs"][0]["specialist"] != "Emergency Medicine"
    assert out["message"].startswith("Pehle: agar saans phoolna abhi bhi hai, to turant 108 par call karein.")
    again = say(client, out, "aur ulti bhi")                      # the number is said once, not on every reply
    assert not again["message"].startswith("Pehle:")


def test_help_arrived(client, breathless):
    out = say(client, breathless, "ambulance aa gayi")
    assert out["kind"] == "info" and out["message"].startswith("Achha hai, madad pahunch gayi.")
    better = say(client, out, "ab theek hoon")
    assert better["kind"] == "info" and "saans phoolna ki jaanch" in better["message"]
    assert better["clarify"]["kind"] == "ask_kind"


def test_cannot_get_through(client, breathless):
    out = say(client, breathless, "108 nahi lag raha")
    assert out["kind"] == "info" and out["message"].startswith("112 par try karein.")


@pytest.mark.parametrize("text", ["ok", "thank you", "theek hai"])
def test_ok_after_an_emergency(client, breathless, text):
    out = say(client, breathless, text)
    assert out["kind"] == "info" and "108" in out["message"] and "Sunkar" not in out["message"]


def test_a_new_red_flag_is_an_emergency_again(client, breathless):
    out = say(client, breathless, "pet mai dard")
    out = say(client, out, "seene mein tez dard aur paseena")
    assert out["kind"] == "emergency" and out["card"]["emergency_reasons"] == ["Chest pain"]
    assert "Lakshan: seene mein dard." in out["message"]
