"""Vita's planner (vita_graph/planner.py, control switch mode "planner"): the model plans with tools around one picture
of the patient's needs. A scripted model stands in for Groq; each step is one model answer with its tool calls."""
import json

import pytest

pytest.importorskip("langgraph")
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from hms_router.adapters.http import create_app  # noqa: E402
from hms_router.adapters.settings import ServiceSettings  # noqa: E402
from hms_router.vita_graph.engine import Breaker, VitaRuntime  # noqa: E402
from hms_router.vita_graph.planner import merge_needs  # noqa: E402
from tests.test_vita_doctor_chat import Fake1Hms, make_directory  # noqa: E402
from tests.test_vita_graph import ScriptedLLM  # noqa: E402


def calls(*pairs):
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(pairs)])


def reply(message, options=None):
    return calls(("reply", {"message": message, "options": options or []}))


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        yield c


@pytest.fixture
def planner(client):
    a = client.app.state

    def use(*steps):
        model = ScriptedLLM(*steps)
        a.vita_runtime = VitaRuntime(model, a.redis, route_service=lambda: a.route)
        a.vita_runtime.set_control({"mode": "planner"})
        return model
    yield use
    a.vita_runtime.set_control({"mode": "agent"})
    a.vita_runtime = VitaRuntime(None, a.redis, route_service=lambda: a.route)


def vita(client, text, sid=None, **kw):
    kw.setdefault("city", "Kolkata")
    r = client.post("/v2/agent/chat", json={"text": text, "session_id": sid, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def names(out):
    return [d["name"] for d in (out.get("doctors") or {}).get("items") or []]


def tool_results(model, step):
    return [m.content for m in model.seen[step] if isinstance(m, ToolMessage)]


# ------------------------------------------------------------------------------------------- many needs at once
def test_everything_in_one_message_is_one_search(client, planner):
    model = planner(
        calls(("note_needs", {"who": "parent", "doctor_gender": "female", "budget": "low", "specialty": "cardiology"}),
              ("find_doctors", {"specialty": "Cardiologist"})),
        reply("Ye mahila heart doctor hain, kam fees wale pehle."))
    out = vita(client, "meri maa ke liye koi lady heart doctor, sasta wala")
    assert out["kind"] == "doctor_list" and out["engine"] == "planner"
    assert out["doctors"]["filters"]["gender"] == "female" and out["doctors"]["filters"]["sort"] == "fee"
    assert names(out) == ["Dr. Neha Sharma"] and len(model.seen) == 2


def test_needs_said_earlier_shape_a_later_search(client, planner):
    planner(calls(("note_needs", {"doctor_gender": "female"})),
            reply("Zaroor, lady doctor hi dhoondhungi. Kis tarah ka doctor chahiye?"))
    sid = vita(client, "mujhe sirf lady doctor chahiye")["session_id"]
    planner(calls(("find_doctors", {"specialty": "Cardiologist"})), reply("Ye heart ki lady doctor hain."))
    out = vita(client, "heart wali", sid)
    assert names(out) == ["Dr. Neha Sharma"]                          # the gender came from the needs


def test_any_removes_a_preference():
    assert merge_needs({"doctor_gender": "female", "budget": "low"}, {"doctor_gender": "any"}) == {"budget": "low"}


# ------------------------------------------------------------------------------------------- symptoms
def test_a_clear_complaint_is_assessed_without_questions(client, planner):
    planner(calls(("note_needs", {"complaint_en": "toothache for 2 days, severe", "in_a_hurry": True}),
                  ("assess_symptoms", {})),
            reply("Daant ke dard ke liye Dentist sahi rahenge; neeche doctor hain."))
    out = vita(client, "2 din se daant mein bahut dard, jaldi doctor batao")
    assert out["kind"] == "recommendation" and out["card"]["outputs"][0]["specialist"] == "Dentist"


def test_the_engine_says_when_it_is_unsure(client, planner):
    model = planner(calls(("note_needs", {"complaint_en": "fever and sore throat for 2 days"}), ("assess_symptoms", {})),
                    reply("Ye bachche ke liye hai ya bade ke liye?"))
    vita(client, "2 din se bukhar aur gale mein dard hai")
    result = json.loads(tool_results(model, 1)[-1])
    assert (result.get("confident") is False or result.get("engine_unsure")) and result["engine_question"], result


def test_never_endless_questions(client, planner):
    q = [calls(("note_needs", {"complaint_en": "headache"})), reply("Kab se?"),
         calls(("note_needs", {"complaint_en": "headache, 2 days"})), reply("Kitna tez?"),
         calls(("note_needs", {"complaint_en": "headache, 2 days, mild"})), reply("Aur kuch?"),
         reply("Aur kya hota hai?")]
    planner(*q)
    sid = vita(client, "sar dard")["session_id"]
    vita(client, "2 din se", sid)
    vita(client, "halka", sid)
    out = vita(client, "pata nahi", sid)
    assert out["kind"] == "recommendation"                            # assessed instead of a 4th question


# ------------------------------------------------------------------------------------------- safety, choosing
def test_danger_signs_end_the_turn_at_once(client, planner):
    model = planner(calls(("note_needs", {"complaint_en": "father is confused and very drowsy",
                                          "danger_signs": ["sudden confusion"]}),
                          ("find_doctors", {"specialty": "Neurologist"})))
    out = vita(client, "papa kuch ajeeb baatein kar rahe hain, bahut neend mein hain")
    assert out["kind"] == "emergency" and "108" in out["message"] and len(model.seen) <= 1
    assert not out.get("doctors")


def test_choose_by_position(client, planner):
    planner(calls(("find_doctors", {"specialty": "Dermatologist"})), reply("Ye skin doctor hain."))
    sid = vita(client, "skin doctor")["session_id"]
    planner(calls(("choose_doctor", {"position": 2})))
    out = vita(client, "jinka experience zyada hai unhe book karo", sid)
    assert out["kind"] == "doctor_selected" and out["engine"] == "planner+step"


def test_unsafe_reply_is_replaced(client, planner):
    planner(calls(("find_doctors", {"specialty": "Dermatologist"})), reply("Calpol lo aur Dr. Fake se milo"))
    out = vita(client, "skin doctor")
    assert "Calpol" not in out["message"] and out["engine"] == "planner+template" and names(out)


def test_safety_steps_still_come_first(client, planner):
    model = planner()
    out = vita(client, "I don't want to live anymore")
    assert out["kind"] == "emergency" and out["engine"] == "step" and model.seen == []


# ------------------------------------------------------------------------------------------- switch, failures
def test_planner_is_per_channel(client, planner):
    model = planner(calls(("find_doctors", {"specialty": "Dermatologist"})), reply("Ye skin doctor hain."))
    rt = client.app.state.vita_runtime
    rt.set_control({"mode": "agent", "channels": {"whatsapp": "planner"}})
    out = vita(client, "skin doctor", channel="whatsapp")
    assert out["engine"] == "planner" and len(model.seen) == 2
    assert client.get("/v2/agent/admin/control").json()["channels"] == {"whatsapp": "planner"}


def test_model_failure_falls_back(client, planner):
    planner(RuntimeError("groq down"))
    out = vita(client, "kuch samajh nahi aa raha")
    assert out["engine"] == "fallback"
    rt = client.app.state.vita_runtime
    rt.breaker = Breaker(limit=1, cooldown=60)
    rt.breaker.fail()
    assert vita(client, "cardiologist")["kind"] == "doctor_list"       # searches still work while it is down
