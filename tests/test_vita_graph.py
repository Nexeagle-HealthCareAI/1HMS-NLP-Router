"""Vita on LangGraph with a tool-calling LLM (hms_router/vita_graph). A scripted model stands in for Groq."""
import pytest

pytest.importorskip("langgraph")
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from hms_router.adapters.http import create_app  # noqa: E402
from hms_router.adapters.settings import ServiceSettings  # noqa: E402
from hms_router.vita_graph import build_vita_graph  # noqa: E402
from hms_router.vita_graph import checker  # noqa: E402
from hms_router.vita_graph.engine import Breaker, VitaRuntime  # noqa: E402
from tests.test_vita_doctor_chat import Fake1Hms, make_directory  # noqa: E402


class ScriptedLLM:
    """Returns the scripted AIMessages in order; records what it was sent. Running out = a failure."""

    def __init__(self, *steps):
        self.steps, self.seen = list(steps), []

    def bind_tools(self, tools, **_):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        if not self.steps:
            raise RuntimeError("LLM should not have been called")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(messages) if callable(step) else step


def call(name, i=0, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}-{name}"}])


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        yield c


@pytest.fixture
def llm(client):
    holder = {}

    def use(*steps):
        holder["llm"] = ScriptedLLM(*steps)
        client.app.state.vita_runtime = VitaRuntime(holder["llm"], build_vita_graph(),
                                                    route_service=lambda: client.app.state.route)
        return holder["llm"]
    yield use
    client.app.state.vita_runtime = None


def vita(client, text, sid=None, **kw):
    r = client.post("/v2/agent/chat", json={"text": text, "session_id": sid, "city": "Kolkata", **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------------------------------- off / rules first
def test_without_llm_vita_is_the_rules_engine(client):
    out = vita(client, "hello")
    assert out["engine"] == "rules"


@pytest.mark.parametrize("text", ["I don't want to live anymore", "seene mein dard aur pasina aa raha hai",
                                  "मैं मरना चाहता हूँ"])
def test_danger_never_waits_for_the_llm(client, llm, text):
    model = llm()                                       # any call would fail the turn
    out = vita(client, text)
    assert out["kind"] == "emergency" and "108" in out["message"] and out["engine"] == "rules"
    assert model.seen == []


def test_doctor_tap_goes_to_rules(client, llm):
    llm(call("find_doctors", specialty="Dermatologist"), call("reply", 1, message="Ye skin ke doctor hain."))
    first = vita(client, "skin doctor chahiye")
    doc = first["doctors"]["items"][0]["doctorId"]
    out = vita(client, "Choose", first["session_id"], select_doctor_id=doc)
    assert out["kind"] == "doctor_selected" and out["engine"] == "rules"


# ------------------------------------------------------------------------------------------- the agent
def test_greeting_answered_by_the_llm(client, llm):
    llm(call("reply", message="Namaste! Main Vita hoon. Aapko kya takleef hai?", options=["Bukhar", "Sar dard"]))
    out = vita(client, "hii")
    assert out["engine"] == "agent" and out["kind"] == "follow_up"
    assert out["message"].startswith("Namaste")
    assert [o["id"] for o in out["clarify"]["options"]] == ["say:Bukhar", "say:Sar dard"]


@pytest.mark.parametrize("text", ["mujhe 3 din se bukhar hai", "मुझे तीन दिन से बुखार है"])
def test_symptoms_through_assess_tool(client, llm, text):
    """The LLM's English reading fixes Hindi-script fever, which route-symptom alone sent to Cardiology."""
    llm(call("assess_symptoms", symptoms="fever for 3 days", age_group="adult"),
        call("reply", 1, message="Bukhar ke liye General Physician sahi rahenge. Neeche doctor hain."))
    out = vita(client, text)
    assert out["engine"] == "agent" and out["kind"] == "recommendation"
    assert out["card"]["outputs"][0]["specialist"] == "General Physician"
    assert out["doctors"]["items"]


def test_assess_tool_raises_urgency_from_its_own_reading(client, llm):
    model = llm(call("assess_symptoms", symptoms="severe chest pain with sweating"))
    out = vita(client, "ajeeb sa lag raha hai, bahut ghabrahat")
    assert out["kind"] == "emergency" and "108" in out["message"] and len(model.seen) == 1


def test_llm_can_raise_an_emergency(client, llm):
    llm(call("raise_emergency", signs=["fainting"]))
    out = vita(client, "papa baar baar gir ja rahe hain")
    assert out["kind"] == "emergency" and out["engine"] == "agent" and "108" in out["message"]


def test_find_doctors_with_filters(client, llm):
    llm(call("find_doctors", specialty="Dermatologist", gender="female"),
        call("reply", 1, message="Ye mahila skin doctor hain, jaise Dr. Anita Sen."))
    out = vita(client, "koi lady skin specialist")
    assert out["kind"] == "doctor_list" and out["engine"] == "agent"
    assert {d["name"] for d in out["doctors"]["items"]} <= {"Dr. Anita Sen", "Dr. Priya Bose"}


def test_near_me_without_location_asks_where(client, llm):
    model = llm(call("find_doctors", specialty="Dermatologist", near_me=True))
    first = vita(client, "paas mein skin doctor", city=None)
    assert first["clarify"]["kind"] == "location" and len(model.seen) == 1
    llm()                                               # the answer to "where are you?" stays with the rules
    out = vita(client, "Kolkata", first["session_id"], city="Kolkata")
    assert out["engine"] == "rules" and out["kind"] == "doctor_list"


def test_booking_and_visit_go_through_vita_flow(client, llm):
    llm(call("vita_flow", message="insaan se baat karni hai"))
    out = vita(client, "mujhe kisi real person se baat karni hai please")
    assert out["kind"] == "handoff" and out["engine"] == "agent"


def test_memory_is_sent_on_the_next_turn(client, llm):
    model = llm(call("reply", message="Aapko kya takleef hai?"))
    sid = vita(client, "hello ji")["session_id"]
    llm(call("reply", message="Kitne din se?"))
    from hms_router.vita_graph.engine import VitaRuntime  # noqa: F401
    second = client.app.state.vita_runtime.llm
    vita(client, "pet mein dard", sid)
    texts = [m.content for m in second.seen[0]]
    assert "hello ji" in texts and "Aapko kya takleef hai?" in texts and model is not second


# ------------------------------------------------------------------------------------------- the checker
@pytest.mark.parametrize("bad", ["Paracetamol 500 mg le lijiye, aaram milega.", "Aapko dengue ho sakta hai.",
                                 "Dr. Ramesh Gupta best hain, unhe dikhayein.", "Call 9876543210 for help.",
                                 "Aapka appointment book ho gaya hai."])
def test_unsafe_llm_reply_never_reaches_the_patient(client, llm, bad):
    llm(call("reply", message=bad))
    out = vita(client, "kuch batao")
    assert out["engine"] == "rules-fallback" and bad not in out["message"]


def test_blocked_reply_after_a_list_keeps_the_list_with_vitas_own_words(client, llm):
    llm(call("find_doctors", specialty="Dermatologist"), call("reply", 1, message="Calpol lo aur Dr. Fake se milo"))
    out = vita(client, "skin doctor dikhao")
    assert out["engine"] == "agent+template" and out["doctors"]["items"] and "Calpol" not in out["message"]


@pytest.mark.parametrize("ok", ["Bukhar ke liye General Physician ko dikhana sahi rahega.",
                                "Main dawai ki salah nahi de sakti, doctor batayenge. Kya main doctor dhoondh doon?",
                                "Aapko kya takleef hai?", "Ye doctors Kolkata mein hain."])
def test_checker_allows_ordinary_replies(ok):
    assert checker.problems(ok, set()) == []


# ------------------------------------------------------------------------------------------- failures
def test_llm_failure_falls_back_and_opens_the_breaker(client, llm):
    llm(RuntimeError("groq down"))
    out = vita(client, "hello")
    assert out["engine"] == "rules-fallback" and out["message"]
    rt = client.app.state.vita_runtime
    rt.breaker = Breaker(limit=1, cooldown=60)
    rt.breaker.fail()
    assert vita(client, "hello")["engine"] == "rules"


def test_text_without_a_tool_call_is_checked_and_used(client, llm):
    llm(AIMessage(content="Namaste! Batayein, main kaise madad karoon?"))
    out = vita(client, "namaste")
    assert out["engine"] == "agent" and out["message"].startswith("Namaste")


def test_whatsapp_contract(client, llm):
    llm(call("reply", message="Kya takleef hai?", options=["Bukhar", "Khansi"]))
    out = vita(client, "hi", channel="whatsapp")
    msg = out["whatsapp"][0]
    assert msg["type"] == "interactive" and msg["interactive"]["type"] == "button"


@pytest.mark.parametrize("text", ["haath jal gaya garam tel se", "main 6 mahine pregnant hoon, pair mein sujan hai"])
def test_warning_sign_questions_stay_with_the_rules(client, llm, text):
    model = llm()
    out = vita(client, text)
    assert out["engine"] == "rules" and out["clarify"]["kind"] == "warning_signs" and model.seen == []
