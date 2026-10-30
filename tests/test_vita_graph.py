"""Vita on LangGraph (hms_router/vita_graph): the whole conversation as one graph with saved state, learning and the
promotion gate. A scripted model stands in for Groq: each step is (understanding, reply) as tool calls."""
import json

import pytest

pytest.importorskip("langgraph")
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from hms_router.adapters.http import create_app  # noqa: E402
from hms_router.adapters.settings import ServiceSettings  # noqa: E402
from hms_router.vita_graph import checker  # noqa: E402
from hms_router.vita_graph.engine import Breaker, VitaEngine, VitaRuntime  # noqa: E402
from tests.test_vita_doctor_chat import Fake1Hms, make_directory  # noqa: E402


class ScriptedLLM:
    """Returns the scripted messages in order (understand, respond, understand, ...); records what it was sent."""

    def __init__(self, *steps):
        self.steps, self.seen = list(steps), []

    def bind_tools(self, tools, **_):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        if not self.steps:
            raise RuntimeError("LLM should not have been called")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def u(intent="symptoms", symptoms="", enough=False, danger=(), lang="hinglish", **kw):
    args = {"intent": intent, "symptoms_en": symptoms, "enough_to_choose_doctor": enough,
            "danger_signs": list(danger), "language": lang, **kw}
    return AIMessage(content="", tool_calls=[{"name": "record_understanding", "args": args, "id": "u"}])


def say(message, options=None):
    return AIMessage(content="", tool_calls=[{"name": "reply", "args": {"message": message, "options": options or []},
                                              "id": "r"}])


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
        client.app.state.vita_runtime = VitaRuntime(holder["llm"], client.app.state.redis,
                                                    route_service=lambda: client.app.state.route)
        return holder["llm"]
    yield use
    client.app.state.vita_runtime = VitaRuntime(None, client.app.state.redis,
                                                route_service=lambda: client.app.state.route)


def vita(client, text, sid=None, **kw):
    kw.setdefault("city", "Kolkata")
    r = client.post("/v2/agent/chat", json={"text": text, "session_id": sid, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def engine(client) -> VitaEngine:
    from hms_router.application.doctor_chat import DoctorChat
    a = client.app.state
    return VitaEngine(DoctorChat(a.vita_agent, a.doctors, a.redis, booking_client=None), a.vita_runtime)


def primary(out):
    return next(o["specialist"] for o in out["card"]["outputs"] if o["role"] == "PRIMARY")


# ------------------------------------------------------------------------------------------- rules first
def test_without_llm_vita_is_the_rules_engine_and_still_traced(client):
    out = vita(client, "hello")
    assert out["engine"] == "rules"
    assert client.app.state.vita_runtime.traces.turns(out["session_id"])


@pytest.mark.parametrize("text", ["I don't want to live anymore", "seene mein dard aur pasina aa raha hai",
                                  "मैं मरना चाहता हूँ", "haath jal gaya garam tel se"])
def test_danger_and_warning_signs_never_wait_for_the_llm(client, llm, text):
    model = llm()
    out = vita(client, text)
    assert out["engine"] == "rules" and model.seen == []
    assert out["kind"] == "emergency" or out["clarify"]["kind"] == "warning_signs"


# ------------------------------------------------------------------------------------------- the conversation
def test_greeting_is_written_by_the_llm(client, llm):
    llm(u("greeting"), say("Namaste! Main Vita hoon. Aapko kya takleef hai?", ["Bukhar", "Sar dard"]))
    out = vita(client, "hii")
    assert out["engine"] == "agent" and out["kind"] == "follow_up" and out["message"].startswith("Namaste")
    assert [o["id"] for o in out["clarify"]["options"]] == ["say:Bukhar", "say:Sar dard"]


def test_two_turn_triage_then_advice_with_doctors(client, llm):
    model = llm(u("symptoms", "fever"), say("Bukhar kab se hai?"),
                u("symptoms", "fever for 3 days", enough=True),
                say("3 din ka bukhar hai to General Physician ko dikhana sahi rahega. Neeche doctor hain."))
    first = vita(client, "bukhar hai")
    assert first["kind"] == "follow_up" and first["engine"] == "agent"
    out = vita(client, "3 din se", first["session_id"])
    assert out["kind"] == "recommendation" and primary(out) == "General Physician" and out["doctors"]["items"]
    # the second understanding call saw the first turn of the conversation
    assert "Bukhar kab se hai?" in model.seen[2][1].content
    st = engine(client).state(out["session_id"])
    assert len(st["messages"]) == 4 and st["triage"]["routed"] and st["triage"]["symptoms_en"] == "fever for 3 days"


def test_never_more_than_two_questions(client, llm):
    llm(u("symptoms", "headache"), say("Kab se?"), u("symptoms", "headache"), say("Kitna tez?"),
        u("symptoms", "headache, mild"), say("Sar dard ke liye ye doctor dekh sakte hain."))
    sid = vita(client, "sar dard")["session_id"]
    vita(client, "pata nahi", sid)
    out = vita(client, "halka", sid)
    assert out["kind"] == "recommendation"                      # routed after two questions, whatever the model said


@pytest.mark.parametrize("text,reading,kw,want", [
    ("मुझे तीन दिन से बुखार है", "fever for 3 days", {"lang": "hindi"}, "General Physician"),
    ("mere 2 saal ke bete ko tez bukhar hai", "high fever in a 2 year old", {"who": "child", "age_group": "child"},
     "Paediatrician"),
    ("daant mein bahut dard hai", "severe toothache", {}, "Dentist"),
])
def test_the_llm_reading_chooses_the_specialist_through_route_symptom(client, llm, text, reading, kw, want):
    llm(u("symptoms", reading, enough=True, **kw), say("Ye doctor sahi rahenge."))
    out = vita(client, text)
    assert out["kind"] == "recommendation" and primary(out) == want


def test_llm_can_add_urgency(client, llm):
    model = llm(u("symptoms", "repeated fainting", danger=["fainting"]))
    out = vita(client, "papa baar baar gir ja rahe hain")
    assert out["kind"] == "emergency" and out["engine"] == "agent" and "108" in out["message"] and len(model.seen) == 1


def test_red_flag_in_the_llm_reading_is_an_emergency(client, llm):
    llm(u("symptoms", "severe chest pain with sweating", enough=True))
    out = vita(client, "ajeeb sa lag raha hai, bahut ghabrahat")
    assert out["kind"] == "emergency" and "108" in out["message"]


def test_find_doctors_with_filters(client, llm):
    llm(u("find_doctor", doctor={"specialty": "Dermatologist", "gender": "female"}),
        say("Ye mahila skin doctor hain, jaise Dr. Anita Sen."))
    out = vita(client, "koi lady skin specialist")
    assert out["kind"] == "doctor_list" and out["engine"] == "agent"
    assert {d["name"] for d in out["doctors"]["items"]} <= {"Dr. Anita Sen", "Dr. Priya Bose"}


def test_near_me_without_location_asks_where_then_rules_take_the_answer(client, llm):
    llm(u("find_doctor", doctor={"specialty": "Dermatologist", "near_me": True}))
    first = vita(client, "paas mein skin doctor", city=None)
    assert first["clarify"]["kind"] == "location"
    model = llm()
    out = vita(client, "Kolkata", first["session_id"])
    assert out["engine"] == "rules" and out["kind"] == "doctor_list" and model.seen == []


def test_booking_and_handoff_go_to_the_tested_flow(client, llm):
    llm(u("human"))
    out = vita(client, "mujhe kisi real person se baat karni hai please")
    assert out["kind"] == "handoff" and out["engine"] == "agent+flow"


def test_a_new_complaint_after_advice_starts_triage_again(client, llm):
    llm(u("symptoms", "toothache", enough=True), say("Dentist sahi rahenge."),
        u("symptoms", "itchy skin rash"), say("Khujli kab se hai?"))
    sid = vita(client, "daant dard")["session_id"]
    out = vita(client, "aur skin pe khujli bhi hai", sid)
    st = engine(client).state(sid)
    assert out["kind"] == "follow_up" and st["triage"] == {"symptoms_en": "itchy skin rash", "asked": 1}


# ------------------------------------------------------------------------------------------- the checker
@pytest.mark.parametrize("bad", ["Paracetamol 500 mg le lijiye, aaram milega.", "Aapko dengue ho sakta hai.",
                                 "Dr. Ramesh Gupta best hain, unhe dikhayein.", "Call 9876543210 for help.",
                                 "Aapka appointment book ho gaya hai."])
def test_unsafe_llm_reply_never_reaches_the_patient(client, llm, bad):
    llm(u("question", question_topic="anything"), say(bad))
    out = vita(client, "kuch batao")
    assert bad not in out["message"] and out["engine"] == "rules-fallback"


def test_blocked_reply_after_doctors_keeps_vitas_own_words(client, llm):
    llm(u("find_doctor", doctor={"specialty": "Dermatologist"}), say("Calpol lo aur Dr. Fake se milo"))
    out = vita(client, "skin doctor dikhao")
    assert out["engine"] == "agent+template" and out["doctors"]["items"] and "Calpol" not in out["message"]


@pytest.mark.parametrize("ok", ["Bukhar ke liye General Physician ko dikhana sahi rahega.",
                                "Main dawai ki salah nahi de sakti, doctor batayenge. Kya main doctor dhoondh doon?",
                                "Aapko kya takleef hai?", "Ye doctors Kolkata mein hain."])
def test_checker_allows_ordinary_replies(ok):
    assert checker.problems(ok, set()) == []


# ------------------------------------------------------------------------------------------- failures, control
def test_llm_failure_falls_back_to_rules_and_opens_the_breaker(client, llm):
    llm(RuntimeError("groq down"))
    out = vita(client, "hello")
    assert out["engine"] == "rules-fallback" and out["message"]
    rt = client.app.state.vita_runtime
    rt.breaker = Breaker(limit=1, cooldown=60)
    rt.breaker.fail()
    assert vita(client, "hello")["engine"] == "rules"


def test_control_switch_per_channel(client, llm):
    model = llm()
    r = client.put("/v2/agent/admin/control", json={"mode": "agent", "channels": {"whatsapp": "rules"}})
    assert r.status_code == 200
    assert vita(client, "hello", channel="whatsapp")["engine"] == "rules" and model.seen == []
    assert client.get("/v2/agent/admin/control").json()["channels"] == {"whatsapp": "rules"}
    client.put("/v2/agent/admin/control", json={"mode": "agent"})


def test_state_survives_a_restart(client, llm):
    llm(u("symptoms", "cough"), say("Khansi kab se hai?"))
    sid = vita(client, "khansi hai")["session_id"]
    model = llm(u("symptoms", "cough for 2 weeks", enough=True), say("Phephde ke doctor sahi rahenge."))
    out = vita(client, "2 hafte se", sid)                       # a new runtime: state read back from Redis
    assert out["kind"] == "recommendation" and "khansi hai" in model.seen[0][1].content


def test_whatsapp_contract(client, llm):
    llm(u("greeting"), say("Kya takleef hai?", ["Bukhar", "Khansi"]))
    msg = vita(client, "hi", channel="whatsapp")["whatsapp"][0]
    assert msg["type"] == "interactive" and msg["interactive"]["type"] == "button"


# ------------------------------------------------------------------------------------------- learning
def test_patient_memory_with_and_without_consent(client, llm):
    rt = None
    llm(u("symptoms", "toothache", enough=True, who="self"), say("Dentist sahi rahenge."))
    e = engine(client)
    rt = e.runtime
    out = e.handle("daant dard", None, city="Purnia", verified_mobile="9876543210", remember=True)
    mem = rt.memory.load("9876543210")
    assert mem["city"] == "Purnia" and mem["specialties"] == ["Dentist"] and mem["consent"] and out["kind"]
    llm(u("greeting"), say("Namaste!"))
    e = engine(client)
    e.handle("hello", None, verified_mobile="9876543210", remember=False)
    mem = e.runtime.memory.load("9876543210")
    assert "specialties" not in mem and mem["city"] == "Purnia"           # consent withdrawn: health facts dropped
    assert e.runtime.memory.forget("9876543210") and e.runtime.memory.load("9876543210") == {}


def test_returning_patients_city_is_used(client, llm):
    rt = VitaRuntime(None, client.app.state.redis)
    rt.memory.update("9123456780", consent=False, city="Kolkata", language="hinglish")
    model = llm(u("find_doctor", doctor={"specialty": "Dermatologist"}), say("Ye skin doctor hain."))
    out = engine(client).handle("skin doctor", None, verified_mobile="9123456780")
    assert out["kind"] == "doctor_list" and out["doctors"]["filters"].get("city") == "Kolkata"
    assert "Kolkata" in model.seen[1][1].content                  # the reply step saw the memory


def test_feedback_review_correct_and_recall(client, llm):
    llm(u("symptoms", "toothache", enough=True), say("Dentist ko dikhayein."))
    sid = vita(client, "daant mein dard")["session_id"]
    assert client.post("/v2/agent/feedback", json={"session_id": sid, "rating": "up"}).json() == {"ok": True}
    listed = client.get("/v2/agent/admin/conversations", params={"outcome": "up"}).json()["conversations"]
    assert any(c["sessionId"] == sid for c in listed)
    detail = client.get(f"/v2/agent/admin/conversations/{sid}").json()
    assert detail["turns"][0]["decision"] == "advise" and detail["turns"][0]["tools"] == ["assess_symptoms"]
    ex = client.post("/v2/agent/admin/examples", json={"session_id": sid, "turn": 0,
                                                       "reply": "Daant ke dard ke liye Dentist sahi rahenge.",
                                                       "specialist": "Dentist", "note": "warmer"}).json()
    assert ex["source"] == "corrected"
    rt = client.app.state.vita_runtime
    rt.examples.promote({"test": True})                         # (the gate is tested below)
    model = llm(u("symptoms", "tooth pain", enough=True), say("Dentist ko dikhayein."))
    vita(client, "daant mein bahut dard hai")
    assert "Daant ke dard ke liye Dentist sahi rahenge." in model.seen[1][1].content     # recalled into the prompt
    assert client.delete(f"/v2/agent/admin/examples/{ex['id']}").json() == {"deleted": True}


def test_promotion_gate_runs_the_suite(client):
    client.app.state.vita_runtime = VitaRuntime(None, client.app.state.redis,
                                                route_service=lambda: client.app.state.route)
    client.post("/v2/agent/admin/examples", json={"patient": "daant dard", "reply": "Dentist ko dikhayein."})
    rep = client.post("/v2/agent/admin/examples/promote").json()
    assert rep["after"]["total"] >= 15 and rep["after"]["safetyPassed"] and rep["promoted"] == 1
    assert client.get("/v2/agent/admin/examples").json()["draft"] == []
