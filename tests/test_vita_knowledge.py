"""Vita's knowledge base (vita_graph/knowledge.py): the facts Vita may state about Doctor Dekho and each hospital,
found by meaning and words, scoped to the hospital in the conversation, used by the agent's answers and the planner's
look_up tool, and edited by staff through the admin API. Nothing found means "I'll check", never a guess."""
import json

import fakeredis
import pytest

pytest.importorskip("langgraph")
from hms_router.vita_graph.knowledge import KnowledgeBase, load_seed  # noqa: E402
from tests.test_vita_graph import client, llm, say, u, vita  # noqa: E402,F401
from tests.test_vita_planner import calls, reply, tool_results  # noqa: E402


@pytest.fixture
def kb():
    k = KnowledgeBase(fakeredis.FakeRedis(decode_responses=True))
    k.add(title="Sadar Hospital Purnia OPD timings", text="OPD runs Monday to Saturday, 9 am to 2 pm. Closed on Sunday.",
          scope="Sadar Hospital Purnia", tags="timing, sunday, khula, opd, kitne baje")
    return k


def test_the_seed_holds_only_service_facts():
    seed = load_seed()
    assert len(seed) >= 10 and all(e["scope"] == "all" and e["text"] for e in seed)
    text = " ".join(e["text"].lower() for e in seed)
    assert not any(w in text for w in ("tablet", "dose", "paracetamol", "mg "))


@pytest.mark.parametrize("q,title", [
    ("fees kitni hai", "Doctor's fee"), ("kya booking free hai?", "Is booking free / do I pay online"),
    ("ayushman card chalta hai", "Insurance and Ayushman card"), ("video call pe dikha sakte hai",
                                                                   "Online or video consultation"),
    ("appointment cancel kaise kare", "Cancel or change an appointment"), ("OTP kyun chahiye",
                                                                           "Why my number and OTP are needed"),
    ("doctor dekho kya hai", "What Doctor Dekho is"), ("dawai ghar pe milegi?", "Lab tests and medicine delivery")])
def test_questions_find_their_fact(kb, q, title):
    assert kb.search(q)[0]["title"] == title


def test_a_hospitals_own_fact_comes_first_and_stays_with_that_hospital(kb):
    assert kb.search("Sunday ko khula hai?", hospital="Sadar Hospital Purnia")[0]["title"].startswith("Sadar")
    assert all(not f["title"].startswith("Sadar") for f in kb.search("Sunday ko khula hai?"))
    assert all(not f["title"].startswith("Sadar") for f in kb.search("Sunday ko khula hai?", hospital="Apollo Kolkata"))


def test_nothing_known_is_nothing_returned(kb):
    assert kb.search("parking hai kya") == [] and kb.search("") == []


def test_staff_can_remove_what_they_added(kb):
    sid = next(e["id"] for e in kb.entries() if e["source"] == "staff")
    assert kb.remove(sid) and not any(e["source"] == "staff" for e in kb.entries())


# ------------------------------------------------------------------------------------------- in the conversation
def test_agent_answers_from_knowledge(client, llm):
    model = llm(u("question"), say("Booking free hai; fees aap hospital mein dete hain."))
    vita(client, "kya booking free hai?")
    told = json.loads(model.seen[1][1].content)
    assert told["decision"] == "answer"
    assert told["facts"]["knowledge"][0]["title"] == "Is booking free / do I pay online"


def test_agent_is_told_when_nothing_is_known(client, llm):
    model = llm(u("question"), say("Main team se pata karke batati hoon."))
    vita(client, "parking hai kya")
    told = json.loads(model.seen[1][1].content)
    assert told["facts"]["knowledge"].startswith("nothing in Vita's knowledge")


def test_planner_looks_up(client):
    from hms_router.vita_graph.engine import VitaRuntime
    from tests.test_vita_graph import ScriptedLLM
    a = client.app.state
    model = ScriptedLLM(calls(("look_up", {"question": "ayushman card chalta hai?"})),
                        reply("Ayushman ki jaankari jaldi aa rahi hai; abhi hospital se pooch lijiye."))
    a.vita_runtime = VitaRuntime(model, a.redis, route_service=lambda: a.route)
    a.vita_runtime.set_control({"mode": "planner"})
    try:
        vita(client, "ayushman card chalta hai?")
        facts = json.loads(tool_results(model, 1)[0])["facts"]
        assert facts[0]["title"] == "Insurance and Ayushman card"
    finally:
        a.vita_runtime.set_control({"mode": "agent"})
        a.vita_runtime = VitaRuntime(None, a.redis, route_service=lambda: a.route)


def test_admin_api_adds_searches_and_removes(client):
    e = client.post("/v2/agent/admin/knowledge", json={"title": "City Clinic parking", "text": "Free parking at the "
                    "back gate.", "scope": "City Clinic", "tags": "parking, gaadi"}).json()
    found = client.get("/v2/agent/admin/knowledge", params={"q": "parking hai kya", "hospital": "City Clinic"}).json()
    assert found["search"][0]["title"] == "City Clinic parking"
    assert client.delete(f"/v2/agent/admin/knowledge/{e['id']}").json() == {"deleted": True}
