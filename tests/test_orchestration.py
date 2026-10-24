"""LangGraph triage graph (optional dependency - skipped when langgraph is not installed)."""
import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from hms_router.orchestration.graph import build_triage_graph  # noqa: E402
from hms_router.orchestration.tools import as_langchain_tools, route_symptom_payload  # noqa: E402


@pytest.fixture(scope="module")
def graph(service):
    return build_triage_graph(service, InMemorySaver())


def say(graph, thread, text):
    return graph.invoke({"messages": [text]}, {"configurable": {"thread_id": thread}})


def test_emergency_short_circuits(graph):
    s = say(graph, "e1", "seene mein dard ho raha hai")
    assert s["done"] and "108" in s["reply"] and "result" not in s       # triage never ran


def test_asks_age_then_answers_without_repeating(graph):
    first = say(graph, "a1", "pet mein dard aur ulti")
    assert not first["done"] and "bachche" in first["reply"]
    second = say(graph, "a1", "bade ke liye")
    assert second["age_group"] == "adult" and second["done"] and "Gastroenterology" in second["reply"]


def test_never_asks_the_same_question_twice(graph):
    say(graph, "r1", "ghutne mein dard")
    s = say(graph, "r1", "pata nahi")
    assert len(s["asked"]) == len(set(s["asked"]))


def test_unknown_text_suggests_gp(graph):
    assert "General Physician" in say(graph, "g1", "asdf qwer")["reply"]


def test_tools(service):
    payload = route_symptom_payload(service, "dant mein bahut dard hai")
    assert payload["decision"] == "answer" and payload["specialties"][0]["specialtyId"] == "dentistry"
    tool = as_langchain_tools(service)[0]
    assert tool.name == "route_symptom" and tool.invoke({"text": "dant dard"})["specialties"]
