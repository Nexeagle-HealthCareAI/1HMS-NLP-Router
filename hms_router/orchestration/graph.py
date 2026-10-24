"""A first LangGraph graph: symptom triage with the Safety Guard as a fixed first node.

    START -> safety -> (emergency) emergency_reply -> END
                    -> triage -> answer_reply | clarify_reply | gp_reply -> END

Deterministic today: no LLM node, replies from templates. Multi-turn via a checkpointer keyed by
thread_id - a clarifying answer is appended to what the patient said before and triage re-runs.
Not served over HTTP yet (design doc phase 5); build it in a worker or a test:

    graph = build_triage_graph(build_route_symptom_service())
    graph.invoke({"messages": ["pet mein dard aur ulti"]}, {"configurable": {"thread_id": "p-123"}})

Requires the optional dependency `langgraph` (pip install "langgraph>=1.0" "langchain-core>=1.0").
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..application.route_symptom import RouteSymptomService
from ..domain.models import Query
from ..severity import EMERGENCY_NUMBER, GENERAL_EMERGENCY_NUMBER
from ..textnorm import preprocess
from .tools import route_symptom_payload

MAX_CLARIFY_TURNS = 2

_T = {  # (english, hinglish) - deterministic wording; needs clinician review before patient use
    "emergency": ("This may be an emergency: {reasons}. Call " + EMERGENCY_NUMBER + " (ambulance) or " + GENERAL_EMERGENCY_NUMBER
                  + ", or go to the nearest emergency room now.",
                  "Yeh emergency ho sakti hai: {reasons}. Abhi " + EMERGENCY_NUMBER + " (ambulance) ya " + GENERAL_EMERGENCY_NUMBER
                  + " par call karein, ya nazdeeki emergency jaayein."),
    "answer": ("Please see a {name} doctor. {reason}", "Aap {name} doctor ko dikhayein. {reason}"),
    "gp": ("I could not match this to a specialty. A General Physician can assess it first.",
           "Main ise kisi specialty se match nahi kar paya. Pehle General Physician ko dikhayein."),
}


_CHILD = {"bachcha", "bachche", "bacche", "bachchi", "beta", "beti", "child", "kid", "baby", "son", "daughter"}
_ADULT = {"bada", "bade", "badi", "adult", "khud", "myself", "main", "mere", "apne", "grown"}


class TriageState(TypedDict, total=False):
    messages: Annotated[list[str], operator.add]     # everything the patient said, oldest first
    language: str
    age_group: str | None
    asked: list[str]                                 # questions already asked - never ask one twice
    emergency: dict
    result: dict
    clarify_turns: int
    reply: str
    done: bool


def _age_from(text: str) -> str | None:
    words = set(preprocess(text).split())
    child, adult = bool(words & _CHILD), bool(words & _ADULT)
    return "child" if child and not adult else "adult" if adult and not child else None


def _say(key: str, state: TriageState, **kw) -> str:
    en, hi = _T[key]
    return (hi if state.get("language") in ("hinglish", "hindi") else en).format(**kw)


def build_triage_graph(service: RouteSymptomService, checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    def joined(state: TriageState) -> str:
        return " . ".join(state.get("messages", []))

    def safety(state: TriageState) -> dict:
        q = Query(raw=joined(state), normalized=preprocess(joined(state)))
        e = service.detector.detect(q)
        return {"emergency": {"flag": e.flag, "reasons": e.reasons, "recommendedSpecialtyIds": e.recommendedSpecialtyIds},
                "language": service._language(q.normalized)}

    def triage(state: TriageState) -> dict:
        age = state.get("age_group")
        last = state.get("result") or {}
        if age is None and (last.get("clarify") or {}).get("kind") == "age":    # the reply answers our age question
            age = _age_from(state["messages"][-1])
        if age is None:
            age = _age_from(joined(state))
        return {"result": route_symptom_payload(service, joined(state), age_group=age), "age_group": age}

    def emergency_reply(state: TriageState) -> dict:
        return {"reply": _say("emergency", state, reasons=", ".join(state["emergency"]["reasons"])), "done": True}

    def answer_reply(state: TriageState) -> dict:
        top = state["result"]["specialties"][0]
        return {"reply": _say("answer", state, name=top["name"], reason=state["result"].get("reason") or "").strip(),
                "done": True}

    def clarify_reply(state: TriageState) -> dict:
        c = state["result"]["clarify"]
        opts = " / ".join(c["options"])
        return {"reply": f"{c['question']} ({opts})", "clarify_turns": state.get("clarify_turns", 0) + 1,
                "asked": [*state.get("asked", []), c["question"]], "done": False}

    def gp_reply(state: TriageState) -> dict:
        return {"reply": _say("gp", state), "done": True}

    def after_safety(state: TriageState) -> str:
        return "emergency_reply" if state["emergency"]["flag"] else "triage"

    def after_triage(state: TriageState) -> str:
        r = state["result"]
        if r["decision"] == "answer":
            return "answer_reply"
        c = r.get("clarify")
        if (r["decision"] == "clarify" and c and state.get("clarify_turns", 0) < MAX_CLARIFY_TURNS
                and c["question"] not in state.get("asked", [])):
            return "clarify_reply"
        return "answer_reply" if r["specialties"] else "gp_reply"     # stop asking: give the best we have

    g = StateGraph(TriageState)
    for name, fn in (("safety", safety), ("triage", triage), ("emergency_reply", emergency_reply),
                     ("answer_reply", answer_reply), ("clarify_reply", clarify_reply), ("gp_reply", gp_reply)):
        g.add_node(name, fn)
    g.add_edge(START, "safety")
    g.add_conditional_edges("safety", after_safety, ["emergency_reply", "triage"])
    g.add_conditional_edges("triage", after_triage, ["answer_reply", "clarify_reply", "gp_reply"])
    for leaf in ("emergency_reply", "answer_reply", "clarify_reply", "gp_reply"):
        g.add_edge(leaf, END)
    return g.compile(checkpointer=checkpointer)
