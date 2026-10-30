"""Vita as a LangGraph graph. One graph run = one patient message.

    START -> guard_in --(danger, a step in progress, a tap)--> rules ----------------------------> finish -> END
                      \\-> agent <-> tools (assess_symptoms, find_doctors, vita_flow, raise_emergency, reply)
                                       \\-> finish (checker; a failed check -> the tool's own text or rules)

guard_in is code, not a model: self-harm and the shared emergency detector run on every message first, and anything
Vita's reviewed flows are in the middle of (booking with OTP, an emergency, "where are you?", "which doctor?",
directions / check-in) stays with them. The LLM can only add urgency (raise_emergency, or assess_symptoms finding
a red flag in its own reading); it never removes it.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, TypedDict

from ..application.journey import wants as journey_wants
from ..application.visit_help import wants as visit_wants
from ..safety.self_harm import mentions_self_harm
from . import checker
from .prompts import CONTEXT, SYSTEM
from .tools import TOOL_NAMES

log = logging.getLogger("hms_router.vita_graph")

MAX_LLM_CALLS = 4
AFTER_EMERGENCY_S = 2 * 3600
STICKY = ("booking", "emergency", "pending", "warn", "which", "which_hospital", "triage_waiting")


class VitaState(TypedDict, total=False):
    route: str                   # rules | agent
    why: str                     # why the rules engine answered (for logs and tests)
    messages: list               # LangChain messages of this turn
    calls: int
    out: dict


def _turn(config) -> Any:
    return config["configurable"]["turn"]


def sticky_reason(turn) -> str | None:
    """Why this message must go to Vita's rules engine, or None for the LLM agent."""
    flow, text, kw = turn.flow, turn.text, turn.kw
    if kw.get("select_doctor_id"):
        return "tap"
    if (kw.get("page") or 1) > 1 or kw.get("new_search"):
        return "tap"
    if mentions_self_harm(text):
        return "self_harm"
    if text.strip().lower() in ("location", "my location") or (turn.ctx.get("located") and len(text.split()) <= 2):
        return "location"
    try:
        if flow.agent.pipeline.detector.detect(text).language == "bangla":
            return "bangla"
    except Exception:
        pass
    st = flow.load(turn.session_id) if turn.session_id else {}
    for k in STICKY:
        if st.get(k):
            return "in_progress:" + k
    ae = st.get("after_emergency") or {}
    if ae and time.time() - float(ae.get("at") or 0) < AFTER_EMERGENCY_S:
        return "after_emergency"
    if flow.visit.enabled and visit_wants(text):
        return "visit"
    if flow.journey.enabled and kw.get("channel") != "whatsapp" and journey_wants(text):
        return "journey"
    if flow._emergency(text):                      # the shared detector (rules + model); fails safe to True
        return "emergency"
    return None


def build_vita_graph():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langgraph.graph import END, START, StateGraph

    from ..adapters.llm.factory import redact

    def guard_in(state: VitaState, config) -> dict:
        turn = _turn(config)
        why = sticky_reason(turn)
        if why:
            return {"route": "rules", "why": why}
        ctx = turn.ctx
        note = CONTEXT.format(channel=turn.kw.get("channel") or "web", city=ctx.get("city") or "unknown",
                              located=bool(ctx.get("located")), hospital=bool(ctx.get("hospital_id")),
                              ui_language=turn.kw.get("ui_language") or "unknown")
        msgs = [SystemMessage(content=SYSTEM + "\n" + note)]
        for h in turn.history:
            msgs.append(HumanMessage(content=redact(h["t"])) if h["r"] == "user" else AIMessage(content=h["t"]))
        msgs.append(HumanMessage(content=redact(turn.text)))
        return {"route": "agent", "messages": msgs, "calls": 0}

    def rules(state: VitaState, config) -> dict:
        turn = _turn(config)
        out = turn.final or turn.flow.handle(turn.text, turn.session_id, **turn.kw)
        out["engine"] = "rules"
        return {"out": out}

    def agent(state: VitaState, config) -> dict:
        turn = _turn(config)
        reply = turn.llm.invoke(state["messages"])
        return {"messages": state["messages"] + [reply], "calls": state.get("calls", 0) + 1}

    def tools(state: VitaState, config) -> dict:
        turn = _turn(config)
        last = state["messages"][-1]
        new = []
        for call in getattr(last, "tool_calls", None) or []:
            name, args = call.get("name"), call.get("args") or {}
            if turn.final is not None or turn.draft is not None:     # the turn is answered: ignore the rest
                result = json.dumps({"skipped": True})
            elif name not in TOOL_NAMES:
                result = json.dumps({"error": f"unknown tool {name}"})
            else:
                result = turn.run(name, args)
            new.append(ToolMessage(content=result, tool_call_id=call.get("id") or name, name=name))
        return {"messages": state["messages"] + new}

    def finish(state: VitaState, config) -> dict:
        turn = _turn(config)
        if state.get("out"):
            return {}
        if turn.final is not None:                                   # a tool answered with Vita's own words
            out = turn.final
            out["engine"] = "agent"
            return {"out": out}
        if turn.draft is None:                                       # the model wrote text instead of calling reply
            last = state["messages"][-1] if state.get("messages") else None
            content = getattr(last, "content", "") if last is not None else ""
            if isinstance(content, str) and content.strip() and not getattr(last, "tool_calls", None):
                turn.draft = {"message": content.strip(), "options": []}
        if turn.draft is not None:
            found = checker.problems(turn.draft["message"], turn.known_names())
            turn.checks = found
            if not found:
                out = turn.compose()
                out["engine"] = "agent"
                return {"out": out}
            log.warning("Vita reply blocked by the checker: %s", found)
        if turn.ui is not None and turn.ui.get("message"):           # the tool's own reviewed wording
            out = dict(turn.ui)
            out["engine"] = "agent+template"
            return {"out": out}
        out = turn.flow.handle(turn.text, turn.session_id, **turn.kw)  # nothing usable: the rules engine answers
        out["engine"] = "rules-fallback"
        return {"out": out}

    def after_guard(state: VitaState) -> str:
        return state["route"]

    def after_agent(state: VitaState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "finish"

    def after_tools(state: VitaState, config) -> str:
        turn = _turn(config)
        if turn.final is not None or turn.draft is not None:
            return "finish"
        return "agent" if state.get("calls", 0) < MAX_LLM_CALLS else "finish"

    g = StateGraph(VitaState)
    for name, fn in (("guard_in", guard_in), ("rules", rules), ("agent", agent), ("tools", tools), ("finish", finish)):
        g.add_node(name, fn)
    g.add_edge(START, "guard_in")
    g.add_conditional_edges("guard_in", after_guard, {"rules": "rules", "agent": "agent"})
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "finish": "finish"})
    g.add_conditional_edges("tools", after_tools, {"agent": "agent", "finish": "finish"})
    g.add_edge("rules", "finish")
    g.add_edge("finish", END)
    return g.compile()
