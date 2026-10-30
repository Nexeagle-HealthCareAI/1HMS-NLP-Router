"""Vita as one LangGraph graph: the whole conversation, one run per patient message, state saved per session
(thread_id = Vita session id) by the Redis checkpointer.

    START -> guard_in ──(danger, a step in progress, a tap, a warning-sign question)──> rules ──────────┐
               └─> recall -> understand (LLM) -> plan ─┬─> emergency ─────────────────────────────────┤
                                                      ├─> flow  (booking, visit, choosing: tested code) ─┤
                                                      ├─> search ─┬─────────────────────────────────────┤
                                                      ├─> route  ─┤  (answered by the tool)               │
                                                      ├─> ask     │                                     │
                                                      └─> answer  └─> respond (LLM) + checker ──────────┴─> remember -> END

State that lives across turns: messages (the conversation), patient (who, age group, pregnant, language), triage
(complaint so far in English, questions asked), stage. Per turn: understanding, decision, facts, out.

Rules that stay in code (never a model's choice): guard_in on every message (self-harm, the shared emergency detector,
warning-sign questions, booking/OTP and other steps in progress); the LLM can only ADD urgency (danger_signs, or
assess_symptoms finding a red flag in its reading); the specialist comes from the route-symptom engine; every LLM reply
passes checker.py or is replaced by Vita's own wording.
"""
from __future__ import annotations

import json
import logging
import operator
import time
from typing import Annotated, Any, TypedDict

from ..application.everyday import warn_kind
from ..application.journey import wants as journey_wants
from ..application.visit_help import wants as visit_wants
from ..safety.self_harm import mentions_self_harm
from ..textnorm import intent_text
from . import checker
from .learning import redact
from .prompts import RESPOND, UNDERSTAND, respond_input, understand_input

log = logging.getLogger("hms_router.vita_graph")

AFTER_EMERGENCY_S = 2 * 3600
MAX_QUESTIONS = 2
STICKY = ("booking", "emergency", "pending", "warn", "which", "which_hospital")


class VitaState(TypedDict, total=False):
    messages: Annotated[list, operator.add]       # [{"role": "patient"|"vita", "text", "kind"?}]
    patient: dict                                 # who, age_group, pregnant, language
    triage: dict                                  # symptoms_en, asked, routed
    stage: str
    route: str                                    # per turn from here on
    why: str
    understanding: dict
    decision: str
    facts: dict
    memory: dict
    examples: list
    out: dict


def _turn(config) -> Any:
    return config["configurable"]["turn"]


def sticky_reason(turn) -> str | None:
    """Why this message must go to Vita's tested rules, or None for the LLM agent."""
    flow, text, kw = turn.flow, turn.text, turn.kw
    if kw.get("select_doctor_id") or (kw.get("page") or 1) > 1 or kw.get("new_search"):
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
    if warn_kind(intent_text(text)):
        return "warning_signs"
    if flow._emergency(text):                       # the shared detector (rules + model); fails safe to True
        return "emergency"
    return None


def _tool_args(msg, name: str) -> dict | None:
    for call in getattr(msg, "tool_calls", None) or []:
        if call.get("name") == name:
            return call.get("args") or {}
    return None


def build_vita_graph(checkpointer=None):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph

    # ------------------------------------------------------------------------------------------------ nodes
    def guard_in(state: VitaState, config) -> dict:
        why = sticky_reason(_turn(config))
        return {"route": "rules", "why": why} if why else {"route": "agent", "why": ""}

    def rules(state: VitaState, config) -> dict:
        turn = _turn(config)
        out = turn.flow.handle(turn.text, turn.session_id, **turn.kw)
        out["engine"] = "rules" if state.get("route") == "rules" else "rules-fallback"
        return {"out": out, "decision": "rules"}

    def recall(state: VitaState, config) -> dict:
        turn = _turn(config)
        memory = turn.rt.memory.load(turn.identity) if turn.identity else {}
        if memory.get("city") and not turn.ctx.get("city"):
            turn.ctx["city"] = memory["city"]                       # a returning patient's city, for search
        return {"memory": memory, "examples": turn.rt.examples.similar(turn.text, k=3)}

    def understand(state: VitaState, config) -> dict:
        turn = _turn(config)
        prompt = understand_input(state.get("messages") or [], state.get("patient") or {}, state.get("triage") or {},
                                  turn.text)
        try:
            msg = turn.rt.llm_understand.invoke([SystemMessage(content=UNDERSTAND),
                                                 HumanMessage(content=redact(prompt))])
            u = _tool_args(msg, "record_understanding")
        except Exception:
            log.exception("understand failed")
            turn.rt.breaker.fail()
            u = None
        if u is None:
            return {"understanding": {"failed": True}}
        patient, triage = dict(state.get("patient") or {}), dict(state.get("triage") or {})
        for f in ("who", "age_group"):
            if u.get(f) and u[f] != "unknown":
                patient[f] = u[f]
        if u.get("pregnant"):
            patient["pregnant"] = True
        if u.get("language"):
            patient["language"] = u["language"]
        if (u.get("symptoms_en") or "").strip():
            if triage.get("routed") and u.get("intent") == "symptoms" and u["symptoms_en"] != triage.get("symptoms_en"):
                triage = {}                                          # a new complaint after advice: start again
            triage["symptoms_en"] = u["symptoms_en"].strip()
        return {"understanding": u, "patient": patient, "triage": triage}

    def plan(state: VitaState, config) -> dict:
        u, triage = state.get("understanding") or {}, state.get("triage") or {}
        if u.get("failed"):
            return {"decision": "rules"}
        intent, doctor = u.get("intent"), {k: v for k, v in (u.get("doctor") or {}).items() if v}
        if u.get("danger_signs"):
            return {"decision": "emergency"}
        if intent in ("choose_or_book", "visit_help", "human"):
            return {"decision": "flow"}
        if intent == "more_doctors" or (intent == "find_doctor" and doctor):
            return {"decision": "search"}
        if intent == "symptoms" or (intent in ("find_doctor", "other") and triage.get("symptoms_en")
                                    and not triage.get("routed")):
            enough = u.get("enough_to_choose_doctor") or triage.get("asked", 0) >= MAX_QUESTIONS
            return {"decision": "route" if enough and triage.get("symptoms_en") else "ask"}
        return {"decision": "answer"}

    def emergency(state: VitaState, config) -> dict:
        turn = _turn(config)
        signs = [s.lower() for s in (state.get("understanding") or {}).get("danger_signs") or []] or ["emergency signs"]
        out = turn.emergency(signs)
        out["engine"] = "agent"
        return {"out": out, "stage": "emergency"}

    def flow(state: VitaState, config) -> dict:
        turn = _turn(config)
        turn.t_vita_flow(turn.text)                                  # the patient's own words, to the tested flow
        out = turn.final
        out["engine"] = "agent+flow"
        return {"out": out, "stage": "booking" if str(out.get("kind", "")).startswith(("book", "doctor_sel")) else
                state.get("stage") or "search"}

    def search(state: VitaState, config) -> dict:
        turn = _turn(config)
        u = state.get("understanding") or {}
        d = {k: v for k, v in (u.get("doctor") or {}).items() if v not in (None, "", False)}
        facts = json.loads(turn.t_find_doctors(more=u.get("intent") == "more_doctors", **d))
        if turn.final is not None:
            out = turn.final
            out["engine"] = "agent+template"
            return {"out": out, "stage": "search"}
        return {"facts": facts, "decision": "doctors", "stage": "search"}

    def route(state: VitaState, config) -> dict:
        turn = _turn(config)
        patient, triage = state.get("patient") or {}, dict(state.get("triage") or {})
        facts = json.loads(turn.t_assess_symptoms(triage.get("symptoms_en", ""), patient.get("age_group"),
                                                  bool(patient.get("pregnant"))))
        triage.update(routed=True, asked=0)
        if turn.final is not None:                                   # an emergency in its reading, or the rules
            out = turn.final
            out["engine"] = "agent"
            return {"out": out, "triage": triage, "stage": "emergency" if out.get("kind") == "emergency" else "triage"}
        return {"facts": facts, "decision": "advise" if facts.get("confident", True) else "advise_unsure",
                "triage": triage, "stage": "advised"}

    def ask(state: VitaState, config) -> dict:
        triage = dict(state.get("triage") or {})
        triage["asked"] = triage.get("asked", 0) + 1
        return {"facts": {"complaint_so_far": triage.get("symptoms_en"), "patient": state.get("patient") or {}},
                "decision": "ask", "triage": triage, "stage": "triage"}

    def answer(state: VitaState, config) -> dict:
        u = state.get("understanding") or {}
        return {"facts": {"intent": u.get("intent"), "topic": u.get("question_topic"),
                          "vita_can": ["find the right kind of doctor from symptoms", "find doctors by name, specialty,"
                                       " hospital or place", "book an appointment", "help on the day of the visit"]},
                "decision": "answer"}

    def respond(state: VitaState, config) -> dict:
        turn = _turn(config)
        patient = state.get("patient") or {}
        prompt = respond_input(decision=state.get("decision") or "answer", facts=state.get("facts") or {},
                               conversation=state.get("messages") or [], patient=patient,
                               memory=state.get("memory") or {}, examples=state.get("examples") or [],
                               text=turn.text, language=patient.get("language"), channel=turn.kw.get("channel") or "web")
        try:
            msg = turn.rt.llm_respond.invoke([SystemMessage(content=RESPOND), HumanMessage(content=redact(prompt))])
            args = _tool_args(msg, "reply")
            if args is None and isinstance(getattr(msg, "content", None), str) and msg.content.strip():
                args = {"message": msg.content.strip()}
        except Exception:
            log.exception("respond failed")
            turn.rt.breaker.fail()
            args = None
        if args:
            turn.t_reply(args.get("message", ""), args.get("options"))
            turn.checks = checker.problems(turn.draft["message"], turn.known_names())
            if not turn.checks:
                out = turn.compose()
                if state.get("decision") == "ask" and not out.get("clarify"):
                    out["kind"], out["question"] = "follow_up", out["message"]
                out["engine"] = "agent"
                turn.rt.breaker.ok()
                return {"out": out}
            log.warning("Vita reply blocked by the checker: %s", turn.checks)
        if turn.ui is not None and turn.ui.get("message"):           # the tool's own reviewed wording
            out = dict(turn.ui)
            out["engine"] = "agent+template"
            return {"out": out}
        out = turn.fresh_rules()                                     # nothing usable: the rules engine answers
        out["engine"] = "rules-fallback"
        return {"out": out}

    def remember(state: VitaState, config) -> dict:
        turn = _turn(config)
        out = state.get("out") or {}
        booking_words = str((out.get("clarify") or {}).get("kind") or "").startswith("book_")
        said = "[booking details]" if booking_words else redact(turn.text)
        patient = dict(state.get("patient") or {})
        if out.get("language"):
            patient.setdefault("language", out["language"])
        if turn.identity:                                            # a returning patient's memory
            specialties, doctors = [], []
            if out.get("kind") == "recommendation":
                specialties = [o.get("specialist") for o in (out.get("card") or {}).get("outputs") or []
                               if o.get("role") == "PRIMARY"]
            if out.get("kind") == "booked":
                sel = (turn.flow.load(out.get("session_id")) or {}).get("selected") or {}
                doctors = [sel["name"]] if sel.get("name") else []
            turn.rt.memory.update(turn.identity, consent=bool(turn.kw.get("remember")),
                                  language=out.get("language"), city=turn.ctx.get("city"), who=patient.get("who"),
                                  specialties=[s for s in specialties if s], doctors=doctors)
        turn.rt.traces.record(out.get("session_id") or turn.session_id, {
            "patient": said, "kind": out.get("kind"), "engine": out.get("engine"), "language": out.get("language"),
            "route": state.get("route"), "why": state.get("why"), "decision": state.get("decision"),
            "understanding": {k: v for k, v in (state.get("understanding") or {}).items() if k != "failed"},
            "tools": turn.tools_used, "checks": turn.checks, "reply": (out.get("message") or "")[:700],
            "specialist": next((o.get("specialist") for o in (out.get("card") or {}).get("outputs") or []
                                if o.get("role") == "PRIMARY"), None),
            "examples": [e.get("id") for e in state.get("examples") or []]})
        return {"messages": [{"role": "patient", "text": said},
                             {"role": "vita", "text": (out.get("message") or "")[:600], "kind": out.get("kind")}],
                "patient": patient}

    # ------------------------------------------------------------------------------------------------ wiring
    g = StateGraph(VitaState)
    for name, fn in (("guard_in", guard_in), ("rules", rules), ("recall", recall), ("understand", understand),
                     ("plan", plan), ("emergency", emergency), ("flow", flow), ("search", search), ("route", route),
                     ("ask", ask), ("answer", answer), ("respond", respond), ("remember", remember)):
        g.add_node(name, fn)
    g.add_edge(START, "guard_in")
    g.add_conditional_edges("guard_in", lambda s: s["route"], {"rules": "rules", "agent": "recall"})
    g.add_edge("recall", "understand")
    g.add_edge("understand", "plan")
    g.add_conditional_edges("plan", lambda s: s["decision"],
                            {d: d for d in ("rules", "emergency", "flow", "search", "route", "ask", "answer")})
    answered = lambda s: "remember" if s.get("out") else "respond"           # noqa: E731
    g.add_conditional_edges("search", answered, {"remember": "remember", "respond": "respond"})
    g.add_conditional_edges("route", answered, {"remember": "remember", "respond": "respond"})
    for n in ("ask", "answer"):
        g.add_edge(n, "respond")
    for n in ("rules", "emergency", "flow", "respond"):
        g.add_edge(n, "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=checkpointer)
