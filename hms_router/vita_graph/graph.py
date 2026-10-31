"""Vita as one LangGraph graph: the whole conversation, one run per patient message, state saved per session
(thread_id = Vita session id) by the Redis checkpointer. There is no other conversation engine.

    START -> self_harm -> tap -> bangla -> emergency_care -> visit -> booking -> picks -> safety -> privacy ─┐
              (each step answers what is its business, or passes the message on)               │   any answer
                                                                                                 │   -> remember
             gate ─(no LLM, switch off, breaker open)─> fallback ───────────────────────────────┤
               ├─(switch "planner")─> recall -> planner (LLM + tools, around the patient's needs) ─┤
               └─> recall -> understand (LLM) -> plan ─┬─> emergency ───────────────────────────┤
                                                      ├─> handoff ─────────────────────────────┤
                                                      ├─> choose (a shown doctor, a day)  ──┬───┤
                                                      ├─> visit_help ─────────────────────┬─┘   │
                                                      ├─> search ──┬────────────────────────────┤
                                                      ├─> route  ──┤ (answered by the tool)     │
                                                      ├─> ask      │                            │
                                                      └─> answer ──┴─> respond (LLM) + checker ─┴─> remember -> END

State that lives across turns: messages (the conversation), patient (who, age group, pregnant, language), triage
(complaint so far in English, questions asked), stage. Per turn: understanding, decision, facts, out.

Rules that stay in code (never a model's choice): the steps above on every message (self-harm, the shared emergency
detector, warning-sign questions, booking/OTP and other steps in progress); the LLM can only ADD urgency (danger_signs,
or assess_symptoms finding a red flag in its reading); the specialist comes from the route-symptom engine; every LLM
reply passes checker.py or is replaced by the tool's own wording or the fallback.
"""
from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, TypedDict

from . import checker
from .learning import redact
from .prompts import RESPOND, UNDERSTAND, respond_input, understand_input

log = logging.getLogger("hms_router.vita_graph")

MAX_QUESTIONS = 2


class VitaState(TypedDict, total=False):
    messages: Annotated[list, operator.add]       # [{"role": "patient"|"vita", "text", "kind"?}]
    patient: dict                                 # who, age_group, pregnant, language
    triage: dict                                  # symptoms_en, asked, routed
    need: dict                                    # the planner's picture of what the patient needs (planner.py)
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


def _tool_args(msg, name: str) -> dict | None:
    for call in getattr(msg, "tool_calls", None) or []:
        if call.get("name") == name:
            return call.get("args") or {}
    return None


def build_vita_graph(checkpointer=None):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph

    # ------------------------------------------------------------------------------------------------ nodes
    # ---------------------------------------------------------------- code steps, in order (application/doctor_chat.py)
    def step(name, run):
        def node(state: VitaState, config) -> dict:
            turn = _turn(config)
            out = run(turn)
            if out is None:
                return {}
            out["engine"] = "step"
            if turn.ctx.get("restarted"):                       # "naya shuru karo": what Vita knew is forgotten too
                return {"out": out, "route": "step", "why": name, "triage": {}, "patient": {}, "need": {},
                        "stage": "start"}
            return {"out": out, "route": "step", "why": name}
        node.__name__ = name
        return node

    STEPS = (
        ("self_harm", lambda t: t.steps.step_self_harm(t.text, t.session_id, t.ctx)),
        ("tap", lambda t: t.steps.step_tap(t.text, t.session_id, t.ctx, t.kw)),
        ("bangla", lambda t: t.steps.step_bangla(t.text, t.session_id)),
        ("emergency_care", lambda t: t.steps.step_emergency_care(t.text, t.session_id, t.ctx)),
        ("visit", lambda t: t.steps.step_visit(t.text, t.session_id, t.ctx)),
        ("booking", lambda t: t.steps.step_booking(t.text, t.session_id, t.ctx)),
        ("picks", lambda t: t.steps.step_picks(t.text, t.session_id, t.ctx)),
        ("safety", lambda t: t.steps.step_safety(t.text, t.session_id, t.ctx)),
        ("privacy", lambda t: t.steps.step_privacy(t.text, t.session_id, t.ctx)),
    )

    def gate(state: VitaState, config) -> dict:
        turn = _turn(config)
        channel = turn.kw.get("channel")
        if not turn.rt.agent_on(channel):
            return {"route": "fallback"}
        return {"route": "planner" if turn.rt.mode(channel) == "planner" and turn.rt.llm_plan is not None else "agent"}

    def planner(state: VitaState, config) -> dict:
        from .planner import run_planner
        turn = _turn(config)
        upd = run_planner(turn, state)
        need, patient = upd.get("need") or {}, dict(state.get("patient") or {})
        for k in ("who", "age_group", "pregnant", "language"):
            if need.get(k) not in (None, ""):
                patient[k] = need[k]
        upd["patient"] = patient
        return upd

    def fallback(state: VitaState, config) -> dict:
        out = _turn(config).fallback()
        out["engine"] = "fallback"
        return {"out": out, "decision": "fallback"}

    def recall(state: VitaState, config) -> dict:
        turn = _turn(config)
        memory = turn.rt.memory.load(turn.identity) if turn.identity else {}
        if memory.get("city") and not turn.ctx.get("city"):
            turn.ctx["city"] = memory["city"]                       # a returning patient's city, for search
        return {"memory": memory, "examples": turn.rt.examples.similar(turn.text, k=3)}

    def understand(state: VitaState, config) -> dict:
        turn = _turn(config)
        prompt = understand_input(state.get("messages") or [], state.get("patient") or {}, state.get("triage") or {},
                                  turn.raw)
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
            return {"decision": "fallback"}
        intent, doctor = u.get("intent"), {k: v for k, v in (u.get("doctor") or {}).items() if v}
        if u.get("danger_signs"):
            return {"decision": "emergency"}
        if intent == "human":
            return {"decision": "handoff"}
        if intent == "choose_or_book":
            return {"decision": "choose"}
        if intent == "visit_help":
            return {"decision": "visit_help"}
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

    def handoff(state: VitaState, config) -> dict:
        turn = _turn(config)
        out = turn.steps.handoff(turn.text, turn.session_id, "asked")
        out["engine"] = "agent+step"
        return {"out": out, "stage": "handoff"}

    def choose(state: VitaState, config) -> dict:
        turn = _turn(config)
        name = ((state.get("understanding") or {}).get("doctor") or {}).get("doctor_name")
        turn.tools_used.append("choose")
        out = turn.steps.choose(turn.text, turn.session_id, turn.ctx, name)
        if out is None:                                  # nothing to choose yet: say so, offer to search
            return {"facts": {"no_doctor_chosen_yet": True, **turn.steps.facts(turn.session_id),
                              "next": "ask which doctor, or offer to find one by need, name or place"},
                    "decision": "answer"}
        out["engine"] = "agent+step"
        return {"out": out, "stage": "booking" if str(out.get("kind", "")).startswith(("book", "doctor_sel")) else
                state.get("stage") or "search"}

    def visit_help(state: VitaState, config) -> dict:
        turn = _turn(config)
        turn.tools_used.append("visit")
        out = turn.steps.step_visit(turn.text, turn.session_id, turn.ctx, asked=True)
        if out is None:
            return {"facts": {"no_booked_visit_found": True, **turn.steps.facts(turn.session_id)},
                    "decision": "answer"}
        out["engine"] = "agent+step"
        return {"out": out, "stage": "visit"}

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
        turn = _turn(config)
        return {"facts": {"intent": u.get("intent"), "topic": u.get("question_topic"),
                          "doctor_in_hand": turn.steps.facts(turn.session_id),
                          "vita_can": ["find the right kind of doctor from symptoms", "find doctors by name, specialty,"
                                       " hospital or place", "book an appointment", "help on the day of the visit"]},
                "decision": "answer"}

    def respond(state: VitaState, config) -> dict:
        turn = _turn(config)
        patient = state.get("patient") or {}
        prompt = respond_input(decision=state.get("decision") or "answer", facts=state.get("facts") or {},
                               conversation=state.get("messages") or [], patient=patient,
                               memory=state.get("memory") or {}, examples=state.get("examples") or [],
                               text=turn.raw, language=patient.get("language"), channel=turn.kw.get("channel") or "web")
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
        out = turn.fallback()                                        # nothing usable: search, or "try again / our team"
        out["engine"] = "fallback"
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
            "need": state.get("need") or {},
            "tools": turn.tools_used, "checks": turn.checks, "reply": (out.get("message") or "")[:700],
            "specialist": next((o.get("specialist") for o in (out.get("card") or {}).get("outputs") or []
                                if o.get("role") == "PRIMARY"), None),
            "examples": [e.get("id") for e in state.get("examples") or []]})
        return {"messages": [{"role": "patient", "text": said},
                             {"role": "vita", "text": (out.get("message") or "")[:600], "kind": out.get("kind")}],
                "patient": patient}

    # ------------------------------------------------------------------------------------------------ wiring
    g = StateGraph(VitaState)
    names = [n for n, _ in STEPS]
    for n, run in STEPS:
        g.add_node(n, step(n, run))
    for name, fn in (("gate", gate), ("fallback", fallback), ("recall", recall), ("planner", planner),
                     ("understand", understand),
                     ("plan", plan), ("emergency", emergency), ("handoff", handoff), ("choose", choose),
                     ("visit_help", visit_help), ("search", search), ("route", route), ("ask", ask),
                     ("answer", answer), ("respond", respond), ("remember", remember)):
        g.add_node(name, fn)
    g.add_edge(START, names[0])
    for here, nxt in zip(names, names[1:] + ["gate"]):
        g.add_conditional_edges(here, lambda s, nxt=nxt: "remember" if s.get("out") else nxt,
                                {"remember": "remember", nxt: nxt})
    g.add_conditional_edges("gate", lambda s: s["route"], {"agent": "recall", "planner": "recall",
                                                            "fallback": "fallback"})
    g.add_conditional_edges("recall", lambda s: "planner" if s.get("route") == "planner" else "understand",
                            {"planner": "planner", "understand": "understand"})
    g.add_edge("planner", "remember")
    g.add_edge("understand", "plan")
    g.add_conditional_edges("plan", lambda s: s["decision"],
                            {d: d for d in ("fallback", "emergency", "handoff", "choose", "visit_help", "search", "route",
                                            "ask", "answer")})
    answered = lambda s: "remember" if s.get("out") else "respond"           # noqa: E731
    for n in ("search", "route", "choose", "visit_help"):
        g.add_conditional_edges(n, answered, {"remember": "remember", "respond": "respond"})
    for n in ("ask", "answer"):
        g.add_edge(n, "respond")
    for n in ("fallback", "emergency", "handoff", "respond"):
        g.add_edge(n, "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=checkpointer)
