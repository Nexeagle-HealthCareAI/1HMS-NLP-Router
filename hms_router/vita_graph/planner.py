"""Vita's planner: the LLM decides how to help, with tools, around one picture of what the patient needs.

The `agent` path (graph.py: understand -> plan -> one node -> respond) labels each message with a fixed intent and code
picks the next move. The planner (control switch mode "planner") lets the model plan instead:

    note_needs      update the patient's needs from anything they say, in any order (who, complaint, urgency, place,
                    day, time, budget, doctor gender, a specialty / doctor / degree / hospital they want)
    find_doctors    search 1HMS with any mix of filters (standing preferences from the needs fill the gaps)
    assess_symptoms the route-symptom engine on the complaint (emergency check first; it says when it is unsure)
    doctor_days     when a doctor sits;  choose_doctor  a doctor on screen;  visit_help  the day of the visit
    talk_to_person  a handoff;  reply  Vita's message (always last)

Usually two model calls a message: [note_needs + a tool] then [reply]. At most MAX_ROUNDS.

What stays code, whatever the model does: the safety steps before it (graph.py), danger signs in note_needs -> the
emergency reply at once (the model can only add urgency), a tool that answers the patient itself (an emergency, a
location question, a doctor chosen, a handoff) ends the turn with its own reviewed wording, every reply passes
checker.py, booking/OTP stay in the booking step, and after MAX_QUESTIONS questions without an answer the complaint is
assessed instead of asked about again.
"""
from __future__ import annotations

import json
import logging

from . import checker
from .learning import redact
from .prompts import NEVER, PERSONA, _fn

log = logging.getLogger("hms_router.vita_planner")

MAX_ROUNDS = 4
MAX_QUESTIONS = 3

PLAN = PERSONA + """

""" + NEVER + """

You help the patient get the right doctor with as little back-and-forth as possible, the way a kind, quick receptionist
would. Each message, decide what helps most NOW and use the tools:
- Call note_needs whenever the message adds or changes anything about the patient or what they want (in the same turn
  as your other tool calls). Danger signs (chest pain with sweating, breathlessness, fainting, stroke signs, heavy
  bleeding, seizure, poisoning, severe injury, pregnancy bleeding, very high fever in a baby, self-harm) go in
  danger_signs; when unsure, include them.
- Someone who knows what they want (a kind of doctor, a name, a degree, a hospital, a place, "near me", cheaper, a lady
  doctor, a day) gets find_doctors at once, with every filter they gave in one search.
- Symptoms: when the main complaint is clear, call assess_symptoms (it chooses the specialist and lists doctors). Ask
  a question only when the answer would change the choice: the engine said it is unsure (use its question), or nothing
  usable was said. Never ask what is already known; never more than one question at a time. Someone in a hurry ("bas
  doctor dikhao") gets doctors now.
- "This one", "the second", "Dr Sen wale ko book karo" -> choose_doctor. "Kab baithte hain / kal milenge?" ->
  doctor_days. Directions, "I've reached", token, queue -> visit_help. Wants a person -> talk_to_person.
- Use what you know about a returning patient (memory) to save them effort, e.g. offer their usual doctor or city.
- Finish with reply: 1-3 short sentences in the patient's language and script, warm and natural; mention one or two
  doctors by name at most (the list is on screen); up to 3 short tap options when they make answering easier. Adapt
  to the person: brief and simple for someone anxious, elderly or in a hurry; reassure a worried parent first.
"""

_STR = {"type": "string"}
NOTE_NEEDS = _fn("note_needs", "Record what the patient's message adds or changes about them and what they want.", {
    "who": {"type": "string", "enum": ["self", "child", "parent", "spouse", "other"]},
    "age_group": {"type": "string", "enum": ["child", "adult"]},
    "pregnant": {"type": "boolean"},
    "complaint_en": {"type": "string", "description": "everything said about their health so far, plain English, with "
                                                       "duration and severity"},
    "danger_signs": {"type": "array", "items": _STR},
    "language": {"type": "string", "enum": ["english", "hinglish", "hindi"]},
    "place": _STR, "near_me": {"type": "boolean"},
    "day": {"type": "string", "description": "today | tomorrow | day after tomorrow | a weekday"},
    "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "evening"]},
    "budget": {"type": "string", "enum": ["low", "any"]},
    "doctor_gender": {"type": "string", "enum": ["female", "male", "any"]},
    "specialty": _STR, "doctor_name": _STR, "degree": _STR, "hospital": _STR,
    "in_a_hurry": {"type": "boolean"},
})
FIND = _fn("find_doctors", "Search doctors on Doctor Dekho with any mix of filters. more=true: the next page.", {
    "specialty": _STR, "doctor_name": _STR, "degree": _STR, "hospital": _STR, "place": _STR,
    "near_me": {"type": "boolean"}, "gender": {"type": "string", "enum": ["female", "male"]},
    "available_today": {"type": "boolean"}, "sort": {"type": "string", "enum": ["fee", "experience"]},
    "more": {"type": "boolean"},
})
ASSESS = _fn("assess_symptoms", "Choose the right specialist for the complaint and list doctors (emergency check "
                                "first). Says when it is unsure and what to ask.", {
    "symptoms": _STR, "age_group": {"type": "string", "enum": ["child", "adult"]}, "pregnant": {"type": "boolean"}})
DAYS = _fn("doctor_days", "When a doctor sits: the chosen one, or by name.", {"doctor_name": _STR, "day": _STR})
CHOOSE = _fn("choose_doctor", "Choose a doctor on screen, by name or position (1 = first).",
             {"doctor_name": _STR, "position": {"type": "integer", "minimum": 1}})
VISIT = _fn("visit_help", "The day of a booked visit: directions, reached, check-in, token, queue.", {})
PERSON = _fn("talk_to_person", "Hand the conversation to a person.", {"reason": _STR})
REPLY = _fn("reply", "Send Vita's message to the patient. Always the last call of a turn.", {
    "message": _STR, "options": {"type": "array", "items": _STR, "maxItems": 3}}, ["message"])
PLAN_TOOLS = [NOTE_NEEDS, FIND, ASSESS, DAYS, CHOOSE, VISIT, PERSON, REPLY]

_KEEP = ("who", "age_group", "pregnant", "complaint_en", "language", "place", "near_me", "day", "time_of_day", "budget",
         "doctor_gender", "specialty", "doctor_name", "degree", "hospital", "in_a_hurry")


def merge_needs(need: dict, said: dict) -> dict:
    out = dict(need)
    for k in _KEEP:
        v = said.get(k)
        if v in (None, ""):
            continue
        if v == "any" and k in ("doctor_gender", "budget"):
            out.pop(k, None)                        # "koi bhi chalega": the preference is gone
            continue
        out[k] = v
    return out


def plan_input(*, need: dict, conversation: list, memory: dict, examples: list, last_question: str | None,
               doctor_in_hand: dict, text: str, channel: str) -> str:
    shots = [{"patient_said": e.get("patient"), "vita_replied": e.get("reply"), "right_doctor": e.get("specialist"),
              "note": e.get("note")} for e in examples]
    return json.dumps({"latest_message": text, "patient_needs_so_far": need, "vita_last_asked": last_question,
                       "doctor_in_hand": doctor_in_hand, "conversation_so_far": conversation[-12:],
                       "returning_patient_memory": memory, "how_vita_answered_similar_patients": shots,
                       "channel": channel}, ensure_ascii=False, default=str)


def _with_needs(name: str, args: dict, need: dict) -> dict:
    """Standing preferences fill what the model left out: a lady doctor, cheaper, the place, today."""
    a = {k: v for k, v in (args or {}).items() if v not in (None, "")}
    if name == "find_doctors" and not a.get("more"):
        if need.get("doctor_gender") in ("female", "male"):
            a.setdefault("gender", need["doctor_gender"])
        if need.get("budget") == "low":
            a.setdefault("sort", "fee")
        if need.get("place") and not a.get("near_me"):
            a.setdefault("place", need["place"])
        if need.get("near_me"):
            a.setdefault("near_me", True)
        if (need.get("day") or "").lower() in ("today", "aaj"):
            a.setdefault("available_today", True)
    if name == "assess_symptoms":
        a.setdefault("symptoms", need.get("complaint_en") or "")
        if need.get("age_group"):
            a.setdefault("age_group", need["age_group"])
        if need.get("pregnant"):
            a.setdefault("pregnant", True)
    return a


def run_planner(turn, state: dict) -> dict:
    """One turn: the model plans with tools until it replies or a tool answers. Returns the graph update."""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    steps, rt = turn.steps, turn.rt
    need = dict(state.get("need") or {})
    st = steps.load(turn.session_id)
    prompt = plan_input(need=need, conversation=state.get("messages") or [], memory=state.get("memory") or {},
                        examples=state.get("examples") or [],
                        last_question=(st.get("last_reply") or {}).get("message"),
                        doctor_in_hand=steps.facts(turn.session_id), text=turn.raw,
                        channel=turn.kw.get("channel") or "web")
    msgs = [SystemMessage(content=PLAN), HumanMessage(content=redact(prompt))]
    replied, used = None, []
    for _ in range(MAX_ROUNDS):
        try:
            ai = rt.llm_plan.invoke(msgs)
        except Exception:
            log.exception("planner call failed")
            rt.breaker.fail()
            break
        calls = list(getattr(ai, "tool_calls", None) or [])
        if not calls and isinstance(getattr(ai, "content", None), str) and ai.content.strip():
            calls = [{"name": "reply", "args": {"message": ai.content.strip()}, "id": "content"}]
        if not calls:
            break
        msgs.append(ai)
        calls.sort(key=lambda c: c.get("name") != "note_needs")         # needs first, so the tools see them
        for c in calls:
            name, args = c.get("name"), c.get("args") or {}
            if name == "note_needs":
                need = merge_needs(need, args)
                signs = [s for s in args.get("danger_signs") or [] if s]
                if signs:                                                # the model can only add urgency
                    turn.final = turn.emergency([s.lower() for s in signs])
                result = "noted"
            elif name == "reply":
                replied, result = args, "sent"
            elif name == "assess_symptoms" and not (args.get("symptoms") or need.get("complaint_en")):
                result = json.dumps({"error": "no complaint yet: ask what the problem is"})
            else:
                used.append(name)
                result = turn.run(name, _with_needs(name, args, need))
            msgs.append(ToolMessage(content=result, tool_call_id=c.get("id") or name))
            if turn.final is not None:
                break
        if turn.final is not None or replied is not None:
            break
    turn.tools_used += [u for u in used if u not in turn.tools_used]
    if "assess_symptoms" in used:
        need["questions"] = 0                                            # assessed: a later complaint starts afresh

    if turn.final is not None:                                           # a tool answered in its own words
        out = turn.final
        out["engine"] = "planner+step"
        return {"out": out, "need": need, "decision": "tool", "stage": _stage(out)}
    asks = bool(replied) and str(replied.get("message", "")).rstrip().endswith(("?", "।?"))
    if asks and not used and need.get("complaint_en") and int(need.get("questions") or 0) >= MAX_QUESTIONS:
        # enough questions: assess what is known instead of asking again
        turn.run("assess_symptoms", _with_needs("assess_symptoms", {}, need))
        need["questions"] = 0
        if turn.final is not None:
            out = turn.final
            out["engine"] = "planner+step"
            return {"out": out, "need": need, "decision": "tool", "stage": _stage(out)}
        replied = None
    if replied:
        turn.t_reply(replied.get("message", ""), replied.get("options"))
        turn.checks = checker.problems(turn.draft["message"], turn.known_names())
        if not turn.checks:
            out = turn.compose()
            if asks and not out.get("clarify") and turn.ui is None:
                out["kind"], out["question"] = "follow_up", out["message"]
            out["engine"] = "planner"
            rt.breaker.ok()
            if asks and turn.ui is None:
                need["questions"] = int(need.get("questions") or 0) + 1
            return {"out": out, "need": need, "decision": "reply", "stage": _stage(out)}
        log.warning("planner reply blocked by the checker: %s", turn.checks)
    if turn.ui is not None and turn.ui.get("message"):                  # the tool's own reviewed wording
        out = dict(turn.ui)
        out["engine"] = "planner+template"
        return {"out": out, "need": need, "decision": "tool", "stage": _stage(out)}
    out = turn.fallback()
    out["engine"] = "fallback"
    return {"out": out, "need": need, "decision": "fallback"}


def _stage(out: dict) -> str:
    kind = str(out.get("kind") or "")
    if kind == "emergency":
        return "emergency"
    if kind.startswith(("book", "doctor_sel")):
        return "booking"
    if kind in ("doctor_list", "recommendation"):
        return "search"
    return "talk"
