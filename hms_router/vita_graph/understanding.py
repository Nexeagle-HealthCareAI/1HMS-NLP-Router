"""How well Vita understands patients, measured on its own (2026-10-31).

The scenario suite (evals.py) checks where a conversation ends up. This checks what Vita UNDERSTOOD at every turn,
separately from what it then did, so a wrong answer can be traced to a misreading or to a wrong action.

Each case in understanding_cases.py is a short conversation; each turn carries the correct reading (only the fields
that matter for that turn). After every turn, Vita's own reading is taken from the graph state and the turn's trace,
in the same shape for both modes:

    agent mode    understanding (intent, complaint, danger signs, doctor filters) + patient + triage
    planner mode  the patient's needs + the tools the model used
    both          what was actually applied (the doctor search's filters, the reply kind)

Fields scored (UNDERSTOOD): who, age_group, pregnant, complaint (must mention / must not mention), urgent (an
emergency reply; "check" = a warning-sign question or an emergency), wants (what the patient asked for), gender,
place, specialty, budget, today, degree, doctor_name.
Fields scored (ACTED): kind, specialist (PRIMARY on the card), chosen (the doctor chosen, by position in the last list).

Critical: an urgent turn that got no emergency reply (missed), and a calm turn that got one (false alarm), are
counted on their own; a missed emergency fails the run however good the rest is.

CLI (inside the router container, with the service's settings):
    python -m hms_router.vita_graph.understanding --mode both --out /tmp/understanding.json
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from ..doctors import specialty_key

log = logging.getLogger("hms_router.vita_understanding")

UNDERSTOOD = ("who", "age_group", "pregnant", "complaint", "urgent", "wants", "gender", "place", "specialty",
              "budget", "today", "degree", "doctor_name")
ACTED = ("kind", "specialist", "chosen")


@dataclass
class Turn:
    say: str
    expect: dict = field(default_factory=dict)


@dataclass
class Case:
    id: str
    category: str
    turns: list[Turn]
    city: str | None = "Kolkata"


def T(say: str, **expect) -> Turn:
    return Turn(say, expect)


def C(id: str, category: str, *turns: Turn, city: str | None = "Kolkata") -> Case:
    return Case(id, category, list(turns), city)


# ------------------------------------------------------------------------------------------------ Vita's reading
_AGENT_WANTS = {"symptoms": "symptoms", "find_doctor": "find_doctor", "more_doctors": "more",
                "choose_or_book": "choose", "visit_help": "visit", "question": "question", "greeting": "smalltalk",
                "thanks_or_bye": "smalltalk", "human": "human", "other": "question"}
_TOOL_WANTS = (("talk_to_person", "human"), ("handoff", "human"), ("visit_help", "visit"), ("choose_doctor", "choose"),
               ("doctor_days", "choose"), ("choose", "choose"), ("assess_symptoms", "symptoms"),
               ("find_doctors", "find_doctor"), ("visit", "visit"))


_KIND_WANTS = {"doctor_list": "find_doctor", "doctor_selected": "choose", "booking": "choose", "handoff": "human",
               "emergency": None}


def reading(out: dict, trace: dict, values: dict) -> dict:
    """Vita's reading of one turn, the same shape whichever mode (or code step) produced it."""
    u = trace.get("understanding") or {}
    need = values.get("need") or {}
    patient = values.get("patient") or {}
    triage = values.get("triage") or {}
    doctor = u.get("doctor") or {}
    applied = (out.get("doctors") or {}).get("filters") or {}
    tools = trace.get("tools") or []
    r: dict = {"engine": out.get("engine"), "kind": out.get("kind")}
    r["who"] = patient.get("who") or u.get("who") or need.get("who")
    r["age_group"] = patient.get("age_group") or u.get("age_group") or need.get("age_group")
    r["pregnant"] = bool(patient.get("pregnant") or u.get("pregnant") or need.get("pregnant"))
    r["complaint"] = (u.get("symptoms_en") or triage.get("symptoms_en") or need.get("complaint_en") or "").lower()
    clar = (out.get("clarify") or {}).get("kind")
    r["urgent"] = "emergency" if out.get("kind") == "emergency" else "warning" if clar == "warning_signs" else "no"
    if u.get("intent"):
        r["wants"] = _AGENT_WANTS.get(u["intent"], u["intent"])
    elif tools:
        r["wants"] = next((w for t, w in _TOOL_WANTS if t in tools), "question")
        if r["wants"] == "find_doctor" and "more" in (trace.get("patient") or "").lower():
            r["wants"] = "more"
    else:                                    # a code step or the no-LLM fallback: what it did is what it read
        r["wants"] = _KIND_WANTS.get(out.get("kind"), None if out.get("engine") == "step" else "question")
    r["gender"] = applied.get("gender") or doctor.get("gender") or (need.get("doctor_gender")
                                                                     if need.get("doctor_gender") != "any" else None)
    r["place"] = (applied.get("city") if applied.get("city") else None) or doctor.get("place") or need.get("place")
    spec = applied.get("specialtyId") or doctor.get("specialty") or need.get("specialty")
    r["specialty"] = specialty_key(spec) if spec else None
    r["budget"] = "low" if (applied.get("sort") == "fee" or doctor.get("sort") == "fee" or need.get("budget") == "low") \
        else None
    r["today"] = bool(applied.get("availableToday") or doctor.get("available_today")
                      or (need.get("day") or "").lower() in ("today", "aaj"))
    r["degree"] = (applied.get("degree") or doctor.get("degree") or need.get("degree") or None)
    r["doctor_name"] = (applied.get("query") or doctor.get("doctor_name") or need.get("doctor_name") or None)
    card = out.get("card") or {}
    r["specialist"] = next((o.get("specialist") for o in card.get("outputs") or [] if o.get("role") == "PRIMARY"), None)
    r["chosen"] = (out.get("doctor") or {}).get("doctorId") or out.get("selectedDoctorId")
    return r


def _any(expected: str, text: str) -> bool:
    return any(alt.strip().lower() in text for alt in expected.split("|"))


def score_turn(expect: dict, r: dict, last_list: list[dict]) -> dict[str, bool]:
    """field -> correct? (only the fields the case states)"""
    s: dict[str, bool] = {}
    for f in ("who", "age_group"):
        if f in expect:
            want = expect[f]
            s[f] = r.get(f) in (want if isinstance(want, (set, list, tuple)) else {want})
    if "pregnant" in expect:
        s["pregnant"] = r["pregnant"] == expect["pregnant"]
    if "complaint" in expect or "not_complaint" in expect:
        ok = all(_any(w, r["complaint"]) for w in expect.get("complaint", []))
        ok = ok and not any(_any(w, r["complaint"]) for w in expect.get("not_complaint", []))
        s["complaint"] = ok
    if "urgent" in expect:
        want = expect["urgent"]
        s["urgent"] = (r["urgent"] == "emergency") if want is True else (r["urgent"] != "emergency") if want is False \
            else r["urgent"] in ("emergency", "warning")
    if "wants" in expect:
        want = expect["wants"]
        s["wants"] = r["wants"] in (want if isinstance(want, (set, list, tuple)) else {want})
    for f in ("gender", "budget", "degree"):
        if f in expect:
            s[f] = (str(r.get(f) or "").lower() == str(expect[f]).lower()) if expect[f] else not r.get(f)
    if "today" in expect:
        s["today"] = r["today"] == expect["today"]
    if "place" in expect:
        s["place"] = bool(r.get("place")) and _any(expect["place"], str(r["place"]).lower())
    if "specialty" in expect:
        want = expect["specialty"]
        s["specialty"] = r.get("specialty") in (want if isinstance(want, (set, list, tuple)) else {want})
    if "doctor_name" in expect:
        s["doctor_name"] = bool(r.get("doctor_name")) and _any(expect["doctor_name"], str(r["doctor_name"]).lower())
    if "kind" in expect:
        want = expect["kind"]
        s["kind"] = r["kind"] in (want if isinstance(want, (set, list, tuple)) else {want})
    if "specialist" in expect:
        want = expect["specialist"]
        s["specialist"] = r["specialist"] in (want if isinstance(want, (set, list, tuple)) else {want})
    if "chosen" in expect:
        pos = int(expect["chosen"])
        s["chosen"] = len(last_list) >= pos and r["chosen"] == last_list[pos - 1].get("doctorId")
    return s


# ------------------------------------------------------------------------------------------------ running
def private_vita(*, lexicon, directory, route_service, llm_understand, llm_respond, llm_plan, mode: str):
    """A Vita with its own in-memory state (never the live conversations): returns (new_engine(), runtime)."""
    import fakeredis
    from langgraph.checkpoint.memory import InMemorySaver

    from ..application.doctor_chat import DoctorChat
    from ..pipeline import HMSRouterPipeline
    from .engine import Breaker, VitaEngine, VitaRuntime
    from .graph import build_vita_graph

    r = fakeredis.FakeRedis(decode_responses=True)
    rt = VitaRuntime(None, r, route_service=route_service, breaker=Breaker(limit=10_000),
                     graph=build_vita_graph(InMemorySaver()))
    rt.llm_understand, rt.llm_respond, rt.llm_plan = llm_understand, llm_respond, llm_plan
    rt.set_control({"mode": mode})
    pipeline = HMSRouterPipeline(r, lexicon)
    svc = route_service() if route_service else None

    def new_engine():
        return VitaEngine(DoctorChat(pipeline, directory, r, booking_client=None,
                                     detector=lambda: getattr(svc, "detector", None)), rt)
    return new_engine, rt


def run(cases: list[Case], *, mode: str = "agent", **vita) -> dict:
    new_engine, rt = private_vita(mode=mode, **vita)
    fields = defaultdict(lambda: [0, 0])                 # field -> [right, scored]
    cats = defaultdict(lambda: [0, 0])                   # category -> [turns fully right, turns]
    missed, false_alarms, results = [], [], []
    t0 = time.perf_counter()
    for case in cases:
        engine, sid, last_list = new_engine(), None, []
        for i, turn in enumerate(case.turns):
            try:
                out = engine.handle(turn.say, sid, city=case.city)
                sid = out.get("session_id") or sid
                trace = (rt.traces.turns(sid) or [{}])[-1]
                snap = rt.graph.get_state({"configurable": {"thread_id": sid}})
                r = reading(out, trace, dict(snap.values or {}) if snap else {})
            except Exception as exc:                     # a crash is a wrong reading, never a failed run
                log.exception("case %s turn %d crashed", case.id, i)
                out, r = {}, {"error": type(exc).__name__, "urgent": "no", "complaint": ""}
            s = score_turn(turn.expect, r, last_list) if "error" not in r else {f: False for f in turn.expect
                                                                                 if f != "not_complaint"}
            for f, ok in s.items():
                fields[f][0] += ok
                fields[f][1] += 1
            if s:
                cats[case.category][0] += all(s.values())
                cats[case.category][1] += 1
            if turn.expect.get("urgent") is True and not s.get("urgent", True):
                missed.append(f"{case.id}#{i + 1}: {turn.say}")
            if turn.expect.get("urgent") is False and not s.get("urgent", True):
                false_alarms.append(f"{case.id}#{i + 1}: {turn.say}")
            items = (out.get("doctors") or {}).get("items") or []
            if items:
                last_list = items
            results.append({"case": case.id, "category": case.category, "turn": i + 1, "said": turn.say,
                            "expected": {k: sorted(v, key=str) if isinstance(v, set) else v for k, v in turn.expect.items()},
                            "read": r, "wrong": [f for f, ok in s.items() if not ok],
                            "reply": (out.get("message") or "")[:200]})
    pct = {f: round(v[0] / v[1], 3) for f, v in fields.items() if v[1]}
    und = [fields[f] for f in UNDERSTOOD if fields[f][1]]
    act = [fields[f] for f in ACTED if fields[f][1]]
    return {
        "mode": mode, "cases": len(cases), "turns": len(results), "seconds": round(time.perf_counter() - t0, 1),
        "understood": round(sum(a for a, _ in und) / max(1, sum(b for _, b in und)), 3),
        "acted": round(sum(a for a, _ in act) / max(1, sum(b for _, b in act)), 3),
        "fields": pct, "fieldCounts": {f: v[1] for f, v in fields.items()},
        "categories": {c: round(v[0] / v[1], 3) for c, v in cats.items() if v[1]},
        "missedEmergencies": missed, "falseAlarms": false_alarms, "safe": not missed,
        "results": results,
    }


def print_report(rep: dict) -> None:
    print(f"\n== {rep['mode']}: understood {rep['understood']:.0%}, acted {rep['acted']:.0%} "
          f"({rep['cases']} conversations, {rep['turns']} turns, {rep['seconds']} s)")
    print(f"   missed emergencies: {len(rep['missedEmergencies'])}   false alarms: {len(rep['falseAlarms'])}")
    for m in rep["missedEmergencies"]:
        print(f"   MISSED  {m}")
    for m in rep["falseAlarms"]:
        print(f"   ALARM   {m}")
    print("   by field:    " + "  ".join(f"{f} {rep['fields'][f]:.0%}" for f in UNDERSTOOD + ACTED if f in rep["fields"]))
    print("   by category: " + "  ".join(f"{c} {v:.0%}" for c, v in sorted(rep["categories"].items(), key=lambda x: x[1])))
    wrong = [r for r in rep["results"] if r["wrong"]]
    for r in wrong[:40]:
        print(f"   WRONG {r['case']}#{r['turn']} [{', '.join(r['wrong'])}] {r['said'][:60]!r}"
              f"\n         read: " + ", ".join(f"{k}={r['read'].get(k)!r}" for k in r["wrong"] if k in r["read"]))
    if len(wrong) > 40:
        print(f"   ... {len(wrong) - 40} more in the report file")


if __name__ == "__main__":                                          # pragma: no cover - manual run
    import argparse
    import json

    from fastapi.testclient import TestClient

    from ..adapters.http import create_app
    from .understanding_cases import CASES
    ap = argparse.ArgumentParser(description="How well Vita understands patients, turn by turn")
    ap.add_argument("--mode", choices=["agent", "planner", "both"], default="both")
    ap.add_argument("--category", help="only this category")
    ap.add_argument("--out", help="write every turn (expected vs read) to this JSON file")
    args = ap.parse_args()
    cases = [c for c in CASES if not args.category or c.category == args.category]
    with TestClient(create_app()) as c:
        a = c.app.state
        rt = a.vita_runtime
        if not rt.has_llm:
            print("No LLM configured (VITA_LLM_PROVIDER / GROQ_API_KEY): only the code steps can be scored.")
        reports = []
        for mode in (["agent", "planner"] if args.mode == "both" else [args.mode]):
            rep = run(cases, mode=mode, lexicon=a.vita_pipeline.lex, directory=a.doctors,
                      route_service=lambda: a.route, llm_understand=rt.llm_understand,
                      llm_respond=rt.llm_respond, llm_plan=rt.llm_plan)
            print_report(rep)
            reports.append(rep)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(reports, f, indent=1, ensure_ascii=False, default=str)
            print(f"\nFull report: {args.out}")
