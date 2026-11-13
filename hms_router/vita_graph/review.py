"""How real conversations with Vita go: read the turn traces Vita keeps (learning.TraceStore, 30 days), score each
conversation, and point at what to fix.

    python -m hms_router.vita_graph.review --days 14 --out /tmp/review.json --cases /tmp/real_cases.py
    GET /v2/agent/admin/review?days=14                  (the summary and the worst conversations, admin scope)

Per conversation:
  outcome           booked > doctor chosen > doctors shown > emergency > handed to a person > ended without any
  turns to doctors  patient messages until the first doctor list or recommendation
  questions         Vita's questions before that point
  friction          what went wrong, turn by turn:
                      repair      the patient had to correct or repeat ("nahi nahi", "maine bola", "kya?", "galat")
                      upset       the patient was annoyed ("bakwas", "samajh nahi aata")
                      repeat_q    Vita asked (almost) the same question twice
                      fallback    answered without the model (model off, failing, or a blocked reply)
                      blocked     the reply checker stopped a model reply
                      lang_flip   Vita's reply language changed without the patient asking
                      long        a reply over 400 characters
                      dead_end    the conversation ended right after Vita asked something, with no outcome
The report counts these across conversations (by channel and language), lists the worst conversations as
transcripts for staff, and drafts an understanding-set case (understanding_cases.py format) for every patient
message that was followed by a repair or an upset, so each real misreading becomes a test.
"""
from __future__ import annotations

import difflib
import json
import re
import statistics
import time
from collections import Counter

from ..application import everyday
from ..textnorm import intent_text, tidy

DOCTORS = {"doctor_list", "recommendation"}
_REPAIR = re.compile(
    r"^(nahi|nhi|no|nope)[ ,]+(nahi|nhi|no)\b|\b(maine|mene|maine to|mai ne)\s+(bola|kaha|likha|bataya)\b|"
    r"\b(already|pehle hi|upar)\s+(bola|bataya|told|said|likha)\b|\bi (said|told you|already)\b|\bgalat\b|\bwrong\b|"
    r"\b(ye|yeh|this is)\s+(nahi|not)\s+(chahiye|what i)\b|\bnot what i\b|\bphir se\b|\bdobara\b|\bagain\b")
FRICTION_WEIGHT = {"repair": 3, "upset": 4, "repeat_q": 2, "fallback": 2, "blocked": 1, "lang_flip": 1, "long": 1,
                   "dead_end": 3}


def _norm(text: str) -> str:
    return intent_text(tidy(text or ""))


def patient_signal(text: str) -> str | None:
    """"repair" | "upset" | None for one patient message."""
    norm = _norm(text)
    move = everyday.detect(norm, text or "")
    if move == "upset":
        return "upset"
    if move in ("what",) or _REPAIR.search(norm):
        return "repair"
    return None


def _question(reply: str) -> bool:
    return (reply or "").rstrip().endswith(("?", "।?"))


def score(turns: list[dict], meta: dict | None = None) -> dict:
    """One conversation (its traced turns, oldest first) -> outcome, speed and friction."""
    meta = meta or {}
    kinds = [t.get("kind") for t in turns]
    first_doc = next((i for i, k in enumerate(kinds) if k in DOCTORS), None)
    if meta.get("booked") or "booked" in kinds:
        outcome = "booked"
    elif "doctor_selected" in kinds or "booking" in kinds:
        outcome = "doctor_chosen"
    elif first_doc is not None:
        outcome = "doctors_shown"
    elif "emergency" in kinds:
        outcome = "emergency"
    elif "handoff" in kinds:
        outcome = "handoff"
    else:
        outcome = "none"
    friction: list[dict] = []
    asked: list[str] = []
    langs = [t.get("language") for t in turns]
    for i, t in enumerate(turns):
        reply = t.get("reply") or ""
        if i > 0:
            sig = patient_signal(t.get("patient") or "")
            if sig:
                friction.append({"turn": i, "kind": sig, "after": (turns[i - 1].get("patient") or "")[:120],
                                 "vita_had_said": (turns[i - 1].get("reply") or "")[:160]})
        if _question(reply):
            if any(difflib.SequenceMatcher(None, reply.lower(), q).ratio() > 0.8 for q in asked):
                friction.append({"turn": i, "kind": "repeat_q"})
            asked.append(reply.lower())
        if t.get("engine") == "fallback":
            friction.append({"turn": i, "kind": "fallback"})
        if t.get("checks"):
            friction.append({"turn": i, "kind": "blocked", "checks": t["checks"]})
        if i > 0 and langs[i] and langs[i - 1] and langs[i] != langs[i - 1] \
                and not re.search(r"\b(hindi|english|angrezi|bangla)\b", _norm(t.get("patient") or "")):
            friction.append({"turn": i, "kind": "lang_flip"})
        if len(reply) > 400:
            friction.append({"turn": i, "kind": "long"})
    if turns and outcome == "none" and _question(turns[-1].get("reply") or ""):
        friction.append({"turn": len(turns) - 1, "kind": "dead_end"})
    questions = sum(1 for t in turns[: first_doc if first_doc is not None else len(turns)] if _question(t.get("reply")))
    return {"turns": len(turns), "outcome": outcome, "turnsToDoctors": None if first_doc is None else first_doc + 1,
            "questionsBeforeDoctors": questions, "friction": friction,
            "frictionScore": sum(FRICTION_WEIGHT.get(f["kind"], 1) for f in friction),
            "channel": next((t.get("channel") for t in turns if t.get("channel")), "web"),
            "language": next((x for x in langs if x), None), "rating": meta.get("rating")}


def load(store, days: int = 14, limit: int = 2000) -> list[tuple[str, list[dict], dict]]:
    """The conversations active in the last `days` (newest first), with their turns and metadata."""
    since = time.time() - days * 86400
    out = []
    for sid in store.r.zrevrangebyscore("vita:conv", "+inf", since, start=0, num=limit):
        turns = store.turns(sid)
        if turns:
            out.append((sid, turns, store.r.hgetall(f"vita:conv:{sid}") or {}))
    return out


def report(convs: list[tuple[str, list[dict], dict]], worst: int = 15) -> dict:
    scored = [(sid, turns, score(turns, meta)) for sid, turns, meta in convs]
    n = len(scored) or 1
    outcomes = Counter(s["outcome"] for _, _, s in scored)
    friction = Counter(f["kind"] for _, _, s in scored for f in s["friction"])
    convs_with = Counter(k for _, _, s in scored for k in {f["kind"] for f in s["friction"]})
    speed = [s["turnsToDoctors"] for _, _, s in scored if s["turnsToDoctors"]]
    turns = [t for _, ts, _ in scored for t in ts]
    engines = Counter(t.get("engine") for t in turns)
    by = {}
    for key in ("channel", "language"):
        groups: dict[str, list] = {}
        for _, _, s in scored:
            groups.setdefault(str(s[key]), []).append(s)
        by[key] = {g: {"conversations": len(v), "reachedDoctors": round(sum(s["outcome"] in ("booked", "doctor_chosen",
                                                                                              "doctors_shown")
                                                                            for s in v) / len(v), 3),
                       "withFriction": round(sum(bool(s["friction"]) for s in v) / len(v), 3)} for g, v in groups.items()}
    ended = Counter((ts[-1].get("decision") or ts[-1].get("kind") or "?") for _, ts, s in scored if s["outcome"] == "none")
    misread = [{"session": sid, "turn": f["turn"] - 1, "patient": f["after"], "vita": f["vita_had_said"],
                "then": (ts[f["turn"]].get("patient") or "")[:120], "signal": f["kind"]}
               for sid, ts, s in scored for f in s["friction"] if f["kind"] in ("repair", "upset")]
    first_msgs = Counter(_norm(ts[0].get("patient") or "")[:40] for _, ts, s in scored if s["outcome"] == "none")
    worst_convs = sorted(scored, key=lambda x: -x[2]["frictionScore"])[:worst]
    return {
        "conversations": len(scored), "turns": len(turns),
        "outcomes": {k: round(v / n, 3) for k, v in outcomes.most_common()},
        "reachedDoctors": round(sum(outcomes[k] for k in ("booked", "doctor_chosen", "doctors_shown")) / n, 3),
        "medianTurnsToDoctors": statistics.median(speed) if speed else None,
        "medianQuestionsBeforeDoctors": statistics.median([s["questionsBeforeDoctors"] for _, _, s in scored
                                                           if s["turnsToDoctors"]]) if speed else None,
        "conversationsWith": {k: round(v / n, 3) for k, v in convs_with.most_common()},
        "frictionEvents": dict(friction.most_common()),
        "engines": {str(k): round(v / max(1, len(turns)), 3) for k, v in engines.most_common()},
        "by": by, "endedWithoutOutcomeAt": dict(ended.most_common(10)),
        "firstMessagesThatWentNowhere": dict(first_msgs.most_common(15)),
        "misreadings": misread[:200],
        "worst": [{"session": sid, **{k: s[k] for k in ("outcome", "turns", "frictionScore", "channel", "language")},
                   "friction": sorted({f["kind"] for f in s["friction"]}),
                   "transcript": [{"patient": t.get("patient"), "vita": (t.get("reply") or "")[:300],
                                   "kind": t.get("kind"), "engine": t.get("engine"), "decision": t.get("decision"),
                                   "understood": {k: v for k, v in (t.get("understanding") or {}).items()
                                                  if k in ("intent", "symptoms_en", "who", "doctor")}}
                                  for t in ts]}
                  for sid, ts, s in worst_convs if s["frictionScore"] > 0],
    }


def draft_cases(rep: dict) -> str:
    """understanding_cases.py lines for every misreading: the patient's words, the conversation before them left out,
    the expected reading left for a person to fill in (a case with no expectation is not scored)."""
    lines = ["# Drafted from real conversations by vita_graph/review.py. Fill in what a careful receptionist would",
             "# read (who, complaint, wants, urgent...), then move the case into understanding_cases.py.",
             "from hms_router.vita_graph.understanding import C, T", "", "REAL = ["]
    seen = set()
    for m in rep.get("misreadings", []):
        text = (m["patient"] or "").strip()
        if not text or text in seen or text.startswith("["):
            continue
        seen.add(text)
        lines.append(f"    # Vita said: {m['vita'][:100]!r} -> patient: {m['then'][:80]!r} ({m['signal']})")
        lines.append(f"    C({('real-' + m['session'][-8:] + '-' + str(m['turn']))!r}, 'real', T({text!r})),  # TODO expectations")
    lines.append("]")
    return "\n".join(lines) + "\n"


def print_report(rep: dict) -> None:
    pct = lambda x: f"{x:.0%}" if isinstance(x, float) else str(x)  # noqa: E731
    print(f"\n{rep['conversations']} conversations, {rep['turns']} turns")
    print(f"reached doctors: {pct(rep['reachedDoctors'])}   median turns to doctors: {rep['medianTurnsToDoctors']}   "
          f"median questions first: {rep['medianQuestionsBeforeDoctors']}")
    print("outcomes:    " + "  ".join(f"{k} {pct(v)}" for k, v in rep["outcomes"].items()))
    print("friction:    " + "  ".join(f"{k} {pct(v)}" for k, v in rep["conversationsWith"].items()))
    print("engines:     " + "  ".join(f"{k} {pct(v)}" for k, v in rep["engines"].items()))
    for key, groups in rep["by"].items():
        print(f"by {key}: " + "  ".join(f"{g}: {v['conversations']} conv, doctors {pct(v['reachedDoctors'])}, "
                                         f"friction {pct(v['withFriction'])}" for g, v in groups.items()))
    print("ended with no outcome after: " + ", ".join(f"{k} ({v})" for k, v in rep["endedWithoutOutcomeAt"].items()))
    for m in rep["misreadings"][:15]:
        print(f"  MISREAD {m['patient']!r}\n          Vita: {m['vita'][:90]!r}\n          then: {m['then']!r}")
    for w in rep["worst"][:5]:
        print(f"\n  WORST {w['session']} ({w['outcome']}, {', '.join(w['friction'])})")
        for t in w["transcript"]:
            print(f"    P: {t['patient']}\n    V: {t['vita'][:140]}  [{t['kind']}/{t['engine']}]")


if __name__ == "__main__":                                          # pragma: no cover - manual run
    import argparse

    from fastapi.testclient import TestClient

    from ..adapters.http import create_app
    ap = argparse.ArgumentParser(description="How real conversations with Vita go")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out", help="full report (JSON)")
    ap.add_argument("--cases", help="draft understanding cases for every misreading (Python)")
    args = ap.parse_args()
    with TestClient(create_app()) as c:
        rep = report(load(c.app.state.vita_runtime.traces, days=args.days))
    print_report(rep)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False, default=str)
        print(f"\nFull report: {args.out}")
    if args.cases:
        with open(args.cases, "w", encoding="utf-8") as f:
            f.write(draft_cases(rep))
        print(f"Draft cases: {args.cases}")
