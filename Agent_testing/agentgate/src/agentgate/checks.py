"""Deterministic checks (steps 3-4 and the must-never rules from step 1).

These are cheap and exact, so they always run before any LLM judge.
Assert on actions and final state, never on exact reply wording.
"""
from __future__ import annotations

import json
import re


def _subset(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _subset(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(_subset(e, a) for e, a in zip(expected, actual)))
    num = (int, float)
    if (isinstance(expected, num) and not isinstance(expected, bool)
            and isinstance(actual, num) and not isinstance(actual, bool)):
        return abs(expected - actual) < 1e-9
    return expected == actual


def _subsequence(wanted: list[str], actual: list[str]) -> bool:
    it = iter(actual)
    return all(any(w == a for a in it) for w in wanted)


def evaluate(case: dict, res, rules: dict) -> tuple[list[dict], list[str]]:
    """Return (checks, must_never_events) for one agent response."""
    exp = case.get("expect") or {}
    checks: list[dict] = []
    events: list[str] = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    if res.error:
        add("agent_responded", False, res.error)
        return checks, events

    names = [c["name"] for c in res.tool_calls]
    mn_rules = rules.get("must_never") or {}
    mn_case = case.get("must_never") or {}

    # Must-never events: zero tolerance, no percentages.
    for tool in list(mn_rules.get("forbidden_tools", [])) + list(mn_case.get("tools", [])):
        if tool in names:
            events.append(f"forbidden tool called: {tool}")
    for pat in list(mn_rules.get("forbidden_text_patterns", [])) + list(mn_case.get("text_patterns", [])):
        if re.search(pat, res.text, re.I):
            events.append(f"forbidden text matched pattern: {pat}")
    if events:
        add("must_never", False, "; ".join(events))

    if "tools_called" in exp:
        missing = [t for t in exp["tools_called"] if t not in names]
        add("tools_called", not missing, f"missing: {missing}; actual: {names}" if missing else "")
    if "tool_order" in exp:
        add("tool_order", _subsequence(exp["tool_order"], names),
            f"wanted order {exp['tool_order']}; actual: {names}")
    if exp.get("tools_not_called"):
        bad = [t for t in exp["tools_not_called"] if t in names]
        add("tools_not_called", not bad, f"called: {bad}" if bad else "")
    for spec in exp.get("tool_args", []):
        ok = any(c["name"] == spec["tool"] and _subset(spec["args"], c.get("arguments") or {})
                 for c in res.tool_calls)
        add(f"tool_args:{spec['tool']}", ok, f"expected args {spec['args']}")

    limit = exp.get("max_tool_calls", rules.get("limits", {}).get("max_tool_calls_per_case"))
    if limit is not None:
        add("tool_call_budget", len(names) <= limit, f"{len(names)} calls (limit {limit})")
    if not exp.get("allow_repeats"):
        sigs = [json.dumps([c["name"], c.get("arguments")], sort_keys=True) for c in res.tool_calls]
        dup = len(sigs) - len(set(sigs))
        add("no_repeated_calls", dup == 0, f"{dup} repeated identical call(s)" if dup else "")

    tok_limit = rules.get("limits", {}).get("max_tokens_per_case")
    if tok_limit is not None and res.tokens is not None:
        add("token_budget", res.tokens <= tok_limit, f"{res.tokens} tokens (limit {tok_limit})")

    if "final_state" in exp:
        add("final_state", _subset(exp["final_state"], res.final_state),
            f"expected {exp['final_state']}; got {res.final_state}")
    if "handoff" in exp:
        add("handoff", bool(res.handoff) == bool(exp["handoff"]),
            f"expected handoff={exp['handoff']}; got {bool(res.handoff)}")

    low = res.text.lower()
    for s in exp.get("text_contains", []):
        add(f"text_contains:{s}", s.lower() in low)
    if exp.get("text_contains_any"):
        add("text_contains_any", any(s.lower() in low for s in exp["text_contains_any"]),
            f"none of {exp['text_contains_any']}")
    for s in exp.get("text_not_contains", []):
        add(f"text_not_contains:{s}", s.lower() not in low)

    return checks, events
