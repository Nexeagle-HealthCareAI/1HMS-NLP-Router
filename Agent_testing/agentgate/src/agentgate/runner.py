"""Runs the dataset against the agent: repeats, deterministic checks, then judge."""
from __future__ import annotations

from agentgate.checks import evaluate


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(-(-p / 100 * len(ordered) // 1)))  # nearest-rank
    return ordered[min(rank, len(ordered)) - 1]


def select_cases(cases: list[dict], tier: str) -> list[dict]:
    if tier == "smoke":
        return [c for c in cases if c.get("tier", "full") == "smoke"]
    return list(cases)


def run_suite(cases, adapter, rules, tier, judge=None, log=lambda *_: None) -> list[dict]:
    run_cfg = rules.get("run", {})
    default_repeats = run_cfg.get("smoke_repeats", 1) if tier == "smoke" else run_cfg.get("repeats", 3)
    min_passes = run_cfg.get("min_passes", 2)
    judge_cfg = rules.get("judge", {})
    fail_below = judge_cfg.get("fail_below", 2)
    records = []

    for case in select_cases(cases, tier):
        cat_rule = (rules.get("categories") or {}).get(case["category"], {})
        repeats = int(case.get("repeat", default_repeats))
        need = repeats if cat_rule.get("zero_tolerance") else min(min_passes, repeats)
        passes, errors, events = 0, 0, []
        latencies, failed, tokens = [], [], []
        first_pass_text, sample = None, None

        for _ in range(repeats):
            res = adapter(case)
            checks, ev = evaluate(case, res, rules)
            latencies.append(res.latency_ms)
            if res.error:
                errors += 1
            if res.tokens is not None:
                tokens.append(res.tokens)
            bad = [c for c in checks if not c["passed"]]
            for e in ev:
                if e not in events:
                    events.append(e)
            if not bad:
                passes += 1
                if first_pass_text is None:
                    first_pass_text = res.text
            else:
                for c in bad:
                    line = f"{c['name']}: {c['detail']}".rstrip(": ")
                    if line not in failed:
                        failed.append(line)
                if sample is None and not ev:
                    sample = res.text[:160]

        passed = passes >= need and not events
        rec = {
            "id": case["id"], "category": case["category"], "kind": case.get("kind", ""),
            "tier": case.get("tier", "full"), "repeats": repeats, "passes": passes,
            "required_passes": need, "passed": passed, "must_never": events,
            "failed_checks": failed[:6], "latencies_ms": [round(x, 1) for x in latencies],
            "errors": errors, "tokens": tokens, "judge": None,
            "sample": "[withheld: forbidden content]" if events else sample,
        }

        if judge is not None and case.get("rubric") and passed and first_pass_text is not None:
            verdict = judge.grade(case, first_pass_text)
            rec["judge"] = verdict
            if verdict.get("status") == "scored":
                if verdict.get("hard_fail") or verdict["score"] < fail_below:
                    rec["passed"] = False
                    rec["failed_checks"].append(
                        f"judge: score {verdict['score']}"
                        f"{', hard-fail' if verdict.get('hard_fail') else ''} - {verdict.get('reason', '')}")
        log(f"  {'PASS' if rec['passed'] else 'FAIL'}  {rec['id']}  ({passes}/{repeats})")
        records.append(rec)
    return records
