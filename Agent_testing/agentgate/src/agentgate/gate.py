"""Release gate (step 7): thresholds from step 1 -> a clear verdict."""
from __future__ import annotations

from agentgate.runner import percentile


def _check(name, status, detail=""):
    return {"name": name, "status": status, "detail": detail}


def decide(records: list[dict], rules: dict, tier: str, judge_enabled: bool,
           rollback_rehearsed: bool = False) -> dict:
    checks = []
    cat_rules = rules.get("categories") or {}
    default_rule = cat_rules.get("default", {"min_pass_rate": 0.9})

    # Per-category thresholds (never one blended score).
    categories = []
    for cat in sorted({r["category"] for r in records}):
        rs = [r for r in records if r["category"] == cat]
        passed = sum(1 for r in rs if r["passed"])
        rate = passed / len(rs)
        rule = cat_rules.get(cat, default_rule)
        zero = bool(rule.get("zero_tolerance"))
        required = 1.0 if zero else float(rule.get("min_pass_rate", default_rule.get("min_pass_rate", 0.9)))
        ok = rate >= required
        categories.append({"category": cat, "cases": len(rs), "passed": passed, "rate": rate,
                           "required": required, "zero_tolerance": zero, "ok": ok})
        checks.append(_check(f"Category '{cat}'", "pass" if ok else "fail",
                             f"{passed}/{len(rs)} passed, need {required:.0%}"
                             f"{' (zero tolerance)' if zero else ''}"))

    # Must-never events: zero tolerance.
    events = [(r["id"], e) for r in records for e in r["must_never"]]
    checks.append(_check("Must-never events", "fail" if events else "pass",
                         f"{len(events)} event(s): " + "; ".join(f"{i}: {e}" for i, e in events[:3])
                         if events else "0 events"))

    # Robustness / cost limits.
    limits = rules.get("limits") or {}
    lat = [x for r in records for x in r["latencies_ms"]]
    calls = sum(r["repeats"] for r in records)
    errs = sum(r["errors"] for r in records)
    p95 = percentile(lat, 95)
    if "max_p95_latency_ms" in limits:
        checks.append(_check("p95 latency", "pass" if p95 <= limits["max_p95_latency_ms"] else "fail",
                             f"{p95:.0f} ms (limit {limits['max_p95_latency_ms']})"))
    err_rate = errs / calls if calls else 0.0
    if "max_error_rate" in limits:
        checks.append(_check("Error rate", "pass" if err_rate <= limits["max_error_rate"] else "fail",
                             f"{err_rate:.1%} (limit {limits['max_error_rate']:.0%})"))

    # Sample-size warning (small samples make percentages noisy).
    min_cases = rules.get("min_cases_per_category")
    small = [f"{c['category']}({c['cases']})" for c in categories
             if min_cases and c["cases"] < min_cases and not c["zero_tolerance"]]
    if small:
        checks.append(_check("Sample size", "warn",
                             f"under {min_cases} cases: {', '.join(small)}; percentages are noisy"))

    judge_avg = None
    if tier == "full":
        jcfg = rules.get("judge") or {}
        rubric_cases = [r for r in records if r["judge"] is not None]
        scored = [r["judge"] for r in rubric_cases if r["judge"].get("status") == "scored"]
        if not judge_enabled:
            checks.append(_check("LLM judge", "warn", "skipped (--no-judge): quality not graded"))
        elif not rubric_cases:
            checks.append(_check("LLM judge", "warn", "no case with a rubric reached the judge"))
        elif len(scored) < len(rubric_cases):
            msg = next((r["judge"].get("reason", "") for r in rubric_cases
                        if r["judge"].get("status") != "scored"), "")
            checks.append(_check("LLM judge", "fail", f"{len(rubric_cases) - len(scored)} case(s) not graded: {msg}"))
        else:
            judge_avg = sum(j["score"] for j in scored) / len(scored)
            need = jcfg.get("min_avg_score", 3.5)
            checks.append(_check("LLM judge", "pass" if judge_avg >= need else "fail",
                                 f"avg {judge_avg:.2f}/5 over {len(scored)} case(s) (need {need})"))
        checks.append(_check("Rollback rehearsed", "pass" if rollback_rehearsed else "pending",
                             "confirmed by operator" if rollback_rehearsed
                             else "manual: re-run with --rollback-rehearsed once done"))

    statuses = {c["status"] for c in checks}
    if tier == "smoke":
        verdict = "SMOKE FAIL" if "fail" in statuses else "SMOKE PASS"
    elif "fail" in statuses:
        verdict = "NO-GO"
    elif "pending" in statuses:
        verdict = "PENDING MANUAL CHECKS"
    else:
        verdict = "GO"

    cases = len(records)
    checks_total = sum(r.get("checks_total", 0) for r in records)
    checks_passed = sum(r.get("checks_passed", 0) for r in records)
    return {
        "verdict": verdict, "checks": checks, "categories": categories,
        "totals": {"cases": cases, "passed": sum(1 for r in records if r["passed"]),
                   "must_never_events": len(events), "p95_latency_ms": round(p95, 1),
                   "error_rate": round(err_rate, 4), "judge_avg": judge_avg,
                   "checks_passed": checks_passed, "checks_total": checks_total,
                   "match_percentage": round(100 * checks_passed / checks_total, 1)
                   if checks_total else 0.0},
    }
