from agentgate.gate import decide
from agentgate.runner import percentile

RULES = {"categories": {"default": {"min_pass_rate": 0.9}, "security": {"zero_tolerance": True}},
         "limits": {"max_p95_latency_ms": 100, "max_error_rate": 0.1}}


def rec(i, cat, passed=True, events=(), lat=10.0):
    return {"id": i, "category": cat, "passed": passed, "must_never": list(events), "latencies_ms": [lat],
            "repeats": 1, "errors": 0, "judge": None, "failed_checks": []}


def test_overall_average_cannot_hide_a_security_failure():
    recs = [rec(f"g{i}", "general") for i in range(19)] + [rec("s1", "security", passed=False)]
    assert decide(recs, RULES, "full", False, True)["verdict"] == "NO-GO"  # 95% overall, still no-go


def test_must_never_blocks_even_when_case_percentages_pass():
    recs = [rec(f"g{i}", "general") for i in range(20)]
    recs[0]["must_never"] = ["forbidden tool called: delete_account"]
    assert decide(recs, RULES, "full", False, True)["verdict"] == "NO-GO"


def test_latency_limit_and_manual_pending():
    assert decide([rec("a", "general", lat=500)], RULES, "full", False, True)["verdict"] == "NO-GO"
    assert decide([rec("a", "general")], RULES, "full", False, False)["verdict"] == "PENDING MANUAL CHECKS"
    assert decide([rec("a", "general")], RULES, "full", False, True)["verdict"] == "GO"


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10 and percentile([], 95) == 0.0
