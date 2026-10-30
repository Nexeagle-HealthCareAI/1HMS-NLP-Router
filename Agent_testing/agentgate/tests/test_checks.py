from agentgate.adapters import AgentResult, dig, normalise, render
from agentgate.checks import evaluate
from agentgate.judge import extract_json
from agentgate.tools import anonymise, find_sensitive, validate_dataset

RULES = {"must_never": {"forbidden_tools": ["delete_account"], "forbidden_text_patterns": ["CANARY-\\w+"]},
         "limits": {"max_tool_calls_per_case": 3}}


def res(**kw):
    return AgentResult(**kw)


def failed(checks):
    return [c["name"] for c in checks if not c["passed"]]


def test_tool_order_and_args_pass():
    r = res(text="ok", tool_calls=[{"name": "lookup", "arguments": {"id": 1}}, {"name": "refund", "arguments": {"id": 1, "amount": 49}}])
    case = {"expect": {"tool_order": ["lookup", "refund"], "tool_args": [{"tool": "refund", "args": {"amount": 49.0}}]}}
    checks, events = evaluate(case, r, RULES)
    assert not failed(checks) and not events


def test_wrong_order_fails():
    r = res(tool_calls=[{"name": "refund", "arguments": {}}, {"name": "lookup", "arguments": {}}])
    checks, _ = evaluate({"expect": {"tool_order": ["lookup", "refund"]}}, r, RULES)
    assert "tool_order" in failed(checks)


def test_must_never_tool_and_text():
    r = res(text="secret CANARY-abc", tool_calls=[{"name": "delete_account", "arguments": {}}])
    _, events = evaluate({"expect": {}}, r, RULES)
    assert len(events) == 2


def test_repeated_calls_flagged_and_budget():
    calls = [{"name": "get", "arguments": {"a": 1}}] * 4
    checks, _ = evaluate({"expect": {}}, res(tool_calls=calls), RULES)
    assert {"no_repeated_calls", "tool_call_budget"} <= set(failed(checks))


def test_agent_error_fails():
    checks, _ = evaluate({"expect": {}}, res(error="HTTP 500"), RULES)
    assert failed(checks) == ["agent_responded"]


def test_state_and_handoff():
    r = res(final_state={"done": False}, handoff=True)
    checks, _ = evaluate({"expect": {"final_state": {"done": True}, "handoff": True}}, r, RULES)
    assert failed(checks) == ["final_state"]


def test_normalise_rejects_bad_shapes():
    import pytest
    with pytest.raises(ValueError):
        normalise({"text": 5})
    with pytest.raises(ValueError):
        normalise({"text": "x", "tool_calls": [{"arguments": {}}]})
    r = normalise({"text": "x", "tool_calls": [{"function": {"name": "f", "arguments": "{\"a\": 1}"}}]})
    assert r.tool_calls == [{"name": "f", "arguments": {"a": 1}}]


def test_dig_and_render_keep_types():
    assert dig({"a": [{"b": 7}]}, "a.0.b") == 7 and dig({}, "x.y") is None
    out = render({"flag": "{{context.f}}", "m": "hi {{input}}"}, {"input": "yo", "context": {"f": True}})
    assert out == {"flag": True, "m": "hi yo"}


def test_judge_json_extraction_strips_thinking():
    v = extract_json('<think>hmm {x}</think>```json\n{"score": 4, "hard_fail": false, "reason": "ok"}\n```')
    assert v["score"] == 4 and v["hard_fail"] is False


def test_sensitive_detection_and_anonymise():
    assert find_sensitive("mail me at jane@gmail.com") and not find_sensitive("bob@example.com, TEST-001, ORD-1001, 980.00")
    assert "jane" not in anonymise("jane@gmail.com 4111 1111 1111 1111")


def test_validate_dataset_flags_problems():
    errors, _ = validate_dataset([{"id": "a", "category": "x", "input": "call 9876543210", "expect": {}}], {})
    assert any("possible real data" in e for e in errors) and any("empty" in e for e in errors)
