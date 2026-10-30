# Contributing

Thanks for helping. The most valuable contribution is a **new golden test case** from a real failure.

1. Fork, then `pip install -e ".[dev]"`.
2. Add cases to `dataset/golden_cases.json` using fake data and test accounts only.
   Run `agentgate validate`; it rejects emails, card-like numbers and phone-like numbers.
3. Assert on actions and final state (`tools_called`, `tool_order`, `tool_args`, `final_state`,
   `handoff`), not exact wording.
4. `pytest -q` and `agentgate run --tier smoke --agent config/agent.demo.yaml` must pass.
5. Open a pull request. Changes to `config/thresholds.yaml` need a written reason, because
   thresholds are frozen quality criteria.

Never commit `.env`, API keys, real customer data or internal names.
