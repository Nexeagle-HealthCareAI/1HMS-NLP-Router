# How agentgate maps to the 8-step framework

| Step | What the framework asks | Where it lives in agentgate |
|---|---|---|
| 1. Define criteria | Pass/fail per category, must-never list, sign-off | `config/thresholds.yaml` (hashed into every report; `signed_off_by` field) |
| 2. Golden dataset | 20-50 versioned cases incl. adversarial, expected tool calls and state | `dataset/golden_cases.json`; `agentgate validate` checks structure, sensitive data, coverage |
| 3. Unit tests, mock tools | Cheap deterministic checks on every PR, schema checks, tool failures | `--tier smoke`; output-contract validation in `adapters.py`; `tool_failure` cases; CI workflow |
| 4. Trajectory checks | Tool choice, arguments, order, repeated calls, repeat runs | `checks.py`; `run.repeats` / `min_passes` in thresholds |
| 5. LLM judge + human review | Rubrics, hard-fail caps, calibrated judge | `judge.py` (runs only after deterministic checks pass, cached); `agentgate calibrate` |
| 6. Security + load | Prompt injection, leaks, p95, error rate, step/token limits | `security-*` cases (zero tolerance), `must_never` rules, `limits` block. **Not covered:** red teaming by a person and real load tests; do those separately |
| 7. Release gate | Per-category thresholds, zero must-never, rollback rehearsed | `gate.py`; verdict GO / NO-GO / PENDING MANUAL CHECKS |
| 8. Release, monitor, feed back | Triage findings into the dataset | `agentgate triage` (anonymises, labels bug/drift/new_scenario, queues for review) |

## Optimisations included
- **Tiered runs:** smoke on every PR (no judge, one run per case); full before release.
- **Pinned runs:** every `results.json` records dataset hash, thresholds hash, judge model and prompt version, commit.
- **Security cases in CI:** adversarial cases are in the smoke tier.
- **Frozen thresholds:** hashed into the report so edits are visible.
- **Small-sample guard:** warning under `min_cases_per_category`; must-never and security are pass/fail.
- **Judge cost:** judge only sees cases that passed deterministic checks; results cached by content.
- **Triage before dataset:** production findings are labelled and anonymised first.
