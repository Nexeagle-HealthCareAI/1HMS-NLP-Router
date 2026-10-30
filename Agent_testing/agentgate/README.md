# agentgate

Test any AI agent before production with the 8-step framework (criteria, golden dataset,
layered tests, LLM judge, release gate, gradual release). Clone it, point one config file at
your agent's API, run one command, and get a **one-page PDF report** with a clear
GO / NO-GO verdict.

## Try it in 60 seconds (no agent, no LLM needed)

```bash
git clone <your-repo-url> agentgate && cd agentgate
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

agentgate validate                                        # check the golden dataset
agentgate run --tier smoke --agent config/agent.demo.yaml # fast tier, deterministic only
```

Open `reports/latest/report.pdf`. See the failure path with a deliberately flawed agent:

```bash
MOCK_AGENT_BUGGY=1 agentgate run --tier full --no-judge --agent config/agent.demo.yaml
```

(Windows PowerShell: `$env:MOCK_AGENT_BUGGY="1"` first.) Sample reports are in `docs/`.

## Test your own agent

1. **Point it at your API.** Edit `config/agent.yaml`: URL, auth header (from env), a request
   template using `{{input}}` / `{{context.key}}`, and a `response_map` that says where your
   JSON response keeps the reply, tool calls, final state and handoff flag. Python agent instead?
   Use `type: python` as in `config/agent.demo.yaml`.
2. **Agree the criteria.** Edit `config/thresholds.yaml` (per-category thresholds, must-never
   tools and text patterns, limits) and get it signed off before the first run.
3. **Write your cases.** Replace `dataset/golden_cases.json` with yours. Start with 20-50:
   normal, ambiguous, tool failure, adversarial. Use fake data and test accounts only.
   Assert on tool calls, arguments, order and final state, not exact wording.
4. **Run.** `agentgate run --tier smoke` on every change; `agentgate run --tier full` before release.

## Commands

| Command | Purpose |
|---|---|
| `agentgate validate` | Check dataset structure, spot real-looking personal data, flag coverage gaps |
| `agentgate run --tier smoke` | Deterministic checks, one run per smoke case. Use in CI |
| `agentgate run --tier full` | All cases, repeated runs, LLM judge, release gate. Add `--no-judge` to skip the judge, `--rollback-rehearsed` once you have rehearsed rollback |
| `agentgate calibrate` | Check the judge against labelled examples (`dataset/judge_gold.json`) before trusting it |
| `agentgate triage --text ... --label bug --category refunds` | Anonymise and queue a production finding for review (step 8) |
| `agentgate report reports/latest/results.json` | Rebuild the PDF |

Exit code is 1 on NO-GO or SMOKE FAIL, so CI can block a merge.

## The LLM judge (Qwen or any other model)

The judge talks to any OpenAI-compatible `/chat/completions` endpoint. Set it in `.env`
(copy `.env.example`): `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`.

- Local server (Ollama, llama.cpp `llama-server`, LM Studio): use its `/v1` URL and the exact
  model name it reports.
- Qwen3.7-Plus: public model trackers list it as a hosted, API-only model, not open weights.
  If you reach it through Alibaba Cloud Model Studio or your own gateway, use that
  OpenAI-compatible URL and key. Your test inputs and agent replies are then sent to that
  provider, so keep real customer data out of the dataset. For a fully offline judge, use a
  local open-weight model instead.
- If the model emits reasoning blocks they are stripped. If your server needs a flag to turn
  thinking off, add it under `extra_body` in `config/judge.yaml`.
- Run `agentgate calibrate` first. In the full tier, a judge that cannot be reached fails the
  gate instead of silently passing.

## What you get after a run

`reports/<run-id>/report.pdf` (one page, always) and `results.json` (every case, check and
judge result), plus a copy in `reports/latest/`. The report lists the verdict, per-category
results, gate checks, failed cases, next actions, and the pinned versions (dataset hash,
thresholds hash, judge model, commit) so a result can be traced and reproduced.

## Layout

```
config/    agent.yaml (API file), judge.yaml, thresholds.yaml
dataset/   golden_cases.json, judge_gold.json
demo/      mock_agent.py (rule-based demo agent; --serve 8000 for HTTP)
src/agentgate/  adapters, checks, judge, runner, gate, report, tools, cli
tests/     tests for agentgate itself
docs/      framework mapping, sample reports
```

## Limits (read before relying on it)

- It automates steps 1-5 and 7, and the repeatable part of 6 and 8. It does **not** replace a
  human red team, real load testing, a rehearsed rollback, or domain-expert review.
- A small dataset gives noisy percentages. Treat must-never and security cases as pass/fail.
- A passing judge score does not verify numbers or IDs; use `final_state` and `tool_args` for those.
- Keep secrets in `.env` or CI secrets, never in the dataset or config files.

MIT licence. Contributions welcome: see `CONTRIBUTING.md`.
