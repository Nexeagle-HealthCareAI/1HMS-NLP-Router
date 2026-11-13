"""agentgate CLI: validate | run | calibrate | triage | report."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from agentgate import __version__
from agentgate.adapters import AutoAgent, build_adapter
from agentgate.config import file_hash, load_dataset, load_dotenv, load_yaml
from agentgate.gate import decide
from agentgate.judge import JUDGE_PROMPT_VERSION, Judge
from agentgate.report import build_pdf
from agentgate.report_xlsx import build_xlsx
from agentgate.runner import run_suite
from agentgate.tools import calibrate, triage, validate_dataset

D = {"agent": "config/agent.yaml", "judge": "config/judge.yaml",
     "rules": "config/thresholds.yaml", "dataset": "dataset/golden_cases.json"}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def cmd_validate(a) -> int:
    rules = load_yaml(a.rules)
    errors, warnings = validate_dataset(load_dataset(a.dataset), rules)
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print("dataset valid" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


def cmd_run(a) -> int:
    load_dotenv()
    rules, cases = load_yaml(a.rules), load_dataset(a.dataset)
    errors, _ = validate_dataset(cases, rules)
    if errors:
        print("Dataset has errors; run `agentgate validate` first.")
        return 2
    judge = None
    if a.tier == "full" and not a.no_judge:
        judge = Judge(load_yaml(a.judge))
    if a.source or a.agent == "auto":
        if not a.source:
            print("error: --source is required when using automatic agent discovery", file=sys.stderr)
            return 2
        adapter = AutoAgent(a.source, endpoint=a.endpoint)
    else:
        if a.endpoint:
            print("error: --endpoint requires automatic discovery with --source", file=sys.stderr)
            return 2
        adapter = build_adapter(load_yaml(a.agent or D["agent"]))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    print(f"agentgate {__version__} | tier={a.tier} | agent={adapter.describe()}")
    try:
        records = run_suite(cases, adapter, rules, a.tier, judge, log=print)
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()
    if not records:
        print("No cases selected for this tier.")
        return 2
    gate = decide(records, rules, a.tier, judge is not None, a.rollback_rehearsed)
    records_by_id = {record["id"]: record for record in records}
    report_cases = []
    for case in cases:
        record = records_by_id.get(case["id"])
        if record is None:
            report_cases.append({
                "id": case["id"],
                "input": case.get("input", ""),
                "expected": case.get("expect") or {},
                "source_constraints": {
                    "must_never": case.get("must_never") or {},
                    **(case.get("source_constraints") or {}),
                },
                "produced": {"text": f"Not executed in {a.tier} tier"},
                "report_status": "NOT RUN",
                "match_percentage": None,
                "failed_checks": [],
                "error": None,
                "check_results": [],
            })
            continue
        report_case = dict(record)
        report_case["source_constraints"] = {
            "must_never": case.get("must_never") or {},
            **(case.get("source_constraints") or {}),
        }
        report_case["report_status"] = "PASS" if record["passed"] else "FAIL"
        report_case["error"] = (record.get("produced") or {}).get("error")
        report_cases.append(report_case)
    res = {"meta": {"run_id": run_id, "tier": a.tier, "agent": adapter.describe(),
                    "judge": judge.describe() if judge else "off",
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "dataset_hash": file_hash(a.dataset), "thresholds_hash": file_hash(a.rules),
                    "commit": _commit(), "version": __version__, "python": platform.python_version()},
           "gate": gate, "cases": records, "report_cases": report_cases}
    out = Path(a.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    build_pdf(res, str(out / "report.pdf"))
    build_xlsx(res, str(out / "report.xlsx"))
    latest = Path(a.out) / "latest"
    shutil.copytree(out, latest, dirs_exist_ok=True)
    print("\nCategory results")
    for c in gate["categories"]:
        print(f"  {c['category']:<12} {c['passed']}/{c['cases']}  need {c['required']:.0%}  {'ok' if c['ok'] else 'FAIL'}")
    for c in gate["checks"]:
        print(f"  [{c['status'].upper():<7}] {c['name']}: {c['detail']}")
    print(f"\nVERDICT: {gate['verdict']}\nExcel:   {out / 'report.xlsx'}\nPDF:     {out / 'report.pdf'}")
    return 1 if gate["verdict"] in ("NO-GO", "SMOKE FAIL") else 0


def cmd_calibrate(a) -> int:
    load_dotenv()
    rules = load_yaml(a.rules)
    rate, lines = calibrate(Judge(load_yaml(a.judge)), a.gold,
                            (rules.get("judge") or {}).get("min_agreement", 0.8))
    print("\n".join(lines))
    return 0 if rate >= (rules.get("judge") or {}).get("min_agreement", 0.8) else 1


def cmd_triage(a) -> int:
    item = triage(a.text, a.label, a.category, a.id, a.out)
    print(f"Saved {item['id']} to {a.out} (anonymised). Add an `expect` block, review it, "
          "then move it into the golden dataset.")
    return 0


def cmd_report(a) -> int:
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    base = Path(a.results).with_name("report")
    pdf, excel = str(base.with_suffix(".pdf")), str(base.with_suffix(".xlsx"))
    build_pdf(res, pdf)
    build_xlsx(res, excel)
    print(f"Wrote {excel}\nWrote {pdf}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agentgate", description="Test AI agents before production.")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check the golden dataset (step 2)")
    v.add_argument("--dataset", default=D["dataset"]); v.add_argument("--rules", default=D["rules"])
    v.set_defaults(fn=cmd_validate)

    r = sub.add_parser("run", help="run the framework and write Excel and PDF reports")
    r.add_argument("--tier", choices=["smoke", "full"], default="smoke")
    r.add_argument("--agent", help="agent YAML config, or 'auto' when used with --source")
    r.add_argument("--source", help="Python agent entrypoint file/directory for automatic discovery")
    r.add_argument("--endpoint", help="POST path to test when --source exposes multiple API endpoints")
    r.add_argument("--judge", default=D["judge"])
    r.add_argument("--rules", default=D["rules"]); r.add_argument("--dataset", default=D["dataset"])
    r.add_argument("--out", default="reports")
    r.add_argument("--no-judge", action="store_true", help="skip the LLM judge in the full tier")
    r.add_argument("--rollback-rehearsed", action="store_true", help="confirm rollback was rehearsed")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("calibrate", help="check the judge against labelled examples (step 5)")
    c.add_argument("--judge", default=D["judge"]); c.add_argument("--rules", default=D["rules"])
    c.add_argument("--gold", default="dataset/judge_gold.json")
    c.set_defaults(fn=cmd_calibrate)

    t = sub.add_parser("triage", help="anonymise and queue a production finding (step 8)")
    t.add_argument("--text", required=True)
    t.add_argument("--label", required=True, choices=["bug", "drift", "new_scenario"])
    t.add_argument("--category", required=True); t.add_argument("--id")
    t.add_argument("--out", default="dataset/pending_review.json")
    t.set_defaults(fn=cmd_triage)

    rp = sub.add_parser("report", help="rebuild Excel and PDF reports from results.json")
    rp.add_argument("results"); rp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, KeyError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
