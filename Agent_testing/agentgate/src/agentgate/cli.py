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
from agentgate.adapters import build_adapter
from agentgate.config import file_hash, load_dataset, load_dotenv, load_yaml
from agentgate.gate import decide
from agentgate.judge import JUDGE_PROMPT_VERSION, Judge
from agentgate.report import build_pdf
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
    adapter = build_adapter(load_yaml(a.agent))
    judge = None
    if a.tier == "full" and not a.no_judge:
        judge = Judge(load_yaml(a.judge))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    print(f"agentgate {__version__} | tier={a.tier} | agent={adapter.describe()}")
    records = run_suite(cases, adapter, rules, a.tier, judge, log=print)
    if not records:
        print("No cases selected for this tier.")
        return 2
    gate = decide(records, rules, a.tier, judge is not None, a.rollback_rehearsed)
    res = {"meta": {"run_id": run_id, "tier": a.tier, "agent": adapter.describe(),
                    "judge": judge.describe() if judge else "off",
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "dataset_hash": file_hash(a.dataset), "thresholds_hash": file_hash(a.rules),
                    "commit": _commit(), "version": __version__, "python": platform.python_version()},
           "gate": gate, "cases": records}
    out = Path(a.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    build_pdf(res, str(out / "report.pdf"))
    latest = Path(a.out) / "latest"
    shutil.copytree(out, latest, dirs_exist_ok=True)
    print("\nCategory results")
    for c in gate["categories"]:
        print(f"  {c['category']:<12} {c['passed']}/{c['cases']}  need {c['required']:.0%}  {'ok' if c['ok'] else 'FAIL'}")
    for c in gate["checks"]:
        print(f"  [{c['status'].upper():<7}] {c['name']}: {c['detail']}")
    print(f"\nVERDICT: {gate['verdict']}\nReport:  {out / 'report.pdf'}")
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
    out = str(Path(a.results).with_name("report.pdf"))
    build_pdf(res, out)
    print(f"Wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agentgate", description="Test AI agents before production.")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check the golden dataset (step 2)")
    v.add_argument("--dataset", default=D["dataset"]); v.add_argument("--rules", default=D["rules"])
    v.set_defaults(fn=cmd_validate)

    r = sub.add_parser("run", help="run the framework and write a one-page report")
    r.add_argument("--tier", choices=["smoke", "full"], default="smoke")
    r.add_argument("--agent", default=D["agent"]); r.add_argument("--judge", default=D["judge"])
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

    rp = sub.add_parser("report", help="rebuild report.pdf from a results.json")
    rp.add_argument("results"); rp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
