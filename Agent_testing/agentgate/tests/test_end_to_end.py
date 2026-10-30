import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent


def run_cli(tmp_path, *args, buggy=False):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    env["MOCK_AGENT_BUGGY"] = "1" if buggy else "0"
    return subprocess.run([sys.executable, "-m", "agentgate", *args, "--agent", "config/agent.demo.yaml",
                           "--out", str(tmp_path)], cwd=ROOT, env=env, capture_output=True, text=True)


def latest(tmp_path):
    return json.loads((tmp_path / "latest" / "results.json").read_text()), tmp_path / "latest" / "report.pdf"


def test_good_agent_smoke_passes_with_one_page(tmp_path):
    p = run_cli(tmp_path, "run", "--tier", "smoke")
    assert p.returncode == 0, p.stdout + p.stderr
    res, pdf = latest(tmp_path)
    assert res["gate"]["verdict"] == "SMOKE PASS"
    assert len(PdfReader(str(pdf)).pages) == 1


def test_good_agent_full_without_judge_is_pending_manual(tmp_path):
    p = run_cli(tmp_path, "run", "--tier", "full", "--no-judge")
    assert p.returncode == 0, p.stdout + p.stderr
    res, _ = latest(tmp_path)
    assert res["gate"]["verdict"] == "PENDING MANUAL CHECKS"
    p = run_cli(tmp_path, "run", "--tier", "full", "--no-judge", "--rollback-rehearsed")
    assert latest(tmp_path)[0]["gate"]["verdict"] == "GO"


def test_buggy_agent_is_no_go_and_never_prints_the_secret(tmp_path):
    p = run_cli(tmp_path, "run", "--tier", "full", "--no-judge", buggy=True)
    assert p.returncode == 1
    res, pdf = latest(tmp_path)
    assert res["gate"]["verdict"] == "NO-GO"
    assert res["gate"]["totals"]["must_never_events"] >= 1
    assert len(PdfReader(str(pdf)).pages) == 1
    assert "CANARY-7f3a" not in json.dumps(res) and "CANARY-7f3a" not in PdfReader(str(pdf)).pages[0].extract_text()


def test_unreachable_judge_fails_closed(tmp_path):
    env = {"JUDGE_BASE_URL": "http://127.0.0.1:9/v1"}
    os.environ.update(env)
    try:
        p = run_cli(tmp_path, "run", "--tier", "full")
    finally:
        os.environ.pop("JUDGE_BASE_URL")
    assert p.returncode == 1 and "NO-GO" in p.stdout


def test_dataset_validates():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    p = subprocess.run([sys.executable, "-m", "agentgate", "validate"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout
