"""Config loading: YAML with ${VAR} / ${VAR:-default} substitution, tiny .env reader."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _subst(value):
    if isinstance(value, str):
        def repl(m):
            env = os.environ.get(m.group(1))
            if env:
                return env
            return m.group(2) if m.group(2) is not None else ""
        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: _subst(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst(v) for v in value]
    return value


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return _subst(yaml.safe_load(f) or {})


def load_dataset(path: str) -> list[dict]:
    if Path(path).suffix.lower() == ".csv":
        return _load_csv_dataset(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"] if isinstance(data, dict) else data


def _load_csv_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = {
        row.get(key, "").strip()
        for row in rows
        for key in ("specialty", "acceptable_alt_specialty")
        if row.get(key, "").strip() and row[key].strip().casefold() != "out of scope"
    }
    cases = []
    for index, row in enumerate(rows, start=1):
        test_input = next((row.get(key, "").strip() for key in (
            "input", "query", "message", "text", "symptoms_hinglish", "symptom", "symptoms"
        ) if row.get(key, "").strip()), "")
        case_id = next((row.get(key, "").strip() for key in ("id", "case_id") if row.get(key, "").strip()),
                       f"csv-{index:04d}")
        explicit = next((row.get(key, "").strip() for key in ("expect", "expected")
                         if row.get(key, "").strip()), "")
        if explicit:
            try:
                expected = json.loads(explicit)
            except json.JSONDecodeError:
                expected = {"text_contains_any": [explicit]}
        elif row.get("expected_output", "").strip():
            output = row["expected_output"].strip()
            if output.casefold() == "no_match":
                expected = {"text_equals": ""}
            else:
                expected = {"text_contains_any": [item.strip() for item in output.split("|") if item.strip()]}
        else:
            expected = _csv_route_expectation(row)

        must_not = row.get("must_not", "").strip()
        forbidden_specialties = [
            label for label in labels
            if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", must_not, re.IGNORECASE)
            and re.search(r"(?:top[- ]?1|first|top specialty)", must_not, re.IGNORECASE)
        ]
        case = {
            "id": case_id,
            "category": row.get("category", "").strip() or "default",
            "kind": {"normal": "normal", "edge": "ambiguous", "must_never": "adversarial"}.get(
                row.get("case_type", "").strip().lower(), "normal"),
            "tier": row.get("tier", "").strip() or "smoke",
            "input": test_input,
            "expect": expected,
            "source_constraints": {
                "must_not": must_not,
                "notes": row.get("notes", "").strip(),
            },
        }
        if forbidden_specialties:
            case["must_never"] = {"text_patterns": [rf"^{re.escape(name)}$" for name in forbidden_specialties]}
        cases.append(case)
    return cases


def _csv_route_expectation(row: dict) -> dict:
    specialty = row.get("specialty", "").strip()
    triage = row.get("triage", "").strip().casefold()
    out_of_scope = specialty.casefold() == "out of scope" or triage == "not specified"
    expected = {"final_state": {
        "noMatch": out_of_scope,
        "emergency": {"flag": triage == "emergency"},
    }}
    if out_of_scope:
        expected["final_state"]["decision"] = "abstain"
    elif specialty:
        accepted = [specialty]
        alternate = row.get("acceptable_alt_specialty", "").strip()
        if alternate and alternate not in accepted:
            accepted.append(alternate)
        expected["text_contains_any"] = accepted
    return expected


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
