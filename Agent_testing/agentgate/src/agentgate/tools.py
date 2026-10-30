"""Dataset validation (step 2), judge calibration (step 5), production triage (step 8)."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

KINDS = {"normal", "ambiguous", "tool_failure", "adversarial"}
EMAIL = re.compile(r"[\w.+-]+@([\w-]+\.)+[\w-]+")
SAFE_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "test.invalid")
LONG_DIGITS = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
PHONE = re.compile(r"(?<![\w.])\+?\d[\d -]{8,}\d(?![\w.])")


def find_sensitive(text: str) -> list[str]:
    hits = []
    for m in EMAIL.finditer(text):
        if not m.group(0).lower().endswith(SAFE_EMAIL_DOMAINS):
            hits.append("email address")
    if LONG_DIGITS.search(text):
        hits.append("long digit sequence (card/ID-like)")
    elif PHONE.search(text):
        hits.append("phone-like number")
    return hits


def anonymise(text: str) -> str:
    text = EMAIL.sub("user@example.com", text)
    text = LONG_DIGITS.sub("[NUMBER]", text)
    return PHONE.sub("[PHONE]", text)


def validate_dataset(cases: list[dict], rules: dict) -> tuple[list[str], list[str]]:
    errors, warnings, ids = [], [], set()
    for i, c in enumerate(cases):
        cid = c.get("id", f"#{i}")
        for key in ("id", "category", "input", "expect"):
            if key not in c:
                errors.append(f"{cid}: missing '{key}'")
        if c.get("id") in ids:
            errors.append(f"{cid}: duplicate id")
        ids.add(c.get("id"))
        if c.get("kind") and c["kind"] not in KINDS:
            errors.append(f"{cid}: kind must be one of {sorted(KINDS)}")
        if c.get("tier", "full") not in ("smoke", "full"):
            errors.append(f"{cid}: tier must be smoke or full")
        if not (c.get("expect") or {}):
            errors.append(f"{cid}: 'expect' is empty; assert tool calls or final state, not only text")
        blob = json.dumps(c)
        for hit in find_sensitive(blob):
            errors.append(f"{cid}: possible real data ({hit}); mask or use test accounts")
    min_cases = rules.get("min_cases_per_category", 0)
    cats = {}
    for c in cases:
        cats[c.get("category")] = cats.get(c.get("category"), 0) + 1
    thin = [f"{k}({v})" for k, v in sorted(cats.items(), key=lambda kv: str(kv[0])) if v < min_cases]
    if thin:
        warnings.append(f"categories under {min_cases} cases: {', '.join(thin)}")
    kinds = {c.get("kind") for c in cases}
    for need in ("tool_failure", "adversarial", "ambiguous"):
        if need not in kinds:
            warnings.append(f"no '{need}' case in the dataset")
    if not any(c.get("kind") == "adversarial" and c.get("tier") == "smoke" for c in cases):
        warnings.append("no adversarial case in the smoke tier; run a few security cases on every PR")
    return errors, warnings


def calibrate(judge, gold_path: str, min_agreement: float) -> tuple[float, list[str]]:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))["examples"]
    lines, agree = [], 0
    for g in gold:
        v = judge.grade_text({"criteria": g["criteria"], "hard_fail": g.get("hard_fail", "none")},
                             g["request"], g["reply"])
        if v.get("status") != "scored":
            lines.append(f"{g['id']}: judge error - {v.get('reason')}")
            continue
        ok = (g["min_score"] <= v["score"] <= g["max_score"]
              and bool(v["hard_fail"]) == bool(g.get("expect_hard_fail", False)))
        agree += ok
        lines.append(f"{g['id']}: {'agree' if ok else 'DISAGREE'} (judge {v['score']}, "
                     f"hard_fail={v['hard_fail']}; expected {g['min_score']}-{g['max_score']})")
    rate = agree / len(gold) if gold else 0.0
    lines.append(f"agreement {rate:.0%} (need {min_agreement:.0%})")
    return rate, lines


def triage(text: str, label: str, category: str, case_id: str | None, out_path: str) -> dict:
    clean = anonymise(text)
    cid = case_id or "prod-" + hashlib.sha256(clean.encode()).hexdigest()[:8]
    item = {"id": cid, "label": label, "category": category, "input": clean,
            "added": date.today().isoformat(), "status": "needs_expected_outcome"}
    p = Path(out_path)
    items = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    items.append(item)
    p.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return item
