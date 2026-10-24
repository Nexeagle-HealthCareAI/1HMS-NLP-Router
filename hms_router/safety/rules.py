"""The red-flag rule file: one source of truth for every Safety Guard.

hms_router/data/red_flags.csv holds (phrase, english, canonical, flag, origin). /route-symptom reads it
today; /chat (via HMSRouterAgent(emergency_rules=...)) and the WhatsApp bot's app/safety.py should read
the same file next, so a rule added once protects every channel.

Starting point only - every row needs clinician review before patients rely on it.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..severity import EMERGENCY, SEVERE_FLAG

RED_FLAGS_PATH = Path(__file__).resolve().parent.parent / "data" / "red_flags.csv"

Rule = tuple[str, str, str, str]            # (phrase, english, canonical, flag) - severity.py's shape

# Where to send someone after a red flag, by canonical red-flag name. Emergency Medicine is always first.
_RECOMMEND: dict[str, list[str]] = {
    "suicidal thoughts": ["emergencymedicine", "psychiatry"],
    "bleeding in pregnancy": ["emergencymedicine", "gynecology"],
    "chest pain": ["emergencymedicine", "cardiology"],
    "heart attack": ["emergencymedicine", "cardiology"],
    "seizure": ["emergencymedicine", "neurology"],
    "daura": ["emergencymedicine", "neurology", "cardiology"],
    "stroke": ["emergencymedicine", "neurology"],
    "facial droop": ["emergencymedicine", "neurology"],
    "numbness": ["emergencymedicine", "neurology"],
    "sudden severe headache": ["emergencymedicine", "neurology"],
    "vision loss": ["emergencymedicine", "ophthalmology"],
    "breathlessness": ["emergencymedicine", "pulmonology"],
    "head injury": ["emergencymedicine", "neurosurgery", "neurology"],
}


def load_red_flags(path: str | Path = RED_FLAGS_PATH) -> list[Rule]:
    with open(path, newline="", encoding="utf-8") as f:
        rules = [(r["phrase"].strip().lower(), r["english"].strip(), r["canonical"].strip().lower(),
                  r["flag"].strip().upper()) for r in csv.DictReader(f) if r["phrase"].strip()]
    bad = [r for r in rules if r[3] not in (EMERGENCY, SEVERE_FLAG)]
    if bad:
        raise ValueError(f"red_flags.csv: unknown flag in {bad[:3]}")
    if not rules:
        raise ValueError("red_flags.csv is empty - refusing to run without a safety net")
    return rules


def recommended_specialties(canonicals: list[str]) -> list[str]:
    out = ["emergencymedicine"]
    for c in canonicals:
        for sid in _RECOMMEND.get(c, []):
            if sid not in out:
                out.append(sid)
    return out
