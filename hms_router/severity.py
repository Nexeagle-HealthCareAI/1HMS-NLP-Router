"""Step 5 - severity classification & emergency flagging.

Standalone. Input is the combined Step 4 state as a plain dict:
    {"symptoms": [...], "negated_symptoms": [...], "intensity": "severe"|"moderate"|"mild"|None,
     "duration": "for 3 days"|None, ...}
plus (optionally) the normalized user messages of the session, so the hardcoded phrase list can run on raw text.

Order of operations (per the architecture doc):
  1. EMERGENCY SAFETY NET - hardcoded phrase/symptom list. Runs first and independently of any model.
  2. Rule-based score (0-100) - symptom weights x intensity, + duration, + combo bonuses.
  3. Optional ML score (e.g. XGBoost) - may RAISE the final score, never lower it.
  4. Band -> level: 0-30 MILD | 31-60 MODERATE | 61-85 SEVERE | 86-100 EMERGENCY

This is triage support, not diagnosis. The weights and phrase list below are a starting point:
have a clinician review them before real patients depend on them.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

EMERGENCY, SEVERE_FLAG = "EMERGENCY", "SEVERE"

# ----------------------------------------------------------------------------- 1. safety net
# (phrase, english meaning, canonical symptom, flag). Phrases are matched on content words only
# (particles dropped), in order, with a small gap allowed and 1-letter spelling slack.
EMERGENCY_RULES: list[tuple[str, str, str, str]] = [
    ("seene dard", "Chest pain", "chest pain", EMERGENCY),
    ("seena dard", "Chest pain", "chest pain", EMERGENCY),
    ("chest pain", "Chest pain", "chest pain", EMERGENCY),
    ("saans nahi aa", "Breathlessness", "breathlessness", EMERGENCY),
    ("saans takleef", "Breathlessness", "breathlessness", EMERGENCY),
    ("saans phoolna", "Breathlessness", "breathlessness", EMERGENCY),
    ("short breath", "Breathlessness", "breathlessness", EMERGENCY),
    ("difficulty breathing", "Breathlessness", "breathlessness", EMERGENCY),
    ("behosh", "Lost consciousness", "unconscious", EMERGENCY),
    ("unconscious", "Lost consciousness", "unconscious", EMERGENCY),
    ("zyada khoon aa", "Excessive bleeding", "excessive bleeding", EMERGENCY),
    ("bahut khoon", "Excessive bleeding", "excessive bleeding", EMERGENCY),
    ("heavy bleeding", "Excessive bleeding", "excessive bleeding", EMERGENCY),
    ("laqwa", "Stroke", "stroke", EMERGENCY),
    ("muh tedha", "Facial drooping / stroke", "facial droop", EMERGENCY),
    ("face drooping", "Facial drooping / stroke", "facial droop", EMERGENCY),
    ("aankhon aage andhera", "Vision loss / blackout", "vision loss", EMERGENCY),
    ("vision loss", "Vision loss / blackout", "vision loss", EMERGENCY),
    ("haath pair sune", "Limb numbness", "numbness", EMERGENCY),
    ("haath pair sunn", "Limb numbness", "numbness", EMERGENCY),
    ("numbness", "Limb numbness", "numbness", EMERGENCY),
    ("sar achanak tez dard", "Sudden severe headache", "sudden severe headache", EMERGENCY),
    ("sir achanak tez dard", "Sudden severe headache", "sudden severe headache", EMERGENCY),
    ("sudden severe headache", "Sudden severe headache", "sudden severe headache", EMERGENCY),
    ("ulti khoon", "Blood in vomit", "blood in vomit", EMERGENCY),
    ("khoon ulti", "Blood in vomit", "blood in vomit", EMERGENCY),       # "khoon ki ulti" (word order flipped)
    ("blood vomit", "Blood in vomit", "blood in vomit", EMERGENCY),
    ("vomiting blood", "Blood in vomit", "blood in vomit", EMERGENCY),
    ("vomit blood", "Blood in vomit", "blood in vomit", EMERGENCY),
    ("chhaati dard", "Chest pain", "chest pain", EMERGENCY),
    ("chhati dard", "Chest pain", "chest pain", EMERGENCY),
    ("saans lene takleef", "Breathlessness", "breathlessness", EMERGENCY),
    ("can t breathe", "Breathlessness", "breathlessness", EMERGENCY),
    ("cant breathe", "Breathlessness", "breathlessness", EMERGENCY),
    ("cannot breathe", "Breathlessness", "breathlessness", EMERGENCY),
    ("trouble breathing", "Breathlessness", "breathlessness", EMERGENCY),
]


def load_extra_rules(path) -> list[tuple[str, str, str, str]]:
    """emergency_keywords.csv (phrase,english,canonical,flag). ADDITIVE ONLY: the built-in list above can never be
    removed or downgraded by a data file - the safety net must not depend on a file being intact."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["phrase"].strip().lower(), r["english"].strip(), r["canonical"].strip().lower(),
                 r["flag"].strip().upper()) for r in csv.DictReader(f)
                if r["phrase"].strip() and r["flag"].strip().upper() in (EMERGENCY, SEVERE_FLAG)]


EMERGENCY_CANONICAL = {r[2]: r[1] for r in EMERGENCY_RULES}

_PARTICLES = {"me", "mein", "mei", "main", "mai", "ka", "ki", "ke", "ko", "se", "hai", "hain", "ho", "hoon",
              "raha", "rahi", "rahe", "hua", "hui", "tha", "thi", "gaya", "gayi", "gaye", "ne", "a", "the", "of"}
_MAX_GAP = 2   # words allowed between consecutive phrase words


def _close(a: str, b: str) -> bool:
    """Equal, or one edit apart (both >=4 letters). Recall matters more than precision for emergencies."""
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4 or abs(len(a) - len(b)) > 1:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 1


def _phrase_in(words: list[str], phrase: list[str]) -> bool:
    def walk(start: int, k: int) -> bool:
        if k == len(phrase):
            return True
        for i in range(start, min(start + (1 if k == 0 else _MAX_GAP + 1), len(words))):
            if _close(words[i], phrase[k]) and walk(i + 1, k + 1):
                return True
        return False
    first = [i for i, w in enumerate(words) if _close(w, phrase[0])]
    return any(walk(i, 0) for i in first)


def emergency_check(state: dict, texts: list[str] | None = None, rules: list | None = None) -> list[dict]:
    """Hardcoded safety net. Returns the list of hits (empty = nothing found). No model involved."""
    rules = rules or EMERGENCY_RULES
    canonical = {r[2]: r[1] for r in rules if r[3] == EMERGENCY}
    symptoms = set(state.get("symptoms") or [])
    negated = set(state.get("negated_symptoms") or []) - symptoms     # user explicitly denied it
    hits: dict[str, dict] = {}

    for canon in symptoms & canonical.keys():                          # channel A: Step 3/4 canonical symptoms
        hits[canon] = {"english": canonical[canon], "flag": EMERGENCY, "via": "symptom"}

    for text in texts or []:                                           # channel B: raw normalized text
        words = [w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if w not in _PARTICLES or w == "nahi"]
        for phrase, english, canon, flag in rules:
            if canon in negated or canon in hits:
                continue
            if _phrase_in(words, [w for w in phrase.split() if w not in _PARTICLES]):
                hits[canon] = {"english": english, "flag": flag, "via": "text", "phrase": phrase}

    # combination red flags (flag SEVERE, not EMERGENCY): stomach pain + fever, at least moderate
    if {"stomach pain", "fever"} <= symptoms and state.get("intensity") in ("moderate", "severe"):
        hits["appendicitis risk"] = {"english": "Appendicitis risk (stomach pain + fever)", "flag": SEVERE_FLAG,
                                     "via": "combo"}
    return [{"canonical": k, **v} for k, v in hits.items()]


# ----------------------------------------------------------------------------- 2. rule-based score
SYMPTOM_WEIGHTS = {
    "itching": 10, "constipation": 15, "cold": 15, "cough": 20, "fatigue": 20, "sore throat": 20,
    "nausea": 25, "weakness": 25, "back pain": 25, "pain": 25, "swelling": 25,
    "headache": 30, "diarrhea": 30, "fever": 35, "vomiting": 35, "dizziness": 35, "stomach pain": 40,
    "blood": 50,
}
_INTENSITY_MULT = {"mild": 0.8, "moderate": 1.0, "severe": 1.25, None: 1.0}
_INTENSITY_LEVEL = {"mild": 1, "moderate": 2, "severe": 3}
_COMBOS = [({"fever", "vomiting"}, 10, "fever with vomiting (dehydration/infection risk)"),
           ({"stomach pain", "vomiting", "fever"}, 10, "abdominal pain + vomiting + fever cluster"),
           ({"headache", "vomiting"}, 8, "headache with vomiting"),
           ({"headache", "dizziness"}, 6, "headache with dizziness")]
_UNIT_DAYS = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}
RULE_CAP = 85          # only the safety net (or a model) can push a case into EMERGENCY


def duration_days(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"(\d+)\s+(hour|day|week|month|year)", text)
    if m:
        return int(m[1]) * _UNIT_DAYS[m[2]]
    if "last week" in text:
        return 7
    if "yesterday" in text or "last night" in text:
        return 1
    return 0.5 if "since" in text else None


def rule_score(state: dict) -> tuple[float, list[str]]:
    syms = [s for s in dict.fromkeys(state.get("symptoms") or []) if s in SYMPTOM_WEIGHTS]
    factors: list[str] = []
    if not syms:
        return 0.0, ["no scorable symptoms"]
    w = sorted((SYMPTOM_WEIGHTS[s] for s in syms), reverse=True)
    score = w[0] + 0.25 * sum(w[1:])                               # worst symptom + a share of the rest
    factors.append(f"base {score:.0f} from {', '.join(syms)}")
    mult = _INTENSITY_MULT.get(state.get("intensity"), 1.0)
    if mult != 1.0:
        factors.append(f"intensity '{state['intensity']}' x{mult}")
    score *= mult
    matched = [(bonus, why) for combo, bonus, why in _COMBOS if combo <= set(syms)]
    if matched:                                                    # overlapping combos count once (the largest)
        bonus, why = max(matched)
        score += bonus
        factors.append(f"+{bonus} {why}")
    days = duration_days(state.get("duration"))
    if days is not None and days >= 3:
        bonus = 12 if days >= 30 else 10 if days >= 14 else 5
        score += bonus
        factors.append(f"+{bonus} persisting about {days:g} days")
    return min(score, RULE_CAP), factors


# ----------------------------------------------------------------------------- 3. ML hook
FEATURE_NAMES = ["n_symptoms", "intensity_level", "duration_days"] + [f"has_{s.replace(' ', '_')}" for s in SYMPTOM_WEIGHTS]


def build_features(state: dict) -> list[float]:
    """Feature vector for training / scoring an XGBoost (or any) severity regressor."""
    syms = set(state.get("symptoms") or [])
    return [float(len(syms)), float(_INTENSITY_LEVEL.get(state.get("intensity"), 0)),
            float(duration_days(state.get("duration")) or 0)] + [float(s in syms) for s in SYMPTOM_WEIGHTS]


class SeverityModel(Protocol):
    def predict(self, X: list[list[float]]) -> list[float]: ...      # returns 0-100


# ----------------------------------------------------------------------------- 4. result + classifier
LEVELS = [(30, "MILD", "Self-care / OPD"), (60, "MODERATE", "GP / Family Physician"),
          (85, "SEVERE", "Specialist - Urgent"), (100, "EMERGENCY", "ER / Ambulance NOW")]
ALERT = "EMERGENCY: Turant 112 par call karein ya nearest ER jaayein. / Call 112 or go to the nearest ER now."


def band(score: float) -> tuple[str, str]:
    for top, level, dispo in LEVELS:
        if score <= top:
            return level, dispo
    return LEVELS[-1][1], LEVELS[-1][2]


@dataclass
class SeverityResult:
    score: int
    level: str
    disposition: str
    emergency: bool
    alert: str | None
    triggered_by: str                       # "safety_net" | "model" | "rules" | "none"
    emergency_hits: list[dict] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    rule_score: int = 0
    ml_score: int | None = None
    factors: list[str] = field(default_factory=list)
    note: str = "Triage support only - not a diagnosis."

    def to_dict(self) -> dict:
        return asdict(self)


class SeverityClassifier:
    def __init__(self, ml_model: SeverityModel | None = None, extra_rules: list | None = None):
        self.ml_model = ml_model
        self.rules = list(EMERGENCY_RULES) + list(extra_rules or [])  # built-ins + additive data rules

    def classify(self, state: dict, texts: list[str] | None = None) -> SeverityResult:
        hits = emergency_check(state, texts, self.rules)              # 1. always first, never skipped
        emerg = [h for h in hits if h["flag"] == EMERGENCY]
        flags = [h["english"] for h in hits if h["flag"] == SEVERE_FLAG]

        rscore, factors = rule_score(state)                           # 2. rules
        final, ml_score, trigger = rscore, None, "rules" if rscore else "none"

        if self.ml_model is not None:                                 # 3. ML can only raise the score
            try:
                ml_score = max(0.0, min(100.0, float(self.ml_model.predict([build_features(state)])[0])))
                if ml_score > final:
                    final, trigger = ml_score, "model"
                    factors.append(f"ML score {ml_score:.0f} raised the result")
            except Exception:                                         # a broken model must never block triage
                factors.append("ML model unavailable - rules and safety net only")

        if flags:
            final = max(final, 70)
            factors.append("red-flag combination -> at least SEVERE")
        if emerg:
            final, trigger = max(final, 90), "safety_net"
            factors.append("hardcoded emergency phrase/symptom: " + ", ".join(h["english"] for h in emerg))

        level, dispo = band(final)
        if level == "EMERGENCY" and trigger not in ("safety_net", "model"):
            trigger = "rules"
        is_emergency = level == "EMERGENCY"
        return SeverityResult(
            score=round(final), level=level, disposition=dispo, emergency=is_emergency,
            alert=ALERT if is_emergency else None, triggered_by=trigger if (final or hits) else "none",
            emergency_hits=emerg, red_flags=flags, rule_score=round(rscore),
            ml_score=None if ml_score is None else round(ml_score), factors=factors)


if __name__ == "__main__":
    clf = SeverityClassifier()
    cases = [
        ("halka khansi", {"symptoms": ["cough"], "intensity": "mild"}, ["halka khansi"]),
        ("pet dard + ulti + bukhar", {"symptoms": ["stomach pain", "vomiting", "fever"]}, ["pet dard ulti bukhar"]),
        ("seene mein dard", {"symptoms": []}, ["seene mein dard ho raha hai"]),
    ]
    for name, st, tx in cases:
        r = clf.classify(st, tx)
        print(f"{name:28} -> {r.score:3} {r.level:9} {r.disposition}  [{r.triggered_by}]")
