"""Step 3 - NER & symptom entity extraction -> Structured Symptom Object.

Cleaned token stream (Step 2)
  -> entity extraction : SYMPTOM | BODY_PART | DURATION | INTENSITY
  -> negation check ("bukhar nahi hai" must NOT become a fever)
  -> Structured Symptom Object (schema from the architecture doc + a few extras)

Backends
  RuleNER  - lexicon/regex based, deterministic, works today. Also the safety net.
  BertNER  - optional HuggingFace token-classification model (fine-tuned on medical Hinglish).
             Its output is merged with RuleNER's, never replaces it.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace as dc_replace
from typing import Protocol

from .lexicon import Lexicon
from .spelling import CorrectionResult

SYMPTOM, BODY_PART, DURATION, INTENSITY = "SYMPTOM", "BODY_PART", "DURATION", "INTENSITY"

_CATEGORY_TO_TYPE = {"symptom": SYMPTOM, "body_part": BODY_PART, "intensity": INTENSITY}
_INTENSITY_LEVEL = {
    # Maps lexicon intensity canonicals -> numeric level (1=mild, 2=moderate, 3=severe).
    # All keys are actual canonical values emitted by hinglish_lexicon.csv for
    # category=intensity entries. The accumulator normalises these to mild/moderate/severe
    # via _LEVEL_NAME. Aligned to actual CSV data (see data/hinglish_lexicon.csv).
    "mild": 1, "little": 1,
    "moderate": 2, "quite": 2, "more": 2,
    "very": 3, "sharp": 3, "severe": 3,
}
_LEVEL_NAME = {1: "mild", 2: "moderate", 3: "severe"}

# symptom -> body part it implies ("sir dard" has no separate body-part word, but it's still the head)
_IMPLIED_BODY_PART = {"headache": "head", "sudden severe headache": "head", "stomach pain": "stomach", "chest pain": "chest",
                      "back pain": "back", "sore throat": "throat", "vision loss": "eye"}
# generic "pain" + a body part -> specific symptom
_PAIN_FOR_PART = {"head": "headache", "throat": "sore throat"}

_NEGATORS_AFTER = {"nahi", "nahin"}          # Hindi: "bukhar nahi hai"
_NEGATORS_BEFORE = {"no", "without", "bina"}  # English / bina: "no fever"
_CLAUSE_BREAK = {"aur", "and", "lekin", "magar", "but"}

_NUM = r"(\d+)"
_UNIT = r"(din|dino|days?|hafte|hafta|weeks?|mahine|mahina|months?|ghante|ghanta|hours?|saal|years?)"
_UNIT_NAME = {"din": "day", "dino": "day", "day": "day", "days": "day", "hafte": "week", "hafta": "week",
              "week": "week", "weeks": "week", "mahine": "month", "mahina": "month", "month": "month",
              "months": "month", "ghante": "hour", "ghanta": "hour", "hour": "hour", "hours": "hour",
              "saal": "year", "year": "year", "years": "year"}
_KAL_PART = {("kal", "raat"): "last night", ("kal", "subah"): "yesterday morning",
             ("kal", "shaam"): "yesterday evening", ("kal", "dopahar"): "yesterday afternoon",
             ("aaj", "subah"): "this morning", ("aaj", "shaam"): "this evening",
             ("aaj", "raat"): "tonight", ("aaj", "dopahar"): "this afternoon"}
_PART = {"subah": "morning", "raat": "night", "shaam": "evening", "dopahar": "afternoon",
         "morning": "morning", "night": "night", "evening": "evening", "afternoon": "afternoon"}


@dataclass
class Entity:
    type: str
    text: str
    canonical: str | None
    start: int            # word offsets into the corrected token stream (end exclusive)
    end: int
    confidence: float = 1.0
    source: str = "rule"
    negated: bool = False


@dataclass
class SymptomObject:
    raw_input: str
    symptoms: list[str]
    body_part: str | None
    duration: str | None
    intensity: str | None
    language_mix: str
    session_id: str
    turn: int
    negated_symptoms: list[str] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    sudden_onset: bool = False

    def to_dict(self, include_entities: bool = False) -> dict:
        d = asdict(self)
        if not include_entities:
            d.pop("entities")
        return d


# ------------------------------------------------------------------ duration
def parse_duration(words: list[str]) -> tuple[str, int, int] | None:
    """Returns (label, start_word, end_word) or None. `words` has number-words already as digits."""
    text = " ".join(words)

    def span(m: re.Match) -> tuple[int, int]:
        s = len(text[: m.start()].split())
        return s, s + len(m.group().split())

    def plural(n: int, unit: str) -> str:
        return f"{n} {unit}{'' if n == 1 else 's'}"

    patterns = [
        (rf"\bpichle\s+{_NUM}\s+{_UNIT}(?:\s+se)?\b", lambda m: f"for the past {plural(int(m[1]), _UNIT_NAME[m[2]])}"),
        (rf"\b(?:for\s+)?{_NUM}\s+{_UNIT}\s+se\b", lambda m: f"for {plural(int(m[1]), _UNIT_NAME[m[2]])}"),
        (rf"\bfor\s+{_NUM}\s+{_UNIT}\b", lambda m: f"for {plural(int(m[1]), _UNIT_NAME[m[2]])}"),
        (rf"\b{_NUM}\s+{_UNIT}\s+(?:ago|pehle)\b", lambda m: f"for {plural(int(m[1]), _UNIT_NAME[m[2]])}"),
        (r"\bpichle\s+(hafte|mahine|saal)\s+se\b", lambda m: f"since last {_UNIT_NAME[m[1]]}"),
        (r"\b(kal|aaj)\s+(subah|raat|shaam|dopahar)\s+se\b", lambda m: f"since {_KAL_PART[(m[1], m[2])]}"),
        (r"\b(subah|raat|shaam|dopahar)\s+se\b", lambda m: f"since {_PART[m[1]]}"),
        (r"\bkal\s+se\b", lambda m: "since yesterday"),
        (r"\baaj\s+se\b", lambda m: "since today"),
        (r"\bsince\s+(yesterday|morning|night|evening|afternoon|last night)\b", lambda m: f"since {m[1]}"),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, text)
        if m:
            s, e = span(m)
            return fmt(m), s, e
    return None


# ------------------------------------------------------------------ backends
class NERBackend(Protocol):
    def extract(self, tokens_text: list[str], correction: CorrectionResult) -> list[Entity]: ...


class RuleNER:
    def extract(self, words: list[str], correction: CorrectionResult) -> list[Entity]:
        ents: list[Entity] = []
        pos = 0
        for t in correction.tokens:
            n = len(t.text.split())
            etype = _CATEGORY_TO_TYPE.get(t.category)
            if etype:
                ents.append(Entity(etype, t.text, t.canonical, pos, pos + n, 0.85 if t.corrected else 1.0))
            pos += n
        dur = parse_duration(words)
        if dur:
            ents.append(Entity(DURATION, " ".join(words[dur[1]:dur[2]]), dur[0], dur[1], dur[2]))
        return sorted(ents, key=lambda e: e.start)


class BertNER:
    """Optional: HuggingFace token-classification model with labels {B,I}-{SYMPTOM,BODY_PART,DURATION,INTENSITY}.

    Not exercised in the test-suite (no fine-tuned model exists yet). Output goes through the same
    canonicalisation + negation as rule entities and is merged under RuleNER, so a weak model can't
    remove anything the rules found.
    """

    def __init__(self, model_path: str, lexicon: Lexicon, min_score: float = 0.6):
        from transformers import pipeline  # lazy import, heavy dependency
        self.pipe = pipeline("token-classification", model=model_path, aggregation_strategy="simple")
        self.lex, self.min_score = lexicon, min_score

    def extract(self, words: list[str], correction: CorrectionResult) -> list[Entity]:
        text = " ".join(words)
        ents = []
        for r in self.pipe(text):
            etype = r["entity_group"].upper()
            if etype not in (SYMPTOM, BODY_PART, DURATION, INTENSITY) or r["score"] < self.min_score:
                continue
            span_text = text[r["start"]:r["end"]].strip()
            s = len(text[: r["start"]].split())
            hit = self.lex.phrase_index.get(" ".join(w for w in span_text.split() if w not in self.lex.particles))
            ents.append(Entity(etype, span_text, hit.canonical if hit else span_text, s, s + len(span_text.split()),
                               float(r["score"]), "bert"))
        return ents


# ------------------------------------------------------------------ extractor
class SymptomExtractor:
    def __init__(self, backends: list | None = None):
        self.backends = backends or [RuleNER()]      # put BertNER *after* RuleNER

    def extract(self, correction: CorrectionResult, raw_input: str, language_mix: str,
                session_id: str, turn: int) -> SymptomObject:
        # number-words ("teen") -> digits so the duration regexes see "3 din se"
        words: list[str] = []
        for t in correction.tokens:
            words += [t.canonical] if t.category == "number" and t.canonical else t.text.split()

        ents = self._merge([b.extract(words, correction) for b in self.backends])
        ents = self._mark_negation(ents, words)
        ents = self._combine_generic_pain(ents)

        live = [e for e in ents if not e.negated]
        levels = [_INTENSITY_LEVEL.get(e.canonical, 0) for e in live if e.type == INTENSITY]
        sudden = any(t.canonical == "sudden" for t in correction.tokens)
        if sudden and levels and max(levels) >= 3:      # "sar mein achanak tez dard" -> red-flag headache (doc, Step 5)
            # Bug #3 fix: use dc_replace() to create a new Entity instead of mutating
            # in-place (ents and live share the same object references; mutation is
            # thread-unsafe and causes downstream consumers to see the changed canonical).
            ents = [
                dc_replace(e, canonical="sudden severe headache")
                if (e.type == SYMPTOM and e.canonical == "headache" and not e.negated)
                else e
                for e in ents
            ]
            live = [e for e in ents if not e.negated]
        symptoms = _unique(e.canonical for e in live if e.type == SYMPTOM and e.canonical)
        negated = _unique(e.canonical for e in ents if e.type == SYMPTOM and e.negated and e.canonical)
        negated = [s for s in negated if s not in symptoms]
        parts = _unique(e.canonical for e in live if e.type == BODY_PART and e.canonical)
        for s in symptoms:                                   # "sir dard" -> head
            if s in _IMPLIED_BODY_PART and _IMPLIED_BODY_PART[s] not in parts:
                parts.append(_IMPLIED_BODY_PART[s])

        durations = [e.canonical for e in ents if e.type == DURATION]
        return SymptomObject(
            raw_input=raw_input, symptoms=symptoms, body_part=parts[0] if parts else None,
            duration=durations[0] if durations else None,
            intensity=_LEVEL_NAME.get(max(levels)) if levels and max(levels) else None,
            language_mix=language_mix, session_id=session_id, turn=turn,
            negated_symptoms=negated, body_parts=parts, entities=ents, sudden_onset=sudden)

    # -- helpers
    @staticmethod
    def _merge(lists: list[list[Entity]]) -> list[Entity]:
        out: list[Entity] = []
        for lst in lists:                                    # earlier backend wins on overlap of same type
            for e in lst:
                if not any(o.type == e.type and o.start < e.end and e.start < o.end for o in out):
                    out.append(e)
        return sorted(out, key=lambda e: (e.start, e.end))

    @staticmethod
    def _mark_negation(ents: list[Entity], words: list[str]) -> list[Entity]:
        """Returns a new list with negation flags set (Bug #3 fix: no in-place mutation)."""
        result: list[Entity] = []
        for e in ents:
            if e.type != SYMPTOM:
                result.append(e)
                continue
            negated = False
            lookahead = words[e.end:min(e.end + 3, len(words))]
            idiomatic_breathlessness = (
                e.canonical == "breathlessness"
                and any(w in _NEGATORS_AFTER for w in lookahead)
                and any(w in {"aa", "a", "raha", "rahi", "rahe"} for w in lookahead)
            )
            # "neend nahi aa rahi", "so nahi pa raha": the "nahi" is part of the complaint (cannot sleep), not a denial
            idiomatic_sleeplessness = (
                e.canonical == "insomnia"
                and any(w in _NEGATORS_AFTER for w in lookahead)
                and (any(w in {"aa", "a", "aati", "aata", "aaya", "aayi", "raha", "rahi", "rahe", "pa", "paa", "paata", "paati",
                               "pata", "pati", "pate", "hoti", "hota"} for w in lookahead) or len(lookahead) == 1)
            )
            if idiomatic_breathlessness or idiomatic_sleeplessness:
                negated = False
            elif e.start > 0 and words[e.start - 1] in _NEGATORS_BEFORE:
                negated = True
            else:
                other_starts = {o.start for o in ents if o is not e}
                for i in range(e.end, min(e.end + 3, len(words))):
                    if words[i] in _CLAUSE_BREAK or i in other_starts:
                        break
                    if words[i] in _NEGATORS_AFTER:
                        negated = True
                        break
            result.append(dc_replace(e, negated=negated) if negated else e)
        return result

    @staticmethod
    def _combine_generic_pain(ents: list[Entity]) -> list[Entity]:
        """'pet mein bahut dard' -> generic 'pain' + body part 'stomach' => 'stomach pain'.
        Bug #3 fix: returns a new list with updated canonicals instead of mutating in-place."""
        parts = [e for e in ents if e.type == BODY_PART]
        result: list[Entity] = []
        for e in ents:
            if e.type == SYMPTOM and e.canonical == "pain" and not e.negated and parts:
                near = min(parts, key=lambda p: min(abs(p.start - e.end), abs(e.start - p.end)))
                if min(abs(near.start - e.end), abs(e.start - near.end)) <= 4:
                    e = dc_replace(e, canonical=_PAIN_FOR_PART.get(near.canonical, f"{near.canonical} pain"))
            result.append(e)
        return result


def _unique(items) -> list[str]:
    return list(dict.fromkeys(items))
