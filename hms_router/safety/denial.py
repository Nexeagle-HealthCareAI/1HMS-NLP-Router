"""Explicit denials ("saans lene mein koi dikkat nahi", "I don't have chest pain") must not raise an emergency.

The shared /chat NER misses these (the spelling stage turns "koi" into "ki"; "don't" becomes "din t"), so the
route-symptom safety stage removes the denied span before its detectors run. Deliberately narrow:

* Hinglish: a problem noun (dikkat, takleef, problem, dard ...) directly followed by nahi/nahin/nhi. The span
  removed runs left from that noun to the previous verb ending (hai, tha, raha ...) or at most 4 words, so
  "seene mein dard hai, saans ki dikkat nahi" keeps "seene mein dard hai".
* English: don't/do not/didn't/haven't/never + have/has/had/get/got/experience and at most 4 following words
  are removed, stopping at a clause break (and but just only now since bas sirf lekin aur ...). Punctuation is
  already gone after normalization, so the word cap is what keeps "never had chest pain before now severe
  chest pain" an emergency.
* Bare "nahi" is never a denial: "saans nahi aa raha" (cannot breathe) stays an emergency, and "nahi ja
  raha / ruk raha" (not going away) is persistence, not denial.
* Loss words ("I don't have any feeling in my arm", "hosh nahi") are never treated as denials.

Detectors can still only add urgency on whatever text remains; nothing here changes /chat.
"""
from __future__ import annotations

import re

from ..domain.models import EmergencyAssessment, Query
from .detectors import EmergencyDetector

_NEG = {"nahi", "nahin", "nhi", "nai", "nahe"}
_PROBLEM = {"dikkat", "dikat", "diqqat", "takleef", "taklif", "takleef", "problem", "pareshani", "pareshaani",
            "samasya", "issue", "dard", "pain", "tension"}
_STOP_LEFT = {"hai", "hain", "he", "h", "tha", "thi", "the", "raha", "rahi", "rahe", "hua", "hui", "hue", "hoti",
              "hota", "aur", "lekin", "magar", "bas", "sirf", "but", "and", "just", "only"}
# "nahi ja raha" / "nahi ruk raha" = does not go away / does not stop -> persistence, not a denial
_PERSIST_AFTER = {"ja", "jaa", "jaata", "jata", "jati", "jaati", "ruk", "rukta", "rukti", "gaya", "gayi", "utar",
                  "utarta", "jaa", "hat", "hatta", "hatti"}
_LOSS = {"feeling", "feel", "sensation", "control", "strength", "vision", "sight", "air", "breath", "pulse",
         "movement", "power", "hosh", "awareness", "voice"}
_EN_AUX = r"(?:don ?t|dont|do not|does not|doesn ?t|didn ?t|did not|haven ?t|have not|hasn ?t|has not|never)"
_EN_VERB = r"(?:have|has|had|get|got|getting|experience|experienced|experiencing)"
_EN_DENIAL = re.compile(rf"\b{_EN_AUX}\s+{_EN_VERB}\b")
_CLAUSE_BREAK = {"and", "but", "just", "only", "bas", "sirf", "lekin", "magar", "aur", "except", "however",
                 "now", "ab", "today", "aaj", "currently", "suddenly", "achanak", "since", "se"}
_MAX_LEFT = 4        # Hinglish: words kept left of the problem noun
_MAX_OBJECT = 4      # English: words after "don't have" that the denial covers


def _clauses(text: str) -> list[list[str]]:
    out: list[list[str]] = []
    for chunk in re.split(r"[,.;!?|]+", text.lower()):
        cur: list[str] = []
        for w in chunk.split():
            if w in _CLAUSE_BREAK and cur:
                out.append(cur)
                cur = []
            cur.append(w)
        if cur:
            out.append(cur)
    return out


def _hinglish_denied(words: list[str]) -> set[int]:
    drop: set[int] = set()
    for i in range(len(words) - 1):
        if words[i] not in _PROBLEM or words[i + 1] not in _NEG:
            continue
        if i + 2 < len(words) and words[i + 2] in _PERSIST_AFTER:
            continue
        start = i
        while start > 0 and i - start < _MAX_LEFT and words[start - 1] not in _STOP_LEFT:
            start -= 1
        span = range(start, i + 2)
        if any(words[j] in _LOSS for j in span):
            continue
        drop.update(span)
        # trailing auxiliaries belong to the denial: "... nahi hai", "... nahi ho rahi"
        j = i + 2
        while j < len(words) and words[j] in {"hai", "hain", "he", "h", "thi", "tha", "ho", "hoti", "hota", "rahi",
                                               "raha", "hui", "hua"}:
            drop.add(j)
            j += 1
    return drop


def _english_denied(words: list[str]) -> set[int]:
    text = " ".join(words)
    m = _EN_DENIAL.search(text)
    if not m:
        return set()
    start = len(text[:m.start()].split())
    verb_end = len(text[:m.end()].split())
    span = range(start, min(len(words), verb_end + _MAX_OBJECT))
    if any(words[j] in _LOSS for j in span):
        return set()
    return set(span)


def strip_denials(normalized: str) -> tuple[str, list[str]]:
    """-> (text with explicitly denied spans removed, the removed spans). Unchanged text if nothing denied."""
    kept, denied = [], []
    for words in _clauses(normalized):
        drop = _hinglish_denied(words) | _english_denied(words)
        kept.append(" ".join(w for j, w in enumerate(words) if j not in drop))
        if drop:
            denied.append(" ".join(words[j] for j in sorted(drop)))
    if not denied:
        return normalized, []
    return " ".join(k for k in kept if k), denied


class DenialAware:
    """Decorator for any EmergencyDetector: it sees the query with explicitly denied spans removed."""

    def __init__(self, inner: EmergencyDetector):
        self.inner = inner

    def detect(self, query: Query) -> EmergencyAssessment:
        kept, denied = strip_denials(query.normalized)
        if not denied:
            return self.inner.detect(query)
        if not re.search(r"[a-z]", kept):
            return EmergencyAssessment(flag=False)
        return self.inner.detect(Query(raw=query.raw, normalized=kept, language=query.language,
                                       age_group=query.age_group))
