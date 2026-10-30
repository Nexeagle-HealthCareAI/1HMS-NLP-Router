"""Spelling-tolerant doctor search (2026-10-30).

Patients type fast, on phones, in two scripts and several spellings: "cardiolgist", "skn doktor", "dr kunal ghose",
"cardiologist kolkatta", "mbs doctor". Each kind of search word is matched to the closest word Vita knows, but only
when that closest word is clear:

    WordIndex   closest known word by edit distance (transpositions count as one typo) or by sound
                (the consonant skeleton from spelling.phonetic_key, same first letter): "ghose" ~ "ghosh",
                "chaterji" ~ "chatterjee". Allowed typos grow with the word: none under 4 letters, 1 up to 6,
                2 up to 12, 3 above. Two equally close words with different meanings -> no guess.
    NameIndex   each word of a doctor-name query -> the closest word in a listed doctor's name, so 1HMS's own
                name match sees a clean spelling.
    CityIndex   a city as typed -> the city as 1HMS spells it (1HMS matches cities exactly), plus old names
                ("calcutta" -> Kolkata).

Nothing here touches symptoms; spelling.SpellingCorrector does that for triage.
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache
from typing import Iterable

from rapidfuzz.distance import OSA

from .spelling import phonetic_key


def allowed_typos(length: int) -> int:
    return 0 if length < 4 else 1 if length <= 6 else 2 if length <= 12 else 3


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


class WordIndex:
    """Known words, each with a meaning. `closest()` returns a known word for a misspelling, or None."""

    def __init__(self, meanings: dict[str, str]):
        self.meanings = {w: m for w, m in meanings.items() if w}
        self._by_len: dict[int, list[str]] = {}
        self._by_sound: dict[str, list[str]] = {}
        for w in self.meanings:
            self._by_len.setdefault(len(w), []).append(w)
            self._by_sound.setdefault(phonetic_key(w), []).append(w)

    def __contains__(self, word: str) -> bool:
        return word in self.meanings

    def __bool__(self) -> bool:
        return bool(self.meanings)

    def closest(self, word: str, max_typos: int | None = None) -> str | None:
        if word in self.meanings:
            return word
        n = len(word)
        k = allowed_typos(n) if max_typos is None else max_typos
        if k <= 0 or n < 3:
            return None
        scored: dict[str, int] = {}
        for m in range(n - k, n + k + 1):
            for w in self._by_len.get(m, ()):
                d = OSA.distance(word, w, score_cutoff=k)
                if d <= k:
                    scored[w] = d
        key = phonetic_key(word)
        if n >= 4 and len(key) >= 3:                # sounds the same: "chaterji" / "chatterjee", "mukharji" / "mukherjee"
            for w in self._by_sound.get(key, ()):   # (a 1-2 letter skeleton says too little: "nahi" is not "neha")
                if w[0] == word[0] and abs(len(w) - n) <= max(2, k):
                    scored[w] = min(scored.get(w, 9), k)
        if not scored:
            return None
        best = min(scored.values())
        tied = [w for w, d in scored.items() if d == best]
        ranked = sorted(tied, key=lambda w: (-_ratio(word, w), w))
        if len({self.meanings[w] for w in tied}) == 1:
            return ranked[0]
        top, second = _ratio(word, ranked[0]), _ratio(word, ranked[1])
        return ranked[0] if top - second >= 0.08 else None       # two different meanings, equally close: no guess


# ------------------------------------------------------------------------------------------------ doctor names
_NAME_NOISE = {"dr", "doctor", "prof", "professor", "mr", "mrs", "ms", "miss", "smt", "shri", "sri"}


def name_tokens(name: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z ]", " ", (name or "").lower()).split() if len(t) >= 2 and t not in _NAME_NOISE]


class NameIndex:
    def __init__(self, names: Iterable[str]):
        self.tokens = WordIndex({t: t for n in names for t in name_tokens(n)})

    def __bool__(self) -> bool:
        return bool(self.tokens)

    def word(self, w: str) -> str | None:
        """The listed name word this word is (a typo of), or None."""
        return self.tokens.closest(w) if len(w) >= 3 else (w if w in self.tokens else None)

    def fix(self, query: str) -> str:
        """'neha sarma' -> 'neha sharma'. Words that match no listed name are kept as typed."""
        if not self.tokens or not query:
            return query
        out = []
        for w in re.sub(r"[^a-z ]", " ", query.lower()).split():
            out.append(self.word(w) or w)
        return " ".join(out)


@lru_cache(maxsize=16)
def name_index(names: tuple[str, ...]) -> NameIndex:
    """NameIndex for a list of names, kept while the list is the same (DoctorDirectory caches it 15 min)."""
    return NameIndex(names)


# ------------------------------------------------------------------------------------------------ cities
# Old and English names people still type; used only when the new name is a city Doctor Dekho has doctors in.
CITY_ALIASES = {
    "calcutta": "kolkata", "bombay": "mumbai", "madras": "chennai", "bangalore": "bengaluru", "banglore": "bengaluru",
    "bengalore": "bengaluru", "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "benares": "varanasi",
    "banaras": "varanasi", "kashi": "varanasi", "allahabad": "prayagraj", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysore": "mysuru", "purnea": "purnia", "orissa": "odisha", "cawnpore": "kanpur",
    "simla": "shimla", "pondicherry": "puducherry", "vizag": "visakhapatnam", "calicut": "kozhikode",
}


class CityIndex:
    def __init__(self, cities: Iterable[str]):
        self.canon: dict[str, str] = {}
        for c in cities:
            key = " ".join(re.sub(r"[^a-z ]", " ", (c or "").lower()).split())
            if key:
                self.canon.setdefault(key, c.strip())
        self.words = WordIndex({k: k for k in self.canon})

    def __bool__(self) -> bool:
        return bool(self.canon)

    def resolve(self, text: str | None) -> str | None:
        """The city as 1HMS spells it, or None when the text is no known city."""
        if not text or not self.canon:
            return None
        key = " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())
        if key in self.canon:
            return self.canon[key]
        old = CITY_ALIASES.get(key) or CITY_ALIASES.get(key.replace(" ", ""))
        if old and old in self.canon:
            return self.canon[old]
        squeezed = key.replace(" ", "")
        for k in self.canon:                                    # "salt lake" / "saltlake"
            if k.replace(" ", "") == squeezed:
                return self.canon[k]
        if len(squeezed) < 4:
            return None
        hit = self.words.closest(key)
        return self.canon[hit] if hit else None


@lru_cache(maxsize=16)
def city_index(cities: tuple[str, ...]) -> CityIndex:
    return CityIndex(cities)
