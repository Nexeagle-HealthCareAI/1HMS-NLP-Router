"""Medical-signal check: does the query contain at least one health word?

Sentence similarity alone cannot tell "cricket score today" from "ankle sprain from sports" (the corpus has
sports-medicine sentences), so the service also asks whether any query word is a known health term.

The vocabulary comes from curated sources only, never from the evaluation sets:
  * data/hinglish_lexicon.csv  (symptoms, body parts - the /chat lexicon, read-only)
  * data/specialty_seed_en.csv (English seed phrases per specialty)
  * data/red_flags.csv         (emergency phrases)
  * _EXTRA below               (generic clinical words those files lack)
Corpus sentences are deliberately not used: they contain everyday words ("cricket", "weather", "office").

Typos: a word of 6+ letters also counts when it is a close spelling of a vocabulary word ("stomache").
Measured 2026-10-19: all 6,292 dataset sentences and every medical query in Doctor Dekho set 1 have a
signal; none of set 1's 10 non-medical queries do.
"""
from __future__ import annotations

import csv
import difflib
import re
from pathlib import Path

from ..textnorm import STOPWORDS, preprocess

DATA = Path(__file__).resolve().parent.parent / "data"
SOURCES = (("hinglish_lexicon.csv", "variant"), ("specialty_seed_en.csv", "text"), ("red_flags.csv", "phrase"))

_EXTRA = """
medicine medicines tablet tablets dawai dawa injection vaccine vaccination teeka
surgery operation test report scan xray ultrasound endoscopy biopsy blood urine stool sugar bp pressure
pain ache hurt hurts hurting sore swelling swollen injury injured wound bleeding fever infection allergy
nail nails naakhoon nakhun mole moles wart warts reflux esophagus gastric ulcer tummy belly
tooth teeth gum gums ear ears nose throat eye eyes skin hair scalp chest heart lung lungs liver kidney
stomach bone bones joint joints muscle muscles nerve nerves spine brain head neck back knee hip shoulder
period periods pregnancy pregnant baby child infant newborn diabetes thyroid cancer tumour tumor lump
cough cold flu breathing breath vomit vomiting nausea diarrhoea diarrhea constipation dizzy dizziness
anxiety anxious depression depressed sad stress sleep insomnia weight obesity diet
chashma chasma specs spectacles glasses lens
"""
# everyday words that the curated sources contain inside longer phrases ("morning stiffness", "after eating")
_NOT_SIGNAL = set("""
after while well old age bad care level management needed running morning night day days week weeks
red dark lower upper left right missed passing fall falls heavy frequent pull root canal ray tear
symptoms problem problems issue issues kam zyada bahut thoda din raat subah shaam baar time hai
good great best near book order phone call help want need like feel feeling
""".split())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", preprocess(text)) if w not in STOPWORDS and len(w) > 2]


class MedicalVocabulary:
    def __init__(self, words: set[str], fuzzy_min_len: int = 6, fuzzy_cutoff: float = 0.88):
        if not words:
            raise ValueError("MedicalVocabulary needs at least one word")
        self.words = frozenset(words)
        self._by_len: dict[int, list[str]] = {}
        for w in self.words:
            self._by_len.setdefault(len(w), []).append(w)
        self.fuzzy_min_len, self.fuzzy_cutoff = fuzzy_min_len, fuzzy_cutoff

    @classmethod
    def from_data(cls, data_dir: str | Path = DATA) -> "MedicalVocabulary":
        words = set(_words(_EXTRA))
        for name, column in SOURCES:
            path = Path(data_dir) / name
            if not path.exists():
                continue
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if (row.get("category") or "").strip() in ("everyday", "function", "particle", "number", "time"):
                        continue                     # everyday words the lexicon lists only to keep them from spell repair
                    words.update(_words(row.get(column) or ""))
        return cls(words - _NOT_SIGNAL)

    def _close(self, word: str) -> bool:
        pool = [w for n in range(len(word) - 1, len(word) + 2) for w in self._by_len.get(n, [])]
        return bool(difflib.get_close_matches(word, pool, n=1, cutoff=self.fuzzy_cutoff))

    def _known(self, word: str) -> bool:
        if word in self.words:
            return True
        for suffix in ("ing", "ed", "es", "s"):                 # coughing, vomited, aches
            stem = word[: -len(suffix)]
            if word.endswith(suffix) and len(stem) >= 3 and (stem in self.words or stem + "e" in self.words):
                return True
        return len(word) >= self.fuzzy_min_len and self._close(word)

    def terms_in(self, normalized: str) -> list[str]:
        return [w for w in re.findall(r"[a-z]+", normalized)
                if w not in STOPWORDS and len(w) > 2 and w not in _NOT_SIGNAL and self._known(w)]

    def has_signal(self, normalized: str) -> bool:
        return bool(self.terms_in(normalized))
