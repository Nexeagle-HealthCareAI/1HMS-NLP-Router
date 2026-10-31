"""Step 2 - text normalization & spelling correction.

Raw normalized text
  -> A. word-level repair : exact -> phonetic key -> RapidFuzz edit distance <= 2
  -> B. phrase matching   : longest n-gram in the symptom lexicon (particles like "me/mein/ka" skipped)
  -> canonical symptom token  OR  unknown token (queued for review, Step 8)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from rapidfuzz import process
from rapidfuzz.distance import OSA, Levenshtein

from .config import Settings
from .lexicon import Lexicon

# ---------------------------------------------------------------- phonetics
_DIGRAPHS = (("chh", "c"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("jh", "j"), ("th", "t"),
             ("dh", "d"), ("bh", "b"), ("ph", "f"), ("sh", "s"), ("ck", "k"))
_LETTERS = str.maketrans({"w": "v", "z": "j", "q": "k", "y": "i"})


def phonetic_key(word: str) -> str:
    """Hindi-aware consonant skeleton (Soundex/Metaphone are English-centric).

    bukhar / bukhaar / bukar -> 'bkr';  sir / sar / ser -> 'sr';  pet / pait -> 'pt'
    """
    w = re.sub(r"(.)\1+", r"\1", word.lower())          # aa->a, ee->e, dd->d
    for src, dst in _DIGRAPHS:
        w = w.replace(src, dst)
    w = w.translate(_LETTERS)
    if not w:
        return w
    return w[0] + re.sub(r"[aeiouh]", "", w[1:])        # keep first letter, drop later vowels/h


# ---------------------------------------------------------------- data classes
@dataclass
class Token:
    source: str                  # what the user typed (normalized, may span several words)
    text: str                    # after spelling repair
    canonical: str | None        # canonical symptom / body part / etc.; None if unknown
    category: str                # symptom | body_part | time | intensity | particle | function | number | unknown
    corrected: bool = False      # True if any word in the span needed spelling repair
    method: str = "exact"        # exact | phonetic | fuzzy | unknown


@dataclass
class CorrectionResult:
    tokens: list[Token] = field(default_factory=list)
    corrected_text: str = ""                    # spelling-fixed Roman Hinglish -> feed to the NER model (Step 3)
    symptoms: list[str] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)
    unknown_tokens: list[str] = field(default_factory=list)


class RedisReviewQueue:
    """Step 8 hook: `LPUSH review_queue`. Stores only the token - never the full sentence (health data)."""

    def __init__(self, redis_client, key: str = "review_queue"):
        self.r, self.key = redis_client, key

    MAX_LEN = 10_000          # bounded: an attacker (or a typo storm) must not grow Redis without limit

    def __call__(self, token: str) -> None:
        if len(token) < 3 or any(ch.isdigit() for ch in token):     # noise: single letters, numbers
            return
        pipe = self.r.pipeline()
        pipe.lpush(self.key, json.dumps({"token": token}, ensure_ascii=False))
        pipe.ltrim(self.key, 0, self.MAX_LEN - 1)
        pipe.execute()


# ---------------------------------------------------------------- corrector
class SpellingCorrector:
    def __init__(self, lexicon: Lexicon, settings: Settings | None = None,
                 unknown_sink: Callable[[str], None] | None = None):
        self.lex = lexicon
        self.cfg = settings or Settings()
        self.unknown_sink = unknown_sink
        self._vocab_list = sorted(lexicon.vocab)
        # fuzzy targets: never very short words - at edit distance 2 anything matches "the", "hai", "ek"
        self._fuzzy_list = sorted(w for w in lexicon.spellable_vocab if len(w) >= 4)
        self._key_index: dict[str, list[str]] = {}
        # everyday words ("kaunsi", "free", "report") are listed only so they are never repaired INTO a symptom;
        # they are never a repair target themselves ("for" must not become "free")
        everyday = {e.variant for e in lexicon.entries if e.category == "everyday"}
        for w in self._vocab_list:
            if w not in everyday:
                self._key_index.setdefault(phonetic_key(w), []).append(w)

    # ---- A. word-level repair
    def repair_word(self, word: str) -> tuple[str, str]:
        """Returns (fixed_word, method). method in exact | alias | phonetic | fuzzy | unknown."""
        if word in self.lex.vocab or word.isdigit():
            return word, "exact"
        if word in self.lex.aliases:                           # curated typo list (spelling_variants.csv)
            return self.lex.aliases[word], "alias"
        if len(word) < 3:
            return word, "unknown"

        # 1) phonetic: same consonant skeleton, closest spelling wins; ties across words -> ambiguous
        cands = self._key_index.get(phonetic_key(word), [])
        scored = sorted((OSA.distance(word, c), c) for c in cands)      # two letters swapped = one typo ("pian")
        scored = [x for x in scored if x[0] <= self.cfg.phonetic_max_distance]
        if scored and (len(scored) == 1 or scored[0][0] < scored[1][0]):
            return scored[0][1], "phonetic"
        if scored:      # tie: fine if every tied candidate means the same thing (seene / seena -> chest)
            tied = [c for d, c in scored if d == scored[0][0]]
            meanings = {self.lex.word_meaning.get(c) for c in tied}
            if len(meanings) == 1 and None not in meanings:
                return tied[0], "phonetic"

        # 2) fuzzy: RapidFuzz edit distance <= 2, long words only, reject ambiguous ties
        if len(word) >= self.cfg.fuzzy_min_word_len:
            max_d = min(self.cfg.fuzzy_max_distance, 1 if len(word) <= 5 else 2)    # short words: 1 typo at most
            hits = process.extract(word, self._fuzzy_list, scorer=OSA.distance,
                                   score_cutoff=max_d, limit=2)
            if hits and (len(hits) == 1 or hits[0][1] < hits[1][1]):
                return hits[0][0], "fuzzy"
        return word, "unknown"

    # ---- B. phrase matching + canonicalisation
    def correct(self, text: str) -> CorrectionResult:
        words = text.split()
        repaired = [self.repair_word(w) for w in words]
        fixed = [r[0] for r in repaired]
        particles = self.lex.particles
        content = [i for i, w in enumerate(fixed) if w not in particles]
        pos = {idx: p for p, idx in enumerate(content)}

        res = CorrectionResult(corrected_text=" ".join(fixed))
        i = 0
        while i < len(words):
            if fixed[i] in particles:
                # Bug #1 fix: use .get() to avoid KeyError when a particle word's stripped
                # phrase_index key differs from the raw word (e.g. multi-word particles).
                entry = self.lex.phrase_index.get(fixed[i])
                canonical = entry.canonical if entry else None
                # Bug #2 fix: "alias" counts as a correction (same as phonetic/fuzzy).
                is_corrected = repaired[i][1] != "exact"
                res.tokens.append(Token(words[i], fixed[i], canonical,
                                        "particle", is_corrected, repaired[i][1]))
                i += 1
                continue
            p, tok = pos[i], None
            for n in range(min(self.lex.max_ngram, len(content) - p), 0, -1):
                idxs = content[p:p + n]
                hit = self.lex.phrase_index.get(" ".join(fixed[j] for j in idxs))
                if hit:
                    end = idxs[-1]
                    methods = [repaired[j][1] for j in range(i, end + 1)]
                    # Bug #2 fix: treat "alias" as a correction method alongside phonetic/fuzzy.
                    bad = [m for m in methods if m in ("phonetic", "fuzzy", "alias")]
                    tok = Token(" ".join(words[i:end + 1]), " ".join(fixed[i:end + 1]), hit.canonical,
                                hit.category, bool(bad), bad[0] if bad else "exact")
                    i = end + 1
                    break
            if tok is None:
                if fixed[i].isdigit():
                    tok = Token(words[i], fixed[i], fixed[i], "number")
                else:
                    tok = Token(words[i], fixed[i], None, "unknown", False, "unknown")
                    res.unknown_tokens.append(fixed[i])
                    if self.unknown_sink:               # Step 8 must see what the user TYPED, not our guess at it
                        self.unknown_sink(words[i])
                i += 1
            res.tokens.append(tok)

        for t in res.tokens:
            if t.category == "symptom" and t.canonical not in res.symptoms:
                res.symptoms.append(t.canonical)
            elif t.category == "body_part" and t.canonical not in res.body_parts:
                res.body_parts.append(t.canonical)
        return res
