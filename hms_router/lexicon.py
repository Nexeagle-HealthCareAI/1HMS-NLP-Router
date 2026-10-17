"""Lexicon loader shared by Step 1 (language detection) and Step 2 (spelling correction)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LEXICON = Path(__file__).parent / "data" / "hinglish_lexicon.csv"
DEFAULT_VARIANTS = Path(__file__).parent / "data" / "spelling_variants.csv"   # doc: spelling_variants.csv


@dataclass(frozen=True)
class Entry:
    variant: str       # normalized lowercase Roman form, may be multi-word
    canonical: str
    category: str      # symptom | body_part | time | intensity | particle | function
    lang: str          # hi | en | amb (also a valid English word)


class Lexicon:
    def __init__(self, entries: list[Entry], aliases: dict[str, str] | None = None):
        self.entries = entries
        self.particles = {e.variant for e in entries if e.category == "particle"}
        # Roman-Hindi marker words used for Hinglish detection (ambiguous words excluded)
        self.hindi_markers = {e.variant for e in entries if e.lang == "hi" and " " not in e.variant}
        # phrase index: particle-stripped variant -> Entry
        self.phrase_index: dict[str, Entry] = {}
        for e in entries:
            key = " ".join(w for w in e.variant.split() if w not in self.particles) or e.variant
            self.phrase_index.setdefault(key, e)
        # Bug #13 fix: default=1 prevents ValueError when phrase_index is empty
        # (e.g. a minimal test lexicon with zero entries).
        self.max_ngram = max((len(k.split()) for k in self.phrase_index), default=1)
        # single-word vocabulary used for word-level spell repair
        self.vocab = {e.variant for e in entries if " " not in e.variant}
        self.spellable_vocab = {
            e.variant for e in entries
            if " " not in e.variant and e.category in {"symptom", "body_part", "intensity", "time"}
        }
        # known typo / alternate spellings -> a word that is in `vocab` (spelling_variants.csv).
        # Aliases pointing at unknown words are dropped, so a bad row can never inject a new word.
        self.aliases = {v: c for v, c in (aliases or {}).items() if c in self.vocab and v not in self.vocab}
        # what a single word *means* (set of canonicals) - lets the corrector accept a spelling tie
        # when every tied candidate means the same thing (e.g. seene / seena -> chest)
        self.word_meaning: dict[str, frozenset[str]] = {}
        for e in entries:
            if " " not in e.variant:
                self.word_meaning[e.variant] = frozenset({e.canonical})

    @classmethod
    def from_csv(cls, path: str | Path | None = None, variants_path: str | Path | None = None) -> "Lexicon":
        with open(path or DEFAULT_LEXICON, newline="", encoding="utf-8") as f:
            rows = [
                Entry(r["variant"].strip().lower(), r["canonical"].strip(), r["category"].strip(), r["lang"].strip())
                for r in csv.DictReader(f)
                if r["variant"].strip()
            ]
        aliases: dict[str, str] = {}
        vp = Path(variants_path) if variants_path else (DEFAULT_VARIANTS if path is None else None)
        if vp and vp.exists():
            with open(vp, newline="", encoding="utf-8") as f:
                aliases = {r["variant"].strip().lower(): r["canonical"].strip().lower()
                           for r in csv.DictReader(f) if r["variant"].strip()}
        return cls(rows, aliases)

    def add_entries(self, new: list[Entry]) -> "Lexicon":
        """Step 8: return a NEW lexicon with extra entries (existing variants always win)."""
        known = {e.variant for e in self.entries}
        return Lexicon(self.entries + [e for e in new if e.variant not in known], self.aliases)
