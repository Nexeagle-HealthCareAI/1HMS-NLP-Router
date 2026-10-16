"""Step 1 - language detection: english | hindi | hinglish.

Roman-script Hindi counts as "hinglish" (matches the doc's own example:
"kal raat se bahut tez sir dard ho raha hai" -> language_mix = hinglish).
fastText lid.176 is *optional*: it is unreliable on Roman-script Hindi, so it is only
used to confirm pure English / pure Devanagari, never to override the marker heuristic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import Lexicon

_DEV = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class LanguageResult:
    language: str      # "english" | "hindi" | "hinglish" | "unknown"
    script: str        # "latin" | "devanagari" | "mixed" | "none"
    method: str        # "script" | "markers" | "fasttext+markers"


class LanguageDetector:
    def __init__(self, lexicon: Lexicon, fasttext_model_path: str | None = None):
        self.lexicon = lexicon
        self._ft = None
        if fasttext_model_path:
            try:
                import fasttext
                self._ft = fasttext.load_model(fasttext_model_path)
            except Exception:  # missing lib / model -> silently fall back to heuristic
                self._ft = None

    def detect(self, text: str) -> LanguageResult:
        dev, lat = len(_DEV.findall(text)), len(_LATIN.findall(text))
        if dev + lat == 0:
            return LanguageResult("unknown", "none", "script")
        if dev and lat:
            return LanguageResult("hinglish", "mixed", "script")
        if dev:
            return LanguageResult("hindi", "devanagari", "script")

        words = re.findall(r"[a-z]+", text.lower())
        hits = sum(w in self.lexicon.hindi_markers for w in words)
        if hits:
            return LanguageResult("hinglish", "latin", "markers")
        method = "markers"
        if self._ft is not None:
            label, prob = self._ft.predict(text.replace("\n", " "))
            method = "fasttext+markers"
            if label[0] == "__label__hi" and prob[0] > 0.8:
                return LanguageResult("hinglish", "latin", method)
        return LanguageResult("english", "latin", method)
