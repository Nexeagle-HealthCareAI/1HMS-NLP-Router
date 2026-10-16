"""Step 1 - script normalization: Devanagari -> lowercase Roman, light text cleanup."""
from __future__ import annotations

import re
import unicodedata

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)

try:  # IndicTrans2 is for sentence-level translation; for Roman<->Devanagari we use indic-transliteration
    from indic_transliteration.sanscript import DEVANAGARI, ITRANS, transliterate
except ImportError:  # pragma: no cover
    transliterate = None


# ITRANS -> casual Roman. ORDER MATTERS: multi-character tokens first, otherwise "N" -> "n" runs before
# ".N" (chandrabindu) is seen and the word is split at the stray "." (e.g. साँस -> "saa ns").
_ITRANS_TO_ROMAN = (
    (".N", "n"), ("~N", "n"), ("~n", "n"), (".m", "n"), (".h", ""),     # nasalisation / halant marks
    (".Dh", "rh"), (".D", "r"),                                          # flapped ड़ ढ़
    ("Ch", "chh"), ("Sh", "sh"),
    ("A", "aa"), ("I", "ee"), ("U", "oo"), ("T", "t"), ("D", "d"), ("N", "n"), ("R", "r"), ("M", "n"),
)


def _dev_word_to_roman(word: str) -> str:
    """Devanagari word -> Hinglish-style Roman (e.g. बुखार -> bukhaar, नहीं -> nahin, साँस -> saans)."""
    if transliterate is None:
        raise RuntimeError("pip install indic-transliteration to handle Devanagari input")
    s = transliterate(word, DEVANAGARI, ITRANS)
    # word-final long ee is written "i" in casual Hinglish (nahi, rahi, ulti), not "ee"
    s = re.sub(r"I(M?)$", lambda m: "i" + ("n" if m[1] else ""), s)
    for src, dst in _ITRANS_TO_ROMAN:
        s = s.replace(src, dst)
    s = s.lower()
    # Hindi schwa deletion: a word ending in a bare consonant has no pronounced final 'a'.
    # NFKC leaves nukta consonants (ज़ ड़ ...) decomposed, so look *through* a trailing nukta.
    base = word.rstrip("\u093c")
    if s.endswith("a") and base and ("\u0915" <= base[-1] <= "\u0939" or "\u0958" <= base[-1] <= "\u095f"):
        s = s[:-1]
    return s


def normalize_script(text: str) -> str:
    """NFKC -> Devanagari to Roman -> lowercase -> strip punctuation -> squeeze elongations."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).translate(_DEV_DIGITS)
    text = DEVANAGARI_RE.sub(lambda m: _dev_word_to_roman(m.group()), text)
    text = text.lower()
    text = re.sub(r"[^\w\s]|_", " ", text)             # punctuation -> space
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)         # "dardddd" -> "dardd" (edit-distance 1 from "dard")
    return re.sub(r"\s+", " ", text).strip()
