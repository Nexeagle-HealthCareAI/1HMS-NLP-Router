"""Query normalization shared by the safety and rag layers (and by training, so both see the same text)."""
from __future__ import annotations

import re

from .script import normalize_script

# "X kam/ruk/band/utar nahi ho raha" means X is NOT going away - it must not read as "no X".
# Rewriting the "nahi" stops both the extractor and the red-flag matcher from treating it as a denial.
_PERSISTENCE = re.compile(r"\b(kam|ruk|band|utar|theek|thik|khatam|chala|chali|ja|jaa)\s+(nahi|nahin|nhi)\b")
PERSIST_TOKEN = "persist"

# Words that carry no specialty signal in either language. Dropping them stops "and / my / hai / mein"
# voting for whichever class happens to use them most.
STOPWORDS = frozenset(
    "the a an and or my i me im is am are was were be been have has had having to of in on at for "
    "with from by it this that its very so but not can cant just also some any get got getting feel "
    "feeling since ago day days week weeks month months year years please doctor".split()
) | frozenset(
    "hai hain ho hota hoti hote raha rahi rahe mein me mai main ka ki ke ko se ne bhi aur ya to tha "
    "thi hua hui gaya gayi kuch bahut bohot kafi kaafi mujhe mera meri mere humko hum".split()
)


def preprocess(text: str) -> str:
    """Devanagari -> Roman, lowercase, punctuation out, then mark persistence phrases."""
    return _PERSISTENCE.sub(rf"\1 {PERSIST_TOKEN}", normalize_script(text or ""))


def has_letters(normalized: str) -> bool:
    return bool(re.search(r"[a-z]", normalized))
