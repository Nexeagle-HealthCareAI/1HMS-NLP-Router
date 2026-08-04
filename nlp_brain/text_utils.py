"""Pure text-analysis helpers -- no ML model, no I/O. Single responsibility:
decide whether a piece of text is clean/usable input, nothing else."""
import re


def clean_text(s: str) -> str:
    """Light normalization: collapse whitespace, strip. Case and punctuation
    are left mostly intact -- they carry signal here (e.g. 'BP', 'CT scan')."""
    return " ".join(str(s).split())


def is_gibberish(text: str) -> bool:
    """Heuristic check for gibberish / nonsensical input.

    Flags text that is empty, too short, has almost no vowels, contains a
    long run of repeated characters, or has a long run of consonants with
    no vowel in between -- all signs of random keyboard mashing rather than
    an actual (Hinglish) symptom description. Intentionally a simple,
    dependency-free heuristic rather than a trained model, so it runs
    instantly before the classifier is bothered."""
    text = clean_text(text)

    if len(text) < 3:
        return True

    letters_only = re.sub(r"[^a-zA-Z]", "", text)
    if len(letters_only) < 3:
        return True

    # Almost no vowels at all -> unlikely to be real words.
    vowels = sum(1 for c in letters_only.lower() if c in "aeiouy")
    vowel_ratio = vowels / len(letters_only)
    if vowel_ratio < 0.15:
        return True

    # Same character repeated 4+ times in a row (e.g. "aaaaa", "asdfff").
    if re.search(r"(.)\1{3,}", text):
        return True

    # A run of 6+ consonants with no vowel in between (e.g. "qwrtplk").
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", text.lower()):
        return True

    return False
