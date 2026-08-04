"""Devanagari -> Roman script conversion, so speech transcribed in Hindi
script matches the Roman-script Hinglish spellings the NLP Brain expects."""
import re

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def devanagari_to_roman(text: str) -> str:
    """Convert any Devanagari-script portion of `text` into Roman letters,
    leaving already-Roman words (English, or Hindi typed phonetically)
    untouched, e.g.:
        "आंखों से धुंधला दिखायी दे रहा है" -> "aankhon se dhundhala dikhayi de raha hai"

    If the text has no Devanagari characters at all, it's returned
    unchanged. If the `indic-transliteration` package isn't installed, the
    original text is returned unchanged and a one-time warning is printed.
    """
    if not _DEVANAGARI_RE.search(text):
        return text

    if not TRANSLITERATION_AVAILABLE:
        print(
            "Note: got Devanagari-script speech but the 'indic-transliteration' "
            "package isn't installed, so it can't be converted to Roman script. "
            "Install it with:\n    pip install -r requirements-voice.txt"
        )
        return text

    roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    roman = roman.replace(".a", "").replace("'", "")
    roman = " ".join(roman.split())
    return roman.lower()
