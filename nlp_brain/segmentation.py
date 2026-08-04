"""Multi-symptom sentence segmentation: a query naming more than one problem
("pet mein dard hai aur sar bhi dukh raha hai") should be routed to a
specialist per symptom, not forced into a single label."""
import re

# Words/punctuation that typically join two separate symptom mentions in one
# sentence. Matched case-insensitively; \b keeps it from matching inside
# other words (e.g. won't split "aurat"). Ported from the original prototype
# (Model_1_Doctor_Dekho.py, deleted in 25df289 during the O(N^2) perf rewrite
# that dropped this mechanism along with the file it lived in).
SEGMENT_SPLIT_PATTERN = re.compile(
    r"\s*(?:,|;|/|\bevam\b|\baur\b|\band\b|\bplus\b|\balso\b|\bwith\b|\bsaath hi\b|\bsath hi\b)\s*",
    re.IGNORECASE,
)


def split_segments(text: str) -> list:
    """Break a sentence into separate symptom mentions.

    "pet mein dard hai aur sar bhi dukh raha hai" -> two segments. A plain
    single-symptom sentence (or one with no recognized separator) just comes
    back as a single segment -- callers don't need to special-case the
    common case; it's just the one-segment case of the same path."""
    raw_parts = SEGMENT_SPLIT_PATTERN.split(text)
    segments = [p.strip() for p in raw_parts if p and p.strip()]

    # Guard against over-splitting on short fragments/stray punctuation
    # (e.g. a trailing "," with nothing meaningful after it).
    segments = [s for s in segments if len(s.split()) >= 2 or len(s) >= 4]

    if not segments:
        segments = [text.strip()]
    return segments
