"""What a message means, as a small set of concepts, so that two messages written differently can be matched:
"pet me jalan" and "acidity hoti hai khane ke baad" share `sym:acidity`; "chamdi ka doctor" and "skin doctor chahiye"
share `spec:dermatology`; "shukriya" and "thank you so much" share `move:thanks`; "mummy ke liye" and "maa ke liye
doctor" share `who:parent`.

Built on Vita's own readers (the Hinglish lexicon, the doctor-search reader, the everyday-talk reader), because the
off-the-shelf multilingual embedding models do not understand Romanised Hindi: tested 31 Oct 2026, the multilingual
MiniLM matched "shukriya" to "acidity hoti hai" and "sar dard" to "bete ko bukhar hai" (1-2 of 13 right).

Retrieval scores concepts first and spelling (character n-grams) second: `similarity()`.
"""
from __future__ import annotations

import functools
import re

from ..application import everyday
from ..doctor_intent import parse
from ..textnorm import intent_text, preprocess, tidy

# who the message is about: the words patients use (Hinglish, English, Hindi script after intent_text)
_WHO = (
    ("child", r"\b(beta|bete|beti|betiyan|bachcha|bachche|bachcho|bachchi|baby|bacche|bache|son|daughter|kid|kids|child"
              r"|munna|munni|ladka|ladki ko|baccha|bachha)\b"),
    ("parent", r"\b(maa|ma|mummy|mumma|mom|mother|amma|ammi|papa|pitaji|father|dad|daddy|abbu|dadi|nani|dada|nana"
               r"|grandmother|grandfather|sasur|saas)\b"),
    ("spouse", r"\b(wife|biwi|patni|husband|pati|shohar|begum|gharwali|gharwale)\b"),
    ("other", r"\b(bhai|bhaiya|behen|bahan|didi|sister|brother|dost|friend|chacha|chachi|mama|mami|uncle|aunty|padosi)\b"),
)
_WHO_RE = [(k, re.compile(p)) for k, p in _WHO]


@functools.lru_cache(maxsize=1)
def _corrector():
    from ..lexicon import Lexicon
    from ..spelling import SpellingCorrector
    return SpellingCorrector(Lexicon.from_csv())


@functools.lru_cache(maxsize=1)
def _word_kinds() -> dict[str, str]:
    return {e.variant: e.category for e in _corrector().lex.entries if " " not in e.variant}


@functools.lru_cache(maxsize=4096)
def concepts(text: str) -> frozenset[str]:
    """The concepts in one message. Never raises: a reader that fails adds nothing."""
    out: set[str] = set()
    if not (text or "").strip():
        return frozenset()
    norm = intent_text(tidy(text))
    try:
        corrector = _corrector()
        res = corrector.correct(preprocess(norm))
        out |= {"sym:" + s.lower() for s in res.symptoms if len(s.split()) <= 3}
        # long imported rows mean themselves ("tez bukhar aa raha hai"): their known words still carry the meaning
        kinds = _word_kinds()
        for w in res.corrected_text.split():
            for canonical in corrector.lex.word_meaning.get(w, ()):
                kind = kinds.get(w)
                if kind == "symptom" and len(canonical.split()) <= 3:
                    out.add("sym:" + canonical.lower())
                elif kind == "body_part":
                    out.add("body:" + canonical.lower())
    except Exception:
        pass
    try:
        move = everyday.detect(norm, text)
        if move:
            out.add("move:" + move)
    except Exception:
        pass
    try:
        ask = parse(text)
        if ask.specialty_key:
            out.add("spec:" + ask.specialty_key)
        if ask.sort == "fee":
            out.add("ask:cheap")
        elif ask.sort:
            out.add("ask:" + ask.sort)
        if ask.gender:
            out.add("ask:" + ask.gender)
        if ask.more:
            out.add("ask:more")
        if ask.available_today or ask.when:
            out.add("ask:when")
        if ask.near or ask.place:
            out.add("ask:where")
        if ask.name_query:
            out.add("ask:name")
        if ask.degree:
            out.add("ask:degree")
        if ask.move and not any(c.startswith("move:") for c in out):
            out.add("move:" + ask.move)
    except Exception:
        pass
    for who, rx in _WHO_RE:
        if rx.search(norm):
            out.add("who:" + who)
    if re.search(r"\b(cancel|radd|reschedule|badal\w*|change)\b", norm) and re.search(r"\b(appointment|booking)\b", norm):
        out.add("ask:cancel")
    return frozenset(out)


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Concept overlap (Dice). Symptoms and the specialty count double: they carry the meaning."""
    if not a or not b:
        return 0.0

    def w(c: str) -> float:
        return 2.0 if c.startswith(("sym:", "spec:")) else 1.0
    shared = sum(w(c) for c in a & b)
    return 2 * shared / (sum(w(c) for c in a) + sum(w(c) for c in b))
