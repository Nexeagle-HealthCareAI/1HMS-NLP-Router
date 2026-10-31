"""How this patient wants Vita to talk, kept for the conversation (and, with consent, for their next visit).

  length      short | normal | detailed   "short me batao", "seedha batao", "bas doctor dikhao" / "detail mein samjhao"
  questions   ok | avoid                  "sawal mat poocho", "no more questions", "bas doctor dikhao"
  tone        formal | casual             how they write: "aap / ji / sir / kripya" vs "tum / yaar / bhai / bro"

Explicit requests win and stay until the patient changes them; tone follows how they write. read() is code (testable,
no model call); the prompts get how_to_reply(), and plan() uses `questions` so "no questions" changes what Vita does,
not only how it sounds.
"""
from __future__ import annotations

import re

from ..textnorm import intent_text, tidy

_SHORT = re.compile(r"\b(short (me|mein|mai)?\s?(batao|bolo|bataiye|likho|answer|reply)|chhot[ae] (jawab|answer|mein)|"
                    r"seedh[ae] (batao|bolo|baat)|to the point|in short|kam (words|shabd)|zyada mat likho|lamba mat likho"
                    r"|keep it short|short answer|be brief|briefly|ek line (mein|me)|jaldi batao|bas (doctor|naam) (batao|dikhao)"
                    r"|sirf doctor (batao|dikhao)|just (show|give|tell) (me )?(the )?doctors?)\b")
_DETAILED = re.compile(r"\b(detail (me|mein|mai|se)|vistar (se|mein)|achhe se samjhao|acche se samjhao|samjha ke batao"
                       r"|explain (in detail|properly|fully)|thoda detail|elaborate|aur detail|in detail|detailed)\b")
_NO_QUESTIONS = re.compile(r"\b((sawal|sawaal|savaal|sawalat|question|questions) (mat|na|nahi|band) (pucho|poocho|puchiye"
                           r"|poochiye|karo|kijiye)|no (more )?questions|stop asking|kitne (sawal|sawaal|question)"
                           r"|bas (doctor|naam) (batao|dikhao)|sirf doctor (batao|dikhao)|seedha doctor|just (show|give)"
                           r"( me)?( the)? doctors?|don ?t ask)\b")
_QUESTIONS_OK = re.compile(r"\b(aap (puch|pooch) (sakti|sakte)|ask me|pucho jo|jo puchna hai pucho)\b")
_FORMAL = re.compile(r"\b(aap|aapka|aapki|aapko|ji|sir|madam|maam|kripya|kripaya|please|shriman|dhanyavaad)\b")
_CASUAL = re.compile(r"\b(tum|tu|tera|teri|tumhara|yaar|yar|bhai|bro|dude|abe|arre|oye)\b")

DEFAULT = {"length": "normal", "questions": "ok", "tone": None}


def read(text: str, prev: dict | None = None) -> dict:
    """The style after this message (prev carried forward)."""
    st = dict(DEFAULT, **(prev or {}))
    norm = intent_text(tidy(text or ""))
    if _SHORT.search(norm):
        st["length"] = "short"
    elif _DETAILED.search(norm):
        st["length"] = "detailed"
    if _NO_QUESTIONS.search(norm):
        st["questions"] = "avoid"
    elif _QUESTIONS_OK.search(norm):
        st["questions"] = "ok"
    formal, casual = len(_FORMAL.findall(norm)), len(_CASUAL.findall(norm))
    if formal > casual:
        st["tone"] = "formal"
    elif casual > formal:
        st["tone"] = "casual"
    return st


def changed(before: dict | None, after: dict) -> bool:
    b = dict(DEFAULT, **(before or {}))
    return any(b.get(k) != after.get(k) for k in ("length", "questions"))


def how_to_reply(style: dict | None, language_pref: str | None = None) -> dict:
    """What the reply prompts are told."""
    st = dict(DEFAULT, **(style or {}))
    out = {"length": {"short": "one short sentence, no extra words",
                      "detailed": "explain a little more (still simple words, up to 5 sentences)"}.get(st["length"],
                                                                                                    "1-3 short sentences"),
           "questions": ("the patient asked not to be asked questions: do not ask; act on what is known"
                         if st["questions"] == "avoid" else "at most one question, only if needed"),
           "tone": {"formal": "respectful (aap, ji)", "casual": "friendly and relaxed, still polite (aap in Hindi)"}
           .get(st["tone"] or "", "warm")}
    if language_pref:
        out["language"] = f"always {language_pref} (the patient asked for it)"
    return out


def digest(messages: list[dict], window: int) -> list[dict]:
    """The turns before the last `window` messages, short: what the patient asked and what Vita did. So earlier
    requests are not forgotten in a long conversation."""
    older = (messages or [])[:-window] if window else list(messages or [])
    out, i = [], 0
    while i < len(older):
        m = older[i]
        if m.get("role") == "patient":
            nxt = older[i + 1] if i + 1 < len(older) and older[i + 1].get("role") == "vita" else {}
            out.append({"patient": (m.get("text") or "")[:100], "vita_did": nxt.get("kind")})
        i += 1
    return out[-12:]
