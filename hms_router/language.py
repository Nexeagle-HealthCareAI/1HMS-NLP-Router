"""Step 1 - language detection: english | hinglish | hindi | bangla, and the conversation's language.

Two questions, answered separately:

1. What is THIS message in?  `LanguageDetector.detect`
   - Script decides first: Devanagari -> hindi, Bengali script -> bangla (both unambiguous).
   - Roman script is scored: Hindi words (hai, mujhe, chahiye, dard, kal...) against English words (I, need,
     have, the...). Mostly English stays English even with one Hindi word ("she has fever since kal");
     Roman-script Hindi is "hinglish" (the doc's own example: "kal raat se bahut tez sir dard ho raha hai").
   - Words that mean nothing about language are ignored: names, numbers, cities, "doctor", specialty names,
     and words that are the same in both ("me" is English *and* "pet me dard"; it votes for neither).
   - `strength`: "strong" (clear evidence), "weak" (one short word: "haan", "ok", "yes") or "none"
     (nothing to go on: "Kolkata", "Dr Anita Sen", "3").

2. What is the CONVERSATION in?  `resolve`
   - The first message with any evidence sets it (or the channel's UI language when the first message is
     neutral). Weak or neutral messages keep it: a city typed in reply, a tapped chip, "ok".
   - A strong message in another language switches it: the patient changed language, Vita follows.

fastText lid.176 is optional and only confirms Roman-script text that has no Hindi words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import Lexicon

_DEV = re.compile(r"[ऀ-ॿ]")
_BENGALI = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[a-z']+")

LANGUAGES = ("english", "hinglish", "hindi", "bangla")

# Roman-script Hindi. Grammar words first (they carry the most signal), then everyday and body words.
HINDI_WORDS = frozenset("""
hai hain ho hoon hu hun tha thi the hua hui hue raha rahi rahe rha rhi gaya gayi gaye kya kyu kyun kyon kaise
kab kahan kaha kidhar kaun kaunsa kaunsi kis kisi kise mujhe mujhko mera meri mere mujh hum humko hamara
hamari hamare humein aap aapka aapki aapke aapko apna apni apne tum tumhara nahi nahin nhi mat haan haa ji
bhi aur ya ka ki ke ko se mein mai main ye yeh wo woh vo voh inme inmein isme ismein unme unmein usme jo jaha
jab tab phir fir wala wali wale waale chahiye chaiye chahie chahte chahti chahta batao bataiye bataye
batayein batado dikhao dikhaiye dikhaye dikhado karo kare karein karna karni kar karke kariye dijiye do dena
lena lijiye milega milegi milenge milna milne baithte baithti baithenge sakta sakti sakte skta pata
dard bukhar bukhaar sar sir pet khansi khaansi ulti dast saans sans seene seena chakkar kamzori kamjori
takleef taklif pareshani jalan khujli sujan soojan zukam jukham gala galaa kaan naak aankh daant haddi dil
kal aaj parso subah shaam sham raat din hafte hafta mahine mahina saal pehle abhi jaldi turant
thoda thodi thode bahut bohot bahot zyada jyada kam acha accha achha acchha theek thik sahi galat
namaste namaskar dhanyawad dhanyavaad shukriya bas sirf koi kuch kuchh sab sabse
paas pas nazdeek nazdik ghar bachcha bachche bachchi beta beti maa papa pitaji mummy didi bhai behen
aurat mahila purush admi
""".split())

ENGLISH_WORDS = frozenset("""
i i'm im i've ive you your you're he she we they my his her our their it its is am are was were be been being
have has had having do does did will would can could should shall may might must need needs want wants
looking look show find give tell please help the a an and or but for with from of in on at near around about
since this that these those which who what when where why how yes no not dont don't doesn't didn't cant can't
hello hey thanks thank okay sorry problem pain fever cough headache stomach chest skin heart child kid baby
mother father mom dad wife husband son daughter today tomorrow yesterday morning evening night week month
some any more less cheap cheaper cheapest female male women woman man available appointment book booking
personal phone call talk person someone among one other another nearby closest best good also too very
feel feeling got getting since days hours just only than there here
""".split())

# Same spelling, both languages (or neither): never evidence.
NEUTRAL = frozenset("me to hi na par pe so us use in man bus din ok okk".split())

# Said in every language in India: they keep the conversation's language, never switch it.
COURTESY = frozenset("yes no ok okay thanks thank you hello hey hi sorry please haan ji nahi".split())
_STRONG_SINGLE = {"namaste", "namaskar", "dhanyawad", "dhanyavaad", "shukriya", "chahiye", "batao", "dikhao"}


@dataclass(frozen=True)
class LanguageResult:
    language: str      # "english" | "hindi" | "hinglish" | "bangla" | "unknown"
    script: str        # "latin" | "devanagari" | "bengali" | "mixed" | "none"
    method: str        # "script" | "markers" | "fasttext+markers" | "session" | "hint"
    strength: str = "strong"   # "strong" | "weak" | "none" - how much this one message says


def _squeeze(w: str) -> str:
    return re.sub(r"(.)\1+", r"\1", w)


class LanguageDetector:
    def __init__(self, lexicon: Lexicon | None = None, fasttext_model_path: str | None = None):
        markers = set(lexicon.hindi_markers) if lexicon is not None else set()
        self.hindi = (HINDI_WORDS | markers) - NEUTRAL - ENGLISH_WORDS
        # "bataoo", "dikhaoo", "chahiyeee": a stretched Hindi word is still Hindi
        self._hindi_squeezed = {_squeeze(w): w for w in self.hindi if len(_squeeze(w)) >= 4}   # not se/pas ("see", "pass")
        self._ft = None
        if fasttext_model_path:
            try:
                import fasttext
                self._ft = fasttext.load_model(fasttext_model_path)
            except Exception:  # missing lib / model -> silently fall back to the word scores
                self._ft = None

    def detect(self, text: str) -> LanguageResult:
        dev, ben, lat = len(_DEV.findall(text)), len(_BENGALI.findall(text)), len(_LATIN.findall(text))
        if dev + ben + lat == 0:
            return LanguageResult("unknown", "none", "script", "none")
        if ben > dev and ben >= lat:
            return LanguageResult("bangla", "bengali", "script", "strong")
        if dev and dev >= lat:
            return LanguageResult("hindi", "devanagari" if not lat else "mixed", "script", "strong")
        if dev:                                     # mostly Roman with some Devanagari: Hindi in Roman letters
            return LanguageResult("hinglish", "mixed", "script", "strong")

        words = [w if w in self.hindi or w in ENGLISH_WORDS else self._hindi_squeezed.get(_squeeze(w), w)
                 for w in _WORD.findall(text.lower())]
        h = sum(w in self.hindi for w in words)
        e = sum(w in ENGLISH_WORDS for w in words)
        if h == 0 and e == 0:
            return LanguageResult("unknown", "latin", "markers", "none")
        if h and (h / (h + e)) >= 0.4:
            lang = "hinglish"
        elif e and h == 0 and self._ft is not None:
            label, prob = self._ft.predict(text.replace("\n", " "))
            lang = "hinglish" if label[0] == "__label__hi" and prob[0] > 0.8 else "english"
        else:
            lang = "english"
        own = self.hindi if lang == "hinglish" else ENGLISH_WORDS
        mine = sum(w in own and w not in COURTESY for w in words)
        theirs = (e if lang == "hinglish" else h)
        strong = (mine >= 2 and mine - theirs >= 1) or (lang == "hinglish" and any(w in _STRONG_SINGLE for w in words))
        return LanguageResult(lang, "latin", "markers", "strong" if strong else "weak")


def resolve(detected: LanguageResult, previous: str | None, hint: str | None = None) -> LanguageResult:
    """The conversation's language after this message (see the module docstring)."""
    prev = previous if previous in LANGUAGES else None
    if detected.strength == "strong" or (detected.strength == "weak" and prev is None):
        return detected
    if prev is not None:
        return LanguageResult(prev, detected.script, "session", detected.strength)
    if hint in LANGUAGES:
        return LanguageResult(hint, detected.script, "hint", detected.strength)
    return LanguageResult("english" if detected.language == "unknown" else detected.language,
                          detected.script, detected.method, detected.strength)


def ui_language(code: str | None) -> str | None:
    """The channel's UI language ("en", "hi", "hinglish", "bn") as a conversation language."""
    return {"en": "english", "english": "english", "hi": "hindi", "hindi": "hindi", "hinglish": "hinglish",
            "bn": "bangla", "bangla": "bangla"}.get((code or "").strip().lower())
