"""What kind of doctor request a /chat message is - rules only, no LLM, no network.

Vita answers five kinds of ask (Vita Chat requirements, "Five ways to ask"):

    symptom     "2 din se bukhar aur khansi"        -> triage first, doctors on the recommendation
    specialty   "skin ka doctor", "cardiologist"    -> doctors straight away
    name        "Dr Sharma", "dr sarma se milna hai" -> fuzzy name search
    degree      "MD medicine wala", "MS ortho"      -> doctors holding that degree
    distance    "paas mein koi doctor", "5 km ke andar"

They combine ("MD cardiologist near Salt Lake"), and a follow-up can refine the last list
("inme se koi paas wala?", "kam fees wala"). `parse()` only reads the text; deciding what to do with the
result (and checking for an emergency first) is the caller's job.

Specialties come out as doctors.specialty_key() keys ("cardiology", "orthopaedics" ...), the same keys
DoctorDirectory maps to 1HMS categories.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .doctors import specialty_key
from .textnorm import preprocess

# ----------------------------------------------------------------------------- vocabulary
DOCTOR_WORDS = {"doctor", "doctors", "docter", "daktar", "dactor", "dr", "specialist", "specialists", "sahab", "saab"}
_LINKS = {"ka", "ke", "ki", "wala", "wale", "wali", "vala", "vale", "vali", "waala"}

# A specialist noun on its own is enough: "cardiologist", "ortho", "ent specialist"
_SPECIALIST_NOUN = re.compile(
    r"\b(cardiologists?|cardio|dermatologists?|derma|orthopa?edic(?:ians?| surgeons?)?|orthopedists?|ortho|"
    r"gyn(?:a)?ecologists?|gyn(?:a)?e|obstetricians?|dentists?|ent(?: specialists?)?|ophthalmologists?|"
    r"neurologists?|neuro|neurosurgeons?|psychiatrists?|urologists?|gastroenterologists?|gastro|"
    r"endocrinologists?|diabetologists?|pulmonologists?|chest specialists?|nephrologists?|oncologists?|"
    r"rheumatologists?|physiotherapists?|physio|p(?:a)?ediatricians?|child specialists?|geriatricians?|"
    r"general physicians?|physicians?|general surgeons?|surgeons?|plastic surgeons?|vascular surgeons?|"
    r"cardiothoracic surgeons?|eye specialists?|skin specialists?)\b")

# Words that name a specialty only next to a doctor word: "dil ka doctor", "skin doctor", "pet ka doctor".
# Alone they are symptoms ("pet dard", "skin pe daane") and go to triage.
_ALIASES: dict[str, str] = {
    "dil": "cardiology", "heart": "cardiology", "hriday": "cardiology",
    "haddi": "orthopaedics", "haddiyon": "orthopaedics", "bone": "orthopaedics", "bones": "orthopaedics",
    "joint": "orthopaedics", "joints": "orthopaedics", "jodo": "orthopaedics",
    "skin": "dermatology", "chamdi": "dermatology", "twacha": "dermatology", "chamadi": "dermatology",
    "aankh": "ophthalmology", "aankhon": "ophthalmology", "ankh": "ophthalmology", "eye": "ophthalmology",
    "eyes": "ophthalmology",
    "daant": "dentistry", "dant": "dentistry", "daanton": "dentistry", "teeth": "dentistry", "tooth": "dentistry",
    "kaan": "ent", "naak": "ent", "gala": "ent", "gale": "ent", "ear": "ent", "nose": "ent", "throat": "ent",
    "bachcha": "paediatrics", "bachche": "paediatrics", "bachcho": "paediatrics", "bachchon": "paediatrics",
    "bacchon": "paediatrics", "baccho": "paediatrics", "bacche": "paediatrics", "child": "paediatrics",
    "children": "paediatrics", "kids": "paediatrics", "baby": "paediatrics", "shishu": "paediatrics",
    "mahila": "gynaecology", "lady": "gynaecology", "ladies": "gynaecology", "women": "gynaecology",
    "stree": "gynaecology", "aurat": "gynaecology", "aurton": "gynaecology",
    "pet": "gastroenterology", "stomach": "gastroenterology", "liver": "gastroenterology",
    "dimag": "neurology", "dimaag": "neurology", "nas": "neurology", "nerve": "neurology", "brain": "neurology",
    "mann": "psychiatry", "mental": "psychiatry",
    "kidney": "nephrology", "gurda": "nephrology", "gurde": "nephrology",
    "sugar": "endocrinology", "diabetes": "endocrinology", "thyroid": "endocrinology", "hormone": "endocrinology",
    "chest": "pulmonology", "lungs": "pulmonology", "lung": "pulmonology", "phephde": "pulmonology",
    "cancer": "oncology",
    "buzurg": "geriatrics", "buzurgon": "geriatrics", "elderly": "geriatrics",
    "family": "general-medicine", "general": "general-medicine", "medicine": "general-medicine",
    "physio": "physiotherapy",
}

_DEGREE = re.compile(r"\b(m\.?b\.?b\.?s|m\.?d|m\.?s|d\.?m|m\.?ch|d\.?n\.?b|b\.?d\.?s|m\.?d\.?s|d\.?g\.?o|d\.?c\.?h)\b")
_DEGREE_CANON = {"mbbs": "MBBS", "md": "MD", "ms": "MS", "dm": "DM", "mch": "MCh", "dnb": "DNB", "bds": "BDS",
                 "mds": "MDS", "dgo": "DGO", "dch": "DCH"}

_NEAR = re.compile(r"\b(near me|nearby|near|nearest|closest|close by|paas|pass mein|aas ?paas|aaspaas|nazdeek|"
                   r"najdeek|nazdik|najdik|around me|mere area|mere ilake)\b")
_RADIUS = re.compile(r"\b(\d{1,3})\s*(?:km|kms|kilometer|kilometers|kilometre|kilometres)\b")

_REFINE = re.compile(r"\b(inme|inmein|inmese|in mein|in me|unme|unmein|isme|ismein|inhi|inhe|in doctors|in mein se|"
                     r"inme se|inmein se|un mein se|ye wale|yeh wale|ussi list)\b")
_CHEAP = re.compile(r"\b(kam fees|kam fee|sasta|sasti|saste|cheap|cheaper|cheapest|low fee|lowest fee|kam paise|"
                    r"kam charge|less fee|affordable)\b")
_EXPERIENCED = re.compile(r"\b(zyada experience|jyada experience|experienced|senior|most experienced|"
                          r"purane|anubhavi)\b")

_NAME_CUE = re.compile(r"\b(se milna|se milni|se milwa|ko dikhana|ko dikhaana|se dikhana|ka appointment|"
                       r"se appointment|ki appointment|appointment with|see dr|meet dr)\b")
# Words that end a "dr <name>" span or can never be a name
_NAME_STOP = DOCTOR_WORDS | _LINKS | {
    "chahiye", "chaahiye", "ke", "ki", "ka", "se", "ko", "paas", "near", "nearby", "hai", "hain", "ho", "batao",
    "bataiye", "dikhao", "dikhaiye", "milna", "mil", "sakta", "sakte", "available", "kaun", "kon", "koi", "acha",
    "accha", "achha", "best", "good", "top", "for", "the", "a", "and", "aur", "ya", "me", "mein", "my", "mera",
    "meri", "please", "plz", "pls", "sir", "madam", "ji", "sahab", "saab", "ne", "bola", "kaha", "list", "show",
    "find", "need", "want", "chahta", "chahti", "hu", "hoon", "appointment", "book", "booking", "today", "aaj",
    "kal", "abhi", "now", "kahan", "kidhar", "wala", "wali", "fees", "fee",
}
# Everyday words the health vocabulary also contains; they never make a message "about symptoms"
_FILLER = {"milna", "milni", "chahiye", "chaahiye", "paas", "andar", "liye", "wala", "wali", "wale", "koi", "kuch",
           "dikhana", "appointment", "batao", "near", "nearby", "fees", "fee", "kam", "sasta", "experience",
           "experienced", "senior", "list", "area", "sheher", "city", "hospital", "clinic", "book", "booking"}


@dataclass
class DoctorAsk:
    specialty_key: str | None = None        # doctors.specialty_key() key
    specialty_text: str | None = None        # the words it came from, for messages and logs
    name_query: str | None = None
    degree: str | None = None                # canonical: MBBS, MD, MS, DM, MCh, DNB, BDS, MDS, DGO, DCH
    near: bool = False
    radius_km: float | None = None
    refine: bool = False                     # builds on the previous list ("inme se ...")
    sort: str | None = None                  # "fee" | "experience" (refinements)
    symptomish: bool = False                 # describes symptoms -> triage decides the specialty
    modes: list[str] = field(default_factory=list)

    @property
    def direct(self) -> bool:
        """Answer with doctors on this turn, without triage."""
        if self.specialty_key or self.name_query or self.degree:
            return True
        if self.near and not self.symptomish:
            return True
        return False


def _words(normalized: str) -> list[str]:
    return normalized.split()


def _find_specialty(text: str) -> tuple[str | None, str | None, set[str]]:
    """(key, matched words, words consumed)."""
    m = _SPECIALIST_NOUN.search(text)
    if m:
        phrase = m.group(1)
        key = specialty_key(phrase) or specialty_key(phrase.replace("ortho", "orthopaedic")
                                                     .replace("gastro", "gastroenterology")
                                                     .replace("derma", "dermatology")
                                                     .replace("cardio", "cardiology")
                                                     .replace("neuro", "neurology")
                                                     .replace("physio", "physiotherapy"))
        if phrase in ("surgeon", "surgeons"):
            key = "general-surgery"
        if phrase in ("physician", "physicians", "general physician", "general physicians"):
            key = "general-medicine"
        if key:
            return key, phrase, set(phrase.split())
    words = _words(text)
    for i, w in enumerate(words):
        if w not in DOCTOR_WORDS or w == "dr":
            continue
        # look back over at most one link word: "dil ka doctor", "skin doctor", "medicine wala doctor"
        j = i - 1
        while j >= 0 and words[j] in _LINKS and i - j <= 2:
            j -= 1
        if j >= 0 and words[j] in _ALIASES:
            return _ALIASES[words[j]], words[j], {words[j]}
    return None, None, set()


def _find_degree(text: str) -> tuple[str | None, set[str]]:
    m = _DEGREE.search(text)
    if not m:
        return None, set()
    raw = m.group(1).replace(".", "")
    return _DEGREE_CANON.get(raw), {m.group(1)}


def _find_name(text: str, names: Iterable[str] | None) -> tuple[str | None, set[str]]:
    words = _words(text)
    # 1) "dr sharma", "doctor anita sen"
    for i, w in enumerate(words):
        if w in ("dr", "doctor", "daktar", "docter"):
            span = []
            for nxt in words[i + 1:i + 3]:
                if nxt in _NAME_STOP or nxt in _ALIASES or len(nxt) < 3 or _SPECIALIST_NOUN.fullmatch(nxt):
                    break
                span.append(nxt)
            if span:
                return " ".join(span), set(span)
    # 2) a listed doctor's name with a meeting cue: "anita sen se milna hai"
    if names and _NAME_CUE.search(text):
        name_tokens = {t for n in names for t in preprocess(n).split() if len(t) >= 3 and t not in DOCTOR_WORDS}
        hits = [w for w in words if len(w) >= 3 and w not in _NAME_STOP and w not in _ALIASES
                and difflib.get_close_matches(w, name_tokens, n=1, cutoff=0.84)]
        if hits:
            return " ".join(hits[:3]), set(hits)
    return None, set()


def parse(text: str, *, names: Iterable[str] | None = None,
          medical_terms: Callable[[str], list[str]] | None = None) -> DoctorAsk:
    """names: listed doctors' names (DoctorDirectory.names()), used only for asks without "dr".
    medical_terms: MedicalVocabulary.terms_in, to tell a symptom description from a plain request."""
    norm = preprocess(text)
    ask = DoctorAsk()
    consumed: set[str] = set()

    ask.specialty_key, ask.specialty_text, used = _find_specialty(norm)
    consumed |= used
    ask.degree, used = _find_degree(norm)
    consumed |= used
    ask.name_query, used = _find_name(norm, names)
    consumed |= used

    m = _RADIUS.search(norm)
    if m:
        ask.near, ask.radius_km = True, float(m.group(1))
    if _NEAR.search(norm):
        ask.near = True

    if _CHEAP.search(norm):
        ask.sort, ask.refine = "fee", True
    elif _EXPERIENCED.search(norm):
        ask.sort, ask.refine = "experience", True
    if _REFINE.search(norm):
        ask.refine = True

    if medical_terms is not None:
        terms = [t for t in medical_terms(norm)
                 if t not in consumed and t not in _FILLER and t not in DOCTOR_WORDS and t not in _NAME_STOP]
        ask.symptomish = bool(terms)

    if ask.symptomish and not (ask.specialty_key or ask.name_query or ask.degree):
        ask.modes.append("symptom")
    if ask.specialty_key:
        ask.modes.append("specialty")
    if ask.name_query:
        ask.modes.append("name")
    if ask.degree:
        ask.modes.append("degree")
    if ask.near:
        ask.modes.append("distance")
    return ask
