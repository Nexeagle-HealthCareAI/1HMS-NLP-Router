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
from .textnorm import intent_text, preprocess

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
    "women": "gynaecology", "womens": "gynaecology", "stree": "gynaecology", "aurat": "gynaecology",
    "aurton": "gynaecology",
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
                     r"inme se|inmein se|un mein se|ye wale|yeh wale|ussi list|among (these|them|those)|of (these|them|those)|"
                     r"from (these|them|those|this list|the list)|in (this|the) list|out of these)\b")
_CHEAP = re.compile(r"\b(kam fees|kam fee|sasta|sasti|saste|cheap|cheaper|cheapest|low fee|lowest fee|kam paise|"
                    r"kam charge|less fee|affordable)\b")
_EXPERIENCED = re.compile(r"\b(zyada experience|jyada experience|experienced|senior|most experienced|"
                          r"purane|anubhavi)\b")

# Conversation moves that are not a search (vision, "How Vita talks")
_PRIVATE_NUMBER = re.compile(r"\b(number|nambar|no\.?|mobile|phone|contact|whatsapp)\b.*\b(do|de|dena|dijiye|chahiye|"
                             r"bhejo|share|send|give|batao|bataiye|milega|hai)\b|\b(give|share|send)\b.*\b(number|mobile|phone|"
                             r"contact)\b|\b(phone|mobile|contact) (number|no)\b|\bdoctor (ka |ki )?(number|mobile|phone|contact)\b")
_OWN_NUMBER = re.compile(r"\b(mera|meri|mere|my|apna|hamara)\b.{0,12}\b(number|mobile|phone|no)\b")
_HUMAN = re.compile(r"\b(human|insaan|insan|real person|kisi (insaan|aadmi|vyakti) se|customer care|customer support|"
                    r"support team|helpline|receptionist|staff se|agent se baat|person se baat|talk to (a )?(person|someone)|"
                    r"call me back|callback|call back)\b")
_JUST_LOOKING = re.compile(r"^((nahi|nahin|no|na) )?(bas |sirf |just )?(pooch raha|pooch rahi|puch raha|puch rahi|dekh raha|dekh rahi|dekh rahe|looking|checking|browsing|time pass|"
                           r"no thanks?|nahi chahiye|abhi nahi|not now|later|baad mein|thanks?|thank you|thanku|shukriya|"
                           r"dhanyavad|dhanyawad|ok bye|bye|ok thanks?|theek hai|thik hai|ok|okay)\b")
_DECLINE = re.compile(r"^(nahi|nahin|nhi|no|na|nope)( (abhi|ji|thanks?|thank you|bas|sir|madam))*( (nahi|nahin|nhi))?$")
_GREETING = re.compile(r"^(hi+|hello|helo|hey|namaste|namaskar|good (morning|afternoon|evening)|kaise ho|hii)\b")
_MORE = re.compile(r"\b(show more|more doctors|aur doctor|aur doctors|aur dikhao|aur dikhaiye|aur batao|dusre doctor|doosre doctor|"
                   r"other doctors|next page|aur options|more options|aur koi)\b")

# "lady doctor", "mahila doctor", "female doctor" -> a female doctor (any specialty); "male doctor" likewise
_GENDER = re.compile(r"\b(lady|ladies|mahila|female|woman|aurat|stree|male|gents|purush|aadmi)\s+(?:(?!ka\b|ke\b|ki\b|rog\b)[a-z]+\s+)?(doctor|doctors|daktar|dr|"
                     r"specialist|physician|surgeon|dentist|gynae\w*|ortho|derma|\w+logists?|\w+icians?)\b"
                     r"|\b(doctor|daktar)\s+(lady|mahila|female)\b")
_FEMALE_WORDS = {"lady", "ladies", "mahila", "female", "woman", "aurat", "stree"}
_TODAY = re.compile(r"\b(aaj|today|abhi)\b.{0,20}\b(available|milenge|mil sakte|mil sakta|baithte|baithenge|hai kya|"
                    r"open|free)\b|\b(available|milenge|baithte)\b.{0,12}\b(aaj|today)\b|\baaj ke doctor\b")
# "Is Dr Sharma available Sunday?", "Dr Sen kal milenge?", "kab baithte hain"
_WHEN_Q = re.compile(r"\b(available|availability|milenge|mil sakte|mil sakti|baithte|baithti|baithenge|aate|aati|"
                     r"aayenge|schedule|timing|timings|kab)\b")
_WEEKDAYS = {"monday": 0, "mon": 0, "somvar": 0, "somwar": 0, "tuesday": 1, "tue": 1, "mangalvar": 1, "mangalwar": 1,
             "wednesday": 2, "wed": 2, "budhvar": 2, "budhwar": 2, "thursday": 3, "thu": 3, "guruvar": 3,
             "guruwar": 3, "brihaspativar": 3, "friday": 4, "fri": 4, "shukravar": 4, "shukrawar": 4,
             "saturday": 5, "sat": 5, "shanivar": 5, "shaniwar": 5, "sunday": 6, "sun": 6, "ravivar": 6,
             "raviwar": 6, "itvar": 6, "itwar": 6}
_RELATIVE_DAYS = {"aaj": 0, "today": 0, "kal": 1, "tomorrow": 1, "parso": 2, "parson": 2}
# "Sadar Hospital Purnia", "AMRI hospital ke doctor", "apollo clinic mein"
_HOSPITAL_WORD = re.compile(r"\b(hospital|hospitals|aspatal|haspatal|asptal|clinic|nursing home|medical college|"
                            r"medical centre|medical center|healthcare|polyclinic)\b")
_HOSPITAL_STOP = {"ke", "ki", "ka", "mein", "me", "se", "wala", "wale", "wali", "doctor", "doctors", "dr", "chahiye",
                  "koi", "ek", "the", "a", "in", "at", "of", "kaun", "kon", "kaunse", "best", "acha", "accha", "achha",
                  "paas", "near", "nearby", "list", "batao", "dikhao", "aur", "and", "ya", "or", "mera", "mere", "my",
                  "mujhe", "hai", "hain", "kya", "jana", "jaana", "go", "for", "pe", "par", "available", "kab",
                  "sarkari", "government", "private", "mahila", "lady"}

_NAME_CUE = re.compile(r"\b(se milna|se milni|se milwa|ko dikhana|ko dikhaana|se dikhana|ka appointment|"
                       r"se appointment|ki appointment|appointment with|see dr|meet dr)\b")
# Words that end a "dr <name>" span or can never be a name
_NAME_STOP = DOCTOR_WORDS | _LINKS | {
    "chahiye", "chaahiye", "ke", "ki", "ka", "se", "ko", "paas", "near", "nearby", "hai", "hain", "ho", "batao",
    "bataiye", "dikhao", "dikhaiye", "milna", "mil", "sakta", "sakte", "available", "kaun", "kon", "koi", "acha",
    "accha", "achha", "best", "good", "top", "for", "the", "a", "and", "aur", "ya", "me", "mein", "my", "mera",
    "meri", "please", "plz", "pls", "sir", "madam", "ji", "sahab", "saab", "ne", "bola", "kaha", "list", "show",
    "among", "these", "those", "them", "this", "from", "of", "out", "any", "one", "only", "just",
    "find", "need", "want", "chahta", "chahti", "hu", "hoon", "appointment", "book", "booking", "today", "aaj",
    "kal", "abhi", "now", "kahan", "kidhar", "wala", "wali", "fees", "fee", "number", "mobile", "phone",
    "contact", "whatsapp", "nambar", "do", "de", "dena", "hospital", "clinic", "aspatal", "haspatal",
    "lady", "mahila", "female", "male", "kab", "milenge", "baithte", "baithenge", "aate", "aati", "parso",
    "tomorrow", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "ravivar",
    "somvar", "mangalvar", "budhvar", "guruvar", "shukravar", "shanivar", "on", "is", "are", "will",
    "yahan", "idhar", "wahan", "udhar", "area", "city", "sheher",
}
# Everyday words the health vocabulary also contains; they never make a message "about symptoms"
_FILLER = {"available", "availability", "aaj", "today", "kal", "kab", "milenge", "hospital", "clinic", "lady",
           "mahila", "female", "male", "baithte", "milna", "milni", "chahiye", "chaahiye", "paas", "andar", "liye", "wala", "wali", "wale", "koi", "kuch",
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
    more: bool = False                       # "aur doctor dikhao": the next page of the last list
    gender: str | None = None                # "female" | "male": "lady doctor", "mahila doctor"
    available_today: bool = False            # "aaj available"
    hospital_query: str | None = None        # "sadar hospital purnia" (the words around "hospital")
    place: str | None = None                 # "near Purnia", "Salt Lake mein": a city / area named, not "near me"
    when: str | None = None                  # availability question: "today" | "+N" days | "dow:N" weekday | "next"
    move: str | None = None                  # not a search: private_number | human | just_looking | greeting
    modes: list[str] = field(default_factory=list)

    @property
    def direct(self) -> bool:
        """Answer with doctors on this turn, without triage."""
        if self.specialty_key or self.name_query or self.degree or self.hospital_query or self.gender:
            return True
        if self.near and not self.symptomish:
            return True
        return False


def _words(normalized: str) -> list[str]:
    return normalized.split()


_PHRASE_SPECIALTY = [(re.compile(r"\b(mahila|stree|aurton ke?) rog\b"), "gynaecology"),
                     (re.compile(r"\b(bachchon|bacchon|bachcho|shishu) rog\b"), "paediatrics"),
                     (re.compile(r"\b(twacha|chamdi|skin) rog\b"), "dermatology"),
                     (re.compile(r"\b(netra|aankh) rog\b"), "ophthalmology"),
                     (re.compile(r"\b(dant|daant) rog\b"), "dentistry")]
_NOT_PLACE = {"me", "here", "home", "area", "yahan", "idhar", "dard", "pain", "hospital", "clinic", "jaldi"}


def _find_specialty(text: str) -> tuple[str | None, str | None, set[str]]:
    """(key, matched words, words consumed)."""
    for pattern, key in _PHRASE_SPECIALTY:
        m = pattern.search(text)
        if m:
            return key, m.group(0), set(m.group(0).split())
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


def _find_name(text: str, names: Iterable[str] | None, exclude: set[str] = frozenset()) -> tuple[str | None, set[str]]:
    words = [w for w in _words(text) if w not in exclude]
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
    norm = intent_text(text)          # Hindi script -> the casual Hinglish these rules read
    ask = DoctorAsk()
    consumed: set[str] = set()

    ask.specialty_key, ask.specialty_text, used = _find_specialty(norm)
    consumed |= used
    ask.degree, used = _find_degree(norm)
    consumed |= used
    ask.place = _find_place(norm, medical_terms)
    place_words = set(ask.place.split()) if ask.place else set()
    ask.name_query, used = _find_name(norm, names, exclude=place_words)
    consumed |= used | place_words

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

    g = _GENDER.search(norm)
    if g:
        word = g.group(1) or g.group(4)
        ask.gender = "female" if word in _FEMALE_WORDS else "male"
        ask.refine = True
    if _TODAY.search(norm):
        ask.available_today = ask.refine = True
    ask.hospital_query = _find_hospital(norm)
    if (ask.name_query or "dr" in norm.split() or "doctor" in norm.split()) and _WHEN_Q.search(norm) \
            and not ask.available_today and not ask.specialty_key:
        ask.when = _find_when(norm)
    elif ask.name_query and ask.available_today:
        ask.when, ask.available_today = "+0", False      # "Dr Sharma aaj milenge?" is a question about one doctor

    if _MORE.search(norm):
        ask.more = ask.refine = True
    if _HUMAN.search(norm):
        ask.move = "human"
    elif _PRIVATE_NUMBER.search(norm) and not _OWN_NUMBER.search(norm):
        ask.move = "private_number"

    if medical_terms is not None:
        terms = [t for t in medical_terms(norm)
                 if t not in consumed and t not in _FILLER and t not in DOCTOR_WORDS and t not in _NAME_STOP]
        ask.symptomish = bool(terms)

    plain = not (ask.specialty_key or ask.name_query or ask.degree or ask.near)
    if ask.move is None and plain:
        m = _JUST_LOOKING.match(norm) or _DECLINE.match(norm)
        rest = norm[m.end():] if m else ""
        if m and not (medical_terms and [t for t in medical_terms(rest) if t not in _FILLER]):
            ask.move, ask.symptomish = "just_looking", False   # "bas dekh raha tha", "nahi abhi nahi"
        elif not ask.symptomish and _GREETING.match(norm) and len(norm.split()) <= 4:
            ask.move = "greeting"
    if ask.symptomish and not (ask.specialty_key or ask.name_query or ask.degree):
        ask.modes.append("symptom")
    if ask.specialty_key:
        ask.modes.append("specialty")
    if ask.name_query:
        ask.modes.append("name")
    if ask.degree:
        ask.modes.append("degree")
    if ask.hospital_query:
        ask.modes.append("hospital")
    if ask.gender:
        ask.modes.append("gender")
    if ask.available_today:
        ask.modes.append("today")
    if ask.when:
        ask.modes.append("availability")
    if ask.place:
        ask.modes.append("place")
    if ask.near:
        ask.modes.append("distance")
    return ask


def _find_place(norm: str, medical_terms) -> str | None:
    """"near Purnia", "in salt lake", "kolkata mein" at the end of the message -> the place, used as the city
    when the patient has not shared a location. Never a symptom, specialty or doctor word."""
    if _HOSPITAL_WORD.search(norm):
        return None
    words = norm.split()
    lead = _leading_place(words, norm, medical_terms)      # Hindi order: "kolkata mein skin doctor"
    if lead:
        return lead
    if len(words) >= 2 and words[-1] in ("mein", "me"):
        tail, cut = words[:-1], None                       # "... kolkata mein"
    elif len(words) >= 2 and any(w in ("near", "in", "at") for w in words[-3:-1]):
        cut = max(i for i, w in enumerate(words) if w in ("near", "in", "at"))
        tail = words[cut + 1:]                             # "... near purnia", "... in salt lake"
    else:
        return None
    for size in (2, 1):                                    # "salt lake" before "lake"
        cand = tail[-size:] if cut is None else tail[:size]
        if len(cand) != size or (cut is not None and len(tail) != size):
            continue
        if any(len(w) < 4 for w in cand) or set(cand) & (_NOT_PLACE | DOCTOR_WORDS | _NAME_STOP | set(_ALIASES)):
            continue
        place = " ".join(cand)
        if _SPECIALIST_NOUN.search(place) or (medical_terms and [t for t in medical_terms(place) if t not in _FILLER]):
            continue
        return place
    return None


_LEAD_PLACE_END = {"mein", "me", "ke"}


def _leading_place(words: list[str], norm: str, medical_terms) -> str | None:
    """'kolkata mein skin doctor', 'purnia ke paas cardiologist', 'salt lake mein lady doctor': a place first,
    as Hindi puts it. Only in a doctor ask (a doctor or specialist word follows), so 'kamar mein dard' stays a
    symptom."""
    if not (set(words) & DOCTOR_WORDS or _SPECIALIST_NOUN.search(norm)):
        return None
    for size in (1, 2):
        if len(words) <= size + 1 or words[size] not in _LEAD_PLACE_END:
            continue
        cand = words[:size]
        if any(len(w) < 4 for w in cand) or set(cand) & (_NOT_PLACE | DOCTOR_WORDS | _NAME_STOP | set(_ALIASES)):
            continue
        place = " ".join(cand)
        if _SPECIALIST_NOUN.search(place) or _GENDER_WORD.search(place) or _find_specialty(place)[0] \
                or (medical_terms and [t for t in medical_terms(place) if t not in _FILLER]):
            continue
        return place
    return None


_GENDER_WORD = re.compile(r"\b(lady|ladies|mahila|female|woman|aurat|stree|male|gents|purush|aadmi|mujhe|mere|meri|"
                          r"inme|isme|unme|aaj|kal|abhi)\b")


def _find_hospital(norm: str) -> str | None:
    """The hospital the patient named: up to three words before "hospital" (its name) and up to two after
    (usually the city). "Sadar Hospital Purnia" -> "sadar hospital purnia"; "koi hospital" -> None."""
    m = _HOSPITAL_WORD.search(norm)
    if not m:
        return None
    before = norm[:m.start()].split()
    after = norm[m.end():].split()
    name: list[str] = []
    for w in reversed(before):
        if w in _HOSPITAL_STOP or w in DOCTOR_WORDS or _SPECIALIST_NOUN.fullmatch(w) or len(name) == 3:
            break
        name.insert(0, w)
    tail: list[str] = []
    for w in after:
        if w in _HOSPITAL_STOP or w in DOCTOR_WORDS or _SPECIALIST_NOUN.fullmatch(w) or len(tail) == 2:
            break
        tail.append(w)
    if not name and not tail:
        return None
    return " ".join([*name, m.group(1), *tail])


def _find_when(norm: str) -> str:
    """'aaj' -> '+0', 'kal' -> '+1', 'sunday' -> 'dow:6', otherwise 'next' (the next working day)."""
    for w in norm.split():
        if w in _RELATIVE_DAYS:
            return f"+{_RELATIVE_DAYS[w]}"
        if w in _WEEKDAYS:
            return f"dow:{_WEEKDAYS[w]}"
    return "next"
