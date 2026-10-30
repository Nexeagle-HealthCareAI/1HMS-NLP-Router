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

import csv
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .doctors import specialty_key
from .search_spelling import CityIndex, NameIndex, WordIndex, name_index
from .textnorm import intent_text, preprocess

# ----------------------------------------------------------------------------- vocabulary
DOCTOR_WORDS = {"doctor", "doctors", "docter", "daktar", "dactor", "dr", "specialist", "specialists", "sahab", "saab"}
_LINKS = {"ka", "ke", "ki", "wala", "wale", "wali", "vala", "vale", "vali", "waala"}
_LOOKBACK_LINKS = _LINKS | {"liye", "lie", "liya"}
# words that may surround a one-word specialty answer ("nahi, dil ka", "actually aankh wala chahiye")
_SHORT_FILLER = _LINKS | {"nahi", "nahin", "nhi", "no", "actually", "bas", "sirf", "to", "toh", "ab", "phir", "please", "plz",
                         "chahiye", "chaiye", "dikhao", "batao", "wala", "doctor", "doctors", "specialist", "mujhe", "hume",
                         "humein", "liye", "ok", "okay", "acha", "accha", "achha"}

# A specialist noun on its own is enough: "cardiologist", "ortho", "ent specialist"
_SPECIALIST_NOUN = re.compile(
    r"\b(cardiologists?|cardio|dermatologists?|derma|orthopa?edic(?:ians?| surgeons?)?|orthopedists?|ortho|"
    r"gyn(?:a)?ecologists?|gyn(?:a)?e|obstetricians?|dentists?|ent(?: specialists?)?|ophthalmologists?|"
    r"neurologists?|neuro|neurosurgeons?|psychiatrists?|urologists?|gastroenterologists?|gastro|"
    r"endocrinologists?|diabetologists?|pulmonologists?|chest specialists?|nephrologists?|oncologists?|"
    r"rheumatologists?|physiotherapists?|physio|p(?:a)?ediatricians?|child specialists?|geriatricians?|"
    r"general physicians?|physicians?|general surgeons?|surgeons?|plastic surgeons?|vascular surgeons?|"
    r"cardiothoracic surgeons?|eye specialists?|skin specialists?|"
    # the specialty itself, as people also write it: "cardiology doctor", "DM cardiology", "dermatology"
    r"cardiology|dermatology|neurology|gastroenterology|orthopa?edics|gyn(?:a)?ecology|obstetrics|ophthalmology|"
    r"urology|nephrology|oncology|pulmonology|psychiatry|endocrinology|diabetology|rheumatology|p(?:a)?ediatrics|"
    r"physiotherapy|dentistry|general medicine|general surgery)\b")

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

# Everyday words for specialties that a doctor signs off (data/everyday_specialty_words.csv): "proposed" and "approved"
# rows are used, "rejected" and "needs decision" rows are not. A word already above keeps its meaning.
EVERYDAY_SPECIALTY_WORDS = Path(__file__).resolve().parent / "data" / "everyday_specialty_words.csv"


def _load_everyday_specialties(path: Path = EVERYDAY_SPECIALTY_WORDS) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {r["word"].strip().lower(): r["specialty_key"].strip() for r in csv.DictReader(f)
                if r.get("word") and r.get("specialty_key") and (r.get("status") or "").strip() in ("proposed", "approved")}


for _w, _k in _load_everyday_specialties().items():
    _ALIASES.setdefault(_w, _k)

# "md", "m.d", and "m d" (what "M.D." becomes once punctuation is removed); letters only, so "m s" is a degree only
# as a whole spaced token run
_DEGREE = re.compile(r"\b(m[. ]?b[. ]?b[. ]?s|m[. ]?d[. ]?s|m[. ]?ch|d[. ]?n[. ]?b|b[. ]?d[. ]?s|d[. ]?g[. ]?o|d[. ]?c[. ]?h|"
                     r"m[. ]?d|m[. ]?s|d[. ]?m)\b")
# a patient naming several degrees means the most specific one ("MBBS, MD" -> MD; "MD DM" -> DM)
_DEGREE_RANK = {"DM": 4, "MCh": 4, "MD": 3, "MS": 3, "MDS": 3, "DNB": 3, "DGO": 2, "DCH": 2, "MBBS": 1, "BDS": 1}
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
                    r"call me back|callback|call back|call me|call karo|call kijiye|mujhe call|baat karao|baat karwao|"
                    r"(koi |kisi )?(aadmi|banda|person|vyakti|staff) se baat|complaint|shikayat|agent please|talk to (a )?human|"
                    r"speak to (a )?(person|human|someone)|real person)\b|^agent$")
_JUST_LOOKING = re.compile(r"^((nahi|nahin|no|na) )?(bas |sirf |just )?(pooch raha|pooch rahi|puch raha|puch rahi|dekh raha|dekh rahi|dekh rahe|jaanna tha|janna tha|looking|checking|browsing|time pass|"
                           r"no thanks?|nahi chahiye|abhi nahi|not now|later|baad mein|thanks?|thank you|thanku|shukriya|"
                           r"dhanyavad|dhanyawad|ok bye|bye|ok thanks?|theek hai|thik hai|ok|okay)\b")
_DECLINE = re.compile(r"^(nahi|nahin|nhi|no|na|nope)( (abhi|ji|thanks?|thank you|bas|sir|madam))*( (nahi|nahin|nhi))?$")
_GREETING = re.compile(r"^(koi hai|suniye|suno|excuse me|(sir|madam|mam|maam|madam ji|sir ji|bhaiya|didi)$|h+i+|h+e+y+a*|h+e+l+o+w?|hlo+|hy+|hola|yo+|namast[ea]+y?|namaskar|pranam|pranaam|"
                       r"salaam|salam|assalam\w*|adaab|sat sri akal|jai shri krishna|radhe radhe|ram ram|"
                       r"good (morning|afternoon|evening|night)|gm|kaise ho|kaisi ho|kaise hain|kya haal|how are you|"
                       r"what'?s up|wassup|sup)\b")
_MORE = re.compile(r"\b(show more|more doctors|aur doctor|aur doctors|aur dikhao|aur dikhaiye|aur batao|dusre doctor|doosre doctor|"
                   r"other doctors|next page|aur options|more options|aur koi)\b")

# "lady doctor", "mahila doctor", "female doctor" -> a female doctor (any specialty); "male doctor" likewise
_GENDER = re.compile(r"\b(lady|mahila|female|woman|aurat|stree|male|gents|purush|aadmi)\s+(?:(?!ka\b|ke\b|ki\b|rog\b)[a-z]+\s+)?(doctor|doctors|daktar|dr|"
                     r"specialist|physician|surgeon|dentist|gynae\w*|ortho|derma|\w+logists?|\w+icians?)\b"
                     r"|\b(doctor|daktar)\s+(lady|mahila|female)\b")
_FEMALE_WORDS = {"lady", "ladies", "mahila", "female", "woman", "aurat", "stree"}
_TODAY = re.compile(r"\b(aaj|today|abhi)\b.{0,20}\b(available|milenge|milega|milegi|mil sakte|mil sakta|baithte|baithenge|hai kya|"
                    r"open|free)\b|\b(available|milenge|baithte)\b.{0,12}\b(aaj|today)\b|\baaj ke doctor\b")
# "Is Dr Sharma available Sunday?", "Dr Sen kal milenge?", "kab baithte hain"
_WHEN_Q = re.compile(r"\b(available|availability|milenge|milengi|milega|milegi|mil sakte|mil sakti|baithte|baithti|baithenge|"
                     r"baithengi|aate|aati|aayenge|aayengi|aaenge|schedule|timing|timings|kab)\b")
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
    # asking words, so "doctor bataoo", "dr suggest karo" are never a doctor called "bataoo" / "suggest"
    "btao", "bata", "bataye", "batayein", "bataiye", "batana", "dikha", "dikhaye", "dikhayein", "dikhana",
    "dikhado", "batado", "dhundo", "dhundho", "dhoondo", "dhoondho", "dhundh", "dhoondh", "dhundiye", "dhoondhiye",
    "chahie", "chaiye", "chiye", "chaahie", "suggest", "recommend", "karo", "kariye", "do", "dijiye", "milega",
    "milegi", "milenge", "sabse", "kaunsa", "konsa", "kaun", "achcha", "acchha", "badhiya", "badiya", "bdiya",
    "accha", "achha", "acha", "koi", "bhi", "ek", "mujhe", "hame", "hume", "humein", "hamein", "nearest", "nazdeek",
    "nazdik", "pass", "lao", "please", "help", "madad", "chahiye", "chaahiye", "chahiyee", "doctors", "doctr", "dctr",
    # answers to Vita's questions, never a name
    "nahi", "nahin", "nai", "haan", "han", "ha", "yes", "no", "nope", "ok", "okay", "theek", "thik", "sahi", "galat",
    "bilkul", "zaroor", "jarur", "shayad", "pata", "malum", "maloom", "din", "dino", "hafte", "mahine", "saal",
}
_NAME_STOP_SQUEEZED = {re.sub(r"(.)\1+", r"\1", w) for w in _NAME_STOP}
# Everyday words the health vocabulary also contains; they never make a message "about symptoms"
_FILLER = {"available", "availability", "aaj", "today", "kal", "kab", "milenge", "hospital", "clinic", "lady",
           "mahila", "female", "male", "baithte", "milna", "milni", "chahiye", "chaahiye", "paas", "andar", "liye", "wala", "wali", "wale", "koi", "kuch",
           "dikhana", "appointment", "batao", "near", "nearby", "fees", "fee", "kam", "sasta", "experience",
           "experienced", "senior", "list", "area", "sheher", "city", "hospital", "clinic", "book", "booking"}


# ----------------------------------------------------------------------------- typing mistakes in search words
# Every specialty word a patient may type, singular and plural. A misspelling of one of these ("cardiolgist",
# "gynacologist", "pediatrision") is read as the word; see search_spelling.WordIndex for how close is close enough.
_SPECIALTY_FORMS = """
cardiologist cardiology cardio dermatologist dermatology derma orthopaedic orthopedic orthopaedics orthopedics
orthopedician orthopaedician orthopedist ortho gynaecologist gynecologist gynaecology gynecology gynae gyne
obstetrician obstetrics dentist dentistry ophthalmologist ophthalmology neurologist neurology neuro neurosurgeon
psychiatrist psychiatry urologist urology gastroenterologist gastroenterology gastro endocrinologist endocrinology
diabetologist diabetology pulmonologist pulmonology nephrologist nephrology oncologist oncology rheumatologist
rheumatology physiotherapist physiotherapy physio paediatrician pediatrician paediatrics pediatrics geriatrician
physician surgeon
""".split()
_PLURAL_OK = {"cardiologist", "dermatologist", "orthopedist", "gynaecologist", "gynecologist", "obstetrician", "dentist",
              "ophthalmologist", "neurologist", "neurosurgeon", "psychiatrist", "urologist", "gastroenterologist",
              "endocrinologist", "diabetologist", "pulmonologist", "nephrologist", "oncologist", "rheumatologist",
              "physiotherapist", "paediatrician", "pediatrician", "geriatrician", "physician", "surgeon"}


def _form_meaning(word: str) -> str:
    if word in ("surgeon", "surgeons"):
        return "general-surgery"
    if word in ("physician", "physicians"):
        return "general-medicine"
    return _find_specialty(word)[0] or word


_LONG_WORDS: dict[str, str] = {}          # filled below, once _find_specialty exists
_SHORT_WORDS: dict[str, str] = {}
_DOCTOR_FORMS = WordIndex({"doctor": "doctor", "doctors": "doctor", "specialist": "doctor", "specialists": "doctor"})
_DOCTOR_SHORT = {"doc": "doctor", "docs": "doctors", "drr": "dr", "dctr": "doctor", "dktr": "doctor", "doctr": "doctor",
                 "docter": "doctor", "doctorr": "doctor"}
_MBBS = re.compile(r"^m+b+s+$")
# Ordinary words that are close to a search word and must never become one ("done" is not "bone", "here" not "heart")
_PLAIN = {"done", "dont", "gone", "tone", "zone", "none", "cone", "lone", "hard", "heard", "hearts", "chess", "cheat",
          "river", "lover", "liver", "dyes", "byes", "skip", "skit", "kin", "kidnap", "cheap", "child", "chill", "kids",
          "bane", "bond", "band", "test", "best", "rest", "nest", "west", "past", "post", "most", "must", "just",
          "ear", "eat", "eye", "dear", "near", "hear", "year", "here", "there", "where", "wear", "bear", "fear",
          "mental", "rental", "dental", "medical", "medicine", "general", "family", "nerve", "nerves", "never",
          "sugar", "super", "lung", "lungs", "long", "brain", "train", "rain", "pain", "chest", "skin", "bone",
          "bones", "heart", "kidney", "stomach", "teeth", "tooth", "baby", "nose", "throat", "joint", "joints",
          "women", "woman", "elderly", "cancer", "thyroid", "hormone", "diabetes", "doctor", "doctors", "dekho",
          "dekha", "dekhe", "dekhna", "dikha", "dawa", "dawai", "dard", "dost", "dono", "dosa", "doctrine", "physical",
          "donor", "dollar", "doctorate", "director", "detail", "details", "special", "specially", "specific",
          "surgery", "surgical", "operation",
          # real specialties Vita has no word for: never "corrected" into a neighbour (radiologist is not cardiologist)
          "radiologist", "radiology", "pathologist", "pathology", "hematologist", "haematologist", "hematology",
          "haematology", "immunologist", "allergist", "sexologist", "audiologist", "cosmetologist", "trichologist",
          "podiatrist", "chiropractor", "homeopath", "homoeopath", "homeopathy", "ayurvedic", "dietician", "dietitian",
          "nutritionist", "psychologist", "counsellor", "counselor", "optometrist", "orthodontist", "periodontist",
          "endodontist", "prosthodontist", "neonatologist", "anaesthetist", "anesthetist", "anesthesiologist",
          "venereologist", "andrologist", "hepatologist", "proctologist", "virologist", "embryologist",
          "histopathologist", "physiologist", "pharmacologist", "sonologist", "intensivist", "neonatology"}
_INDEX: dict[str, WordIndex] = {}


def _indexes() -> tuple[WordIndex, WordIndex]:
    if not _INDEX:
        for w in _SPECIALTY_FORMS:
            meaning = _form_meaning(w)
            target = _LONG_WORDS if len(w) >= 7 else _SHORT_WORDS
            target[w] = meaning
            if w in _PLURAL_OK:
                target[w + "s"] = meaning
        for w, k in _ALIASES.items():
            if len(w) >= 4 and " " not in w:
                _SHORT_WORDS.setdefault(w, k)
        _INDEX["long"], _INDEX["short"] = WordIndex(_LONG_WORDS), WordIndex(_SHORT_WORDS)
    return _INDEX["long"], _INDEX["short"]


def _known_word(w: str) -> bool:
    return (w in _PLAIN or w in _ALIASES or w in DOCTOR_WORDS or w in _NAME_STOP or w in _LINKS or w in _SHORT_FILLER
            or w in _FILLER or w in _TIME_WORDS or w in _WEEKDAYS or w in _HOSPITAL_STOP or w in _NOT_PLACE
            or w in _FEMALE_WORDS or w in _DEGREE_CANON)


def fix_search_words(norm: str, keep: Iterable[str] = (), medical_words: Iterable[str] | None = None) -> str:
    """Repair typing mistakes in the words a doctor search reads: the doctor word ("doktor", "doctr"), specialty
    words ("cardiolgist", "skn doctor", "haddi ka doktor") and MBBS ("mbs", "mbbbs"). `keep`: words never changed
    (listed doctors' and cities' names). medical_words: exact symptom vocabulary, also never changed.
    Short words ("skn", "bone", "heart") are repaired only when the message asks for a doctor, so a typo in a
    symptom is left for the symptom spell-checker."""
    if not norm:
        return norm
    long_ix, short_ix = _indexes()
    keep = set(keep)
    medical = medical_words if medical_words is not None else ()
    words = norm.split()
    out = list(words)
    for i, w in enumerate(words):                         # 1) the doctor word itself
        if w in _DOCTOR_SHORT:
            out[i] = _DOCTOR_SHORT[w]
        elif w not in DOCTOR_WORDS and w not in keep and w not in _PLAIN and 5 <= len(w) <= 12 and w[0] in "ds" \
                and w not in medical:
            hit = _DOCTOR_FORMS.closest(w, max_typos=2 if len(w) >= 6 else 1)
            if hit and hit[0] == w[0]:
                out[i] = hit
    asks_doctor = any(w in DOCTOR_WORDS for w in out) or any(w in long_ix for w in out)
    for i, w in enumerate(out):                           # 2) specialty words and MBBS
        if w in keep or w.isdigit() or w in DOCTOR_WORDS or w in long_ix or w in short_ix:
            continue
        if _MBBS.match(w) and w != "mbbs":
            out[i] = "mbbs"
            continue
        if len(w) < 3 or _known_word(w) or w in medical:
            continue
        hit = long_ix.closest(w) if len(w) >= 6 else None
        if hit is None and asks_doctor:
            if len(w) == 3:                               # "skn doctor": one letter left out of a short word
                cands = [s for s in short_ix.meanings if len(s) == 4 and _subsequence(w, s)]
                hit = cands[0] if len({short_ix.meanings[c] for c in cands}) == 1 else None
            else:
                hit = short_ix.closest(w, max_typos=1) or (long_ix.closest(w) if len(w) >= 5 else None)
        if hit:
            out[i] = hit
    return " ".join(out)


def _subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return short[0] == long[0] and all(ch in it for ch in short)


def close_name_word(w: str, tokens: list[str]) -> bool:
    """A word of a listed doctor's name, allowing typing mistakes and sound-alike spellings ("ghose" / "ghosh")."""
    if difflib.get_close_matches(w, tokens, n=1, cutoff=0.84):
        return True
    return len(w) >= 4 and name_index(tuple(tokens)).word(w) is not None


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
                     (re.compile(r"\bladies (doctor|doctors|dr|daktar)\b"), "gynaecology"),
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
        # look back over link words: "dil ka doctor", "skin doctor", "medicine wala doctor", "bachche ke liye doctor"
        j = i - 1
        while j >= 0 and words[j] in _LOOKBACK_LINKS and i - j <= 3:
            j -= 1
        if j >= 0 and words[j] in _ALIASES:
            return _ALIASES[words[j]], words[j], {words[j]}
    # a short answer without the word doctor: "dil ka", "aankh wala", "nahi, haddi ka", "skin" (a reply to "which doctor?",
    # or a change of mind after a list). Only when nothing else is said, so "dil ki dhadkan tez hai" stays a symptom.
    if 1 <= len(words) <= 4:
        rest = [w for w in words if w not in _SHORT_FILLER]
        if len(rest) == 1 and rest[0] in _ALIASES:
            return _ALIASES[rest[0]], rest[0], {rest[0]}
    return None, None, set()


def _find_degree(text: str) -> tuple[str | None, set[str]]:
    found = []
    for m in _DEGREE.finditer(text):
        canon = _DEGREE_CANON.get(re.sub(r"[. ]", "", m.group(1)))
        if canon:
            found.append((canon, m.group(1)))
    if not found:
        return None, set()
    best = max(found, key=lambda f: _DEGREE_RANK.get(f[0], 0))
    return best[0], {w for _, raw in found for w in raw.replace(".", " ").split()}


def _squeeze(w: str) -> str:
    """bataoo -> batao, chahiyeee -> chahiye, acchha -> acha: typed-out stretching and doubled letters ignored."""
    return re.sub(r"(.)\1+", r"\1", w)


def _is_stop(w: str) -> bool:
    return w in _NAME_STOP or _squeeze(w) in _NAME_STOP_SQUEEZED


def _find_name(text: str, names: Iterable[str] | None, exclude: set[str] = frozenset()) -> tuple[str | None, set[str]]:
    words = [w for w in _words(text) if w not in exclude]
    # 1) "dr sharma", "doctor anita sen"
    for i, w in enumerate(words):
        if w in ("dr", "doctor", "daktar", "docter"):
            span = []
            for nxt in words[i + 1:i + 5]:
                if len(nxt) == 1 and not span:
                    continue                                # an initial: "dr s roy", "dr. a. k. sen"
                if _is_stop(nxt) or nxt in _ALIASES or len(nxt) < 3 or _SPECIALIST_NOUN.fullmatch(nxt):
                    break
                span.append(nxt)
                if len(span) == 2:
                    break
            if span:
                return " ".join(span), set(span)
    # 2) a listed doctor's name with a meeting cue ("anita sen se milna hai"), or the name alone ("neha sharma",
    #    "anita sen ji"): every word left is a name word of one listed doctor
    if names and not _NAME_CUE.search(text):
        rest = [w for w in words if not _is_stop(w) and w not in ("ji", "please", "plz", "sir", "madam")]
        initials = [w for w in rest if len(w) == 1]                  # "s roy", "a k sen": initials, then the surname
        rest = [w for w in rest if len(w) > 1]
        if 1 <= len(rest) <= 3 and len(initials) <= 2 and not any(w in _ALIASES for w in rest):
            for n in names:
                toks = [t for t in preprocess(n).split() if len(t) >= 2 and t not in DOCTOR_WORDS]
                if toks and all(close_name_word(w, toks) for w in rest) and \
                        (len(rest) >= 2 or len(rest[0]) >= 4 or (initials and rest[0] in toks)):
                    return " ".join(rest), set(rest) | set(initials)
    if names and _NAME_CUE.search(text):
        index = name_index(tuple(names))
        hits = [w for w in words if len(w) >= 3 and not _is_stop(w) and w not in _ALIASES and index.word(w)]
        if hits:
            return " ".join(hits[:3]), set(hits)
    return None, set()


def search_text(text: str, *, known_names: Iterable[str] | None = None, cities: CityIndex | None = None,
                medical_terms: Callable[[str], list[str]] | None = None) -> str:
    """intent_text() with typing mistakes in search words repaired (fix_search_words). Listed doctors' and cities'
    names are never changed."""
    keep: set[str] = set()
    if known_names:
        keep |= set(name_index(tuple(known_names)).tokens.meanings)
    if cities:
        keep |= {w for k in cities.canon for w in k.split()}
    medical_words = getattr(getattr(medical_terms, "__self__", None), "words", None)
    return fix_search_words(intent_text(text), keep, medical_words)


def parse(text: str, *, names: Iterable[str] | None = None,
          medical_terms: Callable[[str], list[str]] | None = None,
          known_names: Iterable[str] | None = None, cities: CityIndex | None = None) -> DoctorAsk:
    """names: listed doctors' names (DoctorDirectory.names()), used only for asks without "dr".
    medical_terms: MedicalVocabulary.terms_in, to tell a symptom description from a plain request.
    known_names / cities: every listed doctor's name and city (DoctorDirectory.index()): their words are never
    "corrected", and a city named anywhere in a doctor ask ("skin doctor kolkata", "cardiologist kolkatta") is
    the place to search."""
    # Hindi script -> the casual Hinglish these rules read; typing mistakes in search words repaired
    norm = search_text(text, known_names=known_names or names, cities=cities, medical_terms=medical_terms)
    ask = DoctorAsk()
    consumed: set[str] = set()

    ask.specialty_key, ask.specialty_text, used = _find_specialty(norm)
    consumed |= used
    ask.degree, used = _find_degree(norm)
    consumed |= used
    ask.place = _find_place(norm, medical_terms)
    place_words = set(ask.place.split()) if ask.place else set()
    if cities:
        if ask.place:
            ask.place = (cities.resolve(ask.place) or ask.place).lower()       # "kolkatta", "calcutta" -> "kolkata"
        else:
            ask.place, place_words = _city_in(norm, cities, consumed, known_names or names)
    if ask.place and not (ask.specialty_key or ask.name_query or ask.degree or ask.hospital_query):
        ask.refine = True                        # "Patna mein dikhao": the same search, somewhere else
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
    if _HUMAN.search(norm) and not re.search(r"\b(call|baat|insaan|agent)\b.{0,15}\b(nahi|nahin|mat|no|not)\b", norm):
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
    m = _PLACE_DOCTORS.match(norm)                         # "kishanganj ke doctor", "purnia wale doctors"
    if m:
        cand = m.group(1).split()
        if not (any(len(w) < 4 for w in cand) or set(cand) & (_NOT_PLACE | DOCTOR_WORDS | _NAME_STOP | set(_ALIASES) |
                                                                _TIME_WORDS)
                or _SPECIALIST_NOUN.search(m.group(1)) or _GENDER_WORD.search(m.group(1))
                or (medical_terms and [t for t in medical_terms(m.group(1)) if t not in _FILLER])):
            return m.group(1)
    only = _PLACE_ONLY.match(norm)                         # "patna mein dikhao", "nahi, kolkata me": the same search elsewhere
    if only:
        cand = only.group(1).split()
        if not (any(len(w) < 4 for w in cand) or set(cand) & (_NOT_PLACE | DOCTOR_WORDS | _NAME_STOP | set(_ALIASES) | _TIME_WORDS)
                or _GENDER_WORD.search(only.group(1))
                or (medical_terms and [t for t in medical_terms(only.group(1)) if t not in _FILLER])):
            return only.group(1)
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
        if any(len(w) < 4 for w in cand) or set(cand) & (_NOT_PLACE | DOCTOR_WORDS | _NAME_STOP | set(_ALIASES) | _TIME_WORDS):
            continue
        place = " ".join(cand)
        if _SPECIALIST_NOUN.search(place) or (medical_terms and [t for t in medical_terms(place) if t not in _FILLER]):
            continue
        return place
    return None


def _city_in(norm: str, cities: CityIndex, consumed: set[str], names: Iterable[str] | None) -> tuple[str | None, set[str]]:
    """A listed city named anywhere in a doctor ask, typed as people type it: "skin doctor kolkata",
    "cardiologist kolkatta", "calcutta ka cardiologist". Never a word of the ask itself or of a doctor's name."""
    words = norm.split()
    if not (set(words) & DOCTOR_WORDS or _SPECIALIST_NOUN.search(norm) or consumed):
        return None, set()
    name_words = set(name_index(tuple(names)).tokens.meanings) if names else set()
    for size in (2, 1):
        for i in range(len(words) - size + 1):
            cand = words[i:i + size]
            if set(cand) & consumed or any(_known_word(w) or w in DOCTOR_WORDS for w in cand):
                continue
            text = " ".join(cand)
            exact = text in cities.canon
            if not exact and (any(len(w) < 4 for w in cand) or set(cand) & name_words):
                continue
            hit = cities.resolve(text)
            if hit:
                return hit.lower(), set(cand)
    return None, set()


_LEAD_PLACE_END = {"mein", "me", "ke"}
# "subah mein", "agle hafte mein" say when, not where
_TIME_WORDS = {"subah", "sawere", "savere", "sham", "shaam", "dopahar", "raat", "morning", "evening", "afternoon", "night",
               "hafte", "hafta", "week", "mahine", "mahina", "month", "abhi", "jaldi", "baad", "agle", "agla", "next",
               "this", "today", "tomorrow", "parso", "parson", "ghante", "ghanta", "minute", "monday", "tuesday",
               "wednesday", "thursday", "friday", "saturday", "sunday", "somvar", "mangalvar", "budhvar", "guruvar",
               "shukravar", "shanivar", "ravivar", "itvaar", "time", "samay", "waqt", "date", "tarikh"}
_PLACE_DOCTORS = re.compile(r"^(?:(?:mujhe|hume|koi|achha|acha|accha)\s+)*([a-z]+(?: [a-z]+)?) (?:ke|ka|ki|wale|wala|waale|"
                            r"vale) (?:doctor|doctors|daktar|dr)(?: (?:chahiye|chaiye|dikhao|batao|bataiye|hain|hai))*$")
_PLACE_ONLY = re.compile(r"^(?:(?:nahi|nahin|aur|ab|to|toh|phir|actually|ok|acha|accha)\s+)*([a-z]+(?: [a-z]+)?) (?:mein|me|main)"
                         r"(?: (?:bhi|dikhao|batao|dekho|chahiye|dikhaiye|bataiye|please))*$")


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
        if any(len(w) < 4 for w in cand) or set(cand) & (_NOT_PLACE | DOCTOR_WORDS | _NAME_STOP | set(_ALIASES) | _TIME_WORDS):
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
    if re.search(r"\bday after tomorrow\b", norm):
        return "+2"
    for w in norm.split():
        if w in _RELATIVE_DAYS:
            return f"+{_RELATIVE_DAYS[w]}"
        if w in _WEEKDAYS:
            return f"dow:{_WEEKDAYS[w]}"
    return "next"


def availability_when(norm: str) -> str | None:
    """A question about when the chosen doctor sits, without naming them: "kal milenge kya?", "parso milengi?",
    "monday ko baithte hain?", "is she available tomorrow?", "kab milenge?". Returns the day ('+1', 'dow:0', 'next')
    or None. Needs an availability word, so a bare "kal" (an answer to "which day?") is not a question."""
    if not (_WHEN_Q.search(norm) or _TODAY.search(norm) or re.search(r"\b(is (she|he|the doctor)|are they) (in|free)\b", norm)):
        return None
    has_day = any(w in _RELATIVE_DAYS or w in _WEEKDAYS for w in norm.split()) or "day after tomorrow" in norm
    if not has_day and not re.search(r"\b(kab|when|available|availability|milenge|milengi|baithte|baithti|schedule|timings?)\b", norm):
        return None
    return _find_when(norm)
