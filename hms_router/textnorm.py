"""Query normalization shared by the safety and rag layers (and by training, so both see the same text)."""
from __future__ import annotations

import re
import unicodedata

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


# ------------------------------------------------------------------------------------------------- Hindi script
# normalize_script spells Devanagari formally (डॉक्टर -> "da ktar", का -> "kaa", में -> "men"), but every
# rule in doctor_intent is written in the casual Roman Hinglish patients type. intent_text() bridges the two:
# common words are swapped for their Hinglish spelling BEFORE transliteration, and whatever is left is
# transliterated and given the casual endings ("mahilaa" -> "mahila"). Only text with Devanagari changes.
_DEV_RE = re.compile(r"[ऀ-ॿ]")
_DEV_WORDS = {
    # doctor / hospital words
    "डॉक्टर": "doctor", "डॉक्टरों": "doctors", "डाक्टर": "doctor", "डॉ": "dr", "डॉ.": "dr", "डा.": "dr",
    "हॉस्पिटल": "hospital", "अस्पताल": "hospital", "क्लिनिक": "clinic", "स्पेशलिस्ट": "specialist",
    "विशेषज्ञ": "specialist", "सर्जन": "surgeon", "फीस": "fees", "फ़ीस": "fees", "नंबर": "number", "नम्बर": "number",
    # grammar
    "का": "ka", "की": "ki", "के": "ke", "को": "ko", "से": "se", "में": "mein", "मे": "mein", "पर": "par",
    "है": "hai", "हैं": "hain", "हो": "ho", "था": "tha", "थी": "thi", "थे": "the", "रहा": "raha", "रही": "rahi",
    "रहे": "rahe", "कोई": "koi", "कुछ": "kuch", "और": "aur", "या": "ya", "भी": "bhi", "नहीं": "nahi", "ना": "na",
    "हाँ": "haan", "हां": "haan", "जी": "ji", "मुझे": "mujhe", "मेरा": "mera", "मेरी": "meri", "मेरे": "mere", "लिए": "liye", "जानना": "jaanna", "लिये": "liye",
    "हमें": "humein", "आप": "aap", "आपका": "aapka", "इनमें": "inme", "इनमे": "inme", "इसमें": "isme",
    "उनमें": "unme", "वाला": "wala", "वाली": "wali", "वाले": "wale", "चाहिए": "chahiye", "चाहिये": "chahiye",
    "बताओ": "batao", "बताइए": "bataiye", "बताएं": "batayein", "दिखाओ": "dikhao", "दिखाइए": "dikhaiye",
    "दो": "do", "दीजिए": "dijiye", "करो": "karo", "करें": "karein", "करना": "karna", "करनी": "karni",
    "कीजिए": "kijiye", "बात": "baat", "इंसान": "insaan", "किसी": "kisi", "व्यक्ति": "vyakti",
    "बस": "bas", "सिर्फ": "sirf", "देख": "dekh", "पूछ": "pooch", "धन्यवाद": "dhanyawad", "शुक्रिया": "shukriya",
    "नमस्ते": "namaste", "नमस्कार": "namaskar", "ठीक": "theek",
    # when, near, filters
    "आज": "aaj", "कल": "kal", "परसों": "parso", "अभी": "abhi", "कब": "kab", "पास": "paas", "नज़दीक": "nazdeek",
    "नजदीक": "nazdeek", "आसपास": "aaspaas", "मिलेंगे": "milenge", "मिलेंगी": "milenge", "मिलेगा": "milega",
    "मिलेगी": "milegi", "बैठते": "baithte", "बैठती": "baithti", "उपलब्ध": "available", "सस्ता": "sasta",
    "सस्ते": "saste", "सबसे": "sabse", "कम": "kam", "अनुभवी": "experienced", "और ज्यादा": "aur",
    "लेडी": "lady", "महिला": "mahila", "औरत": "aurat", "स्त्री": "stree", "पुरुष": "purush", "आदमी": "aadmi",
    "सोमवार": "somvar", "मंगलवार": "mangalvar", "बुधवार": "budhvar", "गुरुवार": "guruvar",
    "शुक्रवार": "shukravar", "शनिवार": "shanivar", "रविवार": "ravivar", "इतवार": "itvar",
    # body and specialty words
    "दिल": "dil", "हड्डी": "haddi", "हड्डियों": "haddi", "स्किन": "skin", "त्वचा": "skin", "चमड़ी": "skin",
    "दांत": "daant", "दाँत": "daant", "आंख": "aankh", "आँख": "aankh", "कान": "kaan", "नाक": "naak", "गला": "gala",
    "गले": "gale", "पेट": "pet", "बच्चों": "bachchon", "बच्चे": "bachche", "बच्चा": "bachcha", "बाल": "baal",
    "दिमाग": "dimag", "दिमाग़": "dimag", "नस": "nas", "किडनी": "kidney", "गुर्दा": "kidney", "मन": "mann",
    "शुगर": "sugar", "मधुमेह": "sugar", "रोग": "rog", "दर्द": "dard", "जोड़ों": "jodon", "हृदय": "dil",
    "कार्डियोलॉजिस्ट": "cardiologist", "डर्मेटोलॉजिस्ट": "dermatologist", "गायनेकोलॉजिस्ट": "gynaecologist",
    "ऑर्थोपेडिक": "orthopaedic", "न्यूरोलॉजिस्ट": "neurologist", "जनरल": "general", "फिजिशियन": "physician",
    "डेंटिस्ट": "dentist", "सर्जरी": "surgery",
    # places in and around the service area
    "कोलकाता": "kolkata", "पूर्णिया": "purnia", "पूर्णियाँ": "purnia", "पटना": "patna", "दिल्ली": "delhi",
    "किशनगंज": "kishanganj", "कटिहार": "katihar", "अररिया": "araria", "सिलीगुड़ी": "siliguri",
    "भागलपुर": "bhagalpur", "दरभंगा": "darbhanga", "मुजफ्फरपुर": "muzaffarpur", "रांची": "ranchi",
    "सदर": "sadar",
}
_DEV_PHRASES = {"और डॉक्टर": "aur doctor"}
_TOKEN = re.compile(r"[ऀ-ॿ]+\.?")


def intent_text(text: str) -> str:
    """preprocess(), with Hindi script turned into the casual Hinglish the doctor-search rules read."""
    if not text or not _DEV_RE.search(text):
        return preprocess(text or "")
    t = unicodedata.normalize("NFC", text)
    for src, dst in _DEV_PHRASES.items():
        t = t.replace(src, f" {dst} ")
    t = _TOKEN.sub(lambda m: f" {_DEV_WORDS.get(m.group(), _DEV_WORDS.get(m.group().rstrip('.'), m.group()))} ", t)
    t = t.replace("ॉ", "ो")                            # ऑ/ॉ (English 'o') has no ITRANS form: read it as ो
    out = preprocess(t).split()
    return " ".join(re.sub(r"(?<=[a-z]{2})aa$", "a", w) for w in out)   # mahilaa -> mahila, sastaa -> sasta


# ------------------------------------------------------------------------------------------------- typing tidy-up
# How patients type in Roman Hindi: short forms ("nhi", "kru", "doc chaiye"), stretched letters ("bukharrr", "plzzz"),
# emojis instead of words. tidy() rewrites these to the spelling the rules and the lexicon know, once, at the door,
# so every later step (emergency check, doctor search, triage) reads the same words. Hindi script is left alone.
SHORT_FORMS = {
    # no / yes / ok
    "nhi": "nahi", "nhin": "nahi", "ni": "nahi", "nahii": "nahi", "hn": "haan", "hnji": "haanji", "hanji": "haanji",
    "okk": "ok", "okkk": "ok", "kk": "ok", "k": "ok", "okey": "okay", "tk": "theek", "thk": "theek", "thik": "theek",
    "thx": "thanks", "thnx": "thanks", "thanx": "thanks", "tq": "thanks", "ty": "thanks", "thanku": "thank you",
    "plz": "please", "pls": "please", "plss": "please", "plzz": "please", "gn": "good night",
    # chat shorthand
    "kkrh": "kya kar rahe ho", "kkrhe": "kya kar rahe ho", "kkr": "kya kar rahe ho", "hru": "how are you",
    "wbu": "and you", "hbu": "and you", "gm": "good morning", "wassup": "whats up", "wasup": "whats up",
    "sup": "whats up", "whatsup": "whats up", "hlo": "hello", "hlw": "hello", "helo": "hello", "hii": "hi",
    "rha": "raha", "rhi": "rahi", "rhe": "rahe", "kch": "kuch", "kuchh": "kuch", "tc": "take care", "gud": "good",
    "nite": "night", "mrng": "morning", "mng": "morning", "bbye": "bye bye", "byee": "bye", "sry": "sorry",
    "srry": "sorry", "w8": "wait", "bta": "batao", "btw": "by the way", "ofc": "of course",
    # question words, verbs
    "ky": "kya", "kyaa": "kya", "kyu": "kyun", "kb": "kab", "kaha": "kahan", "kitna": "kitna",
    "kru": "karu", "kroon": "karu", "kro": "karo", "kr": "kar", "krna": "karna", "krni": "karni", "krein": "karein",
    "krdo": "kar do", "kardo": "kar do", "krte": "karte", "pta": "pata", "ptaa": "pata", "btao": "batao",
    "btaye": "bataye", "btaiye": "bataiye", "btana": "batana", "dikhayo": "dikhao", "chaiye": "chahiye",
    "chahie": "chahiye", "chaahiye": "chahiye", "chiye": "chahiye", "chahiyee": "chahiye", "jldi": "jaldi",
    # people, pronouns, fillers
    "mjhe": "mujhe", "muje": "mujhe", "mujhay": "mujhe", "mko": "mujhko", "h": "hai",
    "bhot": "bahut", "bohot": "bahut", "bhut": "bahut", "bahot": "bahut", "bht": "bahut", "yr": "yaar",
    "u": "you", "ur": "your", "bcz": "because", "coz": "because",
    # health words
    "doc": "doctor", "dctr": "doctor", "doctr": "doctor", "docter": "doctor", "daktar": "doctor",
    "hosp": "hospital", "hospitl": "hospital", "dawa": "dawai", "davai": "dawai", "dvai": "dawai", "dwai": "dawai",
    "medicin": "medicine", "apointment": "appointment", "appoinment": "appointment", "appt": "appointment",
    "tbiyat": "tabiyat", "tabiyat": "tabiyat", "tabeeyat": "tabiyat",
}
EMOJI_WORDS = {
    "🙏": "thank you", "👍": "ok", "👌": "ok", "🤒": "bukhar", "🤕": "chot", "😷": "zukam", "🤧": "zukam",
    "🤮": "ulti", "🤢": "ji machlana", "🦷": "daant", "😴": "neend", "🩸": "khoon", "💊": "dawai",
}
_EMOJI_RE = re.compile("|".join(re.escape(e) for e in EMOJI_WORDS))
_RUN3 = re.compile(r"([a-z])\1{2,}")                  # three or more of a letter: "bukharrr", "plzzz"


def tidy(text: str, known: frozenset[str] | set[str] = frozenset()) -> str:
    """Short forms, stretched letters and emojis -> plain words. known: the vocabulary a squeezed word must land in
    ("bataoo" -> "batao" only when "batao" is a word Vita knows). Never touches Hindi script or digits."""
    if not text:
        return text or ""
    t = _EMOJI_RE.sub(lambda m: f" {EMOJI_WORDS[m.group()]} ", text)

    def fix(m: re.Match) -> str:
        w = m.group()
        low = w.lower()
        if low in SHORT_FORMS:
            if low == "k" and len(text.split()) > 1:                  # "k" is only "ok" when it is the whole message
                return w
            return SHORT_FORMS[low]
        if known and low not in known:
            # runs of 3+ anywhere ("bukharrr"), a doubled last letter ("bataoo", "chahiyee"); never a doubled letter
            # inside a word, where it changes the word ("kaam" is not "kam", "paani" is not "pani")
            for cand in (_RUN3.sub(r"\1\1", low), _RUN3.sub(r"\1", low), re.sub(r"([a-z])\1+$", r"\1", _RUN3.sub(r"\1\1", low))):
                if cand != low and (cand in known or cand in SHORT_FORMS):
                    return SHORT_FORMS.get(cand, cand)
        return w

    t = re.sub(r"[A-Za-z]+", fix, t)
    return re.sub(r"\s+", " ", t).strip()
