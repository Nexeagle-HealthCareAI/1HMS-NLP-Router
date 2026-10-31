"""How Vita talks: every phrase in English, Hinglish (Roman Hindi) and Hindi (Devanagari).

The conversation's language (language.resolve) picks the column, so a patient who writes in Hindi script reads
Hindi script, one who writes Hinglish reads Hinglish, and one who writes English reads English, for the whole
chat. Vita speaks in the first person, feminine ("sakti hoon", "jod rahi hoon"), matching the
Doctor Dekho site (decided 2026-10-25).

This module only words things. What Vita decides (which doctors, emergency or not) is decided elsewhere and
never changes with the language.
"""
from __future__ import annotations

import re

from .severity import EMERGENCY_NUMBER, GENERAL_EMERGENCY_NUMBER

CODES = {"english": "en", "hinglish": "hg", "hindi": "hi"}


class Voice:
    """`v(en, hg, hi)` returns the phrase in the conversation's language. Truthy for Hinglish and Hindi, so
    older `x if hi else y` code keeps working (Hindi then reads the Hinglish line)."""

    def __init__(self, language: str | None):
        self.language = language or "english"
        self.code = CODES.get(self.language, "en")

    def __call__(self, en: str, hg: str, hi: str | None = None) -> str:
        return en if self.code == "en" else hg if self.code == "hg" or hi is None else hi

    def __bool__(self) -> bool:
        return self.code != "en"

    @property
    def end(self) -> str:
        """Sentence end: '.' in English and Hinglish, '।' in Hindi."""
        return "।" if self.code == "hi" else "."


# ------------------------------------------------------------------------------------------------ chips
def location_options(v: Voice) -> list[dict]:
    return [{"id": "share_location", "label": v("Share my location", "Location share karein", "लोकेशन शेयर करें")},
            {"id": "type_area", "label": v("Type area or pincode", "Area ya pincode likhein", "इलाका या पिनकोड लिखें")},
            {"id": "use_city", "label": v("Use my city", "Mera city use karein", "मेरा शहर इस्तेमाल करें")}]


def human_option(v: Voice) -> dict:
    return {"id": "human", "label": v("Talk to a person", "Kisi insaan se baat karein", "किसी इंसान से बात करें")}


def ask_kind_options(v: Voice) -> list[dict]:
    return [{"id": "symptom", "label": v("Describe a symptom", "Symptom batayein", "लक्षण बताएँ")},
            {"id": "specialty", "label": v("Choose a specialty", "Specialty chunein", "स्पेशलिटी चुनें")},
            {"id": "name", "label": v("Doctor's name", "Doctor ka naam", "डॉक्टर का नाम")},
            {"id": "degree", "label": v("By degree", "Degree se", "डिग्री से")},
            {"id": "distance", "label": v("Doctors near me", "Paas ke doctor", "पास के डॉक्टर")}]


# ------------------------------------------------------------------------------------------------ specialties
# Plain words first, the specialty in brackets: "bone doctors (Orthopaedic)", "haddi ke doctor (Orthopaedic)"
PLAIN = {  # key: (English, Hinglish, Hindi)
    "cardiology": ("heart", "dil ke", "दिल के"), "dermatology": ("skin", "skin ke", "त्वचा के"),
    "orthopaedics": ("bone", "haddi ke", "हड्डी के"), "gynaecology": ("women's health", "mahila rog", "महिला रोग"),
    "dentistry": ("teeth", "daant ke", "दाँत के"), "ent": ("ear, nose and throat", "kaan-naak-gala ke", "कान-नाक-गले के"),
    "ophthalmology": ("eye", "aankh ke", "आँख के"), "neurology": ("brain and nerve", "dimag aur nas ke", "दिमाग़ और नसों के"),
    "neurosurgery": ("brain surgery", "dimag ki surgery ke", "दिमाग़ की सर्जरी के"),
    "psychiatry": ("mental health", "mann ke", "मन के"), "urology": ("urinary", "peshab ke", "पेशाब के"),
    "gastroenterology": ("stomach", "pet ke", "पेट के"),
    "endocrinology": ("diabetes and hormone", "sugar aur hormone ke", "शुगर और हॉर्मोन के"),
    "pulmonology": ("chest and lung", "chhati aur phephde ke", "छाती और फेफड़ों के"),
    "nephrology": ("kidney", "kidney ke", "किडनी के"), "oncology": ("cancer", "cancer ke", "कैंसर के"),
    "rheumatology": ("joint pain", "jodon ke dard ke", "जोड़ों के दर्द के"),
    "physiotherapy": ("physiotherapy", "physiotherapy ke", "फ़िज़ियोथेरेपी के"),
    "paediatrics": ("child", "bachchon ke", "बच्चों के"), "geriatrics": ("elderly care", "buzurgon ke", "बुज़ुर्गों के"),
    "general-medicine": ("general", "general", "जनरल"),
    "general-surgery": ("general surgery", "general surgery ke", "जनरल सर्जरी के"),
}


def plain(v: Voice, key: str | None) -> str | None:
    row = PLAIN.get(key or "")
    return v(*row) if row else None


# ------------------------------------------------------------------------------------------------ symptoms
SYMPTOMS = {  # canonical English -> (Hinglish, Hindi)
    "fever": ("bukhar", "बुखार"), "headache": ("sar dard", "सिर दर्द"), "cough": ("khansi", "खाँसी"),
    "cold": ("zukam", "ज़ुकाम"), "runny nose": ("naak behna", "नाक बहना"), "sore throat": ("gale mein kharash", "गले में खराश"),
    "throat pain": ("gale mein dard", "गले में दर्द"), "stomach pain": ("pet dard", "पेट दर्द"),
    "teeth pain": ("daant mein dard", "दाँत में दर्द"), "toothache": ("daant mein dard", "दाँत में दर्द"),
    "leg pain": ("pair mein dard", "पैर में दर्द"), "lump": ("gaanth", "गाँठ"),
    "abdominal pain": ("pet dard", "पेट दर्द"), "vomiting": ("ulti", "उल्टी"), "nausea": ("ji machlana", "जी मचलाना"),
    "diarrhoea": ("dast", "दस्त"), "diarrhea": ("dast", "दस्त"), "loose motions": ("dast", "दस्त"),
    "constipation": ("kabz", "कब्ज़"), "acidity": ("acidity", "एसिडिटी"), "chest pain": ("seene mein dard", "सीने में दर्द"),
    "breathlessness": ("saans phoolna", "साँस फूलना"), "dizziness": ("chakkar", "चक्कर"), "weakness": ("kamzori", "कमज़ोरी"),
    "fatigue": ("thakan", "थकान"), "back pain": ("kamar dard", "कमर दर्द"), "joint pain": ("jodon mein dard", "जोड़ों में दर्द"),
    "knee pain": ("ghutne mein dard", "घुटने में दर्द"), "body ache": ("badan dard", "बदन दर्द"), "rash": ("daane", "दाने"),
    "itching": ("khujli", "खुजली"), "ear pain": ("kaan dard", "कान दर्द"), "eye pain": ("aankh mein dard", "आँख में दर्द"),
    "red eye": ("aankh laal hona", "आँख लाल होना"), "toothache": ("daant dard", "दाँत दर्द"),
    "burning urination": ("peshab mein jalan", "पेशाब में जलन"), "swelling": ("sujan", "सूजन"),
    "anxiety": ("ghabrahat", "घबराहट"), "insomnia": ("neend na aana", "नींद न आना"), "unconscious": ("behoshi", "बेहोशी"),
    "excessive bleeding": ("bahut khoon behna", "बहुत खून बहना"), "seizure": ("daura", "दौरा"),
    "stroke": ("lakwa ke lakshan", "लकवे के लक्षण"), "facial droop": ("chehra tedha hona", "चेहरा टेढ़ा होना"),
    "numbness": ("sunnapan", "सुन्नपन"), "palpitations": ("dhadkan tez hona", "धड़कन तेज़ होना"),
    "hair fall": ("baal jhadna", "बाल झड़ना"), "acne": ("muhase", "मुँहासे"), "weight loss": ("vajan ghatna", "वज़न घटना"),
    "blocked nose": ("naak band", "नाक बंद"), "upset stomach": ("pet kharab", "पेट ख़राब"), "gas": ("gas", "गैस"),
    "bloating": ("pet phoolna", "पेट फूलना"), "delayed periods": ("periods late", "माहवारी में देरी"),
    "period pain": ("periods mein dard", "माहवारी में दर्द"), "white discharge": ("safed pani", "सफ़ेद पानी"),
    "high blood pressure": ("BP high", "बीपी ज़्यादा"), "low blood pressure": ("BP low", "बीपी कम"),
    "high blood sugar": ("sugar badhna", "शुगर बढ़ना"), "diabetes": ("sugar", "शुगर"),
    "kidney stone": ("pathri", "पथरी"), "frequent urination": ("baar baar peshab", "बार-बार पेशाब"),
    "ringworm": ("daad", "दाद"), "low mood": ("udaasi", "उदासी"), "watery eyes": ("aankh se paani", "आँख से पानी"),
    "blurred vision": ("dhundhla dikhna", "धुँधला दिखना"), "ear discharge": ("kaan behna", "कान बहना"),
    "mouth ulcers": ("munh mein chhale", "मुँह में छाले"), "weight gain": ("vajan badhna", "वज़न बढ़ना"),
    "loss of appetite": ("bhookh na lagna", "भूख न लगना"), "neck pain": ("gardan dard", "गर्दन दर्द"),
    # the red flags' own names (an emergency's signs), so a Hinglish or Hindi reply never shows them in English
    "suicidal thoughts / self-harm": ("khud ko nuksaan pahunchane ke vichaar", "खुद को नुकसान पहुँचाने के विचार"),
    "facial drooping / stroke": ("chehra tedha hona", "चेहरा टेढ़ा होना"),
    "one-sided weakness / stroke": ("ek taraf kamzori", "एक तरफ़ कमज़ोरी"),
    "slurred speech / stroke": ("bolne mein ladkhadahat", "बोलने में लड़खड़ाहट"),
    "limb numbness": ("haath pair sunn hona", "हाथ-पैर सुन्न होना"), "lost consciousness": ("behoshi", "बेहोशी"),
    "labour pains": ("prasav dard", "प्रसव दर्द"), "waters broken in pregnancy": ("pregnancy mein paani girna", "गर्भ में पानी गिरना"),
    "bleeding in pregnancy": ("pregnancy mein khoon aana", "गर्भावस्था में खून आना"),
    "seizure / convulsions": ("daura / jhatke", "दौरा / झटके"), "seizure or heart attack ('daura')": ("daura", "दौरा"),
    "head injury with vomiting": ("sar par chot ke baad ulti", "सिर पर चोट के बाद उल्टी"),
    "head injury with fainting": ("sar par chot ke baad behoshi", "सिर पर चोट के बाद बेहोशी"),
    "head injury": ("sar par chot", "सिर पर चोट"), "burn injury": ("jalna", "जलना"),
    "pregnancy warning signs": ("pregnancy mein khatre ke lakshan", "गर्भावस्था में ख़तरे के लक्षण"), "acid / chemical burn": ("tezaab se jalna", "तेज़ाब से जलना"),
    "poisoning": ("zeher", "ज़हर"), "poisoning (pesticide)": ("keetnashak zeher", "कीटनाशक ज़हर"),
    "poisoning / overdose": ("zeher / zyada dawai", "ज़हर / ज़्यादा दवाई"), "snake bite": ("saanp ka kaatna", "साँप का काटना"),
    "road accident / trauma": ("accident / gehri chot", "दुर्घटना / गहरी चोट"), "heart attack": ("heart attack", "हार्ट अटैक"),
    "blood in vomit": ("khoon ki ulti", "खून की उल्टी"), "bleeding that won't stop": ("khoon nahi ruk raha", "खून नहीं रुक रहा"),
    "sudden severe headache": ("achanak tez sar dard", "अचानक तेज़ सिर दर्द"),
    "vision loss / blackout": ("achanak dikhna band", "अचानक दिखना बंद"),
    "throat closing (severe allergic reaction)": ("gala band hona (allergy)", "गला बंद होना (एलर्जी)"),
}


def symptom_words(v: Voice, names: list[str]) -> str:
    """['fever', 'headache', 'cough'] -> 'fever, headache and cough' / 'bukhar, sar dard aur khansi'."""
    words = [n if v.code == "en" else (SYMPTOMS.get(n.lower(), (n, n))[0 if v.code == "hg" else 1]) for n in names]
    words = list(dict.fromkeys(w for w in words if w))
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + v(" and ", " aur ", " और ") + words[-1]


_DURATION = re.compile(r"(?:for|since)?\s*(\d+)\s*(hour|day|week|month|year)s?", re.I)
_UNITS = {"hour": ("ghante", "घंटे"), "day": ("din", "दिन"), "week": ("hafte", "हफ़्ते"), "month": ("mahine", "महीने"),
          "year": ("saal", "साल")}
_SINCE = {"morning": ("subah", "सुबह"), "yesterday": ("kal", "कल"), "night": ("raat", "रात"),
          "last night": ("kal raat", "कल रात"), "evening": ("shaam", "शाम")}


def duration_words(v: Voice, duration: str | None) -> str:
    """'for 3 days' -> 'for 3 days' / '3 din se' / '3 दिन से'. Unknown shapes are left out of Hindi and Hinglish."""
    if not duration:
        return ""
    if v.code == "en":
        return duration
    i = 0 if v.code == "hg" else 1
    m = _DURATION.search(duration)
    if m:
        return f"{m[1]} {_UNITS[m[2].lower()][i]} " + ("se" if i == 0 else "से")
    since = duration.lower().replace("since", "").strip()
    if since in _SINCE:
        return _SINCE[since][i] + (" se" if i == 0 else " से")
    return ""


# ------------------------------------------------------------------------------------------------ triage
QUESTIONS = {  # accumulator question key -> (English, Hinglish, Hindi)
    "more": ("Any other symptoms?", "Koi aur symptoms hain?", "कोई और लक्षण हैं?"),
    "duration": ("Since when has this been going on?", "Kitne time se?", "यह कब से हो रहा है?"),
    "intensity": ("How bad is it: mild, moderate or severe?", "Takleef kitni hai - halki, theek-thaak ya bahut tez?",
                  "तकलीफ़ कितनी है: हल्की, ठीक-ठाक या बहुत तेज़?"),
    "describe": ("What's bothering you? Tell me a little more.", "Aapko kya takleef ho rahi hai? Thoda detail mein batayein.",
                 "आपको क्या तकलीफ़ हो रही है? थोड़ा विस्तार से बताएँ।"),
}


def not_understood(v: Voice) -> str:
    """The second time nothing medical was recognised: say so, give an example, and offer the other ways to ask."""
    return v("Sorry, I didn't quite get that. Could you say it another way, like \"headache since 2 days\" or "
             "\"fever and cough\"? Or choose one below.",
             "Maaf kijiye, main samajh nahi paayi. Thoda alag tarah se batayein, jaise \"2 din se sar dard\" ya "
             "\"bukhar aur khansi\". Ya neeche se chunein.",
             "माफ़ कीजिए, मैं समझ नहीं पाई। थोड़ा अलग तरह से बताएँ, जैसे \"2 दिन से सिर दर्द\" या \"बुखार और खाँसी\"। "
             "या नीचे से चुनें।")


def triage_question(v: Voice, key: str | None, symptoms: list[str], first: bool) -> str | None:
    """The next triage question, with a short acknowledgement first so it reads like a person listening:
    'Got it: fever and headache. Since when has this been going on?'"""
    if key not in QUESTIONS:
        return None
    q = v(*QUESTIONS[key])
    if key == "describe" or not symptoms:
        return q
    if first:
        said = symptom_words(v, symptoms)
        ack = v(f"Got it: {said}.", f"Samajh gayi, {said}.", f"समझ गई, {said}।")
    else:
        ack = v("Okay.", "Theek hai.", "ठीक है।")
    return f"{ack} {q}"


def recommendation(v: Voice, *, symptoms: list[str], duration: str | None, specialty_key: str | None,
                   specialist: str | None, severity: str | None, low_confidence: bool, doctors: int,
                   fallback_used: bool) -> str:
    """The triage result as a person would say it: what I heard, who to see, the doctors, what to watch for."""
    said = symptom_words(v, symptoms)
    dur = duration_words(v, duration)
    heard = v(f"Got it: {said}" + (f", {dur}" if dur else "") + ".",
              f"Samajh gayi: {dur + ' ' if dur else ''}{said}.",
              f"समझ गई: {dur + ' ' if dur else ''}{said}।") if said else ""
    name = plain(v, specialty_key)
    spec = specialist or v("a General Physician", "General Physician", "जनरल फ़िज़िशियन")
    who = (f"{name} doctor ({spec})" if v.code == "en" and name and name.lower() not in spec.lower() else
           f"{name} doctor ({spec})" if v.code == "hg" and name and name.lower() not in spec.lower() else
           f"{name} डॉक्टर ({spec})" if v.code == "hi" and name else spec)
    if low_confidence:
        advice = v("I can't tell for sure from this, so a General Physician is a good first step. "
                   "You can also tell me a little more about the problem.",
                   "Isse pakka kehna mushkil hai, isliye pehle General Physician ko dikhana theek rahega. "
                   "Aap thoda aur bata sakte hain.",
                   "इससे पक्का कहना मुश्किल है, इसलिए पहले जनरल फ़िज़िशियन को दिखाना ठीक रहेगा। "
                   "आप थोड़ा और बता सकते हैं।")
    else:
        today = severity == "SEVERE"
        advice = v(f"A {who} is the right one to see for this" + (", and please go today." if today else "."),
                   f"Iske liye {who} ko dikhana sahi rahega" + (", aur aaj hi dikhayein." if today else "."),
                   f"इसके लिए {who} को दिखाना सही रहेगा" + (", और आज ही दिखाएँ।" if today else "।"))
    if doctors:
        found = v("Here are some you can book. Tap Choose to pick one.",
                  "Ye kuch doctor hain jinse appointment le sakte hain. 'Choose' dabakar ek chunein.",
                  "ये कुछ डॉक्टर हैं जिनसे अपॉइंटमेंट ले सकते हैं। 'Choose' दबाकर एक चुनें।")
        if fallback_used:
            found = v("None of them is free right now, so here are General Physicians who can see you first. ",
                      "Abhi woh doctor available nahi hain, isliye ye General Physician pehle dekh sakte hain. ",
                      "अभी वे डॉक्टर उपलब्ध नहीं हैं, इसलिए ये जनरल फ़िज़िशियन पहले देख सकते हैं। ") + found
    else:
        found = ""
    watch = v(f"If it gets worse, or you have chest pain or trouble breathing, call {EMERGENCY_NUMBER} right away.",
              f"Agar takleef badhe, ya seene mein dard ya saans ki dikkat ho, turant {EMERGENCY_NUMBER} par call karein.",
              f"अगर तकलीफ़ बढ़े, या सीने में दर्द या साँस की दिक्कत हो, तो तुरंत {EMERGENCY_NUMBER} पर कॉल करें।")
    note = v("(This is guidance, not a diagnosis.)", "(Ye salah hai, diagnosis nahi.)", "(यह सलाह है, निदान नहीं।)")
    return " ".join(x for x in (heard, advice, found) if x) + "\n\n" + watch + " " + note


TELE_MANAS = "14416"      # India's national mental-health line (Tele-MANAS), also on the Doctor Dekho banner


def emergency(v: Voice, signs: list[str]) -> str:
    """Number first, in the patient's language; non-English replies keep one English line as a backup."""
    head = v(f"This sounds like an emergency. Call {EMERGENCY_NUMBER} (ambulance) now, or {GENERAL_EMERGENCY_NUMBER}. "
             "If you can, go to the nearest emergency room.",
             f"Ye emergency lag rahi hai. Abhi {EMERGENCY_NUMBER} (ambulance) ya {GENERAL_EMERGENCY_NUMBER} par call karein. "
             "Ho sake to turant nazdeeki emergency (ER) jaayein.",
             f"यह इमरजेंसी लग रही है। अभी {EMERGENCY_NUMBER} (एम्बुलेंस) या {GENERAL_EMERGENCY_NUMBER} पर कॉल करें। "
             "हो सके तो तुरंत नज़दीकी इमरजेंसी (ER) जाएँ।")
    seen = symptom_words(v, signs)
    lines = [head]
    if seen:
        lines.append(v(f"Signs: {seen}.", f"Lakshan: {seen}.", f"लक्षण: {seen}।"))
    if any("suicid" in x.lower() or "self-harm" in x.lower() for x in signs):     # a person to talk to, 24x7
        lines.append(v(f"You are not alone. Talk to Tele-MANAS on {TELE_MANAS} (free, 24x7), or someone you trust, right now.",
                       f"Aap akele nahi hain. Abhi Tele-MANAS {TELE_MANAS} par baat karein (free, 24x7), ya kisi apne se.",
                       f"आप अकेले नहीं हैं। अभी टेली-मानस {TELE_MANAS} पर बात करें (मुफ़्त, 24x7), या किसी अपने से।"))
    if v.code != "en":
        lines.append(f"Call {EMERGENCY_NUMBER} (ambulance) or {GENERAL_EMERGENCY_NUMBER} now.")
    return "\n".join(lines)


def nothing_new(v: Voice) -> str:
    """Triage already advised and the message added nothing medical: don't repeat the advice."""
    return v("Would you like one of the doctors above, or is there something else bothering you?",
             "Kya aap upar diye doctors mein se kisi ko dikhana chahenge, ya koi aur takleef hai?",
             "क्या आप ऊपर दिए डॉक्टरों में से किसी को दिखाना चाहेंगे, या कोई और तकलीफ़ है?")


def carry_note(v: Voice, carry: dict) -> str:
    """'sabse sasta doctor' without a kind of doctor: say the filter is kept, then ask which kind."""
    if carry.get("sort") == "fee":
        return v("Okay, I'll show the lowest fees first.", "Theek hai, kam fees wale pehle dikhaungi.",
                 "ठीक है, कम फ़ीस वाले पहले दिखाऊँगी।")
    if carry.get("sort") == "experience":
        return v("Okay, I'll show the most experienced first.", "Theek hai, zyada experience wale pehle dikhaungi.",
                 "ठीक है, ज़्यादा अनुभव वाले पहले दिखाऊँगी।")
    return v("Okay, I'll show doctors available today.", "Theek hai, aaj baithne wale doctor dikhaungi.",
             "ठीक है, आज बैठने वाले डॉक्टर दिखाऊँगी।")


def after_emergency_reminder(v: Voice, signs: str) -> str:
    """Said once, before the answer to whatever the patient asks next after an emergency."""
    if signs:
        return v(f"First: if the {signs} is still there, call {EMERGENCY_NUMBER} now.",
                 f"Pehle: agar {signs} abhi bhi hai, to turant {EMERGENCY_NUMBER} par call karein.",
                 f"पहले: अगर {signs} अभी भी है, तो तुरंत {EMERGENCY_NUMBER} पर कॉल करें।")
    return v(f"First: if it is still an emergency, call {EMERGENCY_NUMBER} now.",
             f"Pehle: agar abhi bhi emergency hai, to turant {EMERGENCY_NUMBER} par call karein.",
             f"पहले: अगर अभी भी इमरजेंसी है, तो तुरंत {EMERGENCY_NUMBER} पर कॉल करें।")


def after_emergency(v: Voice, what: str, signs: str) -> str:
    """The reply to "ambulance aa gayi", "ab theek hoon", "108 nahi lag raha", "thank you" after an emergency."""
    tell = (v(f" Tell the doctor about the {signs}.", f" Doctor ko {signs} ke baare mein zaroor batayein.",
              f" डॉक्टर को {signs} के बारे में ज़रूर बताएँ।") if signs else "")
    if what == "cant_call":
        return v(f"Try {GENERAL_EMERGENCY_NUMBER}. If no one answers, ask someone to take you to the nearest hospital "
                 "emergency right now; don't wait.",
                 f"{GENERAL_EMERGENCY_NUMBER} par try karein. Koi na uthaye to kisi ko saath lekar abhi nazdeeki hospital ki "
                 "emergency mein jaayein, intezaar mat kijiye.",
                 f"{GENERAL_EMERGENCY_NUMBER} पर कोशिश करें। कोई न उठाए तो किसी को साथ लेकर अभी नज़दीकी अस्पताल की "
                 "इमरजेंसी में जाएँ, इंतज़ार मत कीजिए।")
    if what == "arrived":
        return v("Good, help is there." + tell + " I'm here if you need a doctor's appointment later.",
                 "Achha hai, madad pahunch gayi." + tell + " Baad mein doctor ka appointment chahiye to main yahin hoon.",
                 "अच्छा है, मदद पहुँच गई।" + tell + " बाद में डॉक्टर का अपॉइंटमेंट चाहिए तो मैं यहीं हूँ।")
    if what == "better":
        check = v(f" Still, get the {signs} checked by a doctor." if signs else " Still, see a doctor soon.",
                  f" Phir bhi {signs} ki jaanch doctor se zaroor karwayein." if signs else " Phir bhi jaldi doctor ko dikhayein.",
                  f" फिर भी {signs} की जाँच डॉक्टर से ज़रूर करवाएँ।" if signs else " फिर भी जल्दी डॉक्टर को दिखाएँ।")
        return v("Glad you feel better." + check + f" If it comes back, call {EMERGENCY_NUMBER} straight away. "
                 "Shall I find you a doctor?",
                 "Sunkar achha laga." + check + f" Takleef dobara ho to turant {EMERGENCY_NUMBER} par call karein. "
                 "Kya main doctor dhoondhun?",
                 "सुनकर अच्छा लगा।" + check + f" तकलीफ़ दोबारा हो तो तुरंत {EMERGENCY_NUMBER} पर कॉल करें। "
                 "क्या मैं डॉक्टर ढूँढूँ?")
    return v(f"Take care. If it gets worse, call {EMERGENCY_NUMBER}. I'm here if you need a doctor.",
             f"Apna dhyan rakhiye. Takleef badhe to {EMERGENCY_NUMBER} par call karein. Doctor chahiye to main yahin hoon.",
             f"अपना ध्यान रखिए। तकलीफ़ बढ़े तो {EMERGENCY_NUMBER} पर कॉल करें। डॉक्टर चाहिए तो मैं यहीं हूँ।")


# ------------------------------------------------------------------------------------------------ Bangla
BANGLA_NOTE = ("দুঃখিত, আমি এখনও বাংলায় উত্তর দিতে পারি না। দয়া করে হিন্দি বা ইংরেজিতে লিখুন। "
               f"জরুরি অবস্থায় এখনই {EMERGENCY_NUMBER} নম্বরে ফোন করুন।\n"
               "Maaf kijiye, main abhi Bangla mein jawab nahi de sakti. Hindi ya English mein likhein. "
               f"Emergency ho to turant {EMERGENCY_NUMBER} par call karein.\n"
               "Sorry, I can't reply in Bangla yet. Please write in Hindi or English. "
               f"In an emergency, call {EMERGENCY_NUMBER} now.")


def day_label(v: Voice, kind: str) -> str:
    return {"today": v("today", "aaj", "आज"), "tomorrow": v("tomorrow", "kal", "कल")}[kind]
