"""Everyday questions that are not a symptom or a doctor search: who Vita is, fees, "is it free", online consults, my
appointment, reports, timings, insurance, medicine questions, small talk, off-topic asks, services that are coming,
privacy, and the short repair words ("kya?", "aur?", "pehla wala", "wapas").

Each one gets a short, honest answer in the patient's language and is steered back to finding a doctor (catalog:
"Vita: what patients type and how Vita should reply", 26-27 Oct 2026). Answers that need a fact say only what was
confirmed: finding a doctor and booking is free, the fee is paid at the hospital, and online consults, insurance, lab
tests and medicine delivery are coming. No answer gives medical advice, a dose or a diagnosis.

detect() reads intent_text (after textnorm.tidy). The caller decides precedence: a red flag, a booking step, a
specialty or a real symptom always win over these.
"""
from __future__ import annotations

import re

from ..severity import EMERGENCY_NUMBER, GENERAL_EMERGENCY_NUMBER
from ..voice import Voice

_Q = r"(kya|kitni|kitna|kitne|kaise|kab|kahan|kyun|kyu|kaun|how|what|when|where|why|which|hai\?|\?)"

# order matters: the first that matches wins
MOVES: list[tuple[str, re.Pattern]] = [
    # ---- a cry for help with nothing else said (answered by the safety step in every mode)
    ("distress", re.compile(r"^(please |plz |koi )?(bachao|bachaao|bacha lo|bachalo|help me|help karo|please help|help please"
                            r"|helpp+|sos|madad karo|madad chahiye jaldi|jaldi madad|jaldi aao|jaldi aaiye|jaldi bulao"
                            r"|koi hai bachao|emergency hai|urgent hai|jaldi karo)( (please|plz|jaldi|koi|bhai|ji))*$")),
    # ---- social talk: the whole message is one of these (anchored), so a doctor or symptom ask never lands here
    ("thanks", re.compile(r"^(ok |okay |accha |achha )?(thanks?|thank (you|u)|thanku|thnks|ty|shukriya|shukria|dhanyavaa?d"
                          r"|dhanyawaa?d|dhonnobad|dhanyabad|meherbani|shukran|jazakallah\w*|bahut shukriya|bahut dhanyavaad)"
                          r"( (ji|vita|dear|bhai|bhaiya|didi|madam|mam|sir|yaar|so much|a lot|again))*$")),
    ("bye", re.compile(r"^(ok |okay |chalo |accha |achha |theek hai )?(bye|bye bye|byee+|tata|ta ta|alvida|khuda hafiz"
                       r"|allah hafiz|see (you|u|ya)|see you later|phir milte( hain)?|fir milte( hain)?|baad mein baat (karte|karenge)"
                       r"( hain)?|take care|good ?bye|chalta hu|chalta hoon|chalti hu|chalti hoon)( (ji|vita|dear|bhai|bhaiya|didi|madam|mam|sir|yaar|so much|a lot|again))*$")),
    ("later", re.compile(r"^(abhi )?(nahi|nhi|no) ?(abhi|abhi nahi|chahiye|zarurat|zaroorat)?( nahi)?( chahiye| hai)?$"
                         r"|^(abhi nahi chahiye|zarurat nahi|zaroorat nahi|koi zarurat nahi|nahi chahiye|kal dekhte hain"
                         r"|baad mein dekhte hain|baad mein|later|rehne do|rehne dijiye|bas itna hi|bas itna hi tha|bas"
                         r"|no thanks|not now|maybe later|kuch nahi|kuch nai|nothing|kuch nahi bas aise hi|bas aise hi|aise hi"
                         r"|timepass|bas timepass|just checking|just browsing)$")),
    ("wait", re.compile(r"^(ruko|ruko zara|zara ruko|thoda ruko|ek minute|ek min|ek second|ek sec|1 min|1 minute|2 min"
                        r"|do minute|one minute|one sec|wait|wait please|please wait|hold on|abhi aata hu|abhi aata hoon"
                        r"|abhi aati hu|abhi aati hoon|busy hu|busy hoon|thodi der mein)( (ji|yaar|please|na))*$")),
    ("sorry", re.compile(r"^(sorry|sry|oops|galti se|galti se likha|galti ho gayi|galat type (hua|ho gaya)|sorry galat type"
                         r"( hua| ho gaya)?|maaf karna|maaf kijiye|maaf kijiyega|my bad|mistake)( (ji|yaar|sorry))*$")),
    ("name", re.compile(r"^(mera naam|my name is|my name s|naam) (?P<name>[a-z]+( (?!hai$|h$|he$)[a-z]+)?)( hai| h| he)?$"
                        r"|^main (?P<name2>[a-z]+) bol (raha|rahi) (hoon|hu|hun)$|^(this is|i am|i m) (?P<name3>[a-z]+) (here|speaking)$")),
    ("test", re.compile(r"^(test|testing|test test|test message|check|checking|hello hello|hello hello hello|koi hai|koi hai kya"
                        r"|anyone there|is anyone there|kya ye kaam karta hai|kya ye chal raha hai|ye kaam kar raha hai|working"
                        r"|is this working|are you there|aap ho|tum ho|online ho)$")),
    ("personal", re.compile(r"\b(i love you|love you|shaadi karogi|shadi karogi|marry me|date pe chalo\w*|date par chalo\w*"
                            r"|cute ho|sundar ho|beautiful ho|hot ho|gf banogi|girlfriend banogi|kiss|send pic|photo bhejo"
                            r"|pic bhejo|tumhari photo|single ho|tumhari age|kitne saal ki ho)\b")),
    ("what_is_dd", re.compile(r"\bdoctor ?dekho\b.*\b(kya|kaun|what|kaisa|kaise)\b|\bwhat is doctor ?dekho\b")),
    ("about", re.compile(
        r"\b(tum|aap|tu|ap)\s+(kaun|kon)\b|\bwho (are|r) (you|u)\b|\bwhat are you\b"
        r"|\b(your|tumhara|tumhaara|aapka|apka|tera)\s+(naam|name)\b|^(name|naam)\??$"
        r"|\b(insaan|insan|human|robot|bot|machine|real person|ai)\s+(ho|hai|hain|ho kya|hai kya)\b"
        r"|\b(ladki|ladka|male|female|aurat|mard) ho\b|\b(boy|girl|man|woman) (ho|hai)\b|\bare (you|u) (a )?(boy|girl|man|woman)\b"
        r"|\bare (you|u) (a |an )?(bot|human|robot|ai|real|doctor|person)\b|\b(bot|robot) ho\b"
        r"|\b(tum|aap) (doctor|bot|robot|insaan|ai) (ho|hain|hai)\b"
        r"|\b(kya kar sakti|kya kar sakte|kya kya kar|what can (you|u) do|kaise use|how (do i|to) use|how does (this|it) work"
        r"|kaise kaam kar)\w*|\b(who made|kisne banaya|who built|who created|kisne bana)\w*"
        r"|\b(hindi|english|bangla) (samajh|samjh|aati|bol|understand|speak)\w*|^(help|madad|help please|please help)$")),
    ("fees", re.compile(r"\b(fees?|charges?|consultation fee)\b.*\b(kitni|kitna|kitne|kya hai|how much|what)\b"
                        r"|\b(kitni|kitna|kitne|how much)\b.*\b(fees?|charges?|lagega|lagegi|lagti|lagta|paisa|paise|kharcha)\b"
                        r"|^(fees?|charges?)\??$")),
    ("free", re.compile(r"\b(free|muft|muqt|free of cost)\b.*(\bhai\b|\bhain\b|\bkya\b|\?|$)|\bpaise lagte\b|\bpayment\b"
                        r"|\bpay (kaise|kahan|kab|online)\b|\brefund\b|\bpaisa wapas\b|\bpaise wapas\b|\bupi\b")),
    ("coming_soon", re.compile(r"\b(lab test|blood test|test book|test karwana|test karana|sample collection|medicine (home )?delivery"
                               r"|dawai (ghar|delivery|mangwa|mangana|order)|pharmacy|chemist|online dawai|medicine order)\b")),
    ("online", re.compile(r"\b(online|video|tele ?consult\w*|ghar baithe|phone par dikha|call par dikha)\b")),
    ("appointments", re.compile(r"\b(meri|mera|my)\b.{0,20}\b(appointment|booking)\b|\b(appointment|booking)\b.{0,25}"
                                r"\b(cancel|reschedule|badal|badalna|badalni|change|kab hai|status|check)\b"
                                r"|\bcancel kar\w*|\breschedule\b|\bdate badal\w*|\btime badal\w*")),
    ("reports", re.compile(r"\b(report|reports|prescription|parcha|parchi|test result)\b")),
    ("timings", re.compile(r"\b(kitne baje|timing|timings|khulta|khulti|khulte|band hota|band hoti|khula hai|khuli hai"
                           r"|opening time|working hours|open hai|open rehta)\b")),
    ("insurance", re.compile(r"\b(insurance|ayushman|cashless|mediclaim|health card|pmjay|cghs|esic?|tpa)\b")),
    ("ambulance", re.compile(r"\b(ambulance|emergency)\b.{0,15}\b(number|no|call|bulao|bulana|chahiye|kaise)\b"
                             r"|^ambulance$|\b(108|112) (number|kya)\b")),
    ("blood_vaccine", re.compile(r"\bblood (chahiye|bank|donor|donate|ki zarurat|need)\b|\bkhoon chahiye\b"
                                 r"|\b(a|b|o|ab) ?(positive|negative|pos|neg|\+|-) blood\b|\bblood group\b.*\bchahiye\b"
                                 r"|\b(vaccine|vaccination|teeka|tika)\b.*\b(kahan|kab|lagega|lagwana|chahiye|where)\b")),
    ("medicine", re.compile(r"\b(dawai|dawaai|medicine|medicines|tablet|tablets|goli|syrup|dose|dosage|antibiotic|antibiotics"
                            r"|paracetamol|crocin|dolo|calpol|combiflam|disprin|cetirizine|painkiller|gharelu|nuskha|nuskhe"
                            r"|home remedy|side effect|side effects)\b|\binjection (lagwa|le|chahiye)\w*")),
    ("serious", re.compile(r"\b(cancer|tumou?r|tb|serious|khatarnak|dangerous|normal)\b.{0,12}\b(hai kya|to nahi|hai na|kya)\b"
                           r"|\bkya (ye|yeh|mujhe|ise|isko|use|usko)\b.{0,20}\b(cancer|tumou?r|tb|serious|khatarnak|hai|ho gaya|hua)$"
                           r"|\bnormal hai$|\bnormal hai kya\b|\bkya hua hai mujhe\b|\bis (it|this) (serious|normal|cancer)\b")),
    ("what_to_do", re.compile(r"^(ab )?(kya karu|kya karun|kya karoon|kya karein|kya kare|ab kya|what (should|do) i do|what to do)$"
                              r"|\bghar (pe|par|mein) kya (karu|karun|karein|kare)\b|\bkya (karu|karun|karein) ab\b")),
    ("privacy", re.compile(r"\b(number|naam|details|data|otp|address)\b.{0,15}\b(kyun|kyu|why|zaroori|zaruri|safe|kisko|kise"
                           r"|kaun dekh|share hoga|share hogi)\b|\b(kyun|kyu|why)\b.{0,12}\b(number|naam|otp|details)\b"
                           r"|\bmera data\b|\bprivacy\b")),
    ("howareyou", re.compile(r"^((aap|tum|ap) )?(kaise|kaisi|kese|kaisa) (ho|hain|hai|h)\b|^how (are|r) (you|u)\b|^kya haal\b"
                             r"|^kaisa hai\b|^kaisa chal raha\b|^sab (theek|thik|badhiya|badiya|accha|achha|changa)( hai)?\b"
                             r"|^((aap|tum) )?(theek|thik) (ho|hain)\b|^kemon (acho|achen|aachen)\b|^how(s| is) it going\b"
                             r"|^how (you|u) doing\b|^aap sunao\b")),
    ("chitchat", re.compile(r"\bkya (kar|kr) (rahe|rahi|raha) (ho|hai|hain|h)\b|\bkya chal raha\b|\bwhats up\b|^wyd$"
                            r"|^(tum|aap) kahan (ho|hain)$|^kahan (ho|hain)$|^(aur|or) (batao|bta|btao|sunao|bataiye)$"
                            r"|\bwhat (are|r) (you|u) doing\b|\bkya ho raha\b|^(and you|aur aap|aur tum|tum batao|aap batao)$"
                            r"|^(good morning|good evening|good afternoon|suprabhat)\b")),
    ("goodnight", re.compile(r"^(good night|shubh ratri|shubhratri|sweet dreams)\b")),
    ("praise", re.compile(r"^(wah|waah|wow|superb|zabardast|mast|kamaal|kamal|badhiya|badiya|excellent|love it|perfect)( (hai|ji|yaar))*$|"
                          r"\b(thank you so much|thanks a lot|bahut (accha|achha|acha|badhiya|badiya|sahi)|great|awesome|nice|"
                          r"good job|well done|(tum|aap|you) (bahut )?(acchi|achhi|achi|smart|great|nice|helpful)|love you"
                          r"|pyaar|shabash)\b")),
    ("upset", re.compile(r"\b(dimag kharab|dimaag kharab|pareshan mat|time waste|time kharab|irritate|bekaar|bekar|pagal|bewakoof|bevkoof|stupid|useless|bakwas|bkwas|faltu|idiot|dumb|worst|ghatiya"
                         r"|samajh nahi aata (tumhe|aapko)|kuch nahi samajh\w*|kuch samajh nahi aata)\b")),
    ("offtopic", re.compile(r"\b(mausam|weather|cricket|score|match|news|khabar|recipe|khana banana|movie|film|song|gaana"
                            r"|joke|jokes|chutkula|homework|naukri|job|loan|shayari|bore|game|ipl|politics|neta|share market|gana sunao|gaana sunao"
                            r"|khana khaya|khana khaogi|khana khaoge|kya khaya"
                            r"|stock|bitcoin|gf|girlfriend|boyfriend|shaadi)\b")),
    ("what", re.compile(r"^(kya|kya kaha|kya bola|kya matlab|samjha nahi|samjhi nahi|samajh nahi aaya|nahi samjha|nahi samjhi"
                        r"|matlab|matlab kya|what|what do you mean|huh|phir se bolo|dobara bolo|repeat|repeat karo|repeat please|fir se batao"
                        r"|phir se batao|dobara batao|pardon)$")),
    ("yes", re.compile(r"^(haan|haa|ha|han|yes|yeah|ji haan|haanji|haan ji|ji ha)$")),
    ("continue", re.compile(r"^(aur|fir|phir|aage|next|aur batao|then|or|aur kuch)$")),
    ("pick", re.compile(r"^(pehla|pahla|pehle|first|1st|doosra|dusra|doosre|dusre|second|2nd|teesra|tisra|teesre|third|3rd)"
                        r"( (wala|wali|wale|doctor|one|number))?$")),
    ("restart", re.compile(r"^(wapas|back|go back|shuru se|start again|restart|menu|main menu|home|reset|start over"
                           r"|naya search|new search|phir se shuru)$")),
    ("better", re.compile(r"\b(kaunsa|konsa|kaun sa|which( one)?|kon sa|kaun)\b.{0,20}\b(better|accha|achha|acha|best|sahi"
                          r"|badhiya)\b|\bbetter kaun\b|\bkisko (dikhau|dikhaun|choose|chunu)\b")),
    ("distance", re.compile(r"\b(kitna door|kitni door|kitne door|how far|kitni doori|distance)\b")),
    ("ack", re.compile(r"^(accha|achha|acha|achchha|accha ji|hmm|hm|ohh?|oh ok|sahi hai|ok ji|theek hai ji|ji|ji ok|samajh ga(?:ya)"
                       r"|samajh gayi|got it|alright|cool|nahi pata|pata nahi|malum nahi|maloom nahi|idk|dont know|don t know|done|sahi|sahi hai"
                       r"|haan theek|haan theek hai|achha ji|acha ji|achchha ji|ok ok|okay okay|fine|thik hai ji)$")),
]
# Moves that may lead a reply and let the symptom flow go on ("bukhar hai, kaunsi dawai lu?" -> no medicine advice,
# then triage on the fever).
PREFIX_MOVES = {"medicine", "serious"}
# Answers that need no model: the fallback (LLM off or down) can still give them. Medicine / "is it serious" / "what
# should I do" usually come with a symptom, which needs the model, so they are not here.
# Social talk answered before any doctor search when the model is off (the model reads these itself when it is on).
SOCIAL_MOVES = {"thanks", "bye", "later", "wait", "sorry", "name", "test", "personal", "howareyou", "chitchat",
                "goodnight", "praise", "upset", "about"}
NO_MODEL_MOVES = {"thanks", "bye", "later", "wait", "sorry", "name", "test", "personal", "about", "what_is_dd", "fees", "free", "coming_soon", "online", "appointments", "reports", "timings",
                  "insurance", "ambulance", "blood_vaccine", "privacy", "howareyou", "chitchat", "goodnight", "praise",
                  "upset", "offtopic", "what", "ack", "better", "distance"}
# Repair words: they only mean something in context and never beat a doctor or symptom ask.
REPAIR_MOVES = {"what", "continue", "pick", "restart", "better", "distance", "ack", "yes"}

_ORDINAL = {"pehla": 0, "pahla": 0, "pehle": 0, "first": 0, "1st": 0, "doosra": 1, "dusra": 1, "doosre": 1, "dusre": 1,
            "second": 1, "2nd": 1, "teesra": 2, "tisra": 2, "teesre": 2, "third": 2, "3rd": 2}


def detect(norm: str, raw: str = "") -> str | None:
    norm = (norm or "").strip().rstrip("?!. ").strip()
    if not norm:
        return "what" if raw and re.fullmatch(r"\s*[?\uff1f]+\s*", raw) else None      # "???" asks "what do you mean?"
    for name, rx in MOVES:
        if rx.search(norm):
            return name
    return None


def said_name(norm: str) -> str | None:
    """"mera naam rahul hai" -> "Rahul" (only to greet them; never stored)."""
    m = dict(MOVES)["name"].search((norm or "").strip().rstrip("?!. "))
    if not m:
        return None
    n = m.group("name") or m.group("name2") or m.group("name3") or ""
    return n.title() or None


def ordinal(norm: str) -> int | None:
    first = (norm or "").strip().split()[:1]
    return _ORDINAL.get(first[0]) if first else None


def reply(move: str, v: Voice, *, fee: object = None, doctor: str | None = None, name: str | None = None) -> str:
    """The answer, in the patient's language. Booking, lists and buttons are added by the caller."""
    if move == "about":
        return v("I'm Vita, Doctor Dekho's AI assistant. I'm not a person or a doctor. I listen to what's bothering you, "
                 "find the right doctor and book the appointment, in English, Hindi or Hinglish.",
                 "Main Vita hoon, Doctor Dekho ki AI assistant. Main insaan ya doctor nahi hoon. Main aapki takleef sunkar "
                 "sahi doctor dhoondhti hoon aur appointment book karti hoon, Hindi, Hinglish ya English mein.",
                 "मैं Vita हूँ, Doctor Dekho की AI सहायक। मैं इंसान या डॉक्टर नहीं हूँ। मैं आपकी तकलीफ़ सुनकर सही डॉक्टर "
                 "ढूँढती हूँ और अपॉइंटमेंट बुक करती हूँ, हिंदी, हिंग्लिश या अंग्रेज़ी में।")
    if move == "what_is_dd":
        return v("Doctor Dekho lists doctors from hospitals near you. You can see their fees and days, and book an "
                 "appointment for free; you pay the doctor's fee at the hospital. I'm Vita, I help you find the right one.",
                 "Doctor Dekho par aapke paas ke hospitals ke doctor milte hain. Unki fees aur din dekh kar appointment free "
                 "mein book kar sakte hain; doctor ki fees hospital mein deni hoti hai. Main Vita hoon, sahi doctor dhoondhne "
                 "mein madad karti hoon.",
                 "Doctor Dekho पर आपके पास के अस्पतालों के डॉक्टर मिलते हैं। उनकी फ़ीस और दिन देखकर अपॉइंटमेंट मुफ़्त में बुक "
                 "कर सकते हैं; डॉक्टर की फ़ीस अस्पताल में देनी होती है। मैं Vita हूँ, सही डॉक्टर ढूँढने में मदद करती हूँ।")
    if move == "fees":
        if doctor and fee is not None:
            return v(f"{doctor}'s fee is Rs {fee}, paid at the hospital. Booking through Doctor Dekho is free.",
                     f"{doctor} ki fees Rs {fee} hai, jo hospital mein deni hoti hai. Doctor Dekho se booking free hai.",
                     f"{doctor} की फ़ीस {fee} रुपये है, जो अस्पताल में देनी होती है। Doctor Dekho से बुकिंग मुफ़्त है।")
        return v("Each doctor's fee is shown with their name, and you pay it at the hospital. Booking is free. "
                 "Which doctor's fee would you like to see?",
                 "Har doctor ki fees unke naam ke saath dikhti hai, aur fees hospital mein deni hoti hai. Booking free hai. "
                 "Kis tarah ke doctor ki fees dekhni hai?",
                 "हर डॉक्टर की फ़ीस उनके नाम के साथ दिखती है, और फ़ीस अस्पताल में देनी होती है। बुकिंग मुफ़्त है। "
                 "किस तरह के डॉक्टर की फ़ीस देखनी है?")
    if move == "free":
        return v("Finding a doctor and booking an appointment is free. You only pay the doctor's fee at the hospital, so "
                 "there is no online payment or refund.",
                 "Doctor dhoondhna aur appointment book karna free hai. Aap sirf doctor ki fees hospital mein dete hain, "
                 "isliye online payment ya refund nahi hota.",
                 "डॉक्टर ढूँढना और अपॉइंटमेंट बुक करना मुफ़्त है। आप सिर्फ़ डॉक्टर की फ़ीस अस्पताल में देते हैं, इसलिए "
                 "ऑनलाइन पेमेंट या रिफ़ंड नहीं होता।")
    if move == "online":
        return v("Online and video consultations are coming to Doctor Dekho soon. For now, appointments are at the "
                 "hospital. Shall I find a doctor near you?",
                 "Online / video consultation jald hi Doctor Dekho par aa raha hai. Abhi hospital mein milne wale appointment "
                 "hain. Kya main paas ke doctor dhoondhun?",
                 "ऑनलाइन / वीडियो परामर्श जल्द ही Doctor Dekho पर आ रहा है। अभी अस्पताल में मिलने वाले अपॉइंटमेंट हैं। "
                 "क्या मैं पास के डॉक्टर ढूँढूँ?")
    if move == "appointments":
        return v("You can see, cancel or change your appointments under 'My appointments' on Doctor Dekho.",
                 "Apni appointment Doctor Dekho par 'Meri appointments' mein dekh, cancel ya badal sakte hain.",
                 "अपनी अपॉइंटमेंट Doctor Dekho पर 'मेरी अपॉइंटमेंट' में देख, रद्द या बदल सकते हैं।")
    if move == "reports":
        return v("Prescriptions and test reports come from the doctor and the hospital, and only the doctor can explain "
                 "what a report means. Shall I find a doctor?",
                 "Prescription aur report doctor aur hospital dete hain, aur report ka matlab doctor hi batayenge. "
                 "Kya main doctor dhoondhun?",
                 "प्रिस्क्रिप्शन और रिपोर्ट डॉक्टर और अस्पताल देते हैं, और रिपोर्ट का मतलब डॉक्टर ही बताएँगे। "
                 "क्या मैं डॉक्टर ढूँढूँ?")
    if move == "timings":
        return v("Each doctor's days and hours are on their card, and the hospital confirms the exact time. Which doctor "
                 "or kind of doctor should I check?",
                 "Har doctor ke din aur samay unke card par hain, sahi samay hospital confirm karta hai. Kis doctor ya kis tarah "
                 "ke doctor ka samay dekhun?",
                 "हर डॉक्टर के दिन और समय उनके कार्ड पर हैं, सही समय अस्पताल बताता है। किस डॉक्टर या किस तरह के डॉक्टर का "
                 "समय देखूँ?")
    if move == "insurance":
        return v("Insurance and Ayushman details are coming to Doctor Dekho soon. For now, please check with the "
                 "hospital; I can find you a doctor there.",
                 "Insurance / Ayushman ki jaankari jald hi Doctor Dekho par aayegi. Abhi ye hospital se confirm karein; "
                 "main wahan ke doctor dhoondh sakti hoon.",
                 "इंश्योरेंस / आयुष्मान की जानकारी जल्द ही Doctor Dekho पर आएगी। अभी यह अस्पताल से पता करें; मैं वहाँ के "
                 "डॉक्टर ढूँढ सकती हूँ।")
    if move == "coming_soon":
        return v("Lab tests and medicine delivery are coming to Doctor Dekho soon. Right now I can find you a doctor and "
                 "book the appointment.",
                 "Lab test aur dawai ki delivery jald hi Doctor Dekho par aa rahi hai. Abhi main doctor dhoondhne aur "
                 "appointment book karne mein madad kar sakti hoon.",
                 "लैब टेस्ट और दवाई की डिलीवरी जल्द ही Doctor Dekho पर आ रही है। अभी मैं डॉक्टर ढूँढने और अपॉइंटमेंट बुक "
                 "करने में मदद कर सकती हूँ।")
    if move == "blood_vaccine":
        return v("Please ask the nearest hospital or blood bank directly; Doctor Dekho doesn't arrange blood or vaccines. "
                 "I can find you a doctor.",
                 "Iske liye nazdeeki hospital ya blood bank se seedha baat karein; Doctor Dekho blood ya vaccine nahi dilata. "
                 "Main doctor dhoondhne mein madad kar sakti hoon.",
                 "इसके लिए नज़दीकी अस्पताल या ब्लड बैंक से सीधे बात करें; Doctor Dekho खून या टीका नहीं दिलाता। मैं डॉक्टर "
                 "ढूँढने में मदद कर सकती हूँ।")
    if move == "ambulance":
        return v(f"Ambulance: {EMERGENCY_NUMBER}. Any emergency: {GENERAL_EMERGENCY_NUMBER}. If someone is seriously unwell, "
                 "call now.",
                 f"Ambulance: {EMERGENCY_NUMBER}. Koi bhi emergency: {GENERAL_EMERGENCY_NUMBER}. Koi gambhir haalat mein ho "
                 "to abhi call karein.",
                 f"एम्बुलेंस: {EMERGENCY_NUMBER}। कोई भी इमरजेंसी: {GENERAL_EMERGENCY_NUMBER}। कोई गंभीर हालत में हो तो अभी "
                 "कॉल करें।")
    if move == "medicine":
        return v("I can't suggest medicines or doses; a doctor or pharmacist has to decide that.",
                 "Main dawai ya dose ki salah nahi de sakti; ye doctor ya pharmacist hi batayenge.",
                 "मैं दवाई या खुराक की सलाह नहीं दे सकती; यह डॉक्टर या फ़ार्मासिस्ट ही बताएँगे।")
    if move == "what_to_do":
        return v("Tell me what's bothering you and I'll find the right doctor. If it's serious, call "
                 f"{EMERGENCY_NUMBER} now.",
                 "Apni takleef batayein, main sahi doctor dhoondh deti hoon. Haalat gambhir ho to abhi "
                 f"{EMERGENCY_NUMBER} par call karein.",
                 "अपनी तकलीफ़ बताइए, मैं सही डॉक्टर ढूँढ देती हूँ। हालत गंभीर हो तो अभी "
                 f"{EMERGENCY_NUMBER} पर कॉल करें।")
    if move == "serious":
        return v("Only a doctor can tell what it is or how serious it is.",
                 "Ye kya hai aur kitna serious hai, ye doctor hi bata sakte hain.",
                 "यह क्या है और कितना गंभीर है, यह डॉक्टर ही बता सकते हैं।")
    if move == "privacy":
        return v("Your number is used to confirm the appointment and send its details on WhatsApp; the code (OTP) "
                 "checks the number is yours. Your name and number are sent to the hospital for the appointment.",
                 "Number appointment confirm karne aur WhatsApp par details bhejne ke liye hai; OTP se pakka hota hai ki "
                 "number aapka hai. Aapka naam aur number appointment ke liye hospital ko bheja jaata hai.",
                 "नंबर अपॉइंटमेंट पक्की करने और WhatsApp पर जानकारी भेजने के लिए है; OTP से पक्का होता है कि नंबर आपका है। "
                 "आपका नाम और नंबर अपॉइंटमेंट के लिए अस्पताल को भेजा जाता है।")
    if move == "howareyou":
        return v("I'm well, thank you! How can I help you today?",
                 "Main theek hoon, shukriya! Aap batayein, kya madad karun?",
                 "मैं ठीक हूँ, शुक्रिया! आप बताइए, क्या मदद करूँ?")
    if move == "distress":
        return v(f"If this is an emergency, call {EMERGENCY_NUMBER} (ambulance) now. What has happened? Tell me in a few "
                 "words.",
                 f"Agar emergency hai to abhi {EMERGENCY_NUMBER} (ambulance) par call karein. Kya hua hai? Thode shabdon "
                 "mein batayein.",
                 f"अगर इमरजेंसी है तो अभी {EMERGENCY_NUMBER} (एम्बुलेंस) पर कॉल करें। क्या हुआ है? थोड़े शब्दों में बताइए।")
    if move == "thanks":
        return v("You're welcome! Write any time you need a doctor.",
                 "Aapka swagat hai! Doctor ki zaroorat ho to kabhi bhi likhiye.",
                 "आपका स्वागत है! डॉक्टर की ज़रूरत हो तो कभी भी लिखिए।")
    if move == "bye":
        return v("Take care! Write any time you need a doctor.",
                 "Apna dhyan rakhiye! Doctor ki zaroorat ho to kabhi bhi likhiye.",
                 "अपना ध्यान रखिए! डॉक्टर की ज़रूरत हो तो कभी भी लिखिए।")
    if move == "later":
        return v("No problem. I'm here whenever you need a doctor or have a question.",
                 "Koi baat nahi. Jab bhi doctor chahiye ya koi sawaal ho, main yahin hoon.",
                 "कोई बात नहीं। जब भी डॉक्टर चाहिए या कोई सवाल हो, मैं यहीं हूँ।")
    if move == "wait":
        return v("Sure, take your time. I'm here.",
                 "Ji, aaram se. Main yahin hoon.",
                 "जी, आराम से। मैं यहीं हूँ।")
    if move == "sorry":
        return v("No problem! Tell me what's bothering you, or which doctor you need.",
                 "Koi baat nahi! Apni takleef batayein, ya kis tarah ka doctor chahiye.",
                 "कोई बात नहीं! अपनी तकलीफ़ बताइए, या किस तरह का डॉक्टर चाहिए।")
    if move == "name":
        who = f", {name}" if name else ""
        return v(f"Nice to meet you{who}! Tell me what's bothering you, or which doctor you need.",
                 f"Aapse milkar accha laga{who}! Apni takleef batayein, ya kis tarah ka doctor chahiye.",
                 f"आपसे मिलकर अच्छा लगा{who}! अपनी तकलीफ़ बताइए, या किस तरह का डॉक्टर चाहिए।")
    if move == "test":
        return v("Yes, I'm here and working. Tell me what's bothering you, or which doctor you need.",
                 "Ji, main yahin hoon aur kaam kar rahi hoon. Apni takleef batayein, ya kis tarah ka doctor chahiye.",
                 "जी, मैं यहीं हूँ और काम कर रही हूँ। अपनी तकलीफ़ बताइए, या किस तरह का डॉक्टर चाहिए।")
    if move == "personal":
        return v("I'm Vita, an AI assistant for your health. Let's keep to that: do you need a doctor?",
                 "Main Vita hoon, aapki sehat ke liye AI assistant. Isi mein madad kar sakti hoon: kya kisi doctor ki zaroorat hai?",
                 "मैं Vita हूँ, आपकी सेहत के लिए AI सहायक। इसी में मदद कर सकती हूँ: क्या किसी डॉक्टर की ज़रूरत है?")
    if move == "chitchat":
        return v("I'm here to help you find a doctor. Is something bothering you, or do you need a doctor?",
                 "Main yahin hoon, aapki madad ke liye. Koi takleef hai, ya kisi doctor ki zaroorat hai?",
                 "मैं यहीं हूँ, आपकी मदद के लिए। कोई तकलीफ़ है, या किसी डॉक्टर की ज़रूरत है?")
    if move == "goodnight":
        return v("Good night! Write any time you need a doctor.",
                 "Shubh ratri! Doctor ki zaroorat ho to kabhi bhi likhiye.",
                 "शुभ रात्रि! डॉक्टर की ज़रूरत हो तो कभी भी लिखिए।")
    if move == "praise":
        return v("Thank you! I'm here to help with your health.",
                 "Shukriya! Main aapki sehat mein madad ke liye yahin hoon.",
                 "शुक्रिया! मैं आपकी सेहत में मदद के लिए यहीं हूँ।")
    if move == "upset":
        return v("Sorry, I may have misunderstood. Tell me the problem in one line, or write 'talk to a person'.",
                 "Maaf kijiye, shayad main samajh nahi paayi. Apni takleef ek line mein likhein, ya 'insaan se baat' likhein.",
                 "माफ़ कीजिए, शायद मैं समझ नहीं पाई। अपनी तकलीफ़ एक लाइन में लिखें, या 'इंसान से बात' लिखें।")
    if move == "offtopic":
        return v("I can only help with finding a doctor and booking an appointment. Do you need a doctor?",
                 "Main sirf doctor dhoondhne aur appointment mein madad karti hoon. Kya kisi doctor ki zaroorat hai?",
                 "मैं सिर्फ़ डॉक्टर ढूँढने और अपॉइंटमेंट में मदद करती हूँ। क्या किसी डॉक्टर की ज़रूरत है?")
    if move == "better":
        return v("I can't say one doctor is better than another. I can sort them by fee, distance or experience.",
                 "Main kisi doctor ko doosre se better nahi keh sakti. Fees, doori ya experience se sort kar sakti hoon.",
                 "मैं किसी डॉक्टर को दूसरे से बेहतर नहीं कह सकती। फ़ीस, दूरी या अनुभव से क्रम में लगा सकती हूँ।")
    if move == "ack":
        return v("Okay. Anything else I can help with?",
                 "Ji. Aur kuch madad chahiye?",
                 "जी। और कुछ मदद चाहिए?")
    if move == "what":
        return v("Let me put it simply: tell me what's bothering you, or which doctor you need.",
                 "Aasaan shabdon mein: apni takleef batayein, ya kis tarah ka doctor chahiye.",
                 "आसान शब्दों में: अपनी तकलीफ़ बताइए, या किस तरह का डॉक्टर चाहिए।")
    if move == "restart":
        return v("Sure, let's start again. How can I help you find a doctor?",
                 "Theek hai, shuru se karte hain. Aapko kis tarah ka doctor chahiye?",
                 "ठीक है, शुरू से करते हैं। आपको किस तरह का डॉक्टर चाहिए?")
    if move == "distance":
        return v("Share your location and I'll show the nearest doctors first.",
                 "Location share karein, main paas wale doctor pehle dikhaungi.",
                 "लोकेशन शेयर करें, मैं पास वाले डॉक्टर पहले दिखाऊँगी।")
    if move in ("pick", "continue", "yes"):
        return v("Sure. What kind of doctor do you need?", "Ji. Aapko kis tarah ka doctor chahiye?",
                 "जी। आपको किस तरह का डॉक्टर चाहिए?")
    raise KeyError(move)


def sort_options(v: Voice) -> list[dict]:
    """Chips that the doctor-search parser already understands."""
    return [{"id": "sort:fee", "label": v("Lowest fee first", "Kam fees wale pehle", "कम फ़ीस वाले पहले")},
            {"id": "sort:near", "label": v("Nearest first", "Paas wale pehle", "पास वाले पहले")},
            {"id": "sort:experience", "label": v("Most experienced first", "Zyada experience wale pehle", "ज़्यादा अनुभव वाले पहले")}]


def again(v: Voice) -> str:
    """Lead-in when Vita repeats its last question for "kya?" / "samjha nahi"."""
    return v("Let me ask again:", "Main dobara poochti hoon:", "मैं दोबारा पूछती हूँ:")


# ------------------------------------------------------------------------------------------------ warning signs
# A head injury or a burn on its own is not always an emergency, but some are. Vita asks for the warning signs in one
# question; a yes is an emergency, a no gets "see a doctor today" and a general physician. (For doctor review.)
WARN = {
    # pregnancy + swelling: usually harmless swelling of the feet, but with these signs possibly pre-eclampsia
    "pregnancy": re.compile(r"(?=.*\b(pregnan\w*|pregn[ae]nt\w*|pregnency|garbh\w*|pet se (hoon|hu|hun)|delivery (hone|aane) wali|mahine ki pregnancy)\b)"
                            r"(?=.*\b(sujan|soojan|sujaan|sooj\w*|suj (gaye|gayi|gaya|gye|gyi|rahe|raha|rahi)|swell\w*|swollen|"
                            r"phool (gaye|gayi|gaya|rahe)|edema|oedema)\b)"),
    "head": re.compile(r"\b(sar|sir|head|maatha|matha|mathe)\b.{0,12}\b(chot|chhot|injury|lag gayi|lagi|takra\w*|phat\w*|fat gaya)"
                       r"|\b(gir gaya|gir gayi|gira|giri|gir gya)\b.{0,20}\b(sar|sir|head)\b|\bhead injury\b"),
    "burn": re.compile(r"\b(jal gaya|jal gayi|jal gya|jal gyi|jala|jali|jal gaye|burn|burns|burnt|burned)\b"
                       r"|\bgaram (paani|pani|tel|chai|doodh|dal)\b.{0,12}\bgir\w*|\bchhale pad\w*"),
}
WARN_SIGN = {"head": "head injury", "burn": "burn injury", "pregnancy": "pregnancy warning signs"}
# Already a warning sign in the first message: no question, the emergency reply as before. (For doctor review.)
SERIOUS = {
    "burn": re.compile(r"\b(chehr\w*|face|muh|munh|aankh\w*|eyes?|gala|gale|neck|bijli|current|electric\w*|tezaab|acid|chemical|"
                       r"saans|breath\w*|behosh\w*|unconscious|gupt|private|poora|pura|whole|sharir|body|aag mein|bahut jal|"
                       r"zyada jal|kaafi jal|bada hissa|large)\b"),
    "head": re.compile(r"\b(behosh\w*|unconscious|faint\w*|ulti|vomit\w*|khoon|blood|bleed\w*|daura|fit|seizure|"
                       r"hosh|confus\w*|kaan se|naak se|bahut neend|drowsy)\b"),
    "pregnancy": re.compile(r"\b(dhundhla|blur\w*|spots|sar (mein |me |main )?(tez |bahut )?dard|severe headache|daura|fit|"
                            r"seizure|khoon|blood|bleed\w*|chehr\w*|face|haath|hands?|upar pet|upper (stomach|abdomen)|"
                            r"pasli|ribs|kam hil|hil nahi|movement kam|not moving|saans|breath\w*|behosh\w*)\b"),
}
# where "no warning signs" goes
WARN_SPECIALTY = {"head": "general-medicine", "burn": "general-medicine", "pregnancy": "gynaecology"}
_YES = re.compile(r"^(haan|haa|ha|han|yes|ji|ji haan|haanji|hua|hui|hai|hain|ho raha|ho rahi|ho gaya|ho gayi)\b")
_NO = re.compile(r"^(nahi|nahin|na|no|nope|kuch nahi|aisa kuch nahi|nahi hua|nahi hui)\b")


def warn_kind(norm: str) -> str | None:
    return next((k for k, rx in WARN.items() if rx.search(norm or "")), None)


def serious(kind: str, norm: str) -> bool:
    """The first message already names a warning sign for this kind."""
    rx = SERIOUS.get(kind)
    return bool(rx and rx.search(norm or ""))


def warn_answer(norm: str) -> bool | None:
    """True for yes, False for no, None when the message answers something else."""
    n = (norm or "").strip()
    if _NO.match(n):
        return False
    if _YES.match(n):
        return True
    return None


def warn_question(kind: str, v: Voice) -> str:
    if kind == "pregnancy":
        return v("Along with the swelling, is there any of these: a bad headache, blurred vision or seeing spots, "
                 "sudden swelling of the face or hands, pain in the upper stomach, a fit, bleeding, or the baby moving less?",
                 "Sujan ke saath kya inmein se kuch hai: tez sar dard, dhundhla dikhna ya aankhon ke aage dhabbe, chehre ya "
                 "haathon mein achanak sujan, upar pet mein dard, daura, khoon aana, ya baby ka kam hilna?",
                 "सूजन के साथ क्या इनमें से कुछ है: तेज़ सिर दर्द, धुंधला दिखना या आँखों के आगे धब्बे, चेहरे या हाथों में "
                 "अचानक सूजन, ऊपर पेट में दर्द, दौरा, खून आना, या बच्चे का कम हिलना?")
    if kind == "head":
        return v("After the head injury, did any of these happen: fainting, vomiting, a lot of sleepiness, a severe or "
                 "worsening headache, a fit, or bleeding from the nose or ear?",
                 "Sar ki chot ke baad kya inmein se kuch hua: behoshi, ulti, bahut neend, tez ya badhta sar dard, daura, "
                 "ya naak / kaan se khoon?",
                 "सिर की चोट के बाद क्या इनमें से कुछ हुआ: बेहोशी, उल्टी, बहुत नींद, तेज़ या बढ़ता सिर दर्द, दौरा, या नाक / "
                 "कान से खून?")
    return v("Is the burn bigger than the palm of a hand, or on the face, hands, feet or private parts, or from "
             "electricity or acid, or is there trouble breathing?",
             "Kya jala hua hissa hatheli se bada hai, ya chehra, haath, pair ya gupt ang jale hain, ya bijli / tezaab se "
             "jale hain, ya saans mein dikkat hai?",
             "क्या जला हुआ हिस्सा हथेली से बड़ा है, या चेहरा, हाथ, पैर या गुप्त अंग जले हैं, या बिजली / तेज़ाब से जले हैं, "
             "या साँस में दिक्कत है?")


def warn_options(v: Voice) -> list[dict]:
    return [{"id": "warn:yes", "label": v("Yes", "Haan", "हाँ")}, {"id": "warn:no", "label": v("No", "Nahi", "नहीं")}]


def warn_no(kind: str, v: Voice) -> str:
    if kind == "pregnancy":
        return v("Okay. Please see your gynaecologist within a day to check your blood pressure. If a bad headache, "
                 "blurred vision, swelling of the face or hands, upper stomach pain or bleeding starts, call "
                 f"{EMERGENCY_NUMBER} straight away.",
                 "Theek hai. Ek din ke andar apni gynaecologist ko dikhakar BP check karwayein. Agar tez sar dard, dhundhla "
                 f"dikhna, chehre ya haathon mein sujan, upar pet mein dard ya khoon aaye to turant {EMERGENCY_NUMBER} par call karein.",
                 "ठीक है। एक दिन के अंदर अपनी गायनेकोलॉजिस्ट को दिखाकर BP चेक करवाएँ। अगर तेज़ सिर दर्द, धुंधला दिखना, "
                 f"चेहरे या हाथों में सूजन, ऊपर पेट में दर्द या खून आए तो तुरंत {EMERGENCY_NUMBER} पर कॉल करें।")
    if kind == "head":
        return v("Okay. Still, see a doctor today. If vomiting, fainting, a bad headache or a fit starts later, call "
                 f"{EMERGENCY_NUMBER} straight away.",
                 "Theek hai. Phir bhi aaj hi doctor ko dikhayein. Baad mein ulti, behoshi, tez sar dard ya daura ho to turant "
                 f"{EMERGENCY_NUMBER} par call karein.",
                 "ठीक है। फिर भी आज ही डॉक्टर को दिखाएँ। बाद में उल्टी, बेहोशी, तेज़ सिर दर्द या दौरा हो तो तुरंत "
                 f"{EMERGENCY_NUMBER} पर कॉल करें।")
    return v("Okay. Still, have a doctor see it, sooner if the pain or blisters get worse or a fever starts.",
             "Theek hai. Phir bhi doctor ko dikhayein; dard ya chhale badhein ya bukhar aaye to jaldi dikhayein.",
             "ठीक है। फिर भी डॉक्टर को दिखाएँ; दर्द या छाले बढ़ें या बुखार आए तो जल्दी दिखाएँ।")
