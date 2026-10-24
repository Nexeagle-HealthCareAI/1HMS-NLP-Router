"""Vita books the appointment (vision Phase 2): one question at a time, never without an explicit "yes".

After the patient chooses a doctor:

    day      -> the doctor's next working days (1HMS availability)            "Which day suits you?"
    shift    -> that day's shifts (skipped when there is only one)            "Morning or evening?"
    who      -> myself / a saved family member / someone else                 (signed in: saved profile + family)
    name, age, sex                                                             only what is missing
    mobile, otp                                                                guests only: a WhatsApp code to the
                                                                               number; signed-in patients and a channel
                                                                               that verified the number skip both
    confirm  -> "Book Dr X, Skin, Salt Lake Care Kolkata, Tue 27 Oct morning, for Rina (32, F), fee Rs 600?"
    book     -> POST /public/appointments (1HMS); a refusal offers the next free days, never a silent retry

"Cancel", "not now", "rehne do" ends it at any step; the emergency check still runs first on every message
(DoctorChat). The hospital confirms the exact time: a public booking is a request for a day and shift.

Privacy: the patient's details live only in this session's Redis state (30 minutes) and go to 1HMS for the
booking; the OTP token never leaves the server except to the channel that asked for it (to sign the patient
in); telemetry gets no names, numbers or codes.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import date, timedelta
from typing import Any

from ..booking_client import BookingError, HmsPublicClient
from ..doctors import DoctorLookupError
from ..textnorm import intent_text
from ..voice import Voice

log = logging.getLogger("hms_router.booking")

BOOKING_KIND, BOOKED_KIND, FOLLOW_UP = "booking", "booked", "follow_up"
MAX_OTP_TRIES = 3
MAX_DAYS = 4

_CANCEL = re.compile(r"^(cancel|stop|not now|no thanks|nahi chahiye|nahin chahiye|rehne do|rahne do|mat karo|band karo|"
                     r"baad mein|baad me|chhodo|chodo|leave it|never ?mind|abhi nahi|abhi nahin)\b")
_YES = re.compile(r"^(yes|haan|han|ha|ji|ji haan|haanji|ok|okay|theek hai|thik hai|sure|confirm|book|book karo|"
                  r"book kar do|haan book|yes book|kar do|karo)\b")
_CHANGE = re.compile(r"\b(change|badlo|badal|badalna|dusra|doosra|edit)\b")
_MOBILE = re.compile(r"(?:\+?91[\s-]*)?([6-9]\d{9})\b")
_DIGITS = re.compile(r"\b(\d{4,6})\b")
_AGE = re.compile(r"\b(\d{1,3})\s*(saal|sal|years?|yrs?|y|varsh|baras|months?|mahine?|mahina|din|days?)?\b")
_NAME_LEAD = re.compile(r"^(mera naam|my name is|name is|naam|name|patient (ka )?naam|i am|main)\s+", re.I)
_MALE = {"male", "m", "man", "boy", "purush", "aadmi", "ladka", "mard", "gents"}
_FEMALE = {"female", "f", "woman", "girl", "mahila", "aurat", "ladki", "stree", "lady"}
_OTHER = {"other", "anya", "dusra", "doosra", "transgender"}


# ------------------------------------------------------------------------------------------------- helpers
def token_mobile(token: str | None) -> str | None:
    """The mobile in a 1HMS patient token (claim "mobile"). Read without checking the signature: it is only
    used to fill the booking form, and 1HMS checks the token itself when the booking is sent."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        m = _MOBILE.search(str(claims.get("mobile") or ""))
        return m.group(1) if m else None
    except Exception:
        return None


def _mask(mobile: str) -> str:
    return f"{mobile[:2]}xxxxxx{mobile[-2:]}" if len(mobile) >= 6 else "xxxx"


def _norm(text: str) -> str:
    return " ".join(intent_text(text).split())


def _today() -> date:
    from .doctor_chat import _today_ist
    return _today_ist()


_HI_DAYS = ("सोम", "मंगल", "बुध", "गुरु", "शुक्र", "शनि", "रवि")
_HI_MONTHS = ("जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर")
_SHIFT_WORDS = {"morning": ("Subah", "सुबह"), "afternoon": ("Dopahar", "दोपहर"), "evening": ("Shaam", "शाम"),
                "night": ("Raat", "रात")}


def _date_words(v: Voice, d: date) -> str:
    """Mon 26 Oct / सोम, 26 अक्टूबर: a tapped chip in Hindi stays in Hindi script."""
    if v.code == "hi":
        return f"{_HI_DAYS[d.weekday()]}, {d.day} {_HI_MONTHS[d.month - 1]}"
    return d.strftime("%a %d %b")


def _day_label(v: Voice, iso: str) -> str:
    d = date.fromisoformat(iso)
    if d == _today():
        return v("Today", "Aaj", "आज") + f" ({_date_words(v, d)})"
    if d == _today() + timedelta(days=1):
        return v("Tomorrow", "Kal", "कल") + f" ({_date_words(v, d)})"
    return _date_words(v, d)


def _shift_label(sh: dict, v: Voice | None = None) -> str:
    """'Morning 09:00-13:00' in the conversation's language (1HMS names shifts in English)."""
    hours = f"{sh.get('start') or ''}-{sh.get('end') or ''}".strip("-")
    name = sh.get("name") or ""
    words = _SHIFT_WORDS.get(name.strip().lower())
    if v is not None and words and v.code != "en":
        name = words[0 if v.code == "hg" else 1]
    return " ".join(x for x in (name, hours) if x) or "Any time"


def _sex_word(v: Voice, sex: str | None) -> str:
    return {"Male": v("M", "M", "पु"), "Female": v("F", "F", "म"), "Other": v("O", "O", "अ")}.get(sex or "", "")


def _fee(d: dict) -> Any:
    return d.get("discountedFee") if d.get("discountedFee") is not None else d.get("consultFee")


def _sex_of(text: str) -> str | None:
    words = set(_norm(text).split()) | set(text.lower().split())
    if words & _FEMALE or "महिला" in text or "स्त्री" in text:
        return "Female"
    if words & _MALE or "पुरुष" in text:
        return "Male"
    if words & _OTHER or "अन्य" in text:
        return "Other"
    return None


def _clean_name(text: str) -> str | None:
    t = _NAME_LEAD.sub("", text.strip()).strip(" .,!-")
    t = re.sub(r"\s+(hai|he|h)$", "", t, flags=re.I)
    if not (2 <= len(t) <= 60) or re.search(r"\d", t) or len(t.split()) > 5:
        return None
    return " ".join(w[:1].upper() + w[1:] for w in t.split()) if t.isascii() else t


def _age_of(text: str) -> tuple[int, str] | None:
    m = _AGE.search(_norm(text) or text)
    if not m:
        return None
    n, unit = int(m.group(1)), (m.group(2) or "")
    if unit.startswith(("month", "mahin")):
        return (n, "months") if 0 < n < 24 else None
    if unit.startswith(("din", "day")):
        return (max(1, n // 30), "months") if 0 < n < 366 else None
    return (n, "years") if 0 < n <= 120 else None


# ------------------------------------------------------------------------------------------------- the flow
class BookingFlow:
    """Runs one booking conversation. State lives in the DoctorChat session state under "booking"."""

    def __init__(self, directory, client: HmsPublicClient | None):
        self.directory = directory
        self.client = client

    @property
    def enabled(self) -> bool:
        return bool(self.client and self.client.enabled)

    # ---------------------------------------------------------------- entry
    def start(self, st: dict, doctor: dict, v: Voice, ctx: dict) -> dict:
        """The doctor was just chosen: ask for the day. Returns {"message", "clarify", "kind"}."""
        b = {"step": "day", "doctor": _doctor_brief(doctor), "started": int(time.time())}
        st["booking"] = b
        try:
            sched = self.directory.availability(doctor["doctorId"], from_date=_today().isoformat(), days=7)
        except DoctorLookupError:
            sched = None
        days = [d for d in (sched or {}).get("days") or [] if d.get("isAvailable")][:MAX_DAYS]
        if not days:
            st.pop("booking", None)
            msg = v(f"{doctor.get('name')} has no working day in the next 7 days on Doctor Dekho. "
                    "Shall I show other doctors?",
                    f"{doctor.get('name')} agle 7 din mein available nahi hain. Kya main doosre doctor dikhaun?",
                    f"{doctor.get('name')} अगले 7 दिन उपलब्ध नहीं हैं। क्या मैं दूसरे डॉक्टर दिखाऊँ?")
            return self._ask(msg, "book_unavailable",
                             [{"id": "others", "label": v("Other doctors", "Doosre doctor", "दूसरे डॉक्टर")}])
        b["days"] = [{"date": d["date"], "shifts": [s for s in d.get("shifts") or [] if s.get("start")]} for d in days]
        return self._ask_day(b, v)

    # ---------------------------------------------------------------- one turn
    def turn(self, st: dict, text: str, v: Voice, ctx: dict) -> dict | None:
        """Handle one message while booking. Returns {"message", "clarify", "kind", "booking"?} or None when
        the message is not about booking (the caller then treats it as a new ask and the booking ends)."""
        b = st["booking"]
        norm = _norm(text)
        step = b.get("step")

        if _CANCEL.match(norm) or (step in ("day", "shift", "who", "sex", "confirm", "otp", "mobile")
                                    and self._picked(b, text) == "cancel"):
            st.pop("booking", None)
            return self._say(v("Okay, nothing was booked. I'm here if you need a doctor.",
                               "Theek hai, kuch book nahi hua. Doctor chahiye to main yahin hoon.",
                               "ठीक है, कुछ बुक नहीं हुआ। डॉक्टर चाहिए तो मैं यहीं हूँ।"), kind="info")

        handler = getattr(self, f"_on_{step}", None)
        if handler is None:
            st.pop("booking", None)
            return None
        return handler(st, b, text, norm, v, ctx)

    # ---------------------------------------------------------------- steps
    def _on_day(self, st, b, text, norm, v, ctx):
        picked = self._picked(b, text)
        if picked == "others":                       # "show other doctors": leave booking, show more
            st.pop("booking", None)
            return None
        iso = picked if picked and picked.startswith("20") else self._day_from_text(b, norm)
        if not iso:
            return None if self._looks_like_new_ask(text) else self._ask_day(b, v, again=True)
        b["date"] = iso
        shifts = next((d["shifts"] for d in b["days"] if d["date"] == iso), [])
        if len(shifts) <= 1:
            b["shift"] = shifts[0] if shifts else None
            return self._after_when(st, b, v, ctx)
        b["step"] = "shift"
        b["options"] = {_norm(_shift_label(s, v)): f"shift:{i}" for i, s in enumerate(shifts)}
        b["options"].update({_norm(_shift_label(s)): f"shift:{i}" for i, s in enumerate(shifts)})
        q = v("Which time suits you?", "Kaun sa samay theek rahega?", "कौन-सा समय ठीक रहेगा?")
        return self._ask(q, "book_shift", [{"id": f"shift:{i}", "label": _shift_label(s, v)} for i, s in enumerate(shifts)]
                         + [self._not_now(b, v)])

    def _on_shift(self, st, b, text, norm, v, ctx):
        picked = self._picked(b, text)
        shifts = next((d["shifts"] for d in b["days"] if d["date"] == b.get("date")), [])
        idx = None
        if picked and picked.startswith("shift:"):
            idx = int(picked[6:])
        else:
            for i, s in enumerate(shifts):
                if (s.get("name") or "").lower() and (s.get("name") or "").lower() in norm:
                    idx = i
            if idx is None:
                for word, names in (("subah", "morning"), ("sham", "evening"), ("shaam", "evening"),
                                    ("dopahar", "afternoon"), ("सुबह", "morning"), ("शाम", "evening")):
                    if word in norm or word in text:
                        idx = next((i for i, s in enumerate(shifts) if names in (s.get("name") or "").lower()), None)
        if idx is None or idx >= len(shifts):
            if self._looks_like_new_ask(text):
                return None
            return self._ask(v("Please pick a time from these.", "Inmein se ek samay chunein.", "इनमें से एक समय चुनें।"),
                             "book_shift", [{"id": f"shift:{i}", "label": _shift_label(s, v)} for i, s in enumerate(shifts)]
                             + [self._not_now(b, v)])
        b["shift"] = shifts[idx]
        return self._after_when(st, b, v, ctx)

    def _after_when(self, st, b, v, ctx):
        """Day and time are set: who is it for? (signed in: saved profile and family as options)"""
        b["step"] = "who"
        token = ctx.get("patient_token")
        opts: list[dict] = []
        b["people"] = {}
        if token and self.enabled:
            try:
                me = self.client.profile(token, client_ip=ctx.get("client_ip"))
                if me.get("fullName"):
                    b["people"]["self"] = {"name": me["fullName"], "age": me.get("age"),
                                           "ageUnit": me.get("ageUnit"), "sex": me.get("sex")}
                for m in self.client.family(token, client_ip=ctx.get("client_ip"))[:4]:
                    key = f"member:{m.get('memberId')}"
                    b["people"][key] = {"name": m.get("fullName"), "age": m.get("age"), "ageUnit": m.get("ageUnit"),
                                        "sex": m.get("sex"), "relation": m.get("relation")}
            except BookingError:
                log.info("profile / family not available for this patient token")
        me = b["people"].get("self")
        opts.append({"id": "self", "label": v("Myself", "Mere liye", "मेरे लिए") + (f" ({me['name']})" if me else "")})
        for key, p in b["people"].items():
            if key.startswith("member:"):
                opts.append({"id": key, "label": p["name"] + (f" ({p['relation']})" if p.get("relation") else "")})
        opts.append({"id": "other", "label": v("Someone else", "Kisi aur ke liye", "किसी और के लिए")})
        b["options"] = {_norm(o["label"]): o["id"] for o in opts}
        for alias, val in (("myself", "self"), ("me", "self"), ("self", "self"), ("mere liye", "self"),
                           ("mera", "self"), ("khud", "self"), ("apne liye", "self"), ("मेरे लिए", "self"),
                           ("मेरे", "self"), ("family", "other"), ("someone else", "other"), ("kisi aur", "other"),
                           ("किसी और", "other"), ("mummy", "other"), ("papa", "other")):
            b["options"].setdefault(_norm(alias), val)
        return self._ask(v("Who is the appointment for?", "Appointment kiske liye hai?", "अपॉइंटमेंट किसके लिए है?"),
                         "book_who", opts)

    def _on_who(self, st, b, text, norm, v, ctx):
        picked = self._picked(b, text)
        if picked is None:
            for k, val in b.get("options", {}).items():
                if k and re.search(rf"(?<!\w){re.escape(k)}(?!\w)", norm):
                    picked = val
                    break
        if picked is None:
            if self._looks_like_new_ask(text):
                return None
            return self._after_when(st, b, v, ctx)
        person = dict(b.get("people", {}).get(picked) or {})
        b["for"] = "self" if picked == "self" else "family"
        b["patient"] = {k: person.get(k) for k in ("name", "age", "ageUnit", "sex", "relation") if person.get(k)}
        if b["patient"].get("age") and not b["patient"].get("ageUnit"):
            b["patient"]["ageUnit"] = "years"
        return self._next_detail(st, b, v, ctx)

    def _next_detail(self, st, b, v, ctx):
        p = b.setdefault("patient", {})
        self_ = b.get("for") == "self"
        if not p.get("name"):
            b["step"] = "name"
            return self._ask(v("What is your full name?" if self_ else "What is the patient's full name?",
                               "Aapka poora naam?" if self_ else "Patient ka poora naam?",
                               "आपका पूरा नाम?" if self_ else "मरीज़ का पूरा नाम?"), "book_name", [])
        if not p.get("age"):
            b["step"] = "age"
            return self._ask(v(f"How old is {p['name']}?", f"{p['name']} ki umar kitni hai?", f"{p['name']} की उम्र कितनी है?"),
                             "book_age", [])
        if not p.get("sex"):
            b["step"] = "sex"
            opts = [{"id": "Male", "label": v("Male", "Purush", "पुरुष")}, {"id": "Female", "label": v("Female", "Mahila", "महिला")},
                    {"id": "Other", "label": v("Other", "Anya", "अन्य")}]
            b["options"] = {_norm(o["label"]): o["id"] for o in opts}
            return self._ask(v("Male or female?", "Purush ya mahila?", "पुरुष या महिला?"), "book_sex", opts)
        return self._need_mobile(st, b, v, ctx)

    def _on_name(self, st, b, text, norm, v, ctx):
        name = _clean_name(text)
        if not name:
            return self._ask(v("Please type the name only, e.g. Rina Kumari.", "Sirf naam likhein, jaise Rina Kumari.",
                               "सिर्फ़ नाम लिखें, जैसे रीना कुमारी।"), "book_name", [])
        b["patient"]["name"] = name
        return self._next_detail(st, b, v, ctx)

    def _on_age(self, st, b, text, norm, v, ctx):
        age = _age_of(text)
        if not age:
            return self._ask(v("Please type the age in years, e.g. 32.", "Umar saalon mein likhein, jaise 32.",
                               "उम्र सालों में लिखें, जैसे 32।"), "book_age", [])
        b["patient"]["age"], b["patient"]["ageUnit"] = age
        return self._next_detail(st, b, v, ctx)

    def _on_sex(self, st, b, text, norm, v, ctx):
        picked = self._picked(b, text)
        sex = picked if picked in ("Male", "Female", "Other") else _sex_of(text)
        if not sex:
            return self._next_detail(st, b, v, ctx)
        b["patient"]["sex"] = sex
        return self._next_detail(st, b, v, ctx)

    def _need_mobile(self, st, b, v, ctx):
        """Signed in, or a channel that verified the number: no OTP (the patient is already known)."""
        mobile = token_mobile(ctx.get("patient_token")) if ctx.get("patient_token") else None
        if mobile:
            b["mobile"], b["verified"] = mobile, "session"
        elif ctx.get("verified_mobile"):
            b["mobile"], b["verified"] = ctx["verified_mobile"], "channel"
        elif b.get("verified") == "otp" and b.get("mobile"):
            pass
        else:
            b["step"] = "mobile"
            return self._ask(v("Your 10-digit mobile number? I'll send a code on WhatsApp to confirm it's you.",
                               "Aapka 10 digit mobile number? Confirm karne ke liye WhatsApp par code bhejunga.",
                               "आपका 10 अंकों का मोबाइल नंबर? पुष्टि के लिए WhatsApp पर कोड भेजूँगा।"), "book_mobile", [])
        return self._confirm(b, v)

    def _on_mobile(self, st, b, text, norm, v, ctx):
        m = _MOBILE.search(re.sub(r"[\s-]", "", text))
        if not m:
            return self._ask(v("Please type a 10-digit Indian mobile number, e.g. 98765 43210.",
                               "10 digit ka mobile number likhein, jaise 98765 43210.",
                               "10 अंकों का मोबाइल नंबर लिखें, जैसे 98765 43210।"), "book_mobile", [])
        return self._send_code(b, m.group(1), v, ctx)

    def _send_code(self, b, mobile, v, ctx):
        try:
            self.client.send_otp(mobile, client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            return self._ask(v(f"I couldn't send the code: {exc}", f"Code nahi bhej paaya: {exc}",
                               f"कोड नहीं भेज पाया: {exc}"), "book_mobile", [])
        b.update(step="otp", mobile=mobile, otp_tries=0)
        opts = [{"id": "resend", "label": v("Send code again", "Code dobara bhejo", "कोड दोबारा भेजो")},
                {"id": "change_number", "label": v("Change number", "Number badlo", "नंबर बदलो")}]
        b["options"] = {_norm(o["label"]): o["id"] for o in opts}
        return self._ask(v(f"I've sent a code on WhatsApp to {_mask(mobile)}. Please type it here.",
                           f"{_mask(mobile)} par WhatsApp se code bheja hai. Yahan likhein.",
                           f"{_mask(mobile)} पर WhatsApp से कोड भेजा है। यहाँ लिखें।"), "book_otp", opts)

    def _on_otp(self, st, b, text, norm, v, ctx):
        picked = self._picked(b, text)
        if picked == "resend":
            return self._send_code(b, b["mobile"], v, ctx)
        if picked == "change_number":
            b["step"] = "mobile"
            return self._need_mobile(st, b, v, ctx)
        m = _DIGITS.search(text.replace(" ", ""))
        if not m:
            return self._ask(v("Please type the code from WhatsApp (numbers only).", "WhatsApp wala code likhein (sirf number).",
                               "WhatsApp वाला कोड लिखें (सिर्फ़ अंक)।"), "book_otp", self._options_list(b))
        try:
            token, mobile = self.client.verify_otp(b["mobile"], m.group(1), client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            b["otp_tries"] = int(b.get("otp_tries") or 0) + 1
            if exc.unreachable:
                return self._ask(v("I couldn't check the code just now. Please try again.",
                                   "Abhi code check nahi ho paaya. Dobara try karein.",
                                   "अभी कोड जाँच नहीं हो पाया। दोबारा कोशिश करें।"), "book_otp", self._options_list(b))
            if b["otp_tries"] >= MAX_OTP_TRIES:
                return self._ask(v("That code didn't match again. Shall I send a new one?",
                                   "Code phir se match nahi hua. Naya code bhejun?",
                                   "कोड फिर से मेल नहीं खाया। नया कोड भेजूँ?"), "book_otp", self._options_list(b))
            return self._ask(v("That code didn't match. Please check WhatsApp and type it again.",
                               "Code match nahi hua. WhatsApp dekhkar dobara likhein.",
                               "कोड मेल नहीं खाया। WhatsApp देखकर दोबारा लिखें।"), "book_otp", self._options_list(b))
        b.update(mobile=mobile, verified="otp", token=token)
        out = self._confirm(b, v)
        out["patient_session"] = {"token": token, "mobile": mobile}     # the channel may sign the patient in
        return out

    def _confirm(self, b, v):
        b["step"] = "confirm"
        d, p = b["doctor"], b["patient"]
        when = _day_label(v, b["date"]) + (f", {_shift_label(b['shift'], v)}" if b.get("shift") else "")
        where = ", ".join(x for x in (d.get("hospital"), d.get("city")) if x)
        spec = d.get("specialty") or ""
        fee = f"₹{d['fee']}" if d.get("fee") is not None else None
        age = f"{p['age']}{'' if p.get('ageUnit') != 'months' else ' ' + v('months', 'mahine', 'महीने')}"
        who = f"{p['name']} ({age}, {_sex_word(v, p.get('sex'))})"
        msg = v(f"Shall I book {d['name']}" + (f" ({spec})" if spec else "") + f" at {where}, {when}, for {who}"
                + (f", fee {fee} paid at the hospital" if fee else "") + "? The hospital confirms the exact time.",
                f"{d['name']}" + (f" ({spec})" if spec else "") + f", {where}, {when}, {who} ke liye"
                + (f", fees {fee} hospital mein" if fee else "") + " - book karun? Sahi samay hospital confirm karega.",
                f"{d['name']}" + (f" ({spec})" if spec else "") + f", {where}, {when}, {who} के लिए"
                + (f", फ़ीस {fee} अस्पताल में" if fee else "") + " - बुक करूँ? सही समय अस्पताल कन्फ़र्म करेगा।")
        opts = [{"id": "yes", "label": v("Yes, book", "Haan, book karo", "हाँ, बुक करो")},
                {"id": "change", "label": v("Change", "Badlo", "बदलो")},
                {"id": "cancel", "label": v("Cancel", "Rehne do", "रहने दो")}]
        b["options"] = {_norm(o["label"]): o["id"] for o in opts}
        out = self._ask(msg, "book_confirm", opts)
        out["booking"] = self._summary(b, v)
        return out

    def _on_confirm(self, st, b, text, norm, v, ctx):
        picked = self._picked(b, text)
        if picked == "change" or (picked is None and _CHANGE.search(norm)):
            b.pop("date", None)
            b.pop("shift", None)
            return self._ask_day(b, v)
        if not (picked == "yes" or (picked is None and _YES.match(norm))):
            if self._looks_like_new_ask(text):
                return None
            return self._confirm(b, v)
        return self._book(st, b, v, ctx)

    def _book(self, st, b, v, ctx):
        d, p = b["doctor"], b["patient"]
        token = b.get("token") or (ctx.get("patient_token") if b.get("verified") == "session" else None)
        start = (b.get("shift") or {}).get("start")
        patient = {"fullName": p["name"], "mobile": b["mobile"], "age": p["age"], "ageUnit": p.get("ageUnit") or "years",
                   "sex": p.get("sex")}
        if b.get("for") == "family" and p.get("relation"):
            patient["guardianRelation"] = p["relation"]
        try:
            appointment_id = self.client.book(
                doctor_id=d["doctorId"], patient=patient, preferred_date=b["date"],
                preferred_time=f"{start}:00" if start and len(start) == 5 else start,
                reason=f"Booked with Vita ({_shift_label(b['shift'])})" if b.get("shift") else "Booked with Vita",
                token=token, client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            return self._failed(st, b, v, exc)
        summary = self._summary(b, v)
        summary["appointmentId"] = appointment_id
        st["last_booking"] = {"appointmentId": appointment_id, "doctorId": d["doctorId"], "at": int(time.time())}
        st.pop("booking", None)
        where = ", ".join(x for x in (d.get("hospital"), d.get("city")) if x)
        when = summary["when"]
        msg = v(f"Booked. {d['name']}, {where}, {when}. Booking ID: {appointment_id[:8].upper()}. "
                "The hospital will confirm the exact time on WhatsApp. Please bring old prescriptions, reports and an "
                "ID, and pay at the hospital counter.",
                f"Book ho gaya. {d['name']}, {where}, {when}. Booking ID: {appointment_id[:8].upper()}. "
                "Hospital WhatsApp par sahi samay confirm karega. Purane parche, reports aur ID saath layein; "
                "fees hospital counter par dein.",
                f"बुक हो गया। {d['name']}, {where}, {when}। बुकिंग आईडी: {appointment_id[:8].upper()}। "
                "अस्पताल WhatsApp पर सही समय कन्फ़र्म करेगा। पुराने पर्चे, रिपोर्ट और आईडी साथ लाएँ; "
                "फ़ीस अस्पताल काउंटर पर दें।")
        return {"kind": BOOKED_KIND, "message": msg, "clarify": None, "booking": summary}

    def _failed(self, st, b, v, exc: BookingError):
        """Say so, offer the next free days or other doctors; never retry silently."""
        if exc.unreachable:
            opts = [{"id": "yes", "label": v("Try again", "Dobara try karo", "दोबारा कोशिश करो")},
                    {"id": "cancel", "label": v("Not now", "Abhi nahi", "अभी नहीं")}]
            b["options"] = {_norm(o["label"]): o["id"] for o in opts}
            return self._ask(v("I couldn't reach the hospital system, so nothing was booked. If you get a WhatsApp "
                               "confirmation anyway, it went through. Try again?",
                               "Hospital system tak nahi pahunch paaya, isliye kuch book nahi hua. Agar WhatsApp par "
                               "confirmation aaye to booking ho gayi hai. Dobara try karun?",
                               "अस्पताल सिस्टम तक नहीं पहुँच पाया, इसलिए कुछ बुक नहीं हुआ। अगर WhatsApp पर "
                               "कन्फ़र्मेशन आए तो बुकिंग हो गई है। दोबारा कोशिश करूँ?"), "book_confirm", opts)
        others = [d for d in b.get("days", []) if d["date"] != b.get("date")][:2]
        b["step"] = "day"
        opts = [{"id": d["date"], "label": _day_label(v, d["date"])} for d in others]
        opts.append({"id": "others", "label": v("Other doctors", "Doosre doctor", "दूसरे डॉक्टर")})
        b["options"] = {_norm(o["label"]): o["id"] for o in opts}
        return self._ask(v(f"It couldn't be booked: {exc}. Would another day work, or shall I show other doctors?",
                           f"Book nahi ho paaya: {exc}. Koi aur din chalega, ya doosre doctor dikhaun?",
                           f"बुक नहीं हो पाया: {exc}। कोई और दिन चलेगा, या दूसरे डॉक्टर दिखाऊँ?"), "book_day", opts)

    # ---------------------------------------------------------------- small pieces
    def _ask_day(self, b, v, again=False):
        b["step"] = "day"
        opts = [{"id": d["date"], "label": _day_label(v, d["date"])} for d in b.get("days", [])]
        opts.append(self._not_now(b, v))
        b["options"] = {_norm(o["label"]): o["id"] for o in opts}
        b["options"].update({_norm(o["label"].split(" (")[0]): o["id"] for o in opts if " (" in o["label"]})
        q = (v("Please pick a day from these.", "Inmein se ek din chunein.", "इनमें से एक दिन चुनें।") if again else
             v("Which day suits you?", "Kaun sa din theek rahega?", "कौन-सा दिन ठीक रहेगा?"))
        return self._ask(q, "book_day", opts)

    @staticmethod
    def _not_now(b, v):
        return {"id": "cancel", "label": v("Not now", "Abhi nahi", "अभी नहीं")}

    @staticmethod
    def _options_list(b) -> list[dict]:
        return [{"id": val, "label": k} for k, val in (b.get("options") or {}).items()][:3]

    @staticmethod
    def _picked(b, text) -> str | None:
        """A tapped chip (its label as text) or a typed answer equal to an option."""
        opts = b.get("options") or {}
        n = _norm(text)
        if n in opts:
            return opts[n]
        stripped = n.rstrip(" .!")
        return opts.get(stripped)

    @staticmethod
    def _day_from_text(b, norm: str) -> str | None:
        from ..doctor_intent import _find_when
        days = [d["date"] for d in b.get("days", [])]
        when = _find_when(norm)
        today = _today()
        target = None
        if when.startswith("+"):
            target = (today + timedelta(days=int(when[1:]))).isoformat()
        elif when.startswith("dow:"):
            target = (today + timedelta(days=(int(when[4:]) - today.weekday()) % 7)).isoformat()
        else:
            m = re.search(r"\b(\d{1,2})\b", norm)
            if m:
                target = next((d for d in days if int(d[8:]) == int(m.group(1))), None)
        return target if target in days else None

    @staticmethod
    def _looks_like_new_ask(text: str) -> bool:
        from ..doctor_intent import parse
        ask = parse(text)
        return bool(ask.direct or ask.symptomish or ask.move in ("human",))

    @staticmethod
    def _ask(message: str, kind: str, options: list[dict]) -> dict:
        return {"kind": FOLLOW_UP, "message": message,
                "clarify": {"kind": kind, "question": message, "options": options}}

    @staticmethod
    def _say(message: str, kind: str = "info") -> dict:
        return {"kind": kind, "message": message, "clarify": None}

    @staticmethod
    def _summary(b, v) -> dict:
        d, p = b["doctor"], b.get("patient") or {}
        when = _day_label(v, b["date"]) + (f", {_shift_label(b['shift'], v)}" if b.get("shift") else "")
        lat, lng = d.get("lat"), d.get("lng")
        return {"doctorId": d["doctorId"], "doctorName": d["name"], "specialty": d.get("specialty"),
                "hospital": d.get("hospital"), "city": d.get("city"), "date": b["date"],
                "shift": b.get("shift"), "when": when, "fee": d.get("fee"),
                "patientName": p.get("name"), "patientAge": p.get("age"), "patientAgeUnit": p.get("ageUnit"),
                "patientSex": p.get("sex"), "forWhom": b.get("for"), "mobile": _mask(b["mobile"]) if b.get("mobile") else None,
                "mapUrl": f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}" if lat is not None and lng is not None else None}


def _doctor_brief(d: dict) -> dict:
    h = d.get("hospital") or {}
    return {"doctorId": d["doctorId"], "name": d.get("name") or "the doctor",
            "specialty": re.sub(r"\s*\(.*?\)", "", d.get("specialtyName") or "") or None,
            "hospital": h.get("name"), "city": h.get("city"), "lat": h.get("lat"), "lng": h.get("lng"),
            "fee": _fee(d)}
