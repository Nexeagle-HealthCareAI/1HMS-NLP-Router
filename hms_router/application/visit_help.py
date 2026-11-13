"""After booking (vision Phase 4): Vita stays with the patient until they are inside the clinic.

    "rasta batao" / "how do I get there" / "रास्ता बताओ"        -> directions: map link, landmark, entrance, OPD floor
    "pahunch gayi" / "I've reached" / "मैं पहुँच गया"            -> arrival: where to register, the booking ID to quote, check in from
                                                                   the phone (when 1HMS allows it today), OPD, pay at the counter
    "check in" (with the patient's location)                    -> a queue token (1HMS POST /public/tokens checks they are there)
    "mera number kab aayega" / "how long" / "token status"      -> the token, who is being seen now, the expected wait

Which visit: the booking made in this chat, else the one the website says the patient is looking at (`visit_appointment_id`), else a
signed-in patient's next visit today or later (1HMS appointments/mine). The reply is `kind: "visit"` with `visit` (1HMS's visit page
data) and `visit_step` (directions | arrival | checked_in | status). Never answered before the emergency check.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone

from ..booking_client import BookingError, HmsPublicClient
from ..textnorm import intent_text
from ..voice import Voice

log = logging.getLogger("hms_router.visit")

VISIT = "visit"
_ARRIVED = re.compile(r"\b(i('| a)?ve reached|i (have )?(reached|arrived)|i am (here|at the hospital)|i'?m (here|at the hospital)|reached|arrived|"
                      r"pahunch (gaya|gayi|gai|gaye|gya|gyi|chuka|chuki|chuke)|pahuch (gaya|gayi|gaye|gya|gyi)|"
                      r"aa (gaya|gayi|gaye|gya|gyi) (hoon|hun|hu|hain)|hospital (aa|pahunch) (gaya|gayi|gaye)|"
                      r"hospital (mein|me|par) (hoon|hun|hu|hain))\b")
_DIRECTIONS = re.compile(r"\b(direction|directions|route|rasta|raasta|kaise (pahunchu|pahuchu|aaun|aau|jaun|jau|jaaun)|how (do i|to) get|"
                         r"where is (the )?(hospital|clinic|opd)|navigation|navigate|map|location (bhejo|batao|do)|address)\b")
_STATUS = re.compile(r"\b(token|my turn|meri baari|mera number|kitna (time|samay|der)|kab (aayega|aaega|ayega|bulayenge|bulaenge)|how long|"
                     r"wait|queue|line)\b")
_CHECK_IN = re.compile(r"\b(check ?in|checkin|check-in|chek ?in|chekin)\b")
_DEV = {"पहुँच": "pahunch", "पहुंच": "pahunch", "गया": "gaya", "गई": "gayi", "गयी": "gayi", "गए": "gaye", "रास्ता": "rasta",
        "टोकन": "token", "नंबर": "number", "चेक": "check", "इन": "in", "मैं": "main", "हूँ": "hoon", "हूं": "hoon"}


def _norm(text: str) -> str:
    t = intent_text(text or "")
    for k, val in _DEV.items():
        t = t.replace(k, val)
    return t.lower()


def wants(text: str) -> str | None:
    """'arrival' | 'directions' | 'status' | 'checkin' | None."""
    n = _norm(text)
    if _CHECK_IN.search(n):
        return "checkin"
    if _ARRIVED.search(n):
        return "arrival"
    if _DIRECTIONS.search(n):
        return "directions"
    if _STATUS.search(n):
        return "status"
    return None


def _today() -> date:
    return (datetime.now(timezone.utc) + timedelta(minutes=330)).date()


class VisitHelp:
    def __init__(self, client: HmsPublicClient | None):
        self.client = client

    @property
    def enabled(self) -> bool:
        return bool(self.client and self.client.enabled)

    # ------------------------------------------------------------------ which visit
    def appointment_id(self, st: dict, ctx: dict) -> str | None:
        if (st.get("last_booking") or {}).get("appointmentId"):
            return st["last_booking"]["appointmentId"]
        if ctx.get("visit_appointment_id"):
            return ctx["visit_appointment_id"]
        token = ctx.get("patient_token")
        if token:
            try:
                mine = self.client.my_appointments(token, client_ip=ctx.get("client_ip"))
            except BookingError:
                return None
            today = _today().isoformat()
            live = [a for a in mine if (a.get("statusCode") or "").upper() not in ("CANCELLED", "COMPLETED")
                    and str(a.get("apptDate") or "")[:10] >= today]
            live.sort(key=lambda a: (str(a.get("apptDate") or "")[:10], str(a.get("startAt") or "")))
            return live[0]["appointmentId"] if live else None
        return None

    # ------------------------------------------------------------------ one turn
    def turn(self, what: str, st: dict, ctx: dict, v: Voice) -> dict:
        """Answer one visit question. Returns {"kind", "message", "clarify", "visit"?, "visit_step"?}."""
        appt = self.appointment_id(st, ctx)
        if not appt:
            return {"kind": "info", "clarify": None, "message": v(
                "I can't see a booking in this chat. Open 'My appointments' on Doctor Dekho for your visit's directions and check-in, "
                "or tell me which doctor you are visiting.",
                "Is chat mein mujhe koi booking nahi dikh rahi. Doctor Dekho par 'Meri appointments' kholein, wahan rasta aur check-in hai. "
                "Ya batayein kis doctor se milne aaye hain.",
                "इस चैट में मुझे कोई बुकिंग नहीं दिख रही। Doctor Dekho पर 'मेरी अपॉइंटमेंट' खोलें, वहाँ रास्ता और चेक-इन है। "
                "या बताएँ किस डॉक्टर से मिलने आए हैं।")}
        st["visit"] = {"appointmentId": appt}
        try:
            if what == "checkin":
                if ctx.get("lat") is None or ctx.get("lng") is None:
                    st["visit"]["awaiting"] = "checkin"
                    return {"kind": "follow_up", "message": v(
                        "Share your location so I can check you in.", "Check in ke liye apni location share karein.",
                        "चेक-इन के लिए अपनी लोकेशन शेयर करें।"),
                        "clarify": {"kind": "location", "question": None, "options": []}}
                return self._check_in(appt, st, ctx, v)
            visit = self.client.visit(appt, client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            log.warning("visit lookup failed: %s", exc)
            return {"kind": "info", "clarify": None, "degraded": ["visit_lookup"], "message": v(
                "I couldn't open your booking just now. Please show your booking ID at the hospital's registration counter.",
                "Abhi aapki booking khul nahi paayi. Hospital ke registration counter par apna booking ID dikhayein.",
                "अभी आपकी बुकिंग खुल नहीं पाई। अस्पताल के रजिस्ट्रेशन काउंटर पर अपना बुकिंग आईडी दिखाएँ।")}
        if what == "directions":
            return self._directions(visit, v)
        if what == "status":
            return self._status(visit, v)
        return self._arrival(visit, st, v)

    def location_followup(self, st: dict, ctx: dict, v: Voice) -> dict | None:
        """The location the patient shared after "check in" (or after arriving)."""
        pending = (st.get("visit") or {}).get("awaiting")
        if pending != "checkin" or ctx.get("lat") is None:
            return None
        try:
            return self._check_in(st["visit"]["appointmentId"], st, ctx, v)
        except BookingError as exc:
            return self._check_in_refused(st, v, str(exc))

    # ------------------------------------------------------------------ answers
    @staticmethod
    def _out(visit: dict, step: str, message: str, clarify: dict | None = None) -> dict:
        return {"kind": VISIT, "message": message, "clarify": clarify, "visit": visit, "visit_step": step}

    def _directions(self, visit: dict, v: Voice) -> dict:
        g = visit.get("guide") or {}
        where = ", ".join(x for x in (visit.get("hospitalName"), visit.get("address"), visit.get("city")) if x)
        lines = [v(f"Here's the way to {where}. Tap 'Directions' to open the map.",
                   f"{where} ka rasta yeh raha. Map ke liye 'Directions' dabayein.",
                   f"{where} का रास्ता यह रहा। मैप के लिए 'Directions' दबाएँ।")]
        for key, en, hg, hi in (("landmark", "Landmark", "Pehchaan", "पहचान"), ("entranceNote", "Entrance", "Gate", "गेट"),
                                ("opdLocation", "OPD", "OPD", "OPD"), ("parkingNote", "Parking", "Parking", "पार्किंग")):
            if g.get(key):
                lines.append(f"{v(en, hg, hi)}: {g[key]}")
        return self._out(visit, "directions", "\n".join(lines))

    def _arrival(self, visit: dict, st: dict, v: Voice) -> dict:
        if visit.get("cancelled"):
            return self._out(visit, "arrival", v("This booking was cancelled. Please ask at the registration counter.",
                                                 "Yeh booking cancel ho chuki hai. Registration counter par poochein.",
                                                 "यह बुकिंग रद्द हो चुकी है। रजिस्ट्रेशन काउंटर पर पूछें।"))
        g = visit.get("guide") or {}
        ref = visit.get("reference") or ""
        steps = [v("Welcome! Here's what to do:", "Swagat hai! Ab yeh karein:", "स्वागत है! अब यह करें:")]
        n = 1
        if g.get("entranceNote"):
            steps.append(f"{n}. " + v(f"Entrance: {g['entranceNote']}.", f"Gate: {g['entranceNote']}.", f"गेट: {g['entranceNote']}।"))
            n += 1
        token = visit.get("token")
        clarify = None
        if token and token.get("tokenNo"):
            steps.append(f"{n}. " + self._token_line(token, v))
            n += 1
        else:
            desk = g.get("registrationDesk") or v("the registration counter", "registration counter", "रजिस्ट्रेशन काउंटर")
            steps.append(f"{n}. " + v(f"Go to {desk} and give your booking ID {ref}. They will give you a token.",
                                      f"{desk} par jaayein aur booking ID {ref} batayein. Wahan token milega.",
                                      f"{desk} पर जाएँ और बुकिंग आईडी {ref} बताएँ। वहाँ टोकन मिलेगा।"))
            n += 1
            if visit.get("canCheckIn"):
                steps.append(v("Or check in right here: tap 'Check in now' and share your location; I'll get your token.",
                               "Ya yahin check in karein: 'Abhi check in karo' dabakar location share karein, main token le aaungi.",
                               "या यहीं चेक-इन करें: 'अभी चेक-इन करो' दबाकर लोकेशन शेयर करें, मैं टोकन ले आऊँगी।"))
                st.setdefault("visit", {})["awaiting"] = "checkin"
                clarify = {"kind": "visit_checkin", "question": None,
                           "options": [{"id": "visit:checkin", "label": v("Check in now", "Abhi check in karo", "अभी चेक-इन करो")}]}
        if g.get("opdLocation"):
            steps.append(f"{n}. " + v(f"Then go to the OPD: {g['opdLocation']}.", f"Phir OPD jaayein: {g['opdLocation']}.",
                                      f"फिर OPD जाएँ: {g['opdLocation']}।"))
            n += 1
        fee = visit.get("fee")
        pay = (v(f"Pay the fee (Rs {fee:g}) at the counter.", f"Fees (Rs {fee:g}) counter par dein.", f"फ़ीस (Rs {fee:g}) काउंटर पर दें।")
               if isinstance(fee, (int, float)) else v("Pay the fee at the counter.", "Fees counter par dein.", "फ़ीस काउंटर पर दें।"))
        steps.append(f"{n}. " + pay + (f" {g['paymentNote']}" if g.get("paymentNote") else ""))
        if g.get("extraNote"):
            steps.append(g["extraNote"])
        return self._out(visit, "arrival", "\n".join(steps), clarify)

    @staticmethod
    def _token_line(token: dict, v: Voice) -> str:
        no, serving, wait = token.get("tokenNo"), token.get("currentServingTokenNo"), token.get("estimatedWaitMinutes")
        line = v(f"Your token is {no}.", f"Aapka token {no} hai.", f"आपका टोकन {no} है।")
        if serving:
            line += " " + v(f"Now seeing {serving}.", f"Abhi {serving} chal raha hai.", f"अभी {serving} चल रहा है।")
        if wait:
            line += " " + v(f"About {wait} min to go.", f"Lagbhag {wait} minute.", f"लगभग {wait} मिनट।")
        return line

    def _status(self, visit: dict, v: Voice) -> dict:
        token = visit.get("token")
        if token and token.get("tokenNo"):
            return self._out(visit, "status", self._token_line(token, v))
        if visit.get("canCheckIn"):
            return self._out(visit, "status", v("You don't have a token yet. Check in when you reach and I'll tell you the wait.",
                                                "Abhi token nahi hai. Pahunch kar check in karein, main wait time bata dungi.",
                                                "अभी टोकन नहीं है। पहुँचकर चेक-इन करें, मैं इंतज़ार का समय बता दूँगी।"),
                             {"kind": "visit_checkin", "question": None,
                              "options": [{"id": "visit:checkin", "label": v("Check in now", "Abhi check in karo", "अभी चेक-इन करो")}]})
        return self._out(visit, "status", v("You'll get a token at the registration counter on the day of your visit.",
                                            "Visit ke din registration counter par token milega.",
                                            "विज़िट के दिन रजिस्ट्रेशन काउंटर पर टोकन मिलेगा।"))

    def _check_in(self, appt: str, st: dict, ctx: dict, v: Voice) -> dict:
        try:
            res = self.client.check_in(appt, ctx["lat"], ctx["lng"], client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            return self._check_in_refused(st, v, str(exc))
        (st.get("visit") or {}).pop("awaiting", None)
        visit = self.client.visit(appt, client_ip=ctx.get("client_ip"))
        token = visit.get("token") or {"tokenNo": res.get("tokenNo")}
        msg = v("You're checked in. ", "Check in ho gaya. ", "चेक-इन हो गया। ") + self._token_line(token, v)
        if (visit.get("guide") or {}).get("opdLocation"):
            msg += "\n" + v(f"Wait near the OPD: {visit['guide']['opdLocation']}.", f"OPD ke paas intezaar karein: {visit['guide']['opdLocation']}.",
                            f"OPD के पास इंतज़ार करें: {visit['guide']['opdLocation']}।")
        return self._out(visit, "checked_in", msg)

    @staticmethod
    def _check_in_refused(st: dict, v: Voice, reason: str) -> dict:
        (st.get("visit") or {}).pop("awaiting", None)
        far = "don't appear to be at the hospital" in reason
        return {"kind": "info", "clarify": None, "message": (
            v("It looks like you're not at the hospital yet. Try again when you reach, or check in at the registration counter.",
              "Lagta hai aap abhi hospital nahi pahunche. Pahunch kar dobara try karein, ya registration counter par check in karein.",
              "लगता है आप अभी अस्पताल नहीं पहुँचे। पहुँचकर दोबारा कोशिश करें, या रजिस्ट्रेशन काउंटर पर चेक-इन करें।") if far else
            v(f"I couldn't check you in ({reason}). Please check in at the registration counter.",
              f"Check in nahi ho paaya ({reason}). Registration counter par check in karein.",
              f"चेक-इन नहीं हो पाया ({reason})। रजिस्ट्रेशन काउंटर पर चेक-इन करें।"))}
