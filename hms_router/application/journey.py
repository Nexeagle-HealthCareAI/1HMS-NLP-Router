"""Vita "takes you there": from home to the clinic.

After a booking Vita asks how the patient is coming and when they will leave (suggesting a time from the distance), and whether the
hospital may see their arrival time. 1HMS keeps that (PUT /public/appointments/{id}/journey) and sends a "time to leave" WhatsApp.

    "nikal raha hoon" / "on my way"       -> on the way: live navigation on Doctor Dekho, Uber / Ola for a cab, Google Maps transit
                                             for bus or train; 1HMS gets "on the way" (+ ETA when the location is known)
    "kitni der lagegi" / "how far"         -> the ETA (from the navigation's last update, or worked out from the location)
    "late ho jaunga" / "15 min late"       -> tells the hospital (with the patient's OK), with how late
    "pahunch gayi" (visit_help)            -> arrived; then the existing arrival steps and check-in

Nothing here stores or sends the patient's location: 1HMS gets only the travel mode, the leave time, the ETA and the distance left.
Answers are in the patient's language; reply kind "visit" with visit_step "navigate" | "eta" | "late" carries `journey` for the
screens (mode, links, etaMinutes). Planning replies are kind "follow_up" with clarify kinds journey_mode | journey_leave | journey_share.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlencode

from ..booking_client import BookingError, HmsPublicClient
from ..textnorm import intent_text
from ..voice import Voice

log = logging.getLogger("hms_router.journey")

MODES = ("drive", "auto_cab", "transit", "walk")
# road distance ~ 1.4 x straight line in Indian cities; average door-to-door speeds and the wait before moving
_SPEED_KMH = {"drive": 22.0, "auto_cab": 20.0, "transit": 14.0, "walk": 4.5}
_WAIT_MIN = {"drive": 5, "auto_cab": 10, "transit": 15, "walk": 0}
ROAD_FACTOR = 1.4
REGISTRATION_BUFFER_MIN = 15          # reach this early to register before the time
DEFAULT_TRAVEL_MIN = 45               # no location to work from

_MODE_WORDS = (
    ("auto_cab", re.compile(r"\b(auto|cab|taxi|uber|ola|rapido|e ?rickshaw|toto|rickshaw)\b")),
    ("transit", re.compile(r"\b(bus|train|metro|local|tempo|public transport|shared)\b")),
    ("walk", re.compile(r"\b(paidal|pedal|walk|walking|on foot|foot|chal ?kar|pairon se)\b")),
    ("drive", re.compile(r"\b(car|gaadi|gadi|bike|scooty|scooter|motorcycle|apni gaadi|drive|khud chala)\b")),
)
LEAVING = re.compile(r"\b(nikal (raha|rahi|rahe|gaya|gayi|gaye|gya|gyi|chuka|chuki|padi|pada|pade)|nikalne (wala|wali|wale) (hoon|hun|hu|hain)"
                     r"|nikal rha|nikal rhi|chal (diya|di|diye|pada|padi|pade)|ghar se nikal\w*|rawana|leaving (now|home)?|left home"
                     r"|on (my|the) way|start(ing)? navigation|i('| a)?m leaving|heading (out|to the hospital))\b")
ETA = re.compile(r"\b(kitni der|kitna time (lagega|aur)|kitna samay|kab tak pahunch\w*|kitna door|kitni door|how far|how long (to|until|till) "
                 r"(reach|get there|arrive)|eta|arrival time|kitne minute)\b")
LATE = re.compile(r"\b(late (ho|hoon|hun|hu|hounga|hoonga|hunga|hungi|houngi|honge|hoga|hogi|ho jaunga|ho jaungi|ho jayenge|ho raha|ho rahi"
                  r"|ho rahe|ho gaya|ho gayi|ho gaye)|(min|mins|minute|minutes) late|running late|i('| a)?ll be late|be late|der (ho jayegi|ho jaegi"
                  r"|ho jaayegi|ho rahi|hogi)|traffic (mein|me|hai|bahut)|jam (mein|me|hai)|phans (gaya|gayi|gaye|gye)|delay (ho|hai))\b")
_MINUTES = re.compile(r"\b(\d{1,3})\s*(min|mins|minute|minutes|minat|mint)\b")
_HOURS = re.compile(r"\b(\d{1,2}|ek|do|aadha|aadhe|half an?)\s*(ghanta|ghante|ghanton|hour|hours|hr|hrs)\b")
_NO = re.compile(r"^(nahi|nahin|na|no|nope|skip|baad mein|baad me|later|abhi nahi|pata nahi|nahi pata|rehne do|chhodo)\b")
_YES = re.compile(r"^(haan|haa|ha|han|yes|ji|ji haan|haanji|ok|okay|theek hai|thik hai|bata do|bata dijiye|share karo|kar do)\b")


def _now_ist() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=330)


def mode_of(norm: str) -> str | None:
    return next((m for m, rx in _MODE_WORDS if rx.search(norm or "")), None)


def km_between(lat1, lng1, lat2, lng2) -> float | None:
    try:
        p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
        dp, dl = p2 - p1, math.radians(float(lng2) - float(lng1))
    except (TypeError, ValueError):
        return None
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def travel_minutes(mode: str | None, km: float | None) -> int | None:
    """Door-to-door estimate from the straight-line distance: road ~1.4x, a city speed per mode, plus the wait for a cab or bus."""
    if km is None:
        return None
    m = mode if mode in MODES else "drive"
    return int(math.ceil(km * ROAD_FACTOR / _SPEED_KMH[m] * 60 + _WAIT_MIN[m]))


def parse_leave(norm: str, start: datetime | None, now: datetime | None = None) -> datetime | None:
    """'8 baje', '8:30' (or '8 30' after normalising), 'saade 8', 'sava 9', 'paune 9', 'subah 8', 'shaam 5', '8.30 pm', 'abhi',
    'aadhe ghante mein', '20 minute mein' -> a wall-clock time (IST). A bare hour is read as the latest one before the appointment
    ('8' -> 08:00 for a 10:00 visit, 20:00 for a 21:00 one)."""
    n = (norm or "").lower().strip()
    now = now or _now_ist()
    day = (start or now).replace(hour=0, minute=0, second=0, microsecond=0)
    if re.match(r"^(abhi|now|turant|right now)\b", n):
        return now.replace(second=0, microsecond=0)
    later = re.search(r"\b(mein|me|baad|after|later|in)\b", n)
    m = _MINUTES.search(n)
    if m and later:
        return (now + timedelta(minutes=int(m.group(1)))).replace(second=0, microsecond=0)
    h = _HOURS.search(n)
    if h and later:
        word = h.group(1)
        mins = {"ek": 60, "do": 120, "aadha": 30, "aadhe": 30}.get(word) or (
            30 if word.startswith("half") else int(word) * 60 if word.isdigit() else 60)
        return (now + timedelta(minutes=mins)).replace(second=0, microsecond=0)
    t = re.search(r"\b(\d{1,2})(?:[:. ](\d{2}))?\s*(am|pm|baje|bje|o ?clock)?\b", n)
    if not t or (t.group(2) is None and t.group(3) is None and not re.search(r"\b(saade|sade|sava|sawa|paune|subah|shaam|sham|raat|dopahar|morning|evening|afternoon)\b", n)
                 and not re.fullmatch(r"\d{1,2}", n)):
        return None
    hour, minute, suffix = int(t.group(1)), int(t.group(2) or 0), t.group(3)
    if re.search(r"\b(saade|sade)\b", n):
        minute = 30
    elif re.search(r"\b(sava|sawa)\b", n):
        minute = 15
    elif re.search(r"\bpaune\b", n):
        hour, minute = hour - 1, 45
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    if suffix == "pm" or re.search(r"\b(shaam|sham|raat|dopahar|evening|afternoon|night)\b", n):
        hour = hour + 12 if hour < 12 else hour
    elif suffix == "am" or re.search(r"\b(subah|morning|savere|sawere)\b", n):
        hour = hour % 12
    elif hour < 12 and start is not None:
        before = [c for c in (day.replace(hour=hour, minute=minute), day.replace(hour=hour + 12, minute=minute)) if c < start]
        return before[-1] if before else day.replace(hour=hour, minute=minute)
    return day.replace(hour=hour, minute=minute)


def hhmm(t: datetime) -> str:
    return t.strftime("%H:%M")


def say_time(v: Voice, t: datetime) -> str:
    h12 = t.hour % 12 or 12
    clock = f"{h12}:{t.minute:02d}" if t.minute else f"{h12}"
    part = ("subah" if t.hour < 12 else "dopahar" if t.hour < 16 else "shaam" if t.hour < 20 else "raat")
    part_hi = {"subah": "सुबह", "dopahar": "दोपहर", "shaam": "शाम", "raat": "रात"}[part]
    ampm = "am" if t.hour < 12 else "pm"
    return v(f"{clock} {ampm}", f"{part} {clock} baje", f"{part_hi} {clock} बजे")


def mode_words(v: Voice, mode: str | None) -> str:
    return {"drive": v("by car or bike", "car / bike se", "कार / बाइक से"),
            "auto_cab": v("by auto or cab", "auto / cab se", "ऑटो / कैब से"),
            "transit": v("by bus or train", "bus / train se", "बस / ट्रेन से"),
            "walk": v("on foot", "paidal", "पैदल")}.get(mode or "", "")


def mode_options(v: Voice) -> list[dict]:
    return [{"id": "journey:drive", "label": v("Car / bike", "Car / bike", "कार / बाइक")},
            {"id": "journey:auto_cab", "label": v("Auto / cab", "Auto / cab", "ऑटो / कैब")},
            {"id": "journey:transit", "label": v("Bus / train", "Bus / train", "बस / ट्रेन")},
            {"id": "journey:walk", "label": v("On foot", "Paidal", "पैदल")}]


def share_options(v: Voice) -> list[dict]:
    return [{"id": "share:yes", "label": v("Yes, tell them", "Haan, bata do", "हाँ, बता दो")},
            {"id": "share:no", "label": v("No", "Nahi", "नहीं")}]


def links(mode: str | None, visit: dict, ola_token: str | None = None) -> list[dict]:
    """Buttons for the screens: live navigation on Doctor Dekho always; Uber / Ola for a cab; Google Maps transit for bus or train."""
    lat, lng = visit.get("latitude"), visit.get("longitude")
    name = visit.get("hospitalName") or "Hospital"
    address = ", ".join(x for x in (visit.get("hospitalName"), visit.get("address"), visit.get("city")) if x)
    dest = f"{lat},{lng}" if lat is not None and lng is not None else quote(address)
    out = [{"id": "navigate", "kind": "navigate", "url": visit.get("mapUrl") or f"https://www.google.com/maps/dir/?api=1&destination={dest}"}]
    if mode == "auto_cab" and lat is not None and lng is not None:
        out.append({"id": "uber", "kind": "cab", "url": "https://m.uber.com/ul/?" + urlencode(
            {"action": "setPickup", "pickup": "my_location", "dropoff[latitude]": lat, "dropoff[longitude]": lng,
             "dropoff[nickname]": name, "dropoff[formatted_address]": address})})
        ola = {"drop_lat": lat, "drop_lng": lng, "drop_name": name, "dsw": "yes"}
        if ola_token:
            ola["utm_source"] = ola_token
        out.append({"id": "ola", "kind": "cab", "url": "https://book.olacabs.com/?" + urlencode(ola)})
    if mode == "transit":
        out.append({"id": "transit", "kind": "transit",
                    "url": f"https://www.google.com/maps/dir/?api=1&destination={dest}&travelmode=transit"})
    return out


class JourneyHelp:
    def __init__(self, client: HmsPublicClient | None, ola_token: str | None = None):
        self.client = client
        self.ola_token = ola_token

    @property
    def enabled(self) -> bool:
        return bool(self.client and self.client.enabled)

    # ------------------------------------------------------------------ after booking: plan
    def start(self, st: dict, booking: dict, v: Voice) -> dict:
        """Right after "Booked": ask how they are coming. Returns {"message", "clarify"} to add to the booked reply."""
        start = None
        shift = booking.get("shift") or {}
        if booking.get("date") and shift.get("start"):
            try:
                start = datetime.fromisoformat(f"{booking['date']}T{shift['start'][:5]}")
            except ValueError:
                start = None
        st["journey"] = {"appointmentId": booking.get("appointmentId"), "step": "mode", "date": booking.get("date"),
                         "start": start.isoformat() if start else None, "lat": booking.get("lat"), "lng": booking.get("lng"),
                         "hospital": booking.get("hospital")}
        q = v("How will you get there? I'll tell you when to leave and show you the way.",
              "Aap hospital kaise aayenge? Main bataungi kab nikalna hai aur rasta bhi dikhaungi.",
              "आप अस्पताल कैसे आएँगे? मैं बताऊँगी कब निकलना है और रास्ता भी दिखाऊँगी।")
        return {"message": q, "clarify": {"kind": "journey_mode", "question": q, "options": mode_options(v)}}

    def plan_turn(self, text: str, st: dict, ctx: dict, v: Voice) -> dict | None:
        """One answer while planning. None: not an answer (the planning ends and the message goes the usual way)."""
        j = st.get("journey") or {}
        norm = intent_text(text or "")
        step = j.get("step")
        chips = {"mode": mode_options(v), "leave": j.get("leave_options") or [], "share": share_options(v)}.get(step) or []
        tapped = next((o["id"] for o in chips if intent_text(o["label"]) == norm or o["id"] == text), None)
        if tapped:
            text = tapped                                    # the web sends a chip's label as the message
        if _NO.match(norm) and step in ("mode", "leave"):
            j["step"] = None
            return self._say(v("No problem. When you set off, tell me \"leaving now\" and I'll show you the way.",
                               "Koi baat nahi. Nikalte waqt mujhe \"nikal raha hoon\" likhein, main rasta dikha dungi.",
                               "कोई बात नहीं। निकलते समय मुझे \"निकल रहा हूँ\" लिखें, मैं रास्ता दिखा दूँगी।"))
        if step == "mode":
            mode = (text or "").split(":", 1)[1] if (text or "").startswith("journey:") else mode_of(norm)
            if mode not in MODES:
                return None
            j["mode"] = mode
            return self._ask_leave(j, ctx, v)
        if step == "leave":
            start = datetime.fromisoformat(j["start"]) if j.get("start") else None
            picked = (text or "").split(":", 1)[1] if (text or "").startswith("leave:") else None
            leave = datetime.fromisoformat(picked) if picked else parse_leave(norm, start)
            if leave is None:
                if j.get("leave_tries"):
                    return None
                j["leave_tries"] = 1
                return self._ask("journey_leave", v("Please write a time, like \"8:30\" or \"8 baje\".",
                                                    "Samay likhein, jaise \"8:30\" ya \"8 baje\".",
                                                    "समय लिखें, जैसे \"8:30\" या \"8 बजे\"।"), j.get("leave_options") or [])
            if start is not None and leave >= start:
                return self._ask("journey_leave", v(f"That's after your appointment time ({say_time(v, start)}). When will you leave?",
                                                    f"Yeh appointment ({say_time(v, start)}) ke baad hai. Kab nikalenge?",
                                                    f"यह अपॉइंटमेंट ({say_time(v, start)}) के बाद है। कब निकलेंगे?"),
                                 j.get("leave_options") or [])
            j["leaveAt"] = leave.isoformat()
            j["step"] = "share"
            q = v("Shall I let the hospital know when you're on the way and when you'll arrive? (Only the arrival time, never your location.)",
                  "Kya main hospital ko bata doon jab aap raste mein hon aur kab tak pahunchenge? (Sirf pahunchne ka samay, aapki location nahi.)",
                  "क्या मैं अस्पताल को बता दूँ जब आप रास्ते में हों और कब तक पहुँचेंगे? (सिर्फ़ पहुँचने का समय, आपकी लोकेशन नहीं।)")
            return self._ask("journey_share", q, share_options(v))
        if step == "share":
            if text == "share:yes" or _YES.match(norm):
                j["share"] = True
            elif text == "share:no" or _NO.match(norm):
                j["share"] = False
            else:
                return None
            j["step"] = None
            return self._save_plan(j, ctx, v)
        return None

    def _ask_leave(self, j: dict, ctx: dict, v: Voice) -> dict:
        j["step"] = "leave"
        start = datetime.fromisoformat(j["start"]) if j.get("start") else None
        km = km_between(ctx.get("lat"), ctx.get("lng"), j.get("lat"), j.get("lng")) if ctx.get("located") else None
        mins = travel_minutes(j.get("mode"), km)
        opts: list[dict] = []
        if start is not None:
            suggest = start - timedelta(minutes=(mins or DEFAULT_TRAVEL_MIN) + REGISTRATION_BUFFER_MIN)
            suggest = suggest.replace(minute=(suggest.minute // 5) * 5)
            earlier = suggest - timedelta(minutes=15)
            opts = [{"id": f"leave:{suggest.isoformat()}", "label": say_time(v, suggest)},
                    {"id": f"leave:{earlier.isoformat()}", "label": say_time(v, earlier)}]
            if mins is not None:
                why = v(f"It's about {mins} min {mode_words(v, j.get('mode'))} from where you are, and it's good to reach 15 min early.",
                        f"Aapki jagah se {mode_words(v, j.get('mode'))} lagbhag {mins} minute ka rasta hai, aur 15 minute pehle pahunchna achha hai.",
                        f"आपकी जगह से {mode_words(v, j.get('mode'))} लगभग {mins} मिनट का रास्ता है, और 15 मिनट पहले पहुँचना अच्छा है।")
            else:
                why = v("Share your location for a better estimate.", "Location share karein to aur sahi samay bataungi.",
                        "लोकेशन शेयर करें तो और सही समय बताऊँगी।")
            q = v(f"When will you leave? I suggest {say_time(v, suggest)}. {why} I'll remind you on WhatsApp at that time.",
                  f"Kab nikalenge? Mera sujhaav: {say_time(v, suggest)}. {why} Us samay main WhatsApp par yaad dila dungi.",
                  f"कब निकलेंगे? मेरा सुझाव: {say_time(v, suggest)}। {why} उस समय मैं WhatsApp पर याद दिला दूँगी।")
        else:
            q = v("When will you leave? Write a time, like \"8:30\". I'll remind you on WhatsApp then.",
                  "Kab nikalenge? Samay likhein, jaise \"8:30\". Us samay main WhatsApp par yaad dila dungi.",
                  "कब निकलेंगे? समय लिखें, जैसे \"8:30\"। उस समय मैं WhatsApp पर याद दिला दूँगी।")
        j["leave_options"] = opts
        return self._ask("journey_leave", q, opts)

    def _save_plan(self, j: dict, ctx: dict, v: Voice) -> dict:
        leave = datetime.fromisoformat(j["leaveAt"])
        saved = False
        if self.enabled and j.get("appointmentId"):
            try:
                self.client.save_journey(j["appointmentId"], travel_mode=j.get("mode"), leave_at=hhmm(leave) if not j.get("date") or
                                         leave.date().isoformat() == j["date"] else leave.strftime("%Y-%m-%dT%H:%M"),
                                         share=bool(j.get("share")), client_ip=ctx.get("client_ip"))
                saved = True
            except BookingError as exc:
                log.warning("journey not saved: %s", exc)
        when = say_time(v, leave)
        msg = v(f"Done: leaving {when} {mode_words(v, j.get('mode'))}.", f"Theek hai: {when} {mode_words(v, j.get('mode'))} nikalna.",
                f"ठीक है: {when} {mode_words(v, j.get('mode'))} निकलना।")
        if saved:
            msg += " " + v("I'll remind you on WhatsApp then.", "Us samay WhatsApp par yaad dila dungi.", "उस समय WhatsApp पर याद दिला दूँगी।")
        msg += " " + v("When you set off, tell me \"leaving now\" and I'll show you the way.",
                       "Nikalte waqt mujhe \"nikal raha hoon\" likhein, main rasta dikha dungi.",
                       "निकलते समय मुझे \"निकल रहा हूँ\" लिखें, मैं रास्ता दिखा दूँगी।")
        if j.get("share"):
            msg += " " + v("The hospital will see when you're on the way.", "Raste mein hone par hospital ko pata rahega.",
                           "रास्ते में होने पर अस्पताल को पता रहेगा।")
        return self._say(msg)

    # ------------------------------------------------------------------ on the day
    def leaving(self, appt: str, visit: dict, st: dict, ctx: dict, v: Voice) -> dict:
        j = (visit.get("journey") or {})
        mode = j.get("travelMode") or (st.get("journey") or {}).get("mode")
        km = (km_between(ctx.get("lat"), ctx.get("lng"), visit.get("latitude"), visit.get("longitude"))
              if ctx.get("located") else None)
        mins = travel_minutes(mode, km)
        try:
            self.client.journey_progress(appt, status="on_the_way", eta_minutes=mins,
                                         distance_km=round(km * ROAD_FACTOR, 1) if km is not None else None,
                                         client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            log.warning("journey progress not sent: %s", exc)
        where = visit.get("hospitalName") or v("the hospital", "hospital", "अस्पताल")
        lines = [v(f"Safe travels to {where}!", f"Shubh yatra! {where} ka rasta:", f"शुभ यात्रा! {where} का रास्ता:")]
        if mode == "auto_cab":
            lines.append(v("Book a cab with the buttons below; the clinic is already filled in.",
                           "Neeche ke button se cab book karein; hospital ka pata pehle se bhara hai.",
                           "नीचे के बटन से कैब बुक करें; अस्पताल का पता पहले से भरा है।"))
        elif mode == "transit":
            lines.append(v("Tap 'Bus / train route' for the buses and trains.", "Bus aur train ke liye 'Bus / train ka rasta' dabayein.",
                           "बस और ट्रेन के लिए 'बस / ट्रेन का रास्ता' दबाएँ।"))
        else:
            lines.append(v("Tap 'Start navigation' for turn-by-turn directions.", "Mod-dar-mod rasta ke liye 'Rasta shuru karein' dabayein.",
                           "मोड़-दर-मोड़ रास्ते के लिए 'रास्ता शुरू करें' दबाएँ।"))
        if mins is not None:
            lines.append(v(f"About {mins} min.", f"Lagbhag {mins} minute.", f"लगभग {mins} मिनट।"))
        g = visit.get("guide") or {}
        if g.get("landmark"):
            lines.append(v(f"Landmark: {g['landmark']}.", f"Pehchaan: {g['landmark']}.", f"पहचान: {g['landmark']}।"))
        lines.append(v("Running late? Just tell me. When you reach, say \"I've reached\".",
                       "Der ho rahi ho to bata dein. Pahunch kar \"pahunch gaye\" likhein.",
                       "देर हो रही हो तो बता दें। पहुँचकर \"पहुँच गया\" लिखें।"))
        st.setdefault("journey", {}).update({"appointmentId": appt, "onTheWay": True, "mode": mode})
        return {"kind": "visit", "visit_step": "navigate", "visit": visit, "message": "\n".join(lines), "clarify": None,
                "journey": {"mode": mode, "etaMinutes": mins, "links": links(mode, visit, self.ola_token),
                            "appointmentId": appt, "share": bool(j.get("shareWithHospital"))}}

    def eta(self, appt: str, visit: dict, st: dict, ctx: dict, v: Voice) -> dict:
        j = visit.get("journey") or {}
        mins = j.get("etaMinutes")
        km = j.get("distanceKm")
        if mins is None and ctx.get("located"):
            d = km_between(ctx.get("lat"), ctx.get("lng"), visit.get("latitude"), visit.get("longitude"))
            mins, km = travel_minutes(j.get("travelMode"), d), (round(d * ROAD_FACTOR, 1) if d is not None else None)
        if mins is None:
            return {"kind": "follow_up", "visit_step": "eta", "visit": visit, "clarify": {"kind": "location", "question": None, "options": []},
                    "message": v("Share your location and I'll tell you how long it will take.",
                                 "Location share karein, main bata dungi kitna time lagega.",
                                 "लोकेशन शेयर करें, मैं बता दूँगी कितना समय लगेगा।")}
        msg = v(f"About {mins} min to {visit.get('hospitalName') or 'the hospital'}", f"{visit.get('hospitalName') or 'Hospital'} tak lagbhag {mins} minute",
                f"{visit.get('hospitalName') or 'अस्पताल'} तक लगभग {mins} मिनट")
        if km:
            msg += v(f" ({km:g} km left).", f" ({km:g} km baaki).", f" ({km:g} किमी बाकी)।")
        else:
            msg += v(".", ".", "।")
        if j.get("lateByMinutes"):
            msg += " " + v(f"That's about {j['lateByMinutes']} min after your time; the hospital knows.",
                           f"Yeh aapke samay se lagbhag {j['lateByMinutes']} minute baad hai; hospital ko pata hai.",
                           f"यह आपके समय से लगभग {j['lateByMinutes']} मिनट बाद है; अस्पताल को पता है।") if j.get("shareWithHospital") else ""
        return {"kind": "visit", "visit_step": "eta", "visit": visit, "message": msg, "clarify": None,
                "journey": {"mode": j.get("travelMode"), "etaMinutes": mins, "links": links(j.get("travelMode"), visit, self.ola_token),
                            "appointmentId": appt, "share": bool(j.get("shareWithHospital"))}}

    def late(self, appt: str, visit: dict, st: dict, ctx: dict, v: Voice, text: str, consent: bool | None = None) -> dict:
        norm = intent_text(text or "")
        m = _MINUTES.search(norm)
        h = _HOURS.search(norm)
        late_by = int(m.group(1)) if m else (60 if h else None)
        j = visit.get("journey") or {}
        shared = bool(j.get("shareWithHospital"))
        if not shared and consent is None:
            st.setdefault("journey", {}).update({"appointmentId": appt, "awaiting": "late_consent", "lateBy": late_by})
            q = v("Shall I tell the hospital you're running late? (Only how late, not where you are.)",
                  "Kya main hospital ko bata doon ki aap late honge? (Sirf kitni der, aapki location nahi.)",
                  "क्या मैं अस्पताल को बता दूँ कि आप देर से पहुँचेंगे? (सिर्फ़ कितनी देर, आपकी लोकेशन नहीं।)")
            return {"kind": "follow_up", "visit_step": "late", "visit": visit, "message": q,
                    "clarify": {"kind": "journey_late", "question": q,
                                "options": [{"id": "late:yes", "label": v("Yes, tell them", "Haan, bata do", "हाँ, बता दो")},
                                            {"id": "late:no", "label": v("No", "Nahi", "नहीं")}]}}
        if consent is False:
            contact = visit.get("contact")
            return {"kind": "info", "visit_step": "late", "visit": visit, "clarify": None, "message": v(
                "Okay, I won't tell them." + (f" You can call the hospital on {contact}." if contact else ""),
                "Theek hai, main nahi bataungi." + (f" Aap hospital ko {contact} par call kar sakte hain." if contact else ""),
                "ठीक है, मैं नहीं बताऊँगी।" + (f" आप अस्पताल को {contact} पर कॉल कर सकते हैं।" if contact else ""))}
        late_by = late_by if late_by is not None else (st.get("journey") or {}).get("lateBy") or 15
        try:
            if not shared:
                self.client.save_journey(appt, share=True, client_ip=ctx.get("client_ip"))
            self.client.journey_progress(appt, status="late", late_by_minutes=late_by, client_ip=ctx.get("client_ip"))
            told = True
        except BookingError as exc:
            log.warning("late not sent: %s", exc)
            told = False
        (st.get("journey") or {}).pop("awaiting", None)
        contact = visit.get("contact")
        if told:
            msg = v(f"I've told the hospital you'll be about {late_by} min late. No rush; travel safely.",
                    f"Maine hospital ko bata diya ki aap lagbhag {late_by} minute late honge. Aaram se, dhyan se aaiye.",
                    f"मैंने अस्पताल को बता दिया कि आप लगभग {late_by} मिनट देर से पहुँचेंगे। आराम से, ध्यान से आइए।")
        else:
            msg = v("I couldn't reach the hospital system." + (f" Please call them on {contact}." if contact else ""),
                    "Hospital system tak nahi pahunch paayi." + (f" Kripya unhe {contact} par call karein." if contact else ""),
                    "अस्पताल सिस्टम तक नहीं पहुँच पाई।" + (f" कृपया उन्हें {contact} पर कॉल करें।" if contact else ""))
        return {"kind": "info", "visit_step": "late", "visit": visit, "message": msg, "clarify": None}

    def arrived(self, appt: str, ctx: dict) -> None:
        try:
            self.client.journey_progress(appt, status="arrived", client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            log.info("arrival not sent: %s", exc)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ask(kind: str, q: str, options: list[dict]) -> dict:
        return {"kind": "follow_up", "message": q, "clarify": {"kind": kind, "question": q, "options": options}}

    @staticmethod
    def _say(msg: str) -> dict:
        return {"kind": "info", "message": msg, "clarify": None}


def wants(text: str) -> str | None:
    """'leaving' | 'eta' | 'late' | None."""
    n = intent_text(text or "")
    if LEAVING.search(n):
        return "leaving"
    if ETA.search(n):
        return "eta"
    if LATE.search(n):
        return "late"
    return None
