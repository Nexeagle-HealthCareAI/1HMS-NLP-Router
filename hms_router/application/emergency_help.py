"""Emergency takeover (vision Phase 3): when the emergency check fires, show the nearest emergency-ready places and alert them.

The emergency reply already says the number first (108 / 112, in the patient's language). This adds, under it:

    emergency: {
      "places": [{settingId, name, kind, phone, alternatePhone, distanceKm, address, landmark, city, lat, lng, mapUrl}],
      "eventId": "...",            # 1HMS alert event: call taps and a shared number are logged against it
      "alerted": 3,                # places that got "possible emergency nearby" (WhatsApp / 1HMS)
      "repeated": false,           # this chat already alerted in the last 30 minutes: nothing sent again
      "needsLocation": false,      # no point known: ask for the location (112 still shown)
      "numberShared": false,
      "canShareNumber": true       # the channel knows the patient's number (WhatsApp, signed in): one tap shares it
    }

Places come from 1HMS (POST /public/emergency-alerts: verified opt-ins and places a CMS admin added, open now, nearest
first, at most 5). The router never decides that a case is not an emergency here, never books, and the 112 line stays even
when places are found. If 1HMS cannot be reached the reply still goes out with the numbers (degraded: emergency_places).

The patient's point goes to 1HMS for this one lookup and is not stored there (only distances). The patient's number reaches
the hospitals only when they agree ("Share my number"): on WhatsApp with the sender's verified number, on the website with
the number they type (the site sends it itself).
"""
from __future__ import annotations

import logging
import re
import time

from ..booking_client import BookingError, HmsPublicClient
from ..voice import Voice, symptom_words

log = logging.getLogger("hms_router.emergency")

PENDING_S = 30 * 60          # a location shared within 30 minutes of the emergency still gets the nearest places
SHOWN_IN_TEXT = 3            # places written into the message itself (all of them are in `emergency.places`)

_CAPABILITY = [
    ("maternity", re.compile(r"pregnan|labou?r|deliver|garbh|prasav|bleeding in pregnancy")),
    ("paediatric", re.compile(r"\b(child|baby|infant|newborn|kid|bachch?[ae]|shishu)\b")),
    ("cardiac", re.compile(r"chest|heart|cardiac|palpitation|seene|dil\b")),
    ("trauma", re.compile(r"accident|injur|fracture|burn|fall|head (injury|trauma)|road")),
]
SHARE_WORDS = re.compile(r"^\s*(share (my )?number|number share|mera number (share|bhejo|de do)|"
                         r"(मेरा )?नंबर (शेयर करो|शेयर|भेजो)|haan,? number (bhejo|share))", re.I)


def capability_for(signs: list[str], text: str) -> str | None:
    hay = " ".join(signs + [text]).lower()
    for cap, rx in _CAPABILITY:
        if rx.search(hay):
            return cap
    return None


def _map_url(lat, lng) -> str | None:
    return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}" if lat is not None and lng is not None else None


def _place(p: dict) -> dict:
    return {"settingId": p.get("settingId"), "name": p.get("name"), "kind": p.get("kind") or "HOSPITAL",
            "phone": p.get("phone"), "alternatePhone": p.get("alternatePhone"), "distanceKm": p.get("distanceKm"),
            "address": p.get("address"), "landmark": p.get("landmark"), "city": p.get("city"),
            "lat": p.get("latitude"), "lng": p.get("longitude"), "is24x7": p.get("is24x7"),
            "capabilities": p.get("capabilities") or [], "mapUrl": _map_url(p.get("latitude"), p.get("longitude"))}


class EmergencyHelp:
    def __init__(self, client: HmsPublicClient | None):
        self.client = client

    @property
    def enabled(self) -> bool:
        return bool(self.client and self.client.enabled)

    # ------------------------------------------------------------------ the emergency reply
    def attach(self, out: dict, st: dict, ctx: dict, signs: list[str], text: str, v: Voice) -> None:
        """Adds the nearest places to an emergency reply (in place) and remembers the emergency in `st` so a location shared
        next, or "share my number", still acts on it. Never raises."""
        if not self.enabled:
            return
        reason = symptom_words(Voice("english"), signs) or "possible emergency"
        em = {"at": int(time.time()), "reason": reason, "capability": capability_for(signs, text)}
        st["emergency"] = em
        if ctx.get("lat") is None or ctx.get("lng") is None:
            out["emergency"] = {"places": [], "needsLocation": True, "canShareNumber": False}
            out["message"] += "\n" + v("Share your location and I'll show the nearest emergency hospitals and tell them you may be coming.",
                                       "Apni location share karein, main sabse nazdeeki emergency hospital dikhaungi aur unhe bata dungi.",
                                       "अपनी लोकेशन शेयर करें, मैं सबसे नज़दीकी इमरजेंसी अस्पताल दिखाऊँगी और उन्हें बता दूँगी।")
            return
        self._alert(out, em, ctx, v, session_ref=out.get("session_id"))

    def location_followup(self, out: dict, st: dict, ctx: dict, v: Voice) -> bool:
        """The patient shared a location soon after an emergency without places: alert and list now. True when handled."""
        em = st.get("emergency")
        if not (self.enabled and em and not em.get("eventId") and ctx.get("lat") is not None and ctx.get("lng") is not None
                and time.time() - em.get("at", 0) <= PENDING_S):
            return False
        self._alert(out, em, ctx, v, session_ref=out.get("session_id"))
        return True

    def _alert(self, out: dict, em: dict, ctx: dict, v: Voice, session_ref: str | None) -> None:
        try:
            res = self.client.emergency_alert(lat=ctx["lat"], lng=ctx["lng"], reason=em["reason"], capability=em.get("capability"),
                                              channel=ctx.get("channel") or "web", session_ref=session_ref,
                                              client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            log.warning("emergency places unavailable: %s", exc)
            out["degraded"] = list(out.get("degraded") or []) + ["emergency_places"]
            out["emergency"] = {"places": [], "needsLocation": False, "canShareNumber": False}
            return
        places = [_place(p) for p in res.get("places") or []]
        em["eventId"] = res.get("eventId")
        em["places"] = [p["settingId"] for p in places]
        can_share = bool(em["eventId"] and (ctx.get("verified_mobile") or ctx.get("token_mobile")))
        out["emergency"] = {"places": places, "eventId": res.get("eventId"), "alerted": res.get("alerted") or 0,
                            "repeated": bool(res.get("repeated")), "needsLocation": False,
                            "numberShared": bool(em.get("numberShared")), "canShareNumber": can_share}
        if not places:
            out["message"] += "\n" + v("I couldn't find an emergency hospital on Doctor Dekho near you. Please call 112 now.",
                                       "Aapke paas Doctor Dekho par koi emergency hospital nahi mila. Abhi 112 par call karein.",
                                       "आपके पास Doctor Dekho पर कोई इमरजेंसी अस्पताल नहीं मिला। अभी 112 पर कॉल करें।")
            return
        lines = [v("Nearest emergency help:", "Sabse nazdeeki emergency madad:", "सबसे नज़दीकी इमरजेंसी मदद:")]
        for i, p in enumerate(places[:SHOWN_IN_TEXT], 1):
            km = p.get("distanceKm")
            lines.append(f"{i}. {p['name']}" + (f" ({km:g} km)" if isinstance(km, (int, float)) else "") + f" - {p['phone']}")
        if res.get("alerted"):
            lines.append(v("I've told them a patient may be coming.", "Maine unhe bata diya hai ki ek patient aa sakta hai.",
                           "मैंने उन्हें बता दिया है कि एक मरीज़ आ सकता है।"))
        if can_share and not em.get("numberShared"):
            lines.append(v("Want them to call you? Tap 'Share my number'.", "Kya woh aapko call karein? 'Mera number bhejo' dabayein.",
                           "क्या वे आपको कॉल करें? 'मेरा नंबर भेजो' दबाएँ।"))
            out["clarify"] = {"kind": "emergency_share", "question": None,
                              "options": [{"id": "emergency:share", "label": v("Share my number", "Mera number bhejo", "मेरा नंबर भेजो")}]}
        out["message"] += "\n" + "\n".join(lines)

    # ------------------------------------------------------------------ "share my number"
    def wants_share(self, text: str, st: dict) -> bool:
        em = st.get("emergency") or {}
        return bool(em.get("eventId")) and bool(SHARE_WORDS.search(text or ""))

    def share(self, st: dict, ctx: dict, v: Voice) -> dict:
        em = st["emergency"]
        mobile = ctx.get("verified_mobile") or ctx.get("token_mobile")
        if not mobile:
            return {"kind": "info", "clarify": None,
                    "message": v("Please call the hospital directly; I don't have your number here.",
                                 "Kripya hospital ko seedha call karein; mere paas aapka number nahi hai.",
                                 "कृपया अस्पताल को सीधे कॉल करें; मेरे पास आपका नंबर नहीं है।")}
        try:
            self.client.share_emergency_number(em["eventId"], mobile, client_ip=ctx.get("client_ip"))
        except BookingError as exc:
            log.warning("emergency number not shared: %s", exc)
            return {"kind": "info", "clarify": None,
                    "message": v("I couldn't send your number. Please call the hospital or 112 now.",
                                 "Aapka number nahi bhej paayi. Abhi hospital ya 112 par call karein.",
                                 "आपका नंबर नहीं भेज पाई। अभी अस्पताल या 112 पर कॉल करें।")}
        em["numberShared"] = True
        return {"kind": "info", "clarify": None,
                "message": v("Done. The hospitals have your number and may call you. Keep your phone free, and call 112 if it gets worse.",
                             "Ho gaya. Hospitals ke paas aapka number hai, woh call kar sakte hain. Phone free rakhein; halat bigde to 112 par call karein.",
                             "हो गया। अस्पतालों के पास आपका नंबर है, वे कॉल कर सकते हैं। फ़ोन खाली रखें; हालत बिगड़े तो 112 पर कॉल करें।")}
