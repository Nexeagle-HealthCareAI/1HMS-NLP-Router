"""Vita doctor search inside /chat (Vita Chat requirements, Part 2, VITA-2 to VITA-13).

One message in, one reply out. Order of work, every turn:

  1  select_doctor_id?      -> the doctor must be one shown in this session (else 422) -> doctor_selected
  2  parse the ask          doctor_intent.parse: symptom / specialty / name / degree / distance / refine
  3  SAFETY FIRST           any ask that would skip triage is checked with the shared emergency detector;
                            an emergency goes to the agent (emergency card, never a doctor list) - VITA-9
  4  pending location?      the previous turn asked "where are you?" -> lat/lng or "use my city" completes it
  4b conversation moves     "talk to a person" -> handoff; "Dr X ka number do" -> never a private number;
                            "bas dekh raha tha" / "hello" -> a short answer, no push to book
  5  refine / more          "inme se kam fees wala", "aur doctor dikhao" -> last filters + the change
  6  direct ask             specialty / name / degree / distance -> 1HMS search now, kind=doctor_list
  7  otherwise              the triage agent (follow-up questions); on a confident recommendation the
                            PRIMARY specialty's doctors are attached (SECONDARY, then General Physician
                            as fallback, doctors.fallbackUsed=true)

Doctors are exactly what 1HMS returned, in its order (VITA-12); only an explicit "kam fees" sorts them.
1HMS failing never fails /chat: the reply says so and degraded contains "doctor_lookup" (VITA-10).
Patient text is never sent to 1HMS: only the extracted filters are (NFR-2.6). Location is used for the
search in hand and not stored.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from ..doctor_intent import DOCTOR_WORDS, DoctorAsk, parse
from ..doctors import (FALLBACK_SPECIALIST, NEARBY_DEFAULT_KM, WIDE_RADIUS_KM, DoctorDirectory, DoctorLookupError,
                       specialty_key)
from ..severity import EMERGENCY, emergency_check
from ..textnorm import preprocess

log = logging.getLogger("hms_router.doctor_chat")

DOCTOR_LIST, DOCTOR_SELECTED, FOLLOW_UP = "doctor_list", "doctor_selected", "follow_up"
INFO, HANDOFF = "info", "handoff"
SOURCE = "1hms-mcp"
STATE_TTL_S = 1800
MAX_SHOWN = 60

SPECIALTY_NAMES = {
    "cardiology": "Cardiologist", "dermatology": "Dermatologist", "orthopaedics": "Orthopaedic",
    "gynaecology": "Gynaecologist", "dentistry": "Dentist", "ent": "ENT Specialist",
    "ophthalmology": "Eye Specialist", "neurology": "Neurologist", "neurosurgery": "Neurosurgeon",
    "psychiatry": "Psychiatrist", "urology": "Urologist", "gastroenterology": "Gastroenterologist",
    "endocrinology": "Endocrinologist", "pulmonology": "Pulmonologist", "nephrology": "Nephrologist",
    "oncology": "Oncologist", "rheumatology": "Rheumatologist", "physiotherapy": "Physiotherapist",
    "paediatrics": "Paediatrician", "geriatrics": "Geriatrician", "general-medicine": "General Physician",
    "general-surgery": "General Surgeon", "plastic-surgery": "Plastic Surgeon", "vascular-surgery": "Vascular Surgeon",
    "cardiothoracic-surgery": "Cardiothoracic Surgeon", "emergency-medicine": "Emergency Medicine",
    "sports-medicine": "Sports Medicine",
}
# Plain words first, the specialty in brackets (vision: "bone doctor (orthopaedics)", not just "orthopaedics")
PLAIN = {  # key: (English, Hinglish)
    "cardiology": ("heart", "dil ke"), "dermatology": ("skin", "skin ke"), "orthopaedics": ("bone", "haddi ke"),
    "gynaecology": ("women's health", "mahila rog"), "dentistry": ("teeth", "daant ke"),
    "ent": ("ear, nose and throat", "kaan-naak-gala ke"), "ophthalmology": ("eye", "aankh ke"),
    "neurology": ("brain and nerve", "dimag aur nas ke"), "neurosurgery": ("brain surgery", "dimag ki surgery ke"),
    "psychiatry": ("mental health", "mann ke"), "urology": ("urinary", "peshab ke"),
    "gastroenterology": ("stomach", "pet ke"), "endocrinology": ("diabetes and hormone", "sugar aur hormone ke"),
    "pulmonology": ("chest and lung", "chhati aur phephde ke"), "nephrology": ("kidney", "kidney ke"),
    "oncology": ("cancer", "cancer ke"), "rheumatology": ("joint pain", "jodon ke dard ke"),
    "physiotherapy": ("physiotherapy", "physiotherapy ke"), "paediatrics": ("child", "bachchon ke"),
    "geriatrics": ("elderly care", "buzurgon ke"), "general-medicine": ("general", "general"),
    "general-surgery": ("general surgery", "general surgery ke"),
}
LOCATION_OPTIONS = [{"id": "share_location", "label": "Share my location"},
                    {"id": "type_area", "label": "Type area or pincode"},
                    {"id": "use_city", "label": "Use my city"}]
HUMAN_OPTION = {"id": "human", "label": "Talk to a person"}
ASK_KIND_OPTIONS = [{"id": "symptom", "label": "Symptom batayein"}, {"id": "specialty", "label": "Specialty chunein"},
                    {"id": "name", "label": "Doctor ka naam"}, {"id": "degree", "label": "Degree se"},
                    {"id": "distance", "label": "Paas ke doctor"}]

_vocab_lock = threading.Lock()
_vocab = None


def medical_terms(text: str) -> list[str]:
    """MedicalVocabulary.terms_in, loaded once (about 1 s) on first use."""
    global _vocab
    if _vocab is None:
        with _vocab_lock:
            if _vocab is None:
                from ..rag.vocabulary import MedicalVocabulary
                _vocab = MedicalVocabulary.from_data()
    return _vocab.terms_in(text)


class SelectionError(ValueError):
    """select_doctor_id was not shown in this session -> 422."""


def _hinglish(lang: str | None) -> bool:
    return lang in ("hinglish", "hindi")


def _fee(d: dict) -> float:
    fee = d.get("discountedFee") if d.get("discountedFee") is not None else d.get("consultFee")
    return float(fee) if fee is not None else float("inf")


class DoctorChat:
    def __init__(self, agent, directory: DoctorDirectory, redis, *, terms=medical_terms):
        self.agent, self.directory, self.r, self.terms = agent, directory, redis, terms

    # ================================================================== session state
    @staticmethod
    def _key(sid: str) -> str:
        return f"vita:doctor:{sid}"

    def load(self, sid: str | None) -> dict:
        if not sid:
            return {}
        try:
            raw = self.r.get(self._key(sid))
            return json.loads(raw) if raw else {}
        except Exception:
            log.exception("doctor session state unreadable")
            return {}

    def _save(self, sid: str, st: dict) -> None:
        try:
            self.r.set(self._key(sid), json.dumps(st), ex=STATE_TTL_S)
        except Exception:
            log.exception("doctor session state not saved")

    def session_view(self, sid: str) -> dict | None:
        """What GET /session/{id} shows about doctor search (VITA-8)."""
        st = self.load(sid)
        if not st:
            return None
        sel = st.get("selected") or {}
        return {"selectedDoctorId": sel.get("doctorId"), "selectedDoctorName": sel.get("name"),
                "lastFilters": st.get("last"), "doctorsShown": len(st.get("shown") or {})}

    # ================================================================== entry
    def handle(self, text: str, session_id: str | None, *, city: str | None = None, lat: float | None = None,
               lng: float | None = None, hospital_id: str | None = None,
               select_doctor_id: str | None = None, page: int = 1) -> dict:
        ctx = {"city": city, "lat": lat, "lng": lng, "hospital_id": hospital_id,
               "located": lat is not None and lng is not None}
        if select_doctor_id:
            return self._select(text, session_id, select_doctor_id, ctx)

        st = self.load(session_id)
        norm = preprocess(text)
        names = self.directory.names(city) if _wants_names(norm) else None
        ask = parse(text, names=names, medical_terms=self.terms)

        pending = st.get("pending")
        which_id = self._which_pick(norm, st)
        skips_triage = (ask.direct or ask.refine or bool(pending) or which_id is not None or ask.move is not None
                        or page > 1 or _bare_doctor(norm, ask))
        if skips_triage and self._emergency(text):                                       # step 3
            if session_id and (pending or st.get("which")):
                st.pop("pending", None)
                st.pop("which", None)
                self._save(session_id, st)
            return self._agent_turn(text, session_id, ask, ctx, emergency_checked=True)

        if ask.move == "human":                                                            # conversation moves
            return self._handoff(text, session_id, "asked")
        if ask.move == "private_number":
            return self._private_number(text, session_id, ask, st, ctx)
        if ask.move in ("just_looking", "greeting") and not self._mid_triage(session_id):
            return self._small_talk(text, session_id, ask.move)
        if which_id is not None:                                                           # "salt lake wali"
            return self._select(text, session_id, which_id, ctx)

        if pending and not ask.symptomish and not (ask.specialty_key or ask.name_query or ask.degree):   # step 4
            filters = dict(pending)
            filters["locationAsked"] = True
            return self._search_turn(text, session_id, filters, ctx, st)

        last = st.get("last")
        if last and (page > 1 or (ask.more and not (ask.specialty_key or ask.name_query or ask.degree))):  # step 5
            nxt = page if page > 1 else int(st.get("page") or 1) + 1
            return self._search_turn(text, session_id, dict(last), ctx, st, page=nxt)
        if last and ask.refine and not ask.symptomish:
            return self._search_turn(text, session_id, _merge(last, ask), ctx, st)

        if ask.direct:                                                                     # step 6
            return self._search_turn(text, session_id, _filters(ask), ctx, st)
        if _bare_doctor(norm, ask):
            return self._ask_kind(text, session_id, st)
        return self._agent_turn(text, session_id, ask, ctx)                                 # step 7

    # ================================================================== safety
    def _emergency(self, text: str) -> bool:
        texts = [preprocess(text)]
        try:
            shared = self.agent._shared_emergency(text, texts)
            if shared is not None:
                return bool(shared.flag)
            hits = emergency_check({}, texts, self.agent.classifier.rules)
            return any(h["flag"] == EMERGENCY for h in hits)
        except Exception:                                  # cannot tell -> let the agent (and its safety net) decide
            log.exception("emergency pre-check failed; sending the message to the agent")
            return True

    # ================================================================== symptom path (agent)
    def _agent_turn(self, text, session_id, ask: DoctorAsk, ctx, emergency_checked=False) -> dict:
        reply = self.agent.handle(text, session_id)
        out = reply.to_dict()
        card = out.get("card") or {}
        if out.get("kind") != "recommendation" or card.get("low_confidence"):
            return out                                         # follow-ups and emergencies never carry doctors
        outputs = card.get("outputs") or []
        primary = next((o for o in outputs if o.get("role") == "PRIMARY"), None)
        if primary is None:
            return out
        fallbacks = [o["specialist"] for o in outputs if o.get("role") == "SECONDARY" and o.get("specialist")]
        fallbacks.append(FALLBACK_SPECIALIST)
        t0 = time.perf_counter()
        try:
            found = self.directory.for_specialist(
                primary["specialist"], city=ctx["city"], lat=ctx["lat"], lng=ctx["lng"], fallbacks=fallbacks,
                hospital_id=ctx["hospital_id"], raise_errors=True)
        except DoctorLookupError:
            out["degraded"] = list(out.get("degraded") or []) + ["doctor_lookup"]
            return out
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if found is None:
            return out
        modes = ["symptom"] + (["distance"] if ctx["located"] else [])
        filters = {"specialtyKey": specialty_key(found["specialist"]), "specialist": found["specialist"],
                   "modes": modes}
        out["doctors"] = self._block(found["doctors"], found["total"], modes, filters, ctx,
                                     specialty_id=found["specialtyId"], specialist=found["specialist"],
                                     fallback_used=found["fallbackUsed"], ms=ms)
        sid = out["session_id"]
        st = self.load(sid)
        st["last"] = filters
        st["language"], st["turn"] = out.get("language"), out.get("turn")
        st.pop("which", None)
        self._remember(st, found["doctors"])
        self._save(sid, st)
        return out

    # ================================================================== direct search
    def _search_turn(self, text, session_id, filters: dict, ctx, st: dict, page: int = 1) -> dict:
        ing = self.agent.pipeline.ingest(text, session_id)
        sid, lang = ing.session_id, ing.language.language
        if ing.session_id != session_id:
            st = self.load(sid)
        st["language"], st["turn"] = lang, ing.turn
        hi = _hinglish(lang)
        base = {"session_id": sid, "turn": ing.turn, "language": lang}

        if filters.get("near") and not ctx["located"] and not filters.get("locationAsked"):
            st["pending"] = {k: v for k, v in filters.items() if k != "locationAsked"}
            self._save(sid, st)
            q = ("Aap kahan hain? Location share karein, area ya pincode likhein, ya apna city use karein."
                 if hi else "Where are you? Share your location, type your area or pincode, or use your city.")
            return self._reply(base, FOLLOW_UP, q, question=q,
                               clarify={"kind": "location", "question": q, "options": LOCATION_OPTIONS})
        st.pop("pending", None)

        used_city = bool(filters.get("near") and not ctx["located"])
        per = self.directory.limit
        limit = 10 if filters.get("sort") == "fee" else min(10, per * page)   # MCP has no offset yet: fetch, slice
        args = dict(specialty_key=filters.get("specialtyKey"), query=filters.get("query"),
                    degree=filters.get("degree"), city=ctx["city"], hospital_id=ctx["hospital_id"], limit=limit)
        if ctx["located"]:
            near = bool(filters.get("near"))
            args.update(lat=ctx["lat"], lng=ctx["lng"],
                        max_distance_km=(filters.get("radiusKm") or NEARBY_DEFAULT_KM) if near else WIDE_RADIUS_KM)
            if near:
                args["sort"] = "distance"
        if filters.get("sort") == "experience":
            args["sort"] = "experience"

        t0 = time.perf_counter()
        fallback_used = False
        try:
            found = self.directory.search(**args)
            if (found is None or not found["doctors"]) and filters.get("specialtyKey") and not filters.get("query"):
                if filters["specialtyKey"] != "general-medicine":                   # nobody of that specialty
                    gp = self.directory.search(**dict(args, specialty_key="general-medicine"))
                    if gp and gp["doctors"]:
                        found, fallback_used = gp, True
        except DoctorLookupError:
            self._save(sid, st)
            msg = ("Abhi doctors load nahi ho paaye. Thodi der mein dobara try karein, ya Doctor Dekho directory "
                   "dekhein." if hi else
                   "I couldn't load doctors right now. Please try again in a moment, or open the Doctor Dekho directory.")
            out = self._reply(base, DOCTOR_LIST, msg, degraded=["doctor_lookup"])
            out["searchFilters"] = _public_filters(filters)
            return out
        ms = round((time.perf_counter() - t0) * 1000, 1)

        if found is None and not (filters.get("specialtyKey") or filters.get("query") or filters.get("degree")):
            st["pending"] = {k: v for k, v in filters.items() if k != "locationAsked"}   # nothing to search by yet
            self._save(sid, st)
            q = ("Paas ke doctor dhoondhne ke liye location share karein, ya batayein kis tarah ka doctor chahiye."
                 if hi else "To find doctors near you, share your location, or tell me what kind of doctor you need.")
            return self._reply(base, FOLLOW_UP, q, question=q,
                               clarify={"kind": "location", "question": q, "options": LOCATION_OPTIONS})

        items = list(found["doctors"]) if found else []
        if filters.get("sort") == "fee":
            items = sorted(items, key=_fee)                    # stable: equal fees keep 1HMS order
        total = found["total"] if found else 0
        if page > 1 and len(items) <= per * (page - 1):                        # nothing left to show
            st["page"] = page - 1
            self._save(sid, st)
            msg = ("Is list mein aur doctor nahi hain. Doctor Dekho par saare doctor dekh sakte hain, ya filter badlein."
                   if hi else "There are no more doctors in this list. You can browse all doctors on Doctor Dekho, "
                              "or change the filters.")
            return self._reply(base, DOCTOR_LIST, msg, clarify={"kind": "no_more", "question": msg, "options": []})
        items = items[per * (page - 1): per * page]
        modes = list(filters.get("modes") or [])
        block = self._block(items, total, modes, filters, ctx, specialty_id=found["specialtyId"] if found else None,
                            specialist=_spec_name(filters.get("specialtyKey")) if not fallback_used else FALLBACK_SPECIALIST,
                            fallback_used=fallback_used, ms=ms)
        clarify = None
        st["last"] = {k: v for k, v in filters.items() if k not in ("locationAsked",)}
        st["page"] = page
        st.pop("which", None)
        self._remember(st, items)
        block["page"] = page
        block["hasMore"] = total > per * page

        if not items:
            msg, clarify = self._empty_message(filters, hi)
            clarify = self._confused(st, clarify)
        else:
            st["confused"] = 0
            msg = self._list_message(items, total, filters, hi, used_city=used_city and bool(ctx["city"]),
                                     fallback_used=fallback_used, page=page)
            which = self._which_options(items, filters)
            if which:                                   # several doctors share the name: tell them apart
                st["which"] = which
                q = ("Aap kaun se doctor ki baat kar rahe hain?" if hi else "Which one do you mean?")
                clarify = {"kind": "which_doctor", "question": q,
                           "options": [{"id": o["id"], "label": o["label"]} for o in which]}
                msg = msg + " " + q
        self._save(sid, st)
        out = self._reply(base, DOCTOR_LIST, msg, question=clarify["question"] if clarify else None, clarify=clarify)
        out["doctors"] = block
        return out

    # ================================================================== selection
    def _select(self, text, session_id, doctor_id, ctx) -> dict:
        if not session_id:
            raise SelectionError("select_doctor_id needs the session_id the doctor was shown in")
        st = self.load(session_id)
        shown = st.get("shown") or {}
        if doctor_id not in shown:
            raise SelectionError("select_doctor_id was not shown in this session")
        lang = st.get("language") or "english"
        hi = _hinglish(lang)
        base = {"session_id": session_id, "turn": st.get("turn"), "language": lang}
        degraded: list[str] = []
        try:
            doctor = self.directory.get_doctor(doctor_id)
            if doctor is None:
                msg = ("Ye doctor ab Doctor Dekho par available nahi hain. Kripya list se koi aur doctor chunein."
                       if hi else "This doctor is no longer available on Doctor Dekho. Please choose another one.")
                shown.pop(doctor_id, None)
                self._save(session_id, st)
                return self._reply(base, DOCTOR_LIST, msg)
        except DoctorLookupError:
            doctor, degraded = shown[doctor_id], ["doctor_lookup"]       # what was shown; booking re-checks it
        st["selected"] = {"doctorId": doctor_id, "name": doctor.get("name"), "at": int(time.time())}
        self._save(session_id, st)
        name = doctor.get("name") or "the doctor"
        hosp = (doctor.get("hospital") or {}).get("name")
        where = f" ({hosp})" if hosp else ""
        msg = (f"{name}{where} chune gaye. Slot book karne ke liye 'Book a slot' dabayein."
               if hi else f"You chose {name}{where}. Tap 'Book a slot' to pick a time.")
        out = self._reply(base, DOCTOR_SELECTED, msg, degraded=degraded)
        out["doctor"] = doctor
        out["selectedDoctorId"] = doctor_id
        return out

    def clear(self, sid: str) -> None:
        try:
            self.r.delete(self._key(sid))
        except Exception:
            log.exception("doctor session state not cleared")

    # ================================================================== conversation moves
    def _ingest(self, text, session_id):
        ing = self.agent.pipeline.ingest(text, session_id)
        st = self.load(ing.session_id)
        st["language"], st["turn"] = ing.language.language, ing.turn
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": ing.language.language}
        return ing, st, base, _hinglish(ing.language.language)

    def _handoff(self, text, session_id, reason: str) -> dict:
        """A person takes over. The channel (WhatsApp bot, Doctor Dekho) routes the chat on kind=handoff."""
        ing, st, base, hi = self._ingest(text, session_id)
        st["handoff"] = {"reason": reason, "at": int(time.time())}
        self._save(ing.session_id, st)
        msg = ("Main aapko hamari support team se jod raha hoon. Koi jaldi aapse baat karega." if hi else
               "I'm passing you to our support team. Someone will reply to you shortly.")
        out = self._reply(base, HANDOFF, msg)
        out["handoff"] = {"reason": reason}
        return out

    def _private_number(self, text, session_id, ask, st_in, ctx) -> dict:
        """Never a doctor's personal number. Offer to book, or the hospital front desk from 1HMS."""
        ing, st, base, hi = self._ingest(text, session_id)
        doctor = None
        sel = (st.get("selected") or {}).get("doctorId")
        if sel and sel in (st.get("shown") or {}):
            doctor = st["shown"][sel]
        elif ask.name_query and self.directory.enabled:
            try:
                found = self.directory.search(query=ask.name_query, city=ctx["city"], limit=1)
                doctor = (found or {}).get("doctors", [None])[0] if found and found["doctors"] else None
            except DoctorLookupError:
                doctor = None
        if doctor:
            self._remember(st, [doctor])
        self._save(ing.session_id, st)
        name = (doctor or {}).get("name")
        hosp = (doctor or {}).get("hospital") or {}
        desk = hosp.get("contact")                     # the hospital front desk, as 1HMS publishes it
        if hi:
            msg = ("Main doctors ke personal number share nahi kar sakta. "
                   + (f"Main {name} ke saath appointment book kar sakta hoon" if name else
                      "Main aapke liye appointment book kar sakta hoon"))
            msg += f", ya {hosp.get('name')} ka front desk number: {desk}." if desk else "."
        else:
            msg = ("I can't share doctors' personal numbers, but "
                   + (f"I can book {name} for you" if name else "I can book an appointment for you"))
            msg += f", or you can call {hosp.get('name')}'s front desk: {desk}." if desk else "."
        out = self._reply(base, INFO, msg)
        if doctor:
            out["doctor"] = doctor                    # from 1HMS; the card offers Choose / Book
        return out

    def _small_talk(self, text, session_id, move: str) -> dict:
        ing, st, base, hi = self._ingest(text, session_id)
        self._save(ing.session_id, st)
        if move == "greeting":
            q = ("Namaste! Main Vita hoon. Aapko kis tarah ka doctor chahiye?" if hi else
                 "Hello! I'm Vita. How can I help you find a doctor?")
            return self._reply(base, FOLLOW_UP, q, question=q,
                               clarify={"kind": "ask_kind", "question": q, "options": ASK_KIND_OPTIONS})
        msg = ("Koi baat nahi. Jab bhi doctor chahiye ya koi sawaal ho, main yahin hoon." if hi else
               "No problem. I'm here whenever you need a doctor or have a question.")
        return self._reply(base, INFO, msg)

    # ================================================================== helpers
    def _ask_kind(self, text, session_id, st) -> dict:
        ing = self.agent.pipeline.ingest(text, session_id)
        hi = _hinglish(ing.language.language)
        q = ("Kis tarah ka doctor chahiye? Symptom batayein, specialty, doctor ka naam, degree, ya paas ke doctor."
             if hi else "What kind of doctor do you need? Tell me your symptoms, a specialty, a doctor's name, a "
                        "degree, or ask for doctors near you.")
        st = self.load(ing.session_id)
        st["language"], st["turn"] = ing.language.language, ing.turn
        clarify = self._confused(st, {"kind": "ask_kind", "question": q, "options": list(ASK_KIND_OPTIONS)})
        self._save(ing.session_id, st)
        return self._reply({"session_id": ing.session_id, "turn": ing.turn, "language": ing.language.language},
                           FOLLOW_UP, q, question=q, clarify=clarify)

    @staticmethod
    def _reply(base: dict, kind: str, message: str, *, question: str | None = None, clarify: dict | None = None,
               degraded: list[str] | None = None) -> dict:
        """Same top-level fields as AgentReply.to_dict(), so existing clients can read every reply."""
        return {"session_id": base["session_id"], "turn": base.get("turn"), "kind": kind, "message": message,
                "question": question, "card": None, "state": None, "severity": None, "keywords": [],
                "language": base.get("language") or "english", "degraded": degraded or [], "clarify": clarify}

    def _block(self, items, total, modes, filters, ctx, *, specialty_id, specialist, fallback_used, ms) -> dict:
        block = {"mode": modes, "filters": _public_filters(filters, ctx, specialty_id), "items": items,
                 "total": total, "fallbackUsed": fallback_used, "source": SOURCE, "lookupMs": ms,
                 # names kept from the first version of this block (pinned by tests/test_doctor_directory.py)
                 "specialist": specialist, "specialtyId": specialty_id, "doctors": items}
        if ctx.get("hospital_id"):
            own = [d for d in items if d.get("sameHospital")]
            rest = [d for d in items if not d.get("sameHospital")]
            groups = []
            if own:
                groups.append({"title": (own[0].get("hospital") or {}).get("name") or "This hospital",
                               "doctorIds": [d.get("doctorId") for d in own]})
            if rest:
                groups.append({"title": "Other doctors nearby", "doctorIds": [d.get("doctorId") for d in rest]})
            block["groups"] = groups
        return block

    @staticmethod
    def _remember(st: dict, items: list[dict]) -> None:
        shown = st.get("shown") or {}
        for d in items:
            if d.get("doctorId"):
                shown[d["doctorId"]] = d
        if len(shown) > MAX_SHOWN:
            shown = dict(list(shown.items())[-MAX_SHOWN:])
        st["shown"] = shown

    @staticmethod
    def _which_options(items: list[dict], filters: dict) -> list[dict] | None:
        """VITA-4 / vision: a name matches several doctors about equally -> show hospital and city."""
        if "name" not in (filters.get("modes") or []) or len(items) < 2:
            return None
        scores = [d.get("nameMatchScore") for d in items]
        if scores[0] is None or scores[1] is None or scores[0] - scores[1] > 0.1:
            return None
        top = scores[0]
        out = []
        for d in items:
            if d.get("nameMatchScore") is None or top - d["nameMatchScore"] > 0.1:
                continue
            h = d.get("hospital") or {}
            place = ", ".join(x for x in (h.get("name"), h.get("city")) if x)
            out.append({"id": d.get("doctorId"), "label": f"{d.get('name')} · {place}" if place else d.get("name"),
                        "keys": sorted(set(preprocess(f"{h.get('name') or ''} {h.get('city') or ''} "
                                                      f"{d.get('specialtyName') or ''}").split()) - {"hospital", "the"})})
        return out[:4] if len(out) > 1 else None

    @staticmethod
    def _which_pick(norm: str, st: dict) -> str | None:
        """A typed answer to "which one?" ("salt lake wali", "howrah") that fits exactly one option."""
        options = st.get("which") or []
        words = {w for w in norm.split() if len(w) >= 3}
        hits = [o["id"] for o in options if words & set(o.get("keys") or [])]
        return hits[0] if len(hits) == 1 else None

    def _mid_triage(self, session_id: str | None) -> bool:
        """The triage agent asked a question and is waiting: a short "nahi" / "ok" is its answer."""
        if not session_id:
            return False
        try:
            snap = self.agent.accumulator.snapshot(session_id)
            return bool(snap.turn) and not snap.ready and bool(snap.next_question)
        except Exception:
            return False

    @staticmethod
    def _confused(st: dict, clarify: dict | None) -> dict | None:
        """Two dead ends in a row -> also offer a person (vision: repeated confusion hands off)."""
        st["confused"] = int(st.get("confused") or 0) + 1
        if st["confused"] < 2:
            return clarify
        clarify = dict(clarify or {"kind": "no_match", "question": None, "options": []})
        clarify["options"] = [*(clarify.get("options") or []), HUMAN_OPTION]
        return clarify

    @staticmethod
    def _list_message(items, total, filters, hi, *, used_city, fallback_used, page=1) -> str:
        n = len(items)
        spec = _spec_name(filters.get("specialtyKey"))
        plain = PLAIN.get(filters.get("specialtyKey") or "", (None, None))[1 if hi else 0]
        what = (f"{plain} " if plain else f"{spec} ") if spec and not fallback_used else ""
        bracket = f" ({spec})" if spec and plain and plain.lower() not in spec.lower() and not fallback_used else ""
        deg = f"{filters['degree']} " if filters.get("degree") else ""
        near = filters.get("near") and not used_city
        if hi:
            head = (f"{spec} abhi available nahi hain; ye General Physician pehle dekh sakte hain. "
                    if fallback_used and spec else "")
            where = " aapke paas" if near else ""
            city = " (location nahi mili, aapke city ke doctor dikha rahe hain)" if used_city else ""
            core = (f"'{filters['query']}' naam se ye doctor mile" if filters.get("query")
                    else f"Ye {'aur ' if page > 1 else ''}{deg}{what}doctor{bracket} mile{where}")
            return f"{head}{core}{city}. Kisi ek ko chunne ke liye 'Choose' dabayein."
        head = (f"No {spec} is available right now; a General Physician can see you first. "
                if fallback_used and spec else "")
        where = " near you" if near else ""
        city = " (no location shared, so these are in your city)" if used_city else ""
        core = (f"Doctors matching '{filters['query']}'" if filters.get("query")
                else f"Here {'is' if n == 1 else 'are'} {n} {'more ' if page > 1 else ''}{deg}{what}"
                     f"doctor{'s' if n != 1 else ''}{bracket}{where}")
        more = f" ({total} in all)" if total > n else ""
        return f"{head}{core}{more}{city}. Tap Choose to pick one."

    @staticmethod
    def _empty_message(filters, hi) -> tuple[str, dict | None]:
        if filters.get("query"):
            q = filters["query"]
            msg = (f"'{q}' naam ke koi doctor nahi mile. Kya specialty se dhoondhein?" if hi else
                   f"I couldn't find a doctor named '{q}'. Shall I search by specialty instead?")
            return msg, {"kind": "no_match", "question": msg, "options": ASK_KIND_OPTIONS[1:2]}
        spec = _spec_name(filters.get("specialtyKey")) or "matching"
        deg = f" {filters['degree']}" if filters.get("degree") else ""
        where = " paas mein" if filters.get("near") else ""
        msg = (f"Abhi{where} koi{deg} {spec} doctor Doctor Dekho par nahi mile. Doosra area ya specialty try karein."
               if hi else
               f"No{deg} {spec} doctors are listed on Doctor Dekho{' near you' if filters.get('near') else ''} right "
               f"now. Try another area or specialty.")
        return msg, None


# ---------------------------------------------------------------------------------------------- pure helpers
def _wants_names(norm: str) -> bool:
    """Fetch the name index only when the message might name a doctor without 'dr'."""
    from ..doctor_intent import _NAME_CUE
    return bool(_NAME_CUE.search(norm))


def _bare_doctor(norm: str, ask: DoctorAsk) -> bool:
    """'doctor chahiye', 'koi achha doctor batao': asks for a doctor but says nothing about which."""
    return not ask.modes and not ask.refine and any(w in DOCTOR_WORDS for w in norm.split())


def _spec_name(key: str | None) -> str | None:
    return SPECIALTY_NAMES.get(key) if key else None


def _filters(ask: DoctorAsk) -> dict:
    return {"specialtyKey": ask.specialty_key, "query": ask.name_query, "degree": ask.degree, "near": ask.near,
            "radiusKm": ask.radius_km, "sort": ask.sort, "modes": [m for m in ask.modes if m != "symptom"]}


def _merge(last: dict, ask: DoctorAsk) -> dict:
    """'inme se koi paas wala', 'kam fees wala', 'MD wala' -> the previous search plus this change."""
    f = dict(last)
    f.pop("specialist", None)
    modes = set(f.get("modes") or [])
    for attr, key, mode in (("specialty_key", "specialtyKey", "specialty"), ("name_query", "query", "name"),
                            ("degree", "degree", "degree")):
        if getattr(ask, attr):
            f[key] = getattr(ask, attr)
            modes.add(mode)
    if ask.near:
        f["near"], f["radiusKm"] = True, ask.radius_km or f.get("radiusKm")
        modes.add("distance")
    if ask.sort:
        f["sort"] = ask.sort
    f["modes"] = sorted(modes - {"symptom"}) or sorted(modes)
    return f


def _public_filters(filters: dict, ctx: dict | None = None, specialty_id: str | None = None) -> dict:
    """doctors.filters: the filters actually applied (VITA-7). No coordinates are echoed back."""
    out: dict[str, Any] = {}
    if specialty_id:
        out["specialtyId"] = specialty_id
    for k in ("query", "degree", "sort"):
        if filters.get(k):
            out[k] = filters[k]
    if filters.get("near"):
        out["near"] = True
        out["radiusKm"] = filters.get("radiusKm") or NEARBY_DEFAULT_KM
    if ctx:
        if ctx.get("located"):
            out["location"] = "shared"
        elif ctx.get("city"):
            out["city"] = ctx["city"]
        if ctx.get("hospital_id"):
            out["hospitalId"] = ctx["hospital_id"]
    return out
