"""Vita's tested steps: the code the LangGraph conversation (vita_graph/graph.py) calls.

There is no second conversation engine here. The graph decides, on every message, which step (if any) answers:

  step_self_harm      a person first, whatever was going on (Tele-MANAS 14416, 108)
  step_tap            a doctor chosen from a list, the next page, the screen's filter chips
  step_bangla         Bangla is recognised, not yet understood
  step_emergency_care after an emergency: nearest places, sharing the number, "ambulance aa gayi"
  step_visit          after booking: the way to the clinic, the day of the visit
  step_booking        day, time, for whom, details, OTP, confirm
  step_picks          answers to Vita's last question: location, which doctor / hospital, "kal milenge?"
  step_safety         the shared emergency detector and the warning-sign questions, on every message
  then the LLM agent  understands the message; its tools here are read_ask / plain_search (doctor search),
                      choose, handoff, facts, emergency_reply; unavailable() when it cannot understand right now

Doctors are exactly what 1HMS returned, in its order (VITA-12); only an explicit "kam fees" sorts them.
1HMS failing never fails a reply: the reply says so and degraded contains "doctor_lookup" (VITA-10).
Patient text is never sent to 1HMS: only the extracted filters are (NFR-2.6). Location is used for the
search in hand and not stored.
"""
from __future__ import annotations

import copy
import functools
import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .. import voice as V
from ..booking_client import BookingError
from ..doctor_intent import DOCTOR_WORDS, DoctorAsk, availability_when, close_name_word, parse, search_text
from ..doctors import (FALLBACK_SPECIALIST, SPECIALIST_TITLES, NEARBY_DEFAULT_KM, WIDE_RADIUS_KM, DoctorDirectory,
                       DoctorLookupError)
from ..safety.self_harm import REASON as _SELF_HARM, mentions_self_harm
from ..search_spelling import name_tokens
from ..severity import EMERGENCY, emergency_check
from ..textnorm import intent_text, preprocess, tidy
from ..voice import Voice
from .booking import BookingFlow, token_mobile
from . import everyday
from .emergency_help import EmergencyHelp
from .journey import _NO as _JOURNEY_NO, _YES as _JOURNEY_YES, JourneyHelp, wants as journey_wants
from .visit_help import VisitHelp, wants as visit_wants

SELF_HARM_SIGN = _SELF_HARM.lower()
# the words of a plain "no / nothing else" answer (see emergency_signs)
_ANSWER_WORDS = {"nahi", "nahin", "nai", "na", "no", "nope", "aur", "kuch", "koi", "bhi", "hai", "hain", "bas", "itna",
                 "hi", "ji", "nothing", "else", "thats", "that", "s", "it", "all", "theek", "thik", "ok", "okay", "haan",
                 "han", "yes", "aage", "abhi", "toh", "to", "sab", "sirf", "else", "more",
                 # "abhi nahi chahiye", "zarurat nahi", "no thanks, later" (the model scored "abhi nahi chahiye" 0.72)
                 "chahiye", "chaiye", "zarurat", "zaroorat", "need", "dont", "don", "t", "do", "baad", "mein", "later",
                 "please", "thanks", "thank", "you", "rehne", "dijiye", "not", "now", "maybe", "filhal"}
_LANG_NAMES = r"(hindi|hindee|हिंदी|हिन्दी|english|angrezi|angreji|inglish|bangla|bengali|bangali)"
_LANG_ASK = re.compile(
    r"^(please |plz |pls |kya |can (you|u) |could (you|u) )?" + _LANG_NAMES +
    r" ?(mein|me|mai|main|में|te|e|please|plz|pls)? ?(baat karo|baat kariye|baat karein|baat kijiye|bolo|boliye|bolein"
    r"|likho|likhiye|likhein|batao|bataiye|samjhao|reply|reply karo|jawab do|chalega|please|plz|pls|only|me baat karo"
    r"|bol sakti ho|bol sakte ho|aati hai|aati hai kya|बात करो|बोलो|बात कीजिए)?( please| plz| pls| na| ji)?$"
    r"|^(speak|talk|reply|answer|write|chat)( to me| with me)? in " + _LANG_NAMES + r"( please| plz)?$"
    r"|^(can|could|do) (you|u) (speak|talk|understand) " + _LANG_NAMES + r"$"
    r"|^" + _LANG_NAMES + r" (please|plz|pls)$")
_LANG_OF = {"hindi": "hindi", "hindee": "hindi", "हिंदी": "hindi", "हिन्दी": "hindi", "english": "english",
            "angrezi": "english", "angreji": "english", "inglish": "english", "bangla": "bangla", "bengali": "bangla",
            "bangali": "bangla"}


def _language_request(norm: str) -> str | None:
    """"hindi mein baat karo" -> "hindi"; "english please" -> "english". None for anything else, including a doctor
    who speaks a language ("english speaking doctor")."""
    norm = (norm or "").strip().rstrip("?!. ")
    if not norm or re.search(r"\b(doctor|dr|daktar|specialist)\b", norm):
        return None
    m = _LANG_ASK.match(norm)
    if not m:
        return None
    word = next(w for w in re.findall(r"[\w\u0900-\u097F]+", norm) if w in _LANG_OF)
    return _LANG_OF[word]


# words that may be a cry for help: a model-only flag on a message with any of these always stands
_CRY_FOR_HELP = {"help", "bachao", "bachaao", "bacha", "jaldi", "urgent", "emergency", "ambulance", "hospital", "serious",
                 "mar", "marna", "mara", "marne", "maut", "die", "dying", "dead", "khatam", "please", "plz", "sos",
                 "aao", "aaiye", "bulao", "call", "phone", "108", "112"}
AFTER_EMERGENCY_S = 2 * 3600      # how long "ambulance aa gayi", "ab theek hoon" are read as about that emergency
# what patients say after Vita told them to call 108
_AFTER_EMERGENCY = (
    ("cant_call", re.compile(r"\b(call|phone|108|112|number) (nahi|nhi|nahin) (lag|ja|mil|lagg?) ?(raha|rahi|rha|rhi)?|"
                             r"\bambulance (nahi|nhi|nahin) (aa|aayi|aaya|aa rahi|mil)|\bno ambulance|can'?t (call|get through|reach)|"
                             r"\bnot (connecting|answering)|\bkoi (phone|call) nahi utha")),
    ("arrived", re.compile(r"\bambulance (aa gayi|aa gaya|aa gai|agyi|agayi|pahunch gayi|pahunch gaya|aa rahi hai|bula li|bulayi|"
                           r"bula liya|is here|arrived|has come|is coming|on the way)|\b(hospital|emergency|er) (pahunch|pohoch|aa) "
                           r"(gaye|gayi|gaya|gya|gyi|gye)|\b(hospital|er|emergency) (mein|me|main) (hoon|hu|hun|hain|he|hai)\b|"
                           r"\breached (the )?(hospital|er)|\bat the (hospital|er)\b|\bdoctor (aa gaye|dekh rahe)|\bcalled 108|"
                           r"\b(108|112) (call|phone) kar (diya|di|liya|li)\b")),
    ("thanks", re.compile(r"^(ok|okay|thik hai|theek hai|thanks|thank you|thank u|thx|shukriya|dhanyavad|dhanyawad|ji|hmm)\W*$")),
    ("better", re.compile(r"^(ab |abhi |main |mai |me |i am |i'?m |now )*(theek|thik|teek|better|fine|accha|achha|thoda better|"
                          r"behtar)( (hoon|hu|hun|hai|he|hu ab|lag raha|lag rahi|feel kar raha|feel kar rahi|now|ab))*$|"
                          r"\b(feeling better|i'?m fine now|aaram hai|aaram mil gaya|ab theek|ab thik|ab normal|"
                          r"saans (theek|thik|normal) (hai|ho gayi)|dard (kam|chala gaya))")),
)
_LOCATION_ONLY = re.compile(r"^\s*(location|my location|share (my )?location|meri location|location bhej di|मेरी लोकेशन)\s*$", re.I)

log = logging.getLogger("hms_router.doctor_chat")

DOCTOR_LIST, DOCTOR_SELECTED, FOLLOW_UP = "doctor_list", "doctor_selected", "follow_up"
INFO, HANDOFF = "info", "handoff"
SOURCE = "1hms-mcp"
STATE_TTL_S = 1800
MAX_SHOWN = 60

SPECIALTY_NAMES = SPECIALIST_TITLES          # specialty key -> the title patients see (now in doctors.py)
# Plain words for specialties, chips and every phrase live in hms_router/voice.py, in three languages.
PLAIN = {k: v[:2] for k, v in V.PLAIN.items()}     # (English, Hinglish), kept for older callers
_EN = Voice("english")
LOCATION_OPTIONS, HUMAN_OPTION, ASK_KIND_OPTIONS = V.location_options(_EN), V.human_option(_EN), V.ask_kind_options(_EN)

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


@functools.lru_cache(maxsize=1)
def _known_words() -> frozenset[str]:
    """Every word Vita knows: a stretched word is squeezed back only onto one of these ("bataoo" -> "batao")."""
    from ..doctor_intent import _NAME_STOP
    from ..language import HINDI_WORDS
    from ..lexicon import Lexicon
    return frozenset(Lexicon.from_csv().vocab) | DOCTOR_WORDS | _NAME_STOP | HINDI_WORDS


def _hinglish(lang: str | None) -> Voice:
    """The conversation's voice (truthy for Hinglish and Hindi)."""
    return Voice(lang)


def _fee(d: dict) -> float:
    fee = d.get("discountedFee") if d.get("discountedFee") is not None else d.get("consultFee")
    return float(fee) if fee is not None else float("inf")


class DoctorChat:
    def __init__(self, pipeline, directory: DoctorDirectory, redis, *, detector=None, terms=medical_terms,
                 booking_client=None):
        """pipeline: HMSRouterPipeline (sessions, language, spelling). detector: returns the shared emergency
        detector (the route-symptom service's), read per message so a model rebuild is picked up."""
        self.pipeline, self.directory, self.r, self.terms, self.detector = pipeline, directory, redis, terms, detector
        self.booking = BookingFlow(directory, booking_client)
        self.emergency = EmergencyHelp(booking_client)
        self.visit = VisitHelp(booking_client)
        self.journey = JourneyHelp(booking_client, ola_token=os.environ.get("OLA_XAPP_TOKEN") or None)

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

    def _remember_reply(self, out: dict | None, ctx: dict) -> None:
        """What the next message may refer to: whether triage is waiting for an answer ("nahi" answers it), and the
        last question asked ("kya?" / "samjha nahi" repeats it in simpler words)."""
        sid = (out or {}).get("session_id")
        if not sid:
            return
        est = self.load(sid)
        before = (est.get("triage_waiting"), est.get("last_reply"))
        if ctx.get("triage_waiting"):
            est["triage_waiting"] = True
        else:
            est.pop("triage_waiting", None)
        if out.get("kind") not in ("emergency", "handoff") and out.get("message") and not ctx.get("repeated"):
            clarify = out.get("clarify")
            est["last_reply"] = {"kind": out["kind"], "message": out["message"][:600],
                                 "clarify": clarify if clarify and len(json.dumps(clarify)) < 4000 else None}
        if (est.get("triage_waiting"), est.get("last_reply")) != before:
            self._save(sid, est)


    # ================================================================== safety
    # ================================================================== steps (LangGraph nodes call these)
    # Vita's conversation is the LangGraph graph (vita_graph/graph.py). The code below is not a second engine: each
    # step does one tested job when the graph routes a message to it, and returns None when the message is not its
    # business, so the graph moves on (in the order below, then to the LLM agent).
    def context(self, kw: dict) -> dict:
        lat, lng = kw.get("lat"), kw.get("lng")
        tok = kw.get("patient_token")
        self._hint = kw.get("ui_language")
        return {"city": kw.get("city"), "lat": lat, "lng": lng, "hospital_id": kw.get("hospital_id"),
                "located": lat is not None and lng is not None,
                # booking only: the visitor's IP for 1HMS's per-visitor limits, the channel's signed-in patient
                # token (no OTP needed), a number the channel already verified (WhatsApp)
                "client_ip": kw.get("client_ip"), "patient_token": tok, "verified_mobile": kw.get("verified_mobile"),
                "token_mobile": token_mobile(tok) if tok else None, "channel": kw.get("channel") or "web",
                # after booking: the visit the website says the patient is looking at
                "visit_appointment_id": kw.get("visit_appointment_id")}

    @staticmethod
    def tidy(text: str) -> str:
        """"doc chaiye", "nhi pta", "bukharrr", "🤒" -> the words Vita's code knows (the LLM reads the original)."""
        return tidy(text, _known_words())

    def finish(self, out: dict | None, ctx: dict) -> dict | None:
        """Every reply, whoever wrote it: the emergency number said once after an emergency comes first, a step's
        lead line ("No warning signs, so ...") before the message, and what the next message may refer to."""
        reminder = ctx.get("reminder")
        if reminder and out and out.get("kind") != "emergency" and out.get("message"):
            out["message"] = f"{reminder}\n\n{out['message']}"
        lead = ctx.get("lead")
        if lead and out and out.get("kind") != "emergency" and out.get("message"):
            out["message"] = f"{lead} {out['message']}"
        self._remember_reply(out, ctx)
        return out

    def step_self_harm(self, text, sid, ctx) -> dict | None:
        """Every turn, whatever Vita was in the middle of (booking, visit, journey, an earlier emergency): a person
        comes first."""
        if not mentions_self_harm(text):
            return None
        st = self.load(sid)
        if sid and [k for k in ("booking", "pending", "which", "which_hospital", "warn", "last_advice") if st.pop(k, None)]:
            self._save(sid, st)
        return self.emergency_reply(text, sid, ctx, [SELF_HARM_SIGN])

    def step_tap(self, text, sid, ctx, kw) -> dict | None:
        """A doctor chosen from a list, the next page, or the screen re-sending its remaining filters."""
        if kw.get("select_doctor_id"):
            return self._select(text, sid, kw["select_doctor_id"], ctx)
        st = self.load(sid)
        page = int(kw.get("page") or 1)
        if page > 1 and st.get("last"):
            return self._search_turn(text, sid, dict(st["last"]), ctx, st, page=page)
        if kw.get("new_search"):                         # exactly the filters on screen, never merged into the last
            ask = self.read_ask(text, sid)
            if ask.direct or ask.place:
                return self._search_turn(text, sid, _filters(ask), ctx, st)
        return None

    def step_language(self, text, sid) -> dict | None:
        """"hindi mein baat karo", "english please", "bangla bolo": the patient chooses Vita's language. Kept for the
        whole conversation (session context language_pref) whatever language later messages are typed in. A doctor
        who speaks a language ("english bolne wale doctor") is a search, not this."""
        want = _language_request(intent_text(self.tidy(text)))
        if want is None:
            return None
        if want == "bangla":
            return self._bangla(text, sid)
        if want == "hindi" and not re.search(r"[\u0900-\u097F]", text):
            want = "hinglish"                                         # asked in Roman letters: Roman Hindi back
        ing = self._ingest_only(text, sid)
        self.pipeline.sessions.set_context(ing.session_id, language=want, language_pref=want)
        st = self.load(ing.session_id)
        st["language"], st["turn"] = want, ing.turn
        self._save(ing.session_id, st)
        v = Voice(want)
        msg = v("Sure, I'll talk in English. What's bothering you, or which doctor do you need?",
                "Theek hai, ab main Hindi mein baat karungi. Aapko kya takleef hai, ya kaunsa doctor chahiye?",
                "ठीक है, अब मैं हिंदी में बात करूँगी। आपको क्या तकलीफ़ है, या कौन-सा डॉक्टर चाहिए?")
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": want}
        return self._reply(base, FOLLOW_UP, msg, question=msg,
                           clarify={"kind": "ask_kind", "question": msg, "options": V.ask_kind_options(v)})

    def step_bangla(self, text, sid) -> dict | None:
        if self.pipeline.detector.detect(text).language == "bangla":
            return self._bangla(text, sid)
        return None

    def step_emergency_care(self, text, sid, ctx) -> dict | None:
        """Right after an emergency: the nearest places for a shared location, sharing the number, "ambulance aa
        gayi", "ab theek hoon", "call nahi lag raha"."""
        st = self.load(sid)
        if st.get("emergency"):
            out = self._emergency_followup(text, sid, st, ctx)
            if out is not None:
                return out
        if st.get("after_emergency"):
            return self._after_emergency(text, sid, self.load(sid), ctx)
        return None

    def step_visit(self, text, sid, ctx, *, asked: bool = False) -> dict | None:
        """After booking: the way to the clinic (app only for now) and the day of the visit. asked: the agent read
        the message as visit help, so a plain "where is the clinic?" is enough."""
        st = self.load(sid)
        if st.get("booking"):
            return None
        if self.journey.enabled and ctx.get("channel") != "whatsapp":
            out = self._journey_turn(text, sid, st, ctx)
            if out is not None:
                return out
            st = self.load(sid)
        if self.visit.enabled:
            return self._visit_turn(text, sid, st, ctx)
        return None

    def step_booking(self, text, sid, ctx) -> dict | None:
        """One message while booking (day, time, for whom, details, OTP, confirm). An emergency still comes first;
        "number kyun chahiye?" gets its answer and the question again; anything else ends the booking -> None."""
        st = self.load(sid)
        if not st.get("booking"):
            return None
        if self._emergency(text):
            st.pop("booking", None)
            self._save(sid, st)
            return None                                   # the safety step answers it
        side = self._side_question(text, sid, st, ctx)
        if side is not None:
            return side
        mid = ctx["ask"] = self.read_ask(text, sid)
        out = self._booking_turn(text, sid, st, ctx)
        ctx.pop("ask", None)
        if out is not None:
            return out
        st = self.load(sid)
        if mid.more and st.get("last") and not (mid.specialty_key or mid.name_query or mid.degree):
            return self._search_turn(text, sid, dict(st["last"]), ctx, st, page=1)   # the list they chose from, again
        return None

    def step_picks(self, text, sid, ctx) -> dict | None:
        """Answers to Vita's own last question: where are you (location), which of these doctors / hospitals, and
        "kal milenge?" about the doctor just chosen."""
        st = self.load(sid)
        norm = intent_text(text)
        if st.get("selected") and not st.get("booking"):
            ask = self.read_ask(text, sid)
            follow = availability_when(norm)
            if follow and not (ask.when or ask.name_query or ask.specialty_key or ask.symptomish or ask.degree):
                ask.when, ask.available_today, ask.refine = follow, False, False
                return self._availability(text, sid, ask, st, ctx)
        pending = st.get("pending")
        if pending:                                      # the previous turn asked "where are you?"
            ask = self.read_ask(text, sid)
            if not (ask.symptomish or ask.specialty_key or ask.name_query or ask.degree):
                filters = dict(pending)
                filters["locationAsked"] = True
                return self._search_turn(text, sid, filters, ctx, st)
        which_id = self._which_pick(norm, st)
        if which_id is not None:
            return self._select(text, sid, which_id, ctx)
        move = everyday.detect(norm, text)
        prev = st.get("last_reply") or {}
        prev_clarify = prev.get("clarify") or {}
        if move == "yes" and prev_clarify.get("kind") == "book_offer":       # typed "haan" to "Kya main book karoon?"
            doc_id = next((o["id"] for o in prev_clarify.get("options") or [] if o.get("id") not in ("no", None)), None)
            if doc_id and doc_id in (st.get("shown") or {}):
                return self._select(text, sid, doc_id, ctx)
        if move == "pick":                                                   # "pehla wala", "2nd one"
            ids, i = st.get("page_ids") or [], everyday.ordinal(norm)
            if i is not None and i < len(ids):
                return self._select(text, sid, ids[i], ctx)
        if move == "what" and prev.get("message"):                          # "kya?" / "samjha nahi": the question again
            ing, est, base, v = self._ingest(text, sid)
            self._save(ing.session_id, est)
            ctx["repeated"] = True
            msg = f"{everyday.again(v)} {prev['message']}"
            return self._reply(base, prev.get("kind") or FOLLOW_UP, msg, question=msg, clarify=prev.get("clarify"))
        if move == "restart":                                                # "naya shuru karo": a clean start
            ing, est, base, v = self._ingest(text, sid)
            keep = {k: est[k] for k in ("language", "turn") if k in est}
            est.clear()
            est.update(keep)
            self._save(ing.session_id, est)
            try:
                self.pipeline.sessions.clear_history(ing.session_id)
            except Exception:
                log.exception("history not reset on restart")
            ctx["restarted"] = True
            msg = everyday.reply(move, v)
            return self._reply(base, INFO, msg, clarify={"kind": "ask_kind", "question": msg,
                                                         "options": V.ask_kind_options(v)})
        hospital_pick = self._pick_option(norm, st.get("which_hospital"))
        if hospital_pick is not None and st.get("last"):
            opt = next(o for o in st["which_hospital"] if o["id"] == hospital_pick)
            st.pop("which_hospital", None)
            filters = dict(st["last"], hospitalId=opt["id"], hospitalName=opt.get("name"))
            return self._search_turn(text, sid, filters, ctx, st)
        return None

    def step_safety(self, text, sid, ctx) -> dict | None:
        """The shared emergency detector on every message, and the warning-sign questions ("sar pe chot lagi",
        "haath jal gaya", pregnant + swelling) before anyone is sent to 108."""
        st = self.load(sid)
        norm = intent_text(text)
        if st.get("warn"):
            out = self._warn_answer(text, sid, st, ctx, norm)
            if out is not None:
                return out
        ask = self.read_ask(text, sid)
        kind = everyday.warn_kind(norm)
        about_doctor = ask.specialty_key or ask.name_query
        if kind and (kind == "pregnancy" or not about_doctor) and everyday.serious(kind, norm):
            return self._warn_emergency(text, sid, kind)        # "current lag ke jal gaya": no question, 108 now
        if kind and (kind == "pregnancy" or not about_doctor) and not self._red_flag(text):
            # the warning-sign question first; only a red-flag RULE (or self-harm) skips it. The emergency model alone
            # flagged "haath jal gaya garam tel se" (0.90) and sent a minor burn straight to 108.
            return self._warn_ask(text, sid, kind)
        signs = self.emergency_signs(text)
        if signs is not None:
            return self.emergency_reply(text, sid, ctx, signs)
        if everyday.detect(intent_text(self.tidy(text)), text) == "distress":
            return self._distress(text, sid)                       # "bachao", "please help": 108, then what happened
        return None

    def _distress(self, text, sid) -> dict:
        """A cry for help with nothing else said: the emergency number first, then what happened. Not the full
        emergency takeover (no sign was named), so the next message is read normally."""
        ing, st, base, v = self._ingest(text, sid)
        self._save(ing.session_id, st)
        msg = everyday.reply("distress", v)
        return self._reply(base, FOLLOW_UP, msg, question=msg)

    def step_privacy(self, text, sid, ctx) -> dict | None:
        """"Dr Sharma ka number do": never a doctor's personal number; booking, or the hospital front desk from
        1HMS. A rule, so it is code, not the LLM's choice."""
        ask = self.read_ask(text, sid)
        if ask.move != "private_number":
            return None
        return self._private_number(text, sid, ask, self.load(sid), ctx)

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
        if hi.code == "hi":
            msg = ("मैं डॉक्टरों के निजी नंबर शेयर नहीं कर सकती। "
                   + (f"मैं {name} के साथ अपॉइंटमेंट बुक कर सकती हूँ" if name else
                      "मैं आपके लिए अपॉइंटमेंट बुक कर सकती हूँ"))
            msg += f", या {hosp.get('name')} का फ्रंट डेस्क नंबर: {desk}।" if desk else "।"
        elif hi:
            msg = ("Main doctors ke personal number share nahi kar sakti. "
                   + (f"Main {name} ke saath appointment book kar sakti hoon" if name else
                      "Main aapke liye appointment book kar sakti hoon"))
            msg += f", ya {hosp.get('name')} ka front desk number: {desk}." if desk else "."
        else:
            msg = ("I can't share doctors' personal numbers, but "
                   + (f"I can book {name} for you" if name else "I can book an appointment for you"))
            msg += f", or you can call {hosp.get('name')}'s front desk: {desk}." if desk else "."
        out = self._reply(base, INFO, msg)
        if doctor:
            out["doctor"] = doctor                    # from 1HMS; the card offers Choose / Book
        return out

    # ---------------------------------------------------------------- the agent's tools that are plain code
    def read_ask(self, text, sid=None) -> DoctorAsk:
        """What a doctor search in this message asks for (doctor_intent.parse, spelling-tolerant), with every listed
        doctor's name and city (cached 15 min)."""
        memo = self.__dict__.setdefault("_asks", {})            # one reading per message (several steps ask)
        if text in memo:
            return copy.copy(memo[text])
        known, cities = self.directory.index() if self.directory.enabled else ([], None)
        norm = search_text(text, known_names=known, cities=cities, medical_terms=self.terms)
        names = known if _wants_names(norm) else None
        memo[text] = parse(text, names=names, medical_terms=self.terms, known_names=known, cities=cities)
        return copy.copy(memo[text])

    def plain_search(self, text, sid, ctx) -> dict | None:
        """A message that is plainly a doctor search ("cardiologist", "dr neha sharma", "MD near me", "aur dikhao",
        "kam fees wala", "Kishanganj ke doctor"), or a tap on one of Vita's own chips ("No, just checking", "Talk to
        a person"): answered without the LLM. Used when the LLM is off or down; the search bar reads asks the same
        way. None: not plainly one of these."""
        st = self.load(sid)
        ask = self.read_ask(text, sid)
        last = st.get("last")
        if ask.move == "human":
            return self._handoff(text, sid, "asked")
        if ask.move == "just_looking" and not ask.direct:
            ing, est, base, v = self._ingest(text, sid)
            self._save(ing.session_id, est)
            return self._reply(base, INFO, v("No problem. I'm here whenever you need a doctor or have a question.",
                                             "Koi baat nahi. Jab bhi doctor chahiye ya koi sawaal ho, main yahin hoon.",
                                             "कोई बात नहीं। जब भी डॉक्टर चाहिए या कोई सवाल हो, मैं यहीं हूँ।"))
        if ask.symptomish and not (ask.specialty_key or ask.name_query or ask.degree):
            return None
        if ask.when and ask.name_query:
            return self._availability(text, sid, ask, st, ctx)
        if last and ask.more and not (ask.specialty_key or ask.name_query or ask.degree):
            return self._search_turn(text, sid, dict(last), ctx, st, page=int(st.get("page") or 1) + 1)
        if last and ask.refine and not self._said_symptom(text):                   # "inme se kam fees wala"
            return self._search_turn(text, sid, _merge(last, ask), ctx, st)
        if not last and ask.refine and (ask.sort or ask.available_today or ask.place) and not ask.direct \
                and not self._said_symptom(text):        # "sabse sasta doctor", "kishanganj ke doctor": kept for the
            carry = {k: val for k, val in (("sort", ask.sort), ("availableToday", ask.available_today),   # next ask
                                           ("place", ask.place)) if val}
            out = self._ask_kind(text, sid, st)
            est = self.load(out["session_id"])
            est["carry"] = carry
            self._save(out["session_id"], est)
            ctx["lead"] = V.carry_note(Voice(out.get("language") or "english"), carry)
            return out
        if ask.direct:
            f = _filters(ask)
            carry = st.pop("carry", None) if st.get("carry") else None
            for k, val in (carry or {}).items():                                    # "...skin" after "sabse sasta"
                if not f.get(k):
                    f[k] = val
                    f["modes"] = sorted(set(f.get("modes") or []) | {"today" if k == "availableToday" else k})
            return self._search_turn(text, sid, f, ctx, st)
        if _bare_doctor(intent_text(text), ask):
            return self._ask_kind(text, sid, st)
        return None

    def choose(self, text, sid, ctx, doctor_name: str | None = None) -> dict | None:
        """The agent read "this one / book Dr Sen / Dr Sen kal milenge?": a doctor already shown is chosen, a day
        question is answered, or booking goes on. None: nothing to choose yet."""
        st = self.load(sid)
        norm = intent_text(text)
        which_id = self._which_pick(norm, st)
        if which_id is None and doctor_name:
            want = name_tokens(doctor_name)
            hits = [i for i, d in (st.get("shown") or {}).items()
                    if want and all(any(close_name_word(w, [t]) for t in name_tokens(d.get("name", ""))) for w in want)]
            which_id = hits[0] if len(hits) == 1 else None
        ask = self.read_ask(text, sid)
        if ask.when and (ask.name_query or st.get("selected")):
            if not ask.name_query:
                ask.available_today = False
            return self._availability(text, sid, ask, st, ctx)
        if which_id is not None:
            return self._select(text, sid, which_id, ctx)
        sel = (st.get("selected") or {}).get("doctorId")
        if sel and sel in (st.get("shown") or {}) and not st.get("booking"):
            return self._select(text, sid, sel, ctx)          # "book karo" after choosing: booking starts again
        if ask.name_query:
            return self._search_turn(text, sid, _filters(ask), ctx, st)
        return None

    def handoff(self, text, sid, reason: str = "asked") -> dict:
        return self._handoff(text, sid, reason)

    def social(self, text, sid, *, moves=None) -> dict | None:
        """Everyday talk that needs no model ("thanks", "tata", "ruko", "sorry", "mera naam Rahul hai", "test", "tum
        kaun ho"): the ready-made answer in the patient's language. None when the message also carries a symptom or a
        doctor ask, or is not one of these."""
        norm = intent_text(self.tidy(text))
        move = everyday.detect(norm, text)
        if move not in (moves or everyday.NO_MODEL_MOVES) or self._said_symptom(text):
            return None
        ask = self.read_ask(text, sid)
        if move != "personal" and (ask.specialty_key or ask.name_query or ask.degree):
            return None
        words = set(norm.split())
        if move == "about" and words & {"doctor", "doctors", "dr", "daktar"} and not words & {"tum", "aap", "ap", "tu",
                                                                                               "you", "u"}:
            return None                                       # "english bolne wale doctor": a search, not about Vita
        ing, st, base, v = self._ingest(text, sid)
        self._save(ing.session_id, st)
        msg = everyday.reply(move, v, name=everyday.said_name(norm) if move == "name" else None)
        options = ([V.human_option(v)] if move == "upset" else []) + V.ask_kind_options(v)   # upset: a person first
        kind = FOLLOW_UP if move in ("name", "sorry", "test") else INFO
        return self._reply(base, kind, msg, clarify={"kind": "ask_kind", "question": msg, "options": options})

    def unavailable(self, text, sid, ctx) -> dict:
        """When Vita cannot understand right now (the LLM is off or failing): say so, offer a person and the ways to
        search. Emergencies never reach here: the safety step runs first on every message."""
        ing, st, base, v = self._ingest(text, sid)
        self._save(ing.session_id, st)
        ask = self.read_ask(text, sid)
        move = everyday.detect(intent_text(self.tidy(text)), text)
        if ask.move == "greeting" and move not in everyday.SOCIAL_MOVES:   # "hello", "namaste": a hello back
            if ing.language.strength != "strong":                 # "heyyy" does not say which language comes next
                self.pipeline.sessions.set_context(ing.session_id, language="")
                st.pop("language", None)
                self._save(ing.session_id, st)
            q = v("Hello! I'm Vita. How can I help you find a doctor?",
                  "Namaste! Main Vita hoon. Aapko kis tarah ka doctor chahiye?",
                  "नमस्ते! मैं Vita हूँ। आपको किस तरह का डॉक्टर चाहिए?")
            return self._reply(base, FOLLOW_UP, q, question=q,
                               clarify={"kind": "ask_kind", "question": q, "options": V.ask_kind_options(v)})
        out = self.social(text, ing.session_id)                   # "kkrh", "fees kitni", "thanks"
        if out is not None:
            return out
        if not (ask.symptomish or self._said_symptom(text)):     # not medical and not understood: show how to ask
            msg = v("Sorry, I didn't get that. Tell me what's bothering you (like 'fever for 2 days') or which doctor "
                    "you need (like 'skin doctor').",
                    "Maaf kijiye, main samajh nahi paayi. Apni takleef likhiye (jaise 'bukhar 2 din se') ya kaunsa doctor "
                    "chahiye (jaise 'skin doctor').",
                    "माफ़ कीजिए, मैं समझ नहीं पाई। अपनी तकलीफ़ लिखिए (जैसे 'बुखार 2 दिन से') या कौन-सा डॉक्टर चाहिए "
                    "(जैसे 'स्किन डॉक्टर')।")
            return self._reply(base, FOLLOW_UP, msg, question=msg, clarify={
                "kind": "ask_kind", "question": msg, "options": V.ask_kind_options(v) + [V.human_option(v)]})
        msg = v("Sorry, I couldn't follow that just now. Please try again in a minute, or I can connect you to our team. "
                "In an emergency call 108.",
                "Maaf kijiye, abhi main theek se samajh nahi paayi. Ek minute mein dobara likhiye, ya main aapko hamari "
                "team se jod sakti hoon. Emergency ho to 108 par call karein.",
                "माफ़ कीजिए, अभी मैं ठीक से समझ नहीं पाई। एक मिनट में दोबारा लिखिए, या मैं आपको हमारी टीम से जोड़ सकती "
                "हूँ। इमरजेंसी हो तो 108 पर कॉल करें।")
        options = [V.human_option(v)] + [o for o in V.ask_kind_options(v)][:2]
        out = self._reply(base, FOLLOW_UP, msg, question=msg, clarify={"kind": "ask_kind", "question": msg,
                                                                       "options": options})
        out["degraded"] = ["understanding"]
        return out

    def facts(self, sid) -> dict:
        """What the agent may tell the patient about the doctor in hand (fees, hospital), from 1HMS data only."""
        st = self.load(sid)
        d = (st.get("booking") or {}).get("doctor") or {}
        sel = st.get("selected") or {}
        shown = (st.get("shown") or {}).get(sel.get("doctorId") or "", {})
        h = shown.get("hospital") or {}
        out = {"doctor": d.get("name") or sel.get("name"), "fee": d.get("fee") or shown.get("consultFee"),
               "hospital": h.get("name"), "city": h.get("city")}
        return {k: v for k, v in out.items() if v}

    # ---------------------------------------------------------------- emergencies
    def emergency_signs(self, text: str) -> list[str] | None:
        """The shared emergency detector on this message: None when it is not an emergency, else its red flags
        ([] when the model flags it without a sign a patient can read). Fails safe: an error counts as an emergency."""
        texts = [preprocess(text)]
        try:
            shared = self._shared(text, texts)
            if shared is not None:
                if shared.flag and not shared.reasons and shared.source == "model" \
                        and self._only_everyday(texts[0]):
                    # the model alone, no sign, on a message made only of everyday words ("nahi, aur kuch nahi" 0.73,
                    # "abhi nahi chahiye" 0.72, "ruko ek minute" 0.62): talk, not an emergency. A word that could be a
                    # cry for help, any other word, or any rule still counts.
                    return None
                return [r.lower() for r in shared.reasons] if shared.flag else None
            hits = [h["english"].lower() for h in emergency_check({}, texts) if h["flag"] == EMERGENCY]
            return hits or None
        except Exception:
            log.exception("emergency check failed; treating the message as an emergency")
            return []

    def _only_everyday(self, norm: str) -> bool:
        words = set(re.findall(r"[a-z]+", norm))
        if not words or words & _CRY_FOR_HELP:
            return False
        known = self.__dict__.get("_everyday_words")
        if known is None:
            known = self._everyday_words = _ANSWER_WORDS | {e.variant for e in self.pipeline.lex.entries
                                                            if e.category in ("everyday", "particle", "function", "number")
                                                            and " " not in e.variant}
        return words <= known

    def emergency_reply(self, text, sid, ctx, signs: list[str]) -> dict:
        """Vita's emergency reply: the number first in the patient's language, Tele-MANAS for self-harm, the nearest
        emergency-ready places when enabled (vision Phase 3). Never a doctor list."""
        ing = self._ingest_only(text, sid)
        voice = Voice(ing.language.language)
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": ing.language.language}
        out = self._reply(base, "emergency", V.emergency(voice, signs))
        out["card"] = {"kind": "emergency", "emergency_reasons": [s.capitalize() for s in signs],
                       "outputs": [{"role": "EMERGENCY", "specialist": "Emergency Medicine", "confidence": 1.0}],
                       "emergency_signs": [w for w in (V.symptom_words(voice, [x]) for x in signs) if w]}
        self._emergency_done(ing.session_id, signs)
        if self.emergency.enabled:
            est = self.load(ing.session_id)
            est.pop("booking", None)
            self.emergency.attach(out, est, ctx, signs, text, voice)
            self._save(ing.session_id, est)
        return out

    def _shared(self, raw_text: str, session_texts: list[str]):
        """The route-symptom detector's verdict (rules + model, denial-aware). None when it is not loaded yet, so the
        built-in phrase list answers instead."""
        detector = self.detector() if self.detector else None
        if detector is None:
            return None
        from ..domain.models import EmergencyAssessment, Query
        candidates = [raw_text]
        joined = " . ".join(t for t in session_texts if t)
        if joined and joined != preprocess(raw_text):
            candidates.append(joined)
        flagged = [a for a in (detector.detect(Query(raw=t, normalized=preprocess(t))) for t in candidates) if a.flag]
        if not flagged:
            return EmergencyAssessment(flag=False)
        reasons: list[str] = []
        for a in flagged:
            reasons += [r for r in a.reasons if r not in reasons]
        return EmergencyAssessment(flag=True, reasons=reasons, source=flagged[0].source,
                                   score=max(a.score for a in flagged))


    def _red_flag(self, text: str) -> bool:
        """Self-harm, or a red-flag RULE (not the emergency model alone). Fails safe: True when it cannot tell."""
        if mentions_self_harm(text):
            return True
        try:
            shared = self._shared(text, [preprocess(text)])
            if shared is not None:
                return bool(shared.flag and "rules" in (shared.source or ""))
            hits = emergency_check({}, [preprocess(text)])
            return any(h["flag"] == EMERGENCY for h in hits)
        except Exception:
            log.exception("red-flag check failed")
            return True

    def _emergency(self, text: str) -> bool:
        texts = [preprocess(text)]
        try:
            shared = self._shared(text, texts)
            if shared is not None:
                return bool(shared.flag)
            hits = emergency_check({}, texts)
            return any(h["flag"] == EMERGENCY for h in hits)
        except Exception:                                  # cannot tell -> let the agent (and its safety net) decide
            log.exception("emergency pre-check failed; sending the message to the agent")
            return True

    # ================================================================== symptom path (agent)

    def _bangla(self, text, session_id) -> dict:
        """Bangla is recognised, not yet understood: say so in Bangla, Hindi and English, with the emergency
        number, and ask for Hindi or English. The conversation keeps whatever language it had."""
        ing = self._ingest_only(text, session_id)
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": "bangla"}
        return self._reply(base, FOLLOW_UP, V.BANGLA_NOTE, question=V.BANGLA_NOTE)

    # ================================================================== direct search
    def _search_turn(self, text, session_id, filters: dict, ctx, st: dict, page: int = 1) -> dict:
        ing = self._ingest_only(text, session_id)
        sid, lang = ing.session_id, ing.language.language
        if ing.session_id != session_id:
            st = self.load(sid)
        st["language"], st["turn"] = lang, ing.turn
        hi = _hinglish(lang)
        base = {"session_id": sid, "turn": ing.turn, "language": lang}

        place_city = filters.get("place") if not ctx["located"] else None     # "near Purnia" without a location
        if filters.get("near") and not ctx["located"] and not filters.get("locationAsked") and not place_city:
            st["pending"] = {k: v for k, v in filters.items() if k != "locationAsked"}
            self._save(sid, st)
            q = hi("Where are you? Share your location, type your area or pincode, or use your city.",
                   "Aap kahan hain? Location share karein, area ya pincode likhein, ya apna city use karein.",
                   "आप कहाँ हैं? अपनी लोकेशन शेयर करें, इलाका या पिनकोड लिखें, या अपना शहर चुनें।")
            return self._reply(base, FOLLOW_UP, q, question=q,
                               clarify={"kind": "location", "question": q, "options": V.location_options(hi)})
        st.pop("pending", None)
        city = place_city or ctx["city"]

        t0 = time.perf_counter()
        try:
            if filters.get("hospitalQuery") and not filters.get("hospitalId"):           # "Sadar Hospital Purnia"
                picked = self._resolve_hospital(filters, ctx, st, sid, base, hi)
                if isinstance(picked, dict) and "kind" in picked:
                    return picked                                   # asked which hospital, or none found
        except DoctorLookupError:
            return self._lookup_failed(sid, st, base, hi, filters)

        if filters.get("query"):                     # "kunal ghose" -> "kunal ghosh", shown and searched as listed
            filters = dict(filters, query=self.directory.fix_name(filters["query"]))
        used_city = bool(filters.get("near") and not ctx["located"] and not place_city)
        per = self.directory.limit
        fee_sort = filters.get("sort") == "fee"
        own_hospital = filters.get("hospitalId") or ctx["hospital_id"]
        args = dict(specialty_key=filters.get("specialtyKey"), query=filters.get("query"),
                    degree=filters.get("degree"),
                    # a doctor's name is searched in every city unless the patient named a place with it
                    city=None if filters.get("hospitalId") or (filters.get("query") and not place_city) else city,
                    hospital_id=own_hospital, only_at_hospital=bool(filters.get("hospitalId")),
                    gender=filters.get("gender"), available_today=bool(filters.get("availableToday")),
                    # fees are sorted here, over the first 10; everything else pages with 1HMS's offset
                    limit=min(10, per * page) if fee_sort else per,
                    offset=0 if fee_sort else per * (page - 1))
        if ctx["located"]:
            near = bool(filters.get("near"))
            args.update(lat=ctx["lat"], lng=ctx["lng"],
                        max_distance_km=(filters.get("radiusKm") or NEARBY_DEFAULT_KM) if near else WIDE_RADIUS_KM)
            if near:
                args["sort"] = "distance"
        if filters.get("sort") == "experience":
            args["sort"] = "experience"

        fallback_used = widened = False
        try:
            found = self.directory.search(**args)
            if (found is None or (not found["doctors"] and not found["total"])) and args.get("city") \
                    and not ctx["located"] and page == 1:
                wide = self.directory.search(**dict(args, city=None))     # nobody in that city: look in other towns
                if wide and wide["doctors"]:
                    found, widened = wide, True
                    ctx["lead"] = V.widened_note(hi, args["city"], _spec_name(filters.get("specialtyKey")))
            if (found is None or (not found["doctors"] and not found["total"])) and filters.get("specialtyKey") \
                    and not filters.get("query") and not filters.get("hospitalId") and page == 1:
                if filters["specialtyKey"] != "general-medicine":                   # nobody of that specialty
                    gp = self.directory.search(**dict(args, specialty_key="general-medicine"))
                    if gp and gp["doctors"]:
                        found, fallback_used = gp, True
        except DoctorLookupError:
            return self._lookup_failed(sid, st, base, hi, filters)
        ms = round((time.perf_counter() - t0) * 1000, 1)

        if found is None and not any(filters.get(k) for k in ("specialtyKey", "query", "degree", "hospitalId", "gender")):
            st["pending"] = {k: v for k, v in filters.items() if k != "locationAsked"}   # nothing to search by yet
            self._save(sid, st)
            q = hi("To find doctors near you, share your location, or tell me what kind of doctor you need.",
                   "Paas ke doctor dhoondhne ke liye location share karein, ya batayein kis tarah ka doctor chahiye.",
                   "पास के डॉक्टर ढूँढने के लिए अपनी लोकेशन शेयर करें, या बताएँ कि किस तरह का डॉक्टर चाहिए।")
            return self._reply(base, FOLLOW_UP, q, question=q,
                               clarify={"kind": "location", "question": q, "options": V.location_options(hi)})

        items = list(found["doctors"]) if found else []
        total = found["total"] if found else 0
        if fee_sort:
            items = sorted(items, key=_fee)[per * (page - 1): per * page]     # stable: equal fees keep 1HMS order
        if page > 1 and not items:                                            # nothing left to show
            st["page"] = page - 1
            self._save(sid, st)
            msg = hi("There are no more doctors in this list. You can browse all doctors on Doctor Dekho, "
                     "or change the filters.",
                     "Is list mein aur doctor nahi hain. Doctor Dekho par saare doctor dekh sakte hain, ya filter badlein.",
                     "इस लिस्ट में और डॉक्टर नहीं हैं। Doctor Dekho पर सभी डॉक्टर देख सकते हैं, या फ़िल्टर बदलें।")
            return self._reply(base, DOCTOR_LIST, msg, clarify={"kind": "no_more", "question": msg, "options": []})
        items = items[:per]
        filters = dict(filters, city=city) if place_city and not widened else filters
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
            clarify = self._confused(st, clarify, hi)
        else:
            st["confused"] = 0
            msg = self._list_message(items, total, filters, hi, used_city=used_city and bool(ctx["city"]) and not widened,
                                     fallback_used=fallback_used, page=page)
            which = self._which_options(items, filters)
            if which:                                   # several doctors share the name: tell them apart
                st["which"] = which
                q = hi("Which one do you mean?", "Aap kaun se doctor ki baat kar rahe hain?",
                       "आप किस डॉक्टर की बात कर रहे हैं?")
                clarify = {"kind": "which_doctor", "question": q,
                           "options": [{"id": o["id"], "label": o["label"]} for o in which]}
                msg = msg + " " + q
        self._save(sid, st)
        out = self._reply(base, DOCTOR_LIST, msg, question=clarify["question"] if clarify else None, clarify=clarify)
        out["doctors"] = block
        return out

    def _availability(self, text, session_id, ask: DoctorAsk, st_in, ctx) -> dict:
        """ "Is Dr Sharma available Sunday?" -> that day's shifts, or the next working day, from 1HMS. Answers the
        question without pushing a booking; offers it (vision: "Shall I book it?" and accept "no, just checking")."""
        ing, st, base, hi = self._ingest(text, session_id)
        doctor = None
        try:
            if ask.name_query:
                found = self.directory.search(query=ask.name_query, city=ctx["city"], limit=1)
                doctor = found["doctors"][0] if found and found["doctors"] else None
            else:
                sel = (st.get("selected") or {}).get("doctorId")
                doctor = (st.get("shown") or {}).get(sel) if sel else None
            if doctor is None:
                self._save(ing.session_id, st)
                who = f"'{ask.name_query}'"
                msg = (hi(f"I couldn't find a doctor named {who}. Please tell me the doctor's name.",
                          f"{who} naam ke doctor nahi mile. Doctor ka naam batayein.",
                          f"{who} नाम के डॉक्टर नहीं मिले। डॉक्टर का नाम बताएँ।") if ask.name_query else
                       hi("Which doctor do you mean? Please tell me the name.",
                          "Kaun se doctor ke baare mein pooch rahe hain? Naam batayein.",
                          "आप किस डॉक्टर के बारे में पूछ रहे हैं? नाम बताएँ।"))
                return self._reply(base, FOLLOW_UP, msg, question=msg)
            target, label = _target_date(ask.when)
            sched = self.directory.availability(doctor["doctorId"], from_date=target, days=7 if ask.when == "next" else 1)
            nxt = None
            if sched and ask.when != "next" and not any(d["isAvailable"] for d in sched.get("days") or []):
                after = (date.fromisoformat(target) + timedelta(days=1)).isoformat()
                later = self.directory.availability(doctor["doctorId"], from_date=after, days=7)
                nxt = (later or {}).get("nextAvailableDate")
        except DoctorLookupError:
            return self._lookup_failed(ing.session_id, st, base, hi, {"query": ask.name_query})
        self._remember(st, [doctor])
        self._save(ing.session_id, st)
        name = doctor.get("name") or "The doctor"
        if not sched:
            msg = hi(f"{name} is no longer listed on Doctor Dekho.", f"{name} ab Doctor Dekho par available nahi hain.",
                     f"{name} अब Doctor Dekho पर उपलब्ध नहीं हैं।")
            return self._reply(base, INFO, msg)
        day = next((d for d in sched.get("days") or [] if d["isAvailable"]), None) if ask.when == "next" else \
            (sched.get("days") or [None])[0]
        if day and day.get("isAvailable"):
            hours = ", ".join(f"{sh.get('name') or ''} {sh.get('start')}-{sh.get('end')}".strip()
                              for sh in day.get("shifts") or [] if sh.get("start"))
            when = _day_label(day["date"], hi)
            h = f" ({hours})" if hours else ""
            on = "" if day["date"] in _NEAR_DAYS() else "on "
            msg = hi(f"Yes, {name} works {on}{when}{h}. The hospital confirms the exact time. Shall I book an appointment?",
                     f"Haan, {name} {when} baithte hain{h}. Sahi samay hospital confirm karega. Kya main appointment book karoon?",
                     f"हाँ, {name} {when} बैठते हैं{h}। सही समय अस्पताल कन्फ़र्म करेगा। क्या मैं अपॉइंटमेंट बुक करूँ?")
        else:
            when = _day_label(target, hi) if ask.when != "next" else hi("the next 7 days", "agle 7 din", "अगले 7 दिन")
            nxt_txt = _day_label(nxt, hi) if nxt else None
            on = "" if ask.when == "next" or target in _NEAR_DAYS() else "on "
            msg = hi(f"{name} is not available {on}{when}." + (f" Next working day: {nxt_txt}." if nxt_txt else ""),
                     f"{name} {when} available nahi hain." + (f" Agla din: {nxt_txt}." if nxt_txt else ""),
                     f"{name} {when} उपलब्ध नहीं हैं।" + (f" अगला दिन: {nxt_txt}।" if nxt_txt else ""))
            msg += hi(" Shall I book an appointment?", " Kya main appointment book karoon?", " क्या मैं अपॉइंटमेंट बुक करूँ?")
        out = self._reply(base, INFO, msg, question=msg, clarify={"kind": "book_offer", "question": msg, "options": [
            {"id": doctor["doctorId"], "label": hi("Yes, book", "Haan, book karein", "हाँ, बुक करें")},
            {"id": "no", "label": hi("No, just checking", "Nahi, bas jaanna tha", "नहीं, बस जानना था")}]})
        out["doctor"] = doctor
        out["availability"] = {"doctorId": doctor["doctorId"], "days": sched.get("days"),
                               "nextAvailableDate": nxt or sched.get("nextAvailableDate"), "note": sched.get("note")}
        return out

    def _lookup_failed(self, sid, st, base, hi, filters) -> dict:
        self._save(sid, st)
        msg = hi("I couldn't load doctors right now. Please try again in a moment, or open the Doctor Dekho directory.",
                 "Abhi doctors load nahi ho paaye. Thodi der mein dobara try karein, ya Doctor Dekho directory dekhein.",
                 "अभी डॉक्टरों की जानकारी लोड नहीं हो पाई। थोड़ी देर में फिर कोशिश करें, या Doctor Dekho डायरेक्टरी देखें।")
        out = self._reply(base, DOCTOR_LIST, msg, degraded=["doctor_lookup"])
        out["searchFilters"] = _public_filters(filters)
        return out

    def _resolve_hospital(self, filters: dict, ctx, st, sid, base, hi):
        """filters.hospitalQuery -> filters.hospitalId / hospitalName (in place), or a reply: none found, or
        "which one?" when two hospitals match about equally (e.g. two "Sadar Hospital"s, different cities)."""
        query = filters["hospitalQuery"]
        hospitals = self.directory.search_hospitals(query=query, city=filters.get("place"),
                                                    lat=ctx["lat"], lng=ctx["lng"], limit=4)
        if not hospitals and filters.get("place"):
            hospitals = self.directory.search_hospitals(query=query, lat=ctx["lat"], lng=ctx["lng"], limit=4)
        if not hospitals:
            st["last"] = {k: v for k, v in filters.items() if k != "locationAsked"}
            st["confused"] = int(st.get("confused") or 0)
            msg = hi(f"I couldn't find '{query}' on Doctor Dekho. Shall I search by a doctor or specialty instead?",
                     f"'{query}' Doctor Dekho par nahi mila. Kya kisi doctor ya specialty se dhoondhein?",
                     f"'{query}' Doctor Dekho पर नहीं मिला। क्या किसी डॉक्टर या स्पेशलिटी से ढूँढें?")
            clarify = self._confused(st, {"kind": "no_match", "question": msg,
                                          "options": list(V.ask_kind_options(hi)[1:3])}, hi)
            self._save(sid, st)
            out = self._reply(base, DOCTOR_LIST, msg, clarify=clarify)
            out["doctors"] = {"mode": ["hospital"], "filters": {"hospital": query}, "items": [], "total": 0,
                              "fallbackUsed": False, "source": SOURCE, "doctors": [], "specialist": None,
                              "specialtyId": None, "lookupMs": None, "page": 1, "hasMore": False}
            return out
        scores = [h.get("nameMatchScore") or 0 for h in hospitals]
        close = [h for h, sc in zip(hospitals, scores) if scores[0] - sc <= 0.05]
        if len(close) > 1:
            st["which_hospital"] = [{"id": h["hospitalId"], "name": h.get("name"),
                                     "label": ", ".join(x for x in (h.get("name"), h.get("city")) if x),
                                     "keys": sorted(set(preprocess(f"{h.get('name') or ''} {h.get('city') or ''}").split())
                                                    - {"hospital", "the", "clinic"})} for h in close[:4]]
            st["last"] = {k: v for k, v in filters.items() if k != "locationAsked"}
            self._save(sid, st)
            q = hi("Which hospital do you mean?", "Kaun sa hospital?", "आप कौन-से अस्पताल की बात कर रहे हैं?")
            return self._reply(base, FOLLOW_UP, q, question=q, clarify={
                "kind": "which_hospital", "question": q,
                "options": [{"id": o["id"], "label": o["label"]} for o in st["which_hospital"]]})
        filters["hospitalId"], filters["hospitalName"] = hospitals[0]["hospitalId"], hospitals[0].get("name")
        return None

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
                msg = hi("This doctor is no longer available on Doctor Dekho. Please choose another one.",
                         "Ye doctor ab Doctor Dekho par available nahi hain. Kripya list se koi aur doctor chunein.",
                         "यह डॉक्टर अब Doctor Dekho पर उपलब्ध नहीं हैं। कृपया लिस्ट से कोई और डॉक्टर चुनें।")
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
        msg = hi(f"You chose {name}{where}. Tap 'Book a slot' to pick a time.",
                 f"Aapne {name}{where} ko chuna hai. Samay chunne ke liye 'Book a slot' dabayein.",
                 f"आपने {name}{where} को चुना है। समय चुनने के लिए 'Book a slot' दबाएँ।")
        clarify = None
        if self.booking.enabled and not degraded:                     # Vita books it (vision Phase 2)
            started = self.booking.start(st, doctor, hi, ctx)
            clarify = started["clarify"]
            chose = hi(f"You chose {name}{where}.", f"Aapne {name}{where} ko chuna hai.", f"आपने {name}{where} को चुना है।")
            msg = f"{chose} {started['message']}" if st.get("booking") else started["message"]
            if clarify:
                clarify["question"] = msg
            self._save(session_id, st)
        out = self._reply(base, DOCTOR_SELECTED, msg, question=msg if clarify else None, clarify=clarify, degraded=degraded)
        out["doctor"] = doctor
        out["selectedDoctorId"] = doctor_id
        return out

    def _visit_turn(self, text, session_id, st, ctx) -> dict | None:
        """Directions, "I've reached", check in, token status. Only for a known visit, except "I've reached" and "check in", which
        also look up a signed-in patient's visit. The emergency check runs first."""
        what = visit_wants(text)
        located_reply = ((st.get("visit") or {}).get("awaiting") == "checkin" and ctx["located"]
                         and (text == "location" or _LOCATION_ONLY.match(text or "")))
        known = bool((st.get("last_booking") or {}).get("appointmentId") or st.get("visit") or ctx.get("visit_appointment_id"))
        if not (located_reply or (what in ("arrival", "checkin")) or (what and known)):
            return None
        if what in ("directions", "status") and not known:
            return None
        if self._emergency(text):
            return None
        ing = self._ingest_only(text, session_id)
        lang = st.get("language") or ing.language.language
        v = Voice(lang)
        if located_reply and what is None:
            res = self.visit.location_followup(st, ctx, v)
        else:
            res = self.visit.turn(what, st, ctx, v)
            if what == "arrival" and ctx.get("channel") != "whatsapp" and (st.get("journey") or {}).get("appointmentId") and (st.get("visit") or {}).get("appointmentId"):
                self.journey.arrived(st["visit"]["appointmentId"], ctx)       # the hospital's "on the way" becomes "arrived"
                st["journey"] = {"appointmentId": st["visit"]["appointmentId"], "arrived": True}
        if res is None:
            return None
        st["language"], st["turn"] = lang, ing.turn
        self._save(ing.session_id, st)
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": lang}
        out = self._reply(base, res["kind"], res["message"], clarify=res.get("clarify"), degraded=res.get("degraded"))
        if res.get("visit"):
            out["visit"] = res["visit"]
            out["visit_step"] = res.get("visit_step")
        return out

    def _journey_turn(self, text, session_id, st, ctx) -> dict | None:
        """Home to the clinic: the plan right after booking (how, when, may the hospital see the ETA), then "nikal raha hoon",
        "kitni der", "late ho jaunga". None: not about the journey. The emergency check runs first."""
        j = st.get("journey") or {}
        planning = bool(j.get("step"))
        consent = None
        if j.get("awaiting") == "late_consent":
            n = intent_text(text or "")
            if text == "late:yes" or _JOURNEY_YES.match(n):
                consent = True
            elif text == "late:no" or _JOURNEY_NO.match(n):
                consent = False
            else:
                j.pop("awaiting", None)
        what = None if planning or consent is not None else journey_wants(text)
        if not (planning or consent is not None or what):
            return None
        if self._emergency(text):
            if planning:
                j["step"] = None
                self._save(session_id, st)
            return None
        appt = None
        if not planning:
            appt = j.get("appointmentId") or self.visit.appointment_id(st, ctx)
            if not appt:
                return None
        ing = self._ingest_only(text, session_id)
        lang = st.get("language") or ing.language.language
        v = Voice(lang)
        if planning:
            res = self.journey.plan_turn(text, st, ctx, v)
            if res is None:                                                    # a new ask: planning is over
                j["step"] = None
                self._save(session_id, st)
                return None
        else:
            try:
                visit = self.booking_client_visit(appt, ctx)
            except BookingError as exc:
                log.warning("journey visit lookup failed: %s", exc)
                return None
            status = (visit.get("journey") or {}).get("status")
            if what == "eta" and not (j.get("onTheWay") or status in ("on_the_way", "late")):
                return None                                                    # "kitna time" at the hospital: the queue
            if visit.get("cancelled") or visit.get("completed"):
                return None
            if consent is not None or what == "late":
                res = self.journey.late(appt, visit, st, ctx, v, text, consent=consent)
            elif what == "leaving":
                res = self.journey.leaving(appt, visit, st, ctx, v)
            else:
                res = self.journey.eta(appt, visit, st, ctx, v)
        st["language"], st["turn"] = lang, ing.turn
        self._save(ing.session_id, st)
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": lang}
        clarify = res.get("clarify")
        out = self._reply(base, res["kind"], res["message"], question=(clarify or {}).get("question"), clarify=clarify)
        for k in ("visit", "visit_step", "journey"):
            if res.get(k) is not None:
                out[k] = res[k]
        return out

    def booking_client_visit(self, appt: str, ctx: dict) -> dict:
        return self.journey.client.visit(appt, client_ip=ctx.get("client_ip"))

    def _emergency_followup(self, text, session_id, st, ctx) -> dict | None:
        """Right after an emergency: a shared location gets the nearest places; "share my number" sends it to them.
        Anything else goes the normal way (and the emergency check runs on it again)."""
        sharing = self.emergency.wants_share(text, st)
        located = (text == "location" or _LOCATION_ONLY.match(text or "")) and ctx["located"]
        if not (sharing or located):
            return None
        ing = self._ingest_only(text, session_id)
        lang = st.get("language") or ing.language.language
        v = Voice(lang)
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": lang}
        if sharing:
            res = self.emergency.share(st, ctx, v)
            self._save(ing.session_id, st)
            return self._reply(base, res["kind"], res["message"])
        out = self._reply(base, "emergency", V.emergency(v, []))
        if not self.emergency.location_followup(out, st, ctx, v):
            return None
        self._save(ing.session_id, st)
        return out

    # ================================================================== after an emergency
    def _emergency_done(self, sid: str | None, signs: list[str]) -> None:
        """The emergency was said. That triage is over: the next message is judged on its own, so "pet mein dard" or
        "ambulance aa gayi" do not repeat the alarm for the breathlessness said before (a new red flag still raises it)."""
        if not sid:
            return
        try:
            self.pipeline.sessions.clear_history(sid)
        except Exception:
            log.exception("triage not reset after an emergency")
        est = self.load(sid)
        est["after_emergency"] = {"signs": signs, "at": int(time.time())}
        est.pop("asked_describe", None)
        self._save(sid, est)

    def _after_emergency(self, text, session_id, st, ctx) -> dict | None:
        """The message after an emergency: help arrived, feeling better, cannot get through, thanks. A new red flag
        goes the normal way (and is an emergency again); anything else too, with the number said once first."""
        info = st.get("after_emergency") or {}
        if time.time() - info.get("at", 0) > AFTER_EMERGENCY_S:
            st.pop("after_emergency", None)
            self._save(session_id, st)
            return None
        if self._emergency(text):
            return None
        norm = intent_text(text)
        what = next((k for k, rx in _AFTER_EMERGENCY if rx.search(norm)), None)
        ing = self._ingest_only(text, session_id)
        lang = ing.language.language
        v = Voice(lang)
        signs = V.symptom_words(v, info.get("signs") or [])
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": lang}
        st["language"], st["turn"] = lang, ing.turn
        if what is None:
            if not info.get("reminded"):
                info["reminded"] = True
                ctx["reminder"] = V.after_emergency_reminder(v, signs)
            self._save(ing.session_id, st)
            return None
        if what == "cant_call":
            msg = V.after_emergency(v, what, signs)
            out = self._reply(base, "emergency", msg)
            if not self.emergency.location_followup(out, st, ctx, v):  # a shared location still gets the nearest places
                out["kind"] = INFO          # screens show an emergency's numbers, not its text: this text must be read
            self._save(ing.session_id, st)
            return out
        self._save(ing.session_id, st)                     # kept: "ambulance aa gayi" then "ab theek hoon" both fit
        clarify = None
        if what == "better":
            q = V.after_emergency(v, what, signs)
            clarify = {"kind": "ask_kind", "question": q, "options": V.ask_kind_options(v)}
        return self._reply(base, INFO, V.after_emergency(v, what, signs), clarify=clarify)

    # ================================================================== everyday questions

    def _warn_ask(self, text, session_id, kind: str) -> dict:
        ing, st, base, v = self._ingest(text, session_id)
        st["warn"] = {"kind": kind, "at": int(time.time())}
        self._save(ing.session_id, st)
        q = everyday.warn_question(kind, v)
        return self._reply(base, FOLLOW_UP, q, question=q, clarify={"kind": "warning_signs", "question": q,
                                                                   "options": everyday.warn_options(v)})

    def _warn_emergency(self, text, session_id, kind: str) -> dict:
        ing, st, base, v = self._ingest(text, session_id)
        st.pop("warn", None)
        sign = everyday.WARN_SIGN[kind]
        self._save(ing.session_id, st)
        out = self._reply(base, "emergency", V.emergency(v, [sign]))
        out["card"] = {"kind": "emergency", "emergency_reasons": [sign.capitalize()],
                       "emergency_signs": [V.symptom_words(v, [sign])]}
        self._emergency_done(ing.session_id, [sign])
        return out

    def _warn_answer(self, text, session_id, st, ctx, norm) -> dict | None:
        warn = st.pop("warn", None) or {}
        self._save(session_id, st)
        yes = everyday.warn_answer(norm)
        if yes is None or time.time() - warn.get("at", 0) > 3600:
            return None                                              # something else: the usual flow
        kind = warn.get("kind") or "head"
        if yes:                                                      # a warning sign: emergency, as for a red flag
            return self._warn_emergency(text, session_id, kind)
        v = Voice(st.get("language") or "english")
        ctx["lead"] = everyday.warn_no(kind, v)                       # no warning sign: see a doctor, here are some
        filters = {"specialtyKey": everyday.WARN_SPECIALTY.get(kind, "general-medicine"), "modes": ["specialty"]}
        return self._search_turn(text, session_id, filters, ctx, self.load(session_id))

    def _side_question(self, text, session_id, st, ctx) -> dict | None:
        """While booking: "number kyun chahiye?", "fees kitni hai?", "kya?" get their answer, then the booking question
        again. Anything else goes to the booking as before."""
        norm = intent_text(text)
        move = everyday.detect(norm)
        prev = st.get("last_reply")
        if move not in ("privacy", "fees", "free", "about", "what", "online", "insurance") or not prev:
            return None
        if availability_when(norm):                                   # "kal free hain?" is about the day, not the price
            return None
        if self._emergency(text):
            return None
        ing = self._ingest_only(text, session_id)
        lang = st.get("language") or ing.language.language
        v = Voice(lang)
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": lang}
        if move == "what":
            answer = everyday.again(v)
        else:
            d = (st.get("booking") or {}).get("doctor") or {}
            answer = everyday.reply(move, v, fee=d.get("fee"), doctor=d.get("name"))
        ctx["repeated"] = True
        msg = f"{answer}\n\n{prev['message']}" if move != "what" else f"{answer} {prev['message']}"
        clarify = dict(prev["clarify"], question=msg) if prev.get("clarify") else None
        return self._reply(base, prev.get("kind") or FOLLOW_UP, msg, question=msg, clarify=clarify)

    def _said_symptom(self, text: str) -> bool:
        """A symptom the triage lexicon recognises ("bukhar", "pet dard"). Stricter than DoctorAsk.symptomish, which also
        fires on everyday words that happen to sit inside the long symptom phrases ("kar", "aap", "karna")."""
        corrector = getattr(self, "_corrector", None)
        if corrector is None:
            from ..spelling import SpellingCorrector
            corrector = self._corrector = SpellingCorrector(self.pipeline.lex)     # no review-queue side effects
        try:
            return bool(corrector.correct(preprocess(text)).symptoms)
        except Exception:
            return False

    def _lang_of(self, text, session_id, st) -> str:
        return st.get("language") or self.pipeline.detector.detect(text).language or "english"

    def _booking_turn(self, text, session_id, st, ctx) -> dict | None:
        """One message while booking. None: it was not about booking (a new ask), and booking has ended."""
        before = st.get("language")
        ing = self._ingest_only(text, session_id)
        lang = ing.language.language
        if before and lang != before and self.booking._picked(st["booking"], text) is not None:
            # a tapped chip ("Tomorrow (Mon 26 Oct)", "Morning 09:00-13:00") is not the patient changing language
            lang = before
            self.pipeline.sessions.set_context(ing.session_id, language=before)
        st["language"], st["turn"] = lang, ing.turn
        v = Voice(lang)
        res = self.booking.turn(st, text, v, ctx)
        if res is None:
            st.pop("booking", None)
            self._save(session_id, st)
            return None
        if (res["kind"] == "booked" and self.journey.enabled and ctx.get("channel") != "whatsapp"
                and (res.get("booking") or {}).get("appointmentId")):
            plan = self.journey.start(st, res["booking"], v)                  # "how will you get there?"
            res = {**res, "message": f"{res['message']}\n\n{plan['message']}", "clarify": plan["clarify"]}
        self._save(session_id, st)
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": lang}
        clarify = res.get("clarify")
        out = self._reply(base, res["kind"], res["message"], question=clarify["question"] if clarify else None,
                          clarify=clarify)
        if res.get("booking"):
            out["booking"] = res["booking"]
        if res.get("patient_session"):
            out["patient_session"] = res["patient_session"]       # the channel signs the patient in; never logged
        return out

    def clear(self, sid: str) -> None:
        try:
            self.r.delete(self._key(sid))
        except Exception:
            log.exception("doctor session state not cleared")

    # ================================================================== conversation moves
    def _ingest_only(self, text, session_id):
        return self.pipeline.ingest(text, session_id, language_hint=getattr(self, "_hint", None))

    def _ingest(self, text, session_id):
        ing = self._ingest_only(text, session_id)
        st = self.load(ing.session_id)
        st["language"], st["turn"] = ing.language.language, ing.turn
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": ing.language.language}
        return ing, st, base, _hinglish(ing.language.language)

    def _handoff(self, text, session_id, reason: str) -> dict:
        """A person takes over. The channel (WhatsApp bot, Doctor Dekho) routes the chat on kind=handoff."""
        ing, st, base, hi = self._ingest(text, session_id)
        st["handoff"] = {"reason": reason, "at": int(time.time())}
        self._save(ing.session_id, st)
        msg = hi("I'm passing you to our support team. Someone will reply to you shortly.",
                 "Main aapko hamari support team se jod rahi hoon. Koi jaldi aapse baat karega.",
                 "मैं आपको हमारी सपोर्ट टीम से जोड़ रही हूँ। कोई जल्द ही आपसे बात करेगा।")
        out = self._reply(base, HANDOFF, msg)
        out["handoff"] = {"reason": reason}
        return out



    # ================================================================== helpers
    def _ask_kind(self, text, session_id, st) -> dict:
        ing = self._ingest_only(text, session_id)
        hi = _hinglish(ing.language.language)
        q = hi("What kind of doctor do you need? Tell me your symptoms, a specialty, a doctor's name, a degree, "
               "or ask for doctors near you.",
               "Kis tarah ka doctor chahiye? Symptom batayein, specialty, doctor ka naam, degree, ya paas ke doctor.",
               "किस तरह का डॉक्टर चाहिए? लक्षण बताएँ, स्पेशलिटी, डॉक्टर का नाम, डिग्री, या पास के डॉक्टर।")
        st = self.load(ing.session_id)
        st["language"], st["turn"] = ing.language.language, ing.turn
        clarify = self._confused(st, {"kind": "ask_kind", "question": q, "options": V.ask_kind_options(hi)}, hi)
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
        st["page_ids"] = [d["doctorId"] for d in items if d.get("doctorId")]        # "pehla wala", "doosra wala"

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
    def _pick_option(norm: str, options: list[dict] | None) -> str | None:
        """A tapped chip (its label as text) or a typed answer that fits exactly one option."""
        if not options:
            return None
        for o in options:
            if preprocess(o.get("label") or "") == norm:
                return o["id"]
        words = {w for w in norm.split() if len(w) >= 3}
        hits = [o["id"] for o in options if words & set(o.get("keys") or [])]
        return hits[0] if len(hits) == 1 else None

    @staticmethod
    def _which_pick(norm: str, st: dict) -> str | None:
        """A typed answer to "which one?" ("salt lake wali", "howrah") that fits exactly one option."""
        options = st.get("which") or []
        words = {w for w in norm.split() if len(w) >= 3}
        hits = [o["id"] for o in options if words & set(o.get("keys") or [])]
        return hits[0] if len(hits) == 1 else None


    @staticmethod
    def _confused(st: dict, clarify: dict | None, voice: Voice | None = None) -> dict | None:
        """Two dead ends in a row -> also offer a person (vision: repeated confusion hands off)."""
        st["confused"] = int(st.get("confused") or 0) + 1
        if st["confused"] < 2:
            return clarify
        clarify = dict(clarify or {"kind": "no_match", "question": None, "options": []})
        clarify["options"] = [*(clarify.get("options") or []), V.human_option(voice or _EN)]
        return clarify

    @staticmethod
    def _list_message(items, total, filters, hi, *, used_city, fallback_used, page=1) -> str:
        n = len(items)
        spec = _spec_name(filters.get("specialtyKey"))
        plain = V.plain(hi, filters.get("specialtyKey")) if isinstance(hi, Voice) else \
            PLAIN.get(filters.get("specialtyKey") or "", (None, None))[1 if hi else 0]
        what = (f"{plain} " if plain else f"{spec} ") if spec and not fallback_used else ""
        bracket = f" ({spec})" if spec and plain and plain.lower() not in spec.lower() and not fallback_used else ""
        deg = f"{filters['degree']} " if filters.get("degree") else ""
        near = filters.get("near") and not used_city
        hosp, town = filters.get("hospitalName"), filters.get("city")
        lady = {"female": ("female ", "mahila "), "male": ("male ", "purush ")}.get(filters.get("gender") or "", ("", ""))
        sort = filters.get("sort")
        srt = {"fee": (", lowest fee first", " (sabse kam fees pehle)", " (सबसे कम फ़ीस पहले)"),
               "experience": (", most experienced first", " (sabse anubhavi pehle)", " (सबसे अनुभवी पहले)")
               }.get(sort or "", ("", "", ""))
        if isinstance(hi, Voice) and hi.code == "hi":
            head = (f"{spec} अभी उपलब्ध नहीं हैं; ये जनरल फ़िज़िशियन पहले देख सकते हैं। " if fallback_used and spec else "")
            where = (f" {hosp} में" if hosp else f" {town.title()} में" if town else "") + (" आपके पास" if near else "")
            where += " (आज उपलब्ध)" if filters.get("availableToday") else ""
            deg = {"female": "महिला ", "male": "पुरुष "}.get(filters.get("gender") or "", "") + deg
            city = " (लोकेशन नहीं मिली, आपके शहर के डॉक्टर दिखा रहे हैं)" if used_city else ""
            core = (f"'{filters['query']}' नाम से ये डॉक्टर मिले" if filters.get("query")
                    else f"ये {'और ' if page > 1 else ''}{deg}{what}डॉक्टर{bracket} मिले{where}")
            return f"{head}{core}{srt[2]}{city}। किसी एक को चुनने के लिए 'Choose' दबाएँ।"
        if hi:
            head = (f"{spec} abhi available nahi hain; ye General Physician pehle dekh sakte hain. "
                    if fallback_used and spec else "")
            where = (f" {hosp} mein" if hosp else f" {town.title()} mein" if town else "") + (" aapke paas" if near else "")
            where += " (aaj available)" if filters.get("availableToday") else ""
            deg = lady[1] + deg
            city = " (location nahi mili, aapke city ke doctor dikha rahe hain)" if used_city else ""
            core = (f"'{filters['query']}' naam se ye doctor mile" if filters.get("query")
                    else f"Ye {'aur ' if page > 1 else ''}{deg}{what}doctor{bracket} mile{where}")
            return f"{head}{core}{srt[1]}{city}. Kisi ek ko chunne ke liye 'Choose' dabayein."
        head = (f"No {spec} is available right now; a General Physician can see you first. "
                if fallback_used and spec else "")
        where = (f" at {hosp}" if hosp else f" in {town.title()}" if town else "") + (" near you" if near else "")
        where += ", available today" if filters.get("availableToday") else ""
        deg = lady[0] + deg
        city = " (no location shared, so these are in your city)" if used_city else ""
        core = (f"Doctors matching '{filters['query']}'" if filters.get("query")
                else f"Here {'is' if n == 1 else 'are'} {n} {'more ' if page > 1 else ''}{deg}{what}"
                     f"doctor{'s' if n != 1 else ''}{bracket}{where}")
        more = f" ({total} in all)" if total > n else ""
        return f"{head}{core}{more}{srt[0]}{city}. Tap Choose to pick one."

    @staticmethod
    def _empty_message(filters, hi) -> tuple[str, dict | None]:
        v = hi if isinstance(hi, Voice) else Voice("hinglish" if hi else "english")
        if filters.get("query"):
            q = filters["query"]
            msg = v(f"I couldn't find a doctor named '{q}'. Shall I search by specialty instead?",
                    f"'{q}' naam ke koi doctor nahi mile. Kya specialty se dhoondhein?",
                    f"'{q}' नाम के कोई डॉक्टर नहीं मिले। क्या स्पेशलिटी से ढूँढें?")
            return msg, {"kind": "no_match", "question": msg, "options": V.ask_kind_options(v)[1:2]}
        spec = _spec_name(filters.get("specialtyKey")) or ("matching" if not hi else "")
        lady = {"female": (" female", " mahila"), "male": (" male", " purush")}.get(filters.get("gender") or "", ("", ""))
        deg = (lady[1] if hi else lady[0]) + (f" {filters['degree']}" if filters.get("degree") else "")
        hosp, town = filters.get("hospitalName"), filters.get("city")
        if v.code == "hi":
            where = f" {hosp} में" if hosp else f" {town.title()} में" if town else " पास में" if filters.get("near") else ""
            today = " आज" if filters.get("availableToday") else ""
            hdeg = {"female": " महिला", "male": " पुरुष"}.get(filters.get("gender") or "", "") + \
                (f" {filters['degree']}" if filters.get("degree") else "")
            msg = f"अभी{where}{today} कोई{hdeg} {spec} डॉक्टर Doctor Dekho पर नहीं मिले। दूसरा इलाका या फ़िल्टर आज़माएँ।"
        elif hi:
            where = f" {hosp} mein" if hosp else f" {town.title()} mein" if town else " paas mein" if filters.get("near") else ""
            today = " aaj" if filters.get("availableToday") else ""
            msg = f"Abhi{where}{today} koi{deg} {spec} doctor Doctor Dekho par nahi mile. Doosra area ya filter try karein."
        else:
            where = f" at {hosp}" if hosp else f" in {town.title()}" if town else " near you" if filters.get("near") else ""
            today = " available today" if filters.get("availableToday") else ""
            msg = f"No{deg} {spec} doctors{today} are listed on Doctor Dekho{where} right now. Try another area or filter."
        return " ".join(msg.split()), None


# ---------------------------------------------------------------------------------------------- pure helpers
def _wants_names(norm: str) -> bool:
    """Fetch the name index (cached 15 min) only when the message might name a doctor without 'dr': a meeting cue
    ("anita sen se milna hai"), or a short message that could be just a name ("neha sharma", "anita sen ji")."""
    from ..doctor_intent import _NAME_CUE
    words = norm.split()
    # "ghosh doctor", "sharma dr": a surname with the doctor word is a name too
    return bool(_NAME_CUE.search(norm)) or (1 <= len(words) <= 4 and not all(w in DOCTOR_WORDS for w in words))


def _bare_doctor(norm: str, ask: DoctorAsk) -> bool:
    """'doctor chahiye', 'koi achha doctor batao': asks for a doctor but says nothing about which."""
    return not ask.modes and not ask.refine and any(w in DOCTOR_WORDS for w in norm.split())


def _today_ist() -> date:
    return (datetime.now(timezone.utc) + timedelta(minutes=330)).date()


def _NEAR_DAYS() -> tuple[str, str]:  # noqa: N802
    """Today and tomorrow (IST) as ISO dates: said without "on" ("works tomorrow", not "on tomorrow")."""
    today = _today_ist()
    return today.isoformat(), (today + timedelta(days=1)).isoformat()


def _target_date(when: str | None) -> tuple[str, str]:
    """"+1" -> tomorrow, "dow:6" -> the coming Sunday (today if it is Sunday), "next" -> today (IST)."""
    today = _today_ist()
    if when and when.startswith("+"):
        d = today + timedelta(days=int(when[1:]))
    elif when and when.startswith("dow:"):
        d = today + timedelta(days=(int(when[4:]) - today.weekday()) % 7)
    else:
        d = today
    return d.isoformat(), _day_label(d.isoformat())


def _day_label(iso: str | None, hi=False) -> str:
    if not iso:
        return ""
    v = hi if isinstance(hi, Voice) else Voice("hinglish" if hi else "english")
    d = date.fromisoformat(iso)
    today = _today_ist()
    if d == today:
        return V.day_label(v, "today")
    if d == today + timedelta(days=1):
        return V.day_label(v, "tomorrow")
    return d.strftime("%a %d %b")


def _spec_name(key: str | None) -> str | None:
    return SPECIALTY_NAMES.get(key) if key else None


def _filters(ask: DoctorAsk) -> dict:
    return {"specialtyKey": ask.specialty_key, "query": ask.name_query, "degree": ask.degree, "near": ask.near,
            "radiusKm": ask.radius_km, "sort": ask.sort, "gender": ask.gender, "availableToday": ask.available_today,
            "hospitalQuery": ask.hospital_query, "place": ask.place,
            "modes": [m for m in ask.modes if m != "symptom"]}


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
    if ask.gender:
        f["gender"] = ask.gender
        modes.add("gender")
    if ask.available_today:
        f["availableToday"] = True
        modes.add("today")
    if ask.place:
        f["place"] = ask.place
        modes.add("place")
    if ask.hospital_query:
        f["hospitalQuery"], f["hospitalId"], f["hospitalName"] = ask.hospital_query, None, None
        modes.add("hospital")
    if ask.sort:
        f["sort"] = ask.sort
    f["modes"] = sorted(modes - {"symptom"}) or sorted(modes)
    return f


def _public_filters(filters: dict, ctx: dict | None = None, specialty_id: str | None = None) -> dict:
    """doctors.filters: the filters actually applied (VITA-7). No coordinates are echoed back."""
    out: dict[str, Any] = {}
    if specialty_id:
        out["specialtyId"] = specialty_id
    for k in ("query", "degree", "sort", "gender"):
        if filters.get(k):
            out[k] = filters[k]
    if filters.get("availableToday"):
        out["availableToday"] = True
    if filters.get("hospitalId"):
        out["hospital"] = {"hospitalId": filters["hospitalId"], "name": filters.get("hospitalName")}
    if filters.get("near"):
        out["near"] = True
        out["radiusKm"] = filters.get("radiusKm") or NEARBY_DEFAULT_KM
    if ctx:
        if ctx.get("located"):
            out["location"] = "shared"
        elif filters.get("city") or ctx.get("city"):
            out["city"] = filters.get("city") or ctx["city"]
        if ctx.get("hospital_id"):
            out["hospitalId"] = ctx["hospital_id"]
    return out
