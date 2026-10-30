"""The tools Vita's LLM can call. Each one runs Vita's existing, tested code; the LLM only decides which and with
what. Tools that answer the patient themselves (vita_flow, raise_emergency, a location question, an empty list)
end the turn with their own reviewed wording ("final"); the others return facts for the LLM to talk about and
leave their screen parts (doctor list, specialist card) in turn.ui."""
from __future__ import annotations

import json
import logging
import time

from .. import voice as V
from ..application.doctor_chat import DOCTOR_LIST, FOLLOW_UP, DoctorChat
from ..doctor_intent import parse
from ..doctors import FALLBACK_SPECIALIST, DoctorLookupError, specialty_key
from ..domain.models import Query
from ..route_bridge import age_group_of, to_routing
from ..textnorm import preprocess
from ..voice import Voice

log = logging.getLogger("hms_router.vita_tools")


def _fn(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(required), "additionalProperties": False}}}


TOOLS = [
    _fn("assess_symptoms", "Choose the right kind of doctor for the patient's symptoms and show bookable doctors. "
        "Also checks for emergencies.", {
            "symptoms": {"type": "string", "description": "The complaint in plain English, e.g. 'fever for 3 days, "
                                                          "body ache'. Include duration if known."},
            "age_group": {"type": "string", "enum": ["child", "adult", "unknown"],
                          "description": "child = under 14, only if the patient said so"},
            "pregnant": {"type": "boolean", "description": "true only if the patient said she is pregnant"}},
        ["symptoms"]),
    _fn("find_doctors", "Search bookable doctors by specialty, name, degree, hospital, place or distance.", {
        "specialty": {"type": "string", "description": "e.g. Dermatologist, Gynaecologist, Dentist, Paediatrician, "
                                                       "General Physician"},
        "doctor_name": {"type": "string"}, "degree": {"type": "string", "description": "MBBS, MD, MS, DM, BDS ..."},
        "hospital": {"type": "string", "description": "hospital or clinic name the patient said"},
        "place": {"type": "string", "description": "a city or area named by the patient"},
        "near_me": {"type": "boolean"}, "gender": {"type": "string", "enum": ["female", "male"]},
        "available_today": {"type": "boolean"}, "sort": {"type": "string", "enum": ["fee", "experience"]},
        "more": {"type": "boolean", "description": "the next page of the last list"}}),
    _fn("vita_flow", "Hand this message to Vita's booking and visit system: choosing a doctor, booking, OTP, a "
        "doctor's availability on a day, directions, 'I have reached', check-in, token status, talking to a person, "
        "a doctor's number. It answers the patient itself.", {
            "message": {"type": "string", "description": "The patient's message, rephrased simply if needed"}},
        ["message"]),
    _fn("raise_emergency", "The patient may be in danger. Tells them to call 108/112 now and lists the nearest "
        "emergency places. Use whenever in doubt.", {
            "signs": {"type": "array", "items": {"type": "string"},
                      "description": "danger signs in plain English, e.g. ['chest pain', 'sweating']"}},
        ["signs"]),
    _fn("reply", "Send your message to the patient and end the turn.", {
        "message": {"type": "string", "description": "1-3 short sentences in the patient's language"},
        "options": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                    "description": "optional quick-reply buttons, max 20 characters each"}},
        ["message"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


class Turn:
    """Everything one message needs: the rules engine, the request, and what the tools produced."""

    def __init__(self, flow: DoctorChat, text: str, session_id: str | None, kw: dict, ctx: dict, route_service, llm):
        self.flow, self.text, self.session_id, self.kw, self.ctx = flow, text, session_id, kw, ctx
        self.route_service, self.llm = route_service, llm
        self.ui: dict | None = None            # a tool's screen parts (doctors, card), message to be written by the LLM
        self.final: dict | None = None         # a complete reply (from a tool, or a checked LLM reply)
        self.draft: dict | None = None         # the LLM's reply {"message", "options"}
        self.names: set[str] = set()           # doctor names the patient may hear about
        self.tools_used: list[str] = []
        self.checks: list[str] = []
        self._ing = None
        orig = flow._ingest_only

        def once(text, session_id):            # one message = one turn in Vita's history, however many tools run
            if self._ing is None:
                self._ing = orig(text, session_id)
            return self._ing
        flow._ingest_only = once
        flow._hint = kw.get("ui_language")

    @property
    def sid(self) -> str | None:
        return self._ing.session_id if self._ing is not None else self.session_id

    def ingest(self):
        return self.flow._ingest_only(self.text, self.session_id)

    def known_names(self) -> set[str]:
        st = self.flow.load(self.sid) if self.sid else {}
        return self.names | {d.get("name", "") for d in (st.get("shown") or {}).values() if d.get("name")}

    # ------------------------------------------------------------------------------------------------- tools
    def run(self, name: str, args: dict) -> str:
        self.tools_used.append(name)
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return fn(**(args or {}))
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments: {exc}"})

    def t_reply(self, message: str = "", options: list[str] | None = None) -> str:
        self.draft = {"message": (message or "").strip(), "options": [o for o in (options or []) if o][:3]}
        return "sent"

    def t_vita_flow(self, message: str = "") -> str:
        self.final = self.flow.handle(message or self.text, self.session_id, **self.kw)
        return json.dumps({"answered": True})

    def t_raise_emergency(self, signs: list[str] | None = None) -> str:
        self.final = self.emergency([s.lower() for s in (signs or []) if s] or ["emergency signs"])
        return json.dumps({"answered": True})

    def t_find_doctors(self, specialty=None, doctor_name=None, degree=None, hospital=None, place=None, near_me=False,
                       gender=None, available_today=False, sort=None, more=False) -> str:
        flow, sid = self.flow, self.session_id
        st = flow.load(sid)
        page = 1
        if more and st.get("last"):
            filters, page = dict(st["last"]), int(st.get("page") or 1) + 1
        else:
            key = specialty_key(specialty) if specialty else None
            if specialty and not key:                                   # "skin doctor" in words: let the parser try
                key = parse(specialty).specialty_key
            modes = [m for m, on in (("specialty", key), ("name", doctor_name), ("degree", degree),
                                     ("distance", near_me), ("gender", gender), ("today", available_today),
                                     ("hospital", hospital), ("place", place), (sort or "", sort)) if on]
            filters = {"specialtyKey": key, "query": doctor_name, "degree": (degree or "").upper() or None,
                       "near": bool(near_me), "radiusKm": None, "sort": sort, "gender": gender,
                       "availableToday": bool(available_today), "hospitalQuery": hospital, "place": place,
                       "modes": modes}
            if not any(filters[k] for k in ("specialtyKey", "query", "degree", "near", "gender", "hospitalQuery", "place")):
                return json.dumps({"error": "say what kind of doctor, a name, a hospital or a place"})
        out = flow._search_turn(self.text, sid, filters, self.ctx, st, page=page)
        items = list((out.get("doctors") or {}).get("items") or [])
        self.names |= {d.get("name", "") for d in items}
        if out.get("kind") != DOCTOR_LIST or not items or (out.get("clarify") or {}).get("kind") == "which_doctor":
            self.final = out                       # a location question, nothing found, or "which one?": as written
            return json.dumps({"answered": True, "message": out.get("message")})
        self.ui = out
        return json.dumps({"doctors": [_brief(d) for d in items], "total": out["doctors"].get("total"),
                           "has_more": out["doctors"].get("hasMore"),
                           "note": "The list is on screen with a Choose button. Say one short line about it."})

    def t_assess_symptoms(self, symptoms: str = "", age_group: str | None = None, pregnant: bool = False) -> str:
        flow, svc = self.flow, self.route_service() if self.route_service else None
        if svc is None:                                     # engine not ready: Vita's rules triage answers
            return self.t_vita_flow(self.text)
        said = f"{symptoms}{', pregnant' if pregnant else ''}"
        query = f"{said} . {self.text}"
        em = svc.detector.detect(Query(raw=query, normalized=preprocess(query)))
        if em.flag:                                         # the LLM's reading can only add urgency
            self.final = self.emergency([r.lower() for r in em.reasons] or ["emergency signs"])
            return json.dumps({"answered": True, "emergency": True})
        age = age_group if age_group in ("child", "adult") else age_group_of(self.text)
        decision = svc.route(said if said.strip() else self.text, top_k=5, age_group=age)
        routing = to_routing(decision, [symptoms], said, age)
        recs = routing.recommendations
        ing = self.ingest()
        voice = Voice(ing.language.language)
        card = {"kind": "recommendation", "low_confidence": routing.low_confidence, "summary": symptoms,
                "outputs": [{"role": r, "specialist": x.specialist, "confidence": x.confidence}
                            for r, x in zip(("PRIMARY", "SECONDARY", "ALTERNATIVE"), recs)]}
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": ing.language.language}
        if routing.low_confidence:
            self.ui = flow._reply(base, "recommendation", "")
            self.ui["card"] = card
            return json.dumps({"confident": False, "advice": "Not sure which specialist. Suggest a General Physician "
                               "first, or ask one short question about the main problem."})
        primary = recs[0].specialist
        second = recs[1].specialist if len(recs) > 1 and recs[1].source == "route-symptom" else None
        t0 = time.perf_counter()
        out = flow._reply(base, "recommendation", "")
        out["card"] = card
        try:
            found = flow.directory.for_specialist(
                primary, city=self.ctx["city"], lat=self.ctx["lat"], lng=self.ctx["lng"],
                fallbacks=[s for s in (second, FALLBACK_SPECIALIST) if s], hospital_id=self.ctx["hospital_id"],
                raise_errors=True)
        except DoctorLookupError:
            out["degraded"] = ["doctor_lookup"]
            self.ui = out
            return json.dumps({"specialist": primary, "second_option": second, "doctors": [],
                               "note": "Doctor search is down right now; suggest trying again soon."})
        doctors = []
        if found:
            modes = ["symptom"] + (["distance"] if self.ctx["located"] else [])
            filters = {"specialtyKey": specialty_key(found["specialist"]), "specialist": found["specialist"],
                       "modes": modes}
            out["doctors"] = flow._block(found["doctors"], found["total"], modes, filters, self.ctx,
                                         specialty_id=found["specialtyId"], specialist=found["specialist"],
                                         fallback_used=found["fallbackUsed"],
                                         ms=round((time.perf_counter() - t0) * 1000, 1))
            doctors = found["doctors"]
            st = flow.load(ing.session_id)
            st["last"], st["language"], st["turn"] = filters, ing.language.language, ing.turn
            flow._remember(st, doctors)
            flow._save(ing.session_id, st)
            self.names |= {d.get("name", "") for d in doctors}
        out["message"] = V.recommendation(voice, symptoms=[], duration=None, severity=None, low_confidence=False,
                                          specialty_key=specialty_key(primary), specialist=primary,
                                          doctors=len(doctors), fallback_used=bool(found and found["fallbackUsed"]))
        self.ui = out
        return json.dumps({"specialist": primary, "second_option": second,
                           "doctors": [_brief(d) for d in doctors][:5],
                           "fallback_used": bool(found and found["fallbackUsed"]),
                           "note": "Doctors are on screen with a Choose button. In one or two lines say which kind "
                                   "of doctor fits and why in simple words (no disease names)."})

    # ------------------------------------------------------------------------------------------------- helpers
    def emergency(self, signs: list[str]) -> dict:
        """Vita's own emergency reply: the number first, Tele-MANAS for self-harm, nearest places when enabled."""
        flow = self.flow
        ing = self.ingest()
        voice = Voice(ing.language.language)
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": ing.language.language}
        out = flow._reply(base, "emergency", V.emergency(voice, signs))
        out["card"] = {"kind": "emergency", "emergency_reasons": signs,
                       "outputs": [{"role": "EMERGENCY", "specialist": "Emergency Medicine", "confidence": 1.0}],
                       "emergency_signs": [w for w in (V.symptom_words(voice, [x]) for x in signs) if w]}
        flow._emergency_done(ing.session_id, signs)
        if flow.emergency.enabled:
            est = flow.load(ing.session_id)
            est.pop("booking", None)
            flow.emergency.attach(out, est, self.ctx, signs, self.text, voice)
            flow._save(ing.session_id, est)
        return out

    def compose(self) -> dict | None:
        """The checked LLM reply on top of the tool's screen parts (or a plain message)."""
        if self.draft is None:
            return None
        msg, options = self.draft["message"], self.draft["options"]
        if self.ui is not None:
            out = dict(self.ui)
            out["message"] = msg
            if options and not out.get("clarify"):
                out["clarify"] = {"kind": "say", "question": msg, "options": [_opt(o) for o in options]}
            return out
        ing = self.ingest()
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": ing.language.language}
        asks = msg.rstrip().endswith(("?", "؟", "।?"))
        clarify = {"kind": "say", "question": msg, "options": [_opt(o) for o in options]} if options else None
        return self.flow._reply(base, FOLLOW_UP if asks else "info", msg, question=msg if asks else None,
                                clarify=clarify)


def _opt(label: str) -> dict:
    label = label.strip()[:24]
    return {"id": f"say:{label}", "label": label}


def _brief(d: dict) -> dict:
    h = d.get("hospital") or {}
    return {k: v for k, v in (("name", d.get("name")), ("specialty", d.get("specialtyName")),
                              ("hospital", h.get("name")), ("city", h.get("city")),
                              ("fee", d.get("discountedFee") if d.get("discountedFee") is not None else d.get("consultFee")),
                              ("experienceYears", d.get("experienceYears")), ("availableToday", d.get("availableToday")),
                              ("distanceKm", d.get("distanceKm"))) if v is not None}
