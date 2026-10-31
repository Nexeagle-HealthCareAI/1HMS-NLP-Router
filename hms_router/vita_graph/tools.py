"""What the graph's agent nodes call (graph.py). Each tool runs Vita's tested code (application/doctor_chat.py, the
steps). Tools that answer the patient themselves (emergency, a location question, an empty list, choosing a doctor)
set `final` with their own reviewed wording; the others return facts for the LLM to talk about and leave their screen parts (doctor list, specialist
card) in `ui`, which the LLM's checked reply is placed on top of (compose)."""
from __future__ import annotations

import json
import logging
import time

from ..application import everyday
from .. import voice as V
from ..application.doctor_chat import DOCTOR_LIST, FOLLOW_UP, DoctorChat
from ..doctors import FALLBACK_SPECIALIST, DoctorLookupError, specialty_key
from ..domain.models import Query
from ..route_bridge import age_group_of, to_routing
from ..safety.denial import strip_denials
from ..textnorm import preprocess
from ..voice import Voice

log = logging.getLogger("hms_router.vita_tools")


class Turn:
    """Everything one message needs: Vita's steps, the request, and what the tools produced.
    text: the message tidied for the code steps ("doc chaiye" -> "doctor chahiye"); raw: as the patient wrote it,
    for the LLM."""

    def __init__(self, flow: DoctorChat, text: str, session_id: str | None, kw: dict, ctx: dict, rt):
        self.flow = self.steps = flow
        self.raw, self.text = text, flow.tidy(text)
        self.session_id, self.kw, self.ctx, self.rt = session_id, kw, ctx, rt
        self.route_service = rt.route_service if rt is not None else None
        self.identity = ctx.get("verified_mobile") or ctx.get("token_mobile")    # a patient the channel identified
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

    def fallback(self) -> dict:
        """When the LLM agent cannot answer (off, failing, or its reply was blocked): a plain doctor search when the
        message is plainly one, else "please try again, or talk to our team". The safety step has already run."""
        steps = self.flow
        return steps.social(self.text, self.session_id, moves=everyday.SOCIAL_MOVES) or \
            steps.plain_search(self.text, self.session_id, self.ctx) or \
            steps.unavailable(self.text, self.session_id, self.ctx)

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

    def t_raise_emergency(self, signs: list[str] | None = None) -> str:
        self.final = self.emergency([s.lower() for s in (signs or []) if s] or ["emergency signs"])
        return json.dumps({"answered": True})

    def t_find_doctors(self, specialty=None, doctor_name=None, degree=None, hospital=None, place=None, near_me=False,
                       gender=None, available_today=False, sort=None, more=False) -> str:
        self.tools_used.append("find_doctors")
        flow, sid = self.flow, self.session_id
        st = flow.load(sid)
        page = 1
        if more and st.get("last"):
            filters, page = dict(st["last"]), int(st.get("page") or 1) + 1
        else:
            key = specialty_key(specialty) if specialty else None
            if specialty and not key:                 # "skin doctor", "cardiolgist" in words: the spelling-tolerant reader
                key = flow.read_ask(specialty).specialty_key
            modes = [m for m, on in (("specialty", key), ("name", doctor_name), ("degree", degree),
                                     ("distance", near_me), ("gender", gender), ("today", available_today),
                                     ("hospital", hospital), ("place", place), (sort or "", sort)) if on]
            filters = {"specialtyKey": key, "query": doctor_name, "degree": (degree or "").upper() or None,
                       "near": bool(near_me), "radiusKm": None, "sort": sort, "gender": gender,
                       "availableToday": bool(available_today), "hospitalQuery": hospital, "place": place,
                       "modes": modes}
            last = st.get("last")
            if last and not any(filters[k] for k in ("specialtyKey", "query", "degree", "hospitalQuery")):
                # "kam fees wala", "lady doctor", "paas wale", "Patna mein": the same search, narrowed
                change = {k: v for k, v in filters.items() if v and k != "modes"}
                filters = dict(last, **change)
                filters["modes"] = sorted(set(last.get("modes") or []) | set(modes))
            elif not any(filters[k] for k in ("specialtyKey", "query", "degree", "near", "gender", "hospitalQuery",
                                               "place")):
                return json.dumps({"error": "say what kind of doctor, a name, a hospital or a place"})
        out = flow._search_turn(self.text, sid, filters, self.ctx, st, page=page)
        items = list((out.get("doctors") or {}).get("items") or [])
        self.names |= {d.get("name", "") for d in items}
        if out.get("kind") != DOCTOR_LIST or not items or (out.get("clarify") or {}).get("kind") == "which_doctor":
            self.final = out                       # a location question, nothing found, or "which one?": as written
            return json.dumps({"answered": True, "message": out.get("message")})
        lead = self.ctx.pop("lead", None)                 # "none in Purnia, these are in other towns"
        if lead:
            out["message"] = f"{lead} {out.get('message') or ''}".strip()
        self.ui = out
        return json.dumps({"doctors": [_brief(d) for d in items], "total": out["doctors"].get("total"),
                           "note_for_patient": lead,
                           "has_more": out["doctors"].get("hasMore"),
                           "note": "The list is on screen with a Choose button. Say one short line about it."})

    def t_assess_symptoms(self, symptoms: str = "", age_group: str | None = None, pregnant: bool = False) -> str:
        self.tools_used.append("assess_symptoms")
        flow, svc = self.flow, self.route_service() if self.route_service else None
        if svc is None:                                     # the specialty engine is loading: a GP is a safe first step
            return json.dumps({"confident": False, "advice": "Suggest a General Physician first; they can refer on."})
        said = f"{symptoms}{', pregnant' if pregnant else ''}"
        query = f"{said} . {self.text}"
        em = svc.detector.detect(Query(raw=query, normalized=preprocess(query)))
        if em.flag:                                         # the LLM's reading can only add urgency
            self.final = self.emergency([r.lower() for r in em.reasons])
            return json.dumps({"answered": True, "emergency": True})
        age = age_group if age_group in ("child", "adult") else age_group_of(self.text)
        # its reading + their words, minus anything denied ("chest pain nahi hai, bas khansi" is about the cough)
        chosen = strip_denials(preprocess(query if said.strip() else self.text))[0] or preprocess(self.text)
        decision = svc.route(chosen, top_k=5, age_group=age)
        routing = to_routing(decision, [symptoms], said, age)
        recs = routing.recommendations
        ing = self.ingest()
        voice = Voice(ing.language.language)
        card = {"kind": "recommendation", "low_confidence": routing.low_confidence, "summary": symptoms,
                "outputs": [{"role": r, "specialist": x.specialist, "confidence": x.confidence}
                            for r, x in zip(("PRIMARY", "SECONDARY", "ALTERNATIVE"), recs)]}
        base = {"session_id": ing.session_id, "turn": ing.turn, "language": ing.language.language}
        cl = decision.clarify if decision.decision.value == "clarify" else None      # the engine's own doubt
        if routing.low_confidence:
            self.ui = flow._reply(base, "recommendation", "")
            self.ui["card"] = card
            return json.dumps({"confident": False, "advice": "Not sure which specialist. Suggest a General Physician "
                               "first, or ask one short question about the main problem.",
                               "engine_question": cl.question if cl else None,
                               "engine_options": list(cl.options) if cl else []}, ensure_ascii=False)
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
                           "engine_unsure": cl is not None, "engine_question": cl.question if cl else None,
                           "engine_options": list(cl.options) if cl else [],
                           "note": "Doctors are on screen with a Choose button. In one or two lines say which kind "
                                   "of doctor fits and why in simple words (no disease names). If the engine was "
                                   "unsure, you may also ask its question."}, ensure_ascii=False)

    def t_doctor_days(self, doctor_name: str | None = None, day: str | None = None) -> str:
        """Which days / hours a doctor sits (the one chosen, or by name). day: today | tomorrow | day after tomorrow |
        a weekday | next."""
        self.tools_used.append("doctor_days")
        flow, sid = self.flow, self.session_id
        st = flow.load(sid)
        ask = flow.read_ask(self.text, sid)
        ask.name_query = doctor_name or ask.name_query
        ask.when = _when(day) or ask.when or "next"
        ask.available_today = False
        if not ask.name_query and not (st.get("selected") or {}).get("doctorId"):
            return json.dumps({"error": "no doctor chosen yet: ask which doctor, or find one first"})
        self.final = flow._availability(self.text, sid, ask, st, self.ctx)
        return json.dumps({"answered": True})

    def t_choose_doctor(self, doctor_name: str | None = None, position: int | None = None) -> str:
        """Choose a doctor already shown, by name or by position in the list on screen (1 = first); booking starts
        when the channel books."""
        self.tools_used.append("choose_doctor")
        flow, sid = self.flow, self.session_id
        st = flow.load(sid)
        if position:
            ids = st.get("page_ids") or []
            if 1 <= int(position) <= len(ids):
                self.final = flow._select(self.text, sid, ids[int(position) - 1], self.ctx)
                return json.dumps({"answered": True})
            return json.dumps({"error": f"only {len(ids)} doctors are on screen"})
        out = flow.choose(self.text, sid, self.ctx, doctor_name)
        if out is None:
            return json.dumps({"error": "no such doctor shown yet: find doctors first, or ask which one"})
        self.final = out
        return json.dumps({"answered": True})

    def t_visit_help(self) -> str:
        """The day of a booked visit: directions, reaching, check-in, token, queue."""
        self.tools_used.append("visit_help")
        out = self.flow.step_visit(self.text, self.session_id, self.ctx, asked=True)
        if out is None:
            return json.dumps({"error": "no booked visit found for this patient"})
        self.final = out
        return json.dumps({"answered": True})

    def t_talk_to_person(self, reason: str = "asked") -> str:
        self.tools_used.append("handoff")
        self.final = self.flow.handoff(self.text, self.session_id, "asked")
        return json.dumps({"answered": True})

    # ------------------------------------------------------------------------------------------------- helpers
    def emergency(self, signs: list[str]) -> dict:
        """Vita's own emergency reply: the number first, Tele-MANAS for self-harm, nearest places when enabled."""
        self.tools_used.append("emergency")
        self.ingest()
        return self.flow.emergency_reply(self.text, self.session_id, self.ctx, signs)

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


_DAYS = {"today": "+0", "aaj": "+0", "tomorrow": "+1", "kal": "+1", "day after tomorrow": "+2", "parso": "+2",
         "next": "next", "monday": "dow:0", "tuesday": "dow:1", "wednesday": "dow:2", "thursday": "dow:3",
         "friday": "dow:4", "saturday": "dow:5", "sunday": "dow:6"}


def _when(day: str | None) -> str | None:
    return _DAYS.get((day or "").strip().lower()) if day else None
