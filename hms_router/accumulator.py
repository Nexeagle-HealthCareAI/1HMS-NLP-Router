"""Step 4 - multi-turn conversation accumulation (Redis).

Standalone: it only needs a Redis client and the Step 3 "symptom object" as a plain dict, e.g.
    {"symptoms": ["stomach pain"], "negated_symptoms": [], "body_parts": ["stomach"], "body_part": "stomach",
     "duration": None, "intensity": None, "language_mix": "hinglish", "turn": 1}

Redis keys (all expire together; TTL refreshed on every message = sliding 30-minute window):
    session:{id}:symptoms   LIST  one JSON symptom object per turn (capped)
    session:{id}:turns      INT   turn counter
    session:{id}:context    HASH  language, script, intensity (+ internal: asked, done)
The doc also lists `session:{id}:ttl`; a separate key isn't needed because EXPIRE on each key does the job.

Merge rules ("latest statement wins"):
    symptoms      union across turns; a later negation ("bukhar nahi") removes it, a later mention re-adds it
    body_parts    union
    intensity     highest level seen (mild < moderate < severe)
    duration      most recent non-empty value (the user usually answers "Kitne time se?")
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

_LEVEL = {"mild": 1, "moderate": 2, "severe": 3}
_LEVEL_NAME = {v: k for k, v in _LEVEL.items()}

# Only used to skip follow-up questions. Step 5's hardcoded emergency check stays the authority.
DEFAULT_URGENT = frozenset({"chest pain", "breathlessness", "unconscious", "excessive bleeding", "stroke",
                            "facial droop", "vision loss", "blood in vomit", "numbness", "sudden severe headache"})

QUESTIONS = {   # asked in this order, each at most once per session
    "more": "Koi aur symptoms hain?",
    "duration": "Kitne time se?",
    "intensity": "Takleef kitni hai - halki, theek-thaak ya bahut tez?",
    "describe": "Aapko kya takleef ho rahi hai? Thoda detail mein batayein.",
}
_DONE_PHRASES = re.compile(
    r"\b(bas itna hi|bas yahi|bas itna|aur kuch nahi|aur koi nahi|aur nahi|nothing else|no other|that is all|thats all)\b")
_PLAIN_NO = {"nahi", "nahin", "no", "nope", "nahi hai"}


@dataclass
class AccumulatedState:
    session_id: str
    turn: int
    symptoms: list[str] = field(default_factory=list)
    negated_symptoms: list[str] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)
    intensity: str | None = None
    duration: str | None = None
    durations: list[str] = field(default_factory=list)
    language: str | None = None
    first_seen_turn: dict[str, int] = field(default_factory=dict)   # symptom -> turn it first appeared
    ready: bool = False                  # True -> run full analysis (Step 5) on the combined symptoms
    ready_reason: str = ""
    next_question: str | None = None     # None when ready

    def to_dict(self) -> dict:
        return asdict(self)


class SymptomAccumulator:
    def __init__(self, redis_client, ttl_seconds: int = 1800, max_turns: int = 3,
                 max_buffer: int = 20, urgent: frozenset[str] = DEFAULT_URGENT, store_raw: bool = False):
        self.r = redis_client            # create with decode_responses=True
        self.ttl, self.max_turns, self.max_buffer = ttl_seconds, max_turns, max_buffer
        self.urgent, self.store_raw = urgent, store_raw

    # ---------------------------------------------------------------- keys
    @staticmethod
    def _k(sid: str, name: str) -> str:
        return f"session:{sid}:{name}"

    # ---------------------------------------------------------------- write path
    def add(self, session_id: str, symptom_object: dict, user_text: str | None = None,
            turn: int | None = None,
            emergency: "bool | Callable[[AccumulatedState], bool]" = False) -> AccumulatedState:
        """Store this turn's symptom object, merge with earlier turns, decide what happens next.

        user_text : normalized (lowercase Roman) user message - only used to spot "bas, aur kuch nahi".
        turn      : pass the turn number if another component already counts turns (e.g. a session manager);
                    the counter is then SET instead of INCR'd, so nothing is counted twice.
        emergency : True when Step 5's hardcoded safety net has fired - or a callable(merged_state) -> bool that
                    asks it, so the check sees negations from earlier turns. The patient is then never asked
                    follow-up questions; `urgent` below is only a fallback when the safety net isn't wired in.
        """
        if not isinstance(symptom_object.get("symptoms", []), list):
            raise ValueError("symptom_object['symptoms'] must be a list")

        record = {k: symptom_object.get(k) for k in
                  ("turn", "symptoms", "negated_symptoms", "body_parts", "body_part",
                   "duration", "intensity", "language_mix", "script")}
        record["symptoms"] = record["symptoms"] or []
        record["negated_symptoms"] = record["negated_symptoms"] or []
        record["ts"] = time.time()
        if self.store_raw:
            record["raw_input"] = symptom_object.get("raw_input")

        sid = session_id
        pipe = self.r.pipeline(transaction=True)
        pipe.rpush(self._k(sid, "symptoms"), json.dumps(record, ensure_ascii=False))
        pipe.ltrim(self._k(sid, "symptoms"), -self.max_buffer, -1)
        if turn is None:
            pipe.incr(self._k(sid, "turns"))
        else:
            pipe.set(self._k(sid, "turns"), int(turn))
        pipe.execute()

        ctx = self.r.hgetall(self._k(sid, "context"))
        if self._is_done(user_text, ctx.get("last_asked")):
            self.r.hset(self._k(sid, "context"), "done", "1")

        state = self.snapshot(sid)
        flag = emergency(state) if callable(emergency) else bool(emergency)
        self._decide(state, flag)                                 # ready? else which question
        fields = {k: v for k, v in {"language": state.language, "script": record.get("script"),
                                    "intensity": state.intensity}.items() if v}
        if fields:
            self.r.hset(self._k(sid, "context"), mapping=fields)
        for name in ("symptoms", "turns", "context"):
            self.r.expire(self._k(sid, name), self.ttl)                # sliding TTL
        return state

    # ---------------------------------------------------------------- read path
    def snapshot(self, session_id: str) -> AccumulatedState:
        """Combined view of the whole session. Empty state if the session expired."""
        sid = session_id
        records = [json.loads(x) for x in self.r.lrange(self._k(sid, "symptoms"), 0, -1)]
        turns = self.r.get(self._k(sid, "turns"))
        st = AccumulatedState(sid, int(turns) if turns else len(records))

        active: dict[str, int] = {}
        negated: list[str] = []
        level = 0
        for i, rec in enumerate(records, start=1):
            turn_no = rec.get("turn") or i
            for s in rec.get("negated_symptoms") or []:               # negations first, then mentions
                active.pop(s, None)
                if s not in negated:
                    negated.append(s)
            for s in rec.get("symptoms") or []:
                active.setdefault(s, turn_no)
                if s in negated:
                    negated.remove(s)
            for bp in (rec.get("body_parts") or []) + ([rec["body_part"]] if rec.get("body_part") else []):
                if bp not in st.body_parts:
                    st.body_parts.append(bp)
            level = max(level, _LEVEL.get(rec.get("intensity") or "", 0))
            if rec.get("duration"):
                st.duration = rec["duration"]
                if rec["duration"] not in st.durations:
                    st.durations.append(rec["duration"])
            st.language = rec.get("language_mix") or st.language

        st.symptoms = list(active)
        st.first_seen_turn = dict(active)
        st.negated_symptoms = [s for s in negated if s not in active]
        st.intensity = _LEVEL_NAME.get(level)
        return st

    def clear(self, session_id: str) -> None:
        self.r.delete(*(self._k(session_id, n) for n in ("symptoms", "turns", "context")))

    # ---------------------------------------------------------------- decisions
    def _decide(self, st: AccumulatedState, emergency: bool = False) -> None:
        ctx = self.r.hgetall(self._k(st.session_id, "context"))
        asked = set(filter(None, ctx.get("asked", "").split(",")))

        if emergency:                                                   # safety net beats everything, even "no symptoms yet"
            st.ready, st.ready_reason = True, "emergency_safety_net"
        elif st.symptoms and self.urgent.intersection(st.symptoms):
            st.ready, st.ready_reason = True, "urgent_symptom"          # don't make a patient answer questions
        elif st.symptoms and ctx.get("done") == "1":
            st.ready, st.ready_reason = True, "user_done"
        elif st.symptoms and st.turn >= self.max_turns:
            st.ready, st.ready_reason = True, "max_turns"
        elif st.symptoms and st.duration and st.intensity:
            st.ready, st.ready_reason = True, "complete"

        if st.ready:
            return

        if not st.symptoms:
            st.ready_reason = "no_symptoms_after_max_turns" if st.turn >= self.max_turns else "no_symptoms"
            st.next_question = QUESTIONS["describe"]                    # may repeat until max_turns
            self.r.hset(self._k(st.session_id, "context"), "last_asked", "describe")
            return

        wanted = [("more", True), ("duration", st.duration is None), ("intensity", st.intensity is None)]
        pending = [key for key, needed in wanted if needed and key not in asked]
        if not pending:                                                 # everything asked once already
            st.ready, st.ready_reason = True, "followups_exhausted"
            return
        key = pending[0]
        st.next_question = QUESTIONS[key]
        self.r.hset(self._k(st.session_id, "context"),
                    mapping={"asked": ",".join(sorted(asked | {key})), "last_asked": key})

    @staticmethod
    def _is_done(text: str | None, last_asked: str | None) -> bool:
        if not text:
            return False
        t = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        t = re.sub(r"\s+", " ", t)
        return bool(_DONE_PHRASES.search(t)) or (last_asked == "more" and t in _PLAIN_NO)


# ---------------------------------------------------------------- demo (the 3-turn example from the doc)
if __name__ == "__main__":
    import fakeredis

    acc = SymptomAccumulator(fakeredis.FakeRedis(decode_responses=True))
    turns = [
        ("pet dard ho raha hai", {"symptoms": ["stomach pain"], "body_parts": ["stomach"], "language_mix": "hinglish"}),
        ("aur ulti bhi aa rahi hai", {"symptoms": ["vomiting"], "language_mix": "hinglish"}),
        ("subah se, aur bukhaar bhi hai", {"symptoms": ["fever"], "duration": "since morning", "language_mix": "hinglish"}),
    ]
    for text, obj in turns:
        s = acc.add("sess_demo", obj, user_text=text)
        print(f"turn {s.turn}: {s.symptoms} duration={s.duration} ready={s.ready}({s.ready_reason}) -> {s.next_question}")
