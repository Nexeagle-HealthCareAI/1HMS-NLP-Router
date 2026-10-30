"""VitaEngine: what POST /v2/agent/chat calls. Same signature and reply contract as DoctorChat.handle, so the
Doctor Dekho app, the website and WhatsApp need no change.

  * An LLM configured (VITA_LLM_PROVIDER), the control switch on "agent" and the breaker closed -> the LangGraph
    graph (graph.py): the whole conversation, state saved per session by the Redis checkpointer.
  * Otherwise, or any error before an answer exists -> DoctorChat (the rules engine), as before.
Every turn, by either engine, is traced for learning (learning.py).

Control (`vita:control` in Redis, admin API): mode "agent" (default) or "rules", per channel if wanted:
{"mode": "agent", "channels": {"whatsapp": "rules"}}.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from ..application.booking import token_mobile
from ..application.doctor_chat import DoctorChat
from .learning import ExampleStore, TraceStore, redact
from .memory import PatientMemory
from .prompts import REPLY_TOOL, UNDERSTAND_TOOL
from .tools import Turn

log = logging.getLogger("hms_router.vita_engine")

# what one turn starts from: per-turn channels cleared; messages / patient / triage / stage carry over (checkpoint)
TURN_RESET = {"route": None, "why": "", "understanding": {}, "decision": None, "facts": {}, "memory": {},
              "examples": [], "out": None}


class Breaker:
    """After `limit` failures in a row, skip the LLM for `cooldown` seconds."""

    def __init__(self, limit: int = 3, cooldown: float = 60.0):
        self.limit, self.cooldown, self.fails, self.opened = limit, cooldown, 0, 0.0
        self._lock = threading.Lock()

    @property
    def open(self) -> bool:
        with self._lock:
            if self.fails >= self.limit and time.monotonic() - self.opened < self.cooldown:
                return True
            if self.fails >= self.limit:
                self.fails = self.limit - 1                      # half-open: try once
            return False

    def ok(self) -> None:
        with self._lock:
            self.fails = 0

    def fail(self) -> None:
        with self._lock:
            self.fails += 1
            if self.fails >= self.limit:
                self.opened = time.monotonic()


class VitaRuntime:
    """Built once at start-up: the graph with its checkpointer, the model bound to each step's tool, the learning
    stores, patient memory and the breaker. Without an LLM it still traces turns (rules engine)."""

    def __init__(self, llm, redis_client, route_service=None, graph=None, breaker: Breaker | None = None):
        self.r = redis_client
        self.route_service = route_service
        self.breaker = breaker or Breaker()
        self.memory, self.traces, self.examples = PatientMemory(redis_client), TraceStore(redis_client), \
            ExampleStore(redis_client)
        self.llm_understand = self.llm_respond = None
        self.graph = None
        if llm is not None:
            self.llm_understand = _bind(llm, UNDERSTAND_TOOL)
            self.llm_respond = _bind(llm, REPLY_TOOL)
            if graph is None:
                from .checkpoint import RedisCheckpointSaver
                from .graph import build_vita_graph
                graph = build_vita_graph(RedisCheckpointSaver(redis_client))
            self.graph = graph

    @property
    def has_llm(self) -> bool:
        return self.graph is not None and self.llm_understand is not None

    # ---------------------------------------------------------------- control
    def control(self) -> dict:
        try:
            return json.loads(self.r.get("vita:control") or "{}")
        except Exception:
            return {}

    def set_control(self, value: dict) -> dict:
        mode = value.get("mode", "agent")
        channels = {k: v for k, v in (value.get("channels") or {}).items() if v in ("agent", "rules")}
        if mode not in ("agent", "rules"):
            raise ValueError("mode must be agent or rules")
        ctl = {"mode": mode, "channels": channels, "at": int(time.time())}
        self.r.set("vita:control", json.dumps(ctl))
        return ctl

    def mode(self, channel: str | None) -> str:
        ctl = self.control()
        return (ctl.get("channels") or {}).get(channel or "web") or ctl.get("mode") or "agent"


def _bind(llm, tool: dict):
    name = tool["function"]["name"]
    try:
        return llm.bind_tools([tool], tool_choice=name)
    except Exception:                                    # a provider without forced tool choice
        return llm.bind_tools([tool])


class VitaEngine:
    def __init__(self, flow: DoctorChat, runtime: VitaRuntime | None):
        self.flow, self.runtime = flow, runtime
        self.agent = flow.agent                                 # GET/DELETE /session use these

    def load(self, sid):
        return self.flow.load(sid)

    def clear(self, sid):
        self.flow.clear(sid)
        rt = self.runtime
        if rt is not None and rt.graph is not None:
            try:
                rt.graph.checkpointer.delete_thread(sid)
            except Exception:
                log.exception("conversation state not cleared")

    def session_view(self, sid):
        return self.flow.session_view(sid)

    def state(self, sid) -> dict | None:
        """The conversation's saved graph state (admin: what Vita knows, turn by turn)."""
        rt = self.runtime
        if rt is None or rt.graph is None:
            return None
        snap = rt.graph.get_state({"configurable": {"thread_id": sid}})
        return {k: v for k, v in (snap.values or {}).items() if k in ("messages", "patient", "triage", "stage")} \
            if snap else None

    # ---------------------------------------------------------------- entry
    def handle(self, text: str, session_id: str | None, **kw) -> dict:
        rt = self.runtime
        sid, _ = self.flow.agent.pipeline.sessions.open(session_id)       # the graph's thread id
        if rt is None or not rt.has_llm or rt.breaker.open or rt.mode(kw.get("channel")) != "agent":
            out = self.flow.handle(text, sid, **kw)
            out.setdefault("engine", "rules")
            if rt is not None:
                rt.traces.record(out.get("session_id") or sid, {
                    "patient": redact(text), "kind": out.get("kind"), "engine": out["engine"],
                    "language": out.get("language"), "reply": (out.get("message") or "")[:700]})
            return out
        turn = Turn(self.flow, text, sid, kw, _ctx(kw), rt)
        try:
            res = rt.graph.invoke(dict(TURN_RESET), {"configurable": {"thread_id": sid, "turn": turn},
                                                     "recursion_limit": 30})
            out = res["out"]
        except Exception:
            log.exception("Vita graph failed; answering with the rules engine")
            rt.breaker.fail()
            out = turn.final
            if out is None:
                fresh = DoctorChat(self.flow.agent, self.flow.directory, self.flow.r,
                                   booking_client=getattr(self.flow.booking, "client", None))
                out = fresh.handle(text, sid, **kw)
            out["engine"] = "rules-fallback"
        log.info("vita turn engine=%s tools=%s checks=%s", out.get("engine"), turn.tools_used, turn.checks)
        return out


def _ctx(kw: dict) -> dict:
    """The same request context DoctorChat.handle builds."""
    lat, lng = kw.get("lat"), kw.get("lng")
    tok = kw.get("patient_token")
    return {"city": kw.get("city"), "lat": lat, "lng": lng, "hospital_id": kw.get("hospital_id"),
            "located": lat is not None and lng is not None, "client_ip": kw.get("client_ip"),
            "patient_token": tok, "verified_mobile": kw.get("verified_mobile"),
            "token_mobile": token_mobile(tok) if tok else None, "channel": kw.get("channel") or "web",
            "visit_appointment_id": kw.get("visit_appointment_id")}
