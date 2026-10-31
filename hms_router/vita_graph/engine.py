"""VitaEngine: what POST /v2/agent/chat calls. One brain: every message runs through the LangGraph graph (graph.py).

  * The graph's code steps (safety, booking, taps, after an emergency, the visit) answer what is theirs, with or
    without an LLM.
  * Everything else goes to the LLM agent (understand -> plan -> tools -> reply) when an LLM is configured
    (VITA_LLM_PROVIDER), the control switch is "agent" and the breaker is closed.
  * Otherwise the fallback: a plain doctor search when the message is plainly one, else "please try again, or talk
    to our team" (the emergency check has already run). There is no second conversation engine.
Every turn is traced for learning (learning.py).

Control (`vita:control` in Redis, admin API): mode "agent" (default: the model labels the message, code picks the next
move), "planner" (the model plans with tools around the patient's needs: planner.py) or "off" (no LLM: code steps and
the fallback), per channel if wanted: {"mode": "agent", "channels": {"whatsapp": "planner"}}. "rules" is read as "off".
"""
from __future__ import annotations

import json
import logging
import threading
import time

from ..application.doctor_chat import DoctorChat, SelectionError
from .learning import ExampleStore, TraceStore, redact
from .memory import PatientMemory
from .planner import PLAN_TOOLS
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
    stores, patient memory and the breaker. Without an LLM the graph still runs (code steps and the fallback)."""

    def __init__(self, llm, redis_client, route_service=None, graph=None, breaker: Breaker | None = None):
        self.r = redis_client
        self.route_service = route_service
        self.breaker = breaker or Breaker()
        self.memory, self.traces, self.examples = PatientMemory(redis_client), TraceStore(redis_client), \
            ExampleStore(redis_client)
        self.llm_understand = self.llm_respond = self.llm_plan = None
        if llm is not None:
            self.llm_understand = _bind(llm, UNDERSTAND_TOOL)
            self.llm_respond = _bind(llm, REPLY_TOOL)
            self.llm_plan = _bind_all(llm, PLAN_TOOLS)
        if graph is None:
            from .checkpoint import RedisCheckpointSaver
            from .graph import build_vita_graph
            graph = build_vita_graph(RedisCheckpointSaver(redis_client))
        self.graph = graph

    @property
    def has_llm(self) -> bool:
        return self.llm_understand is not None and self.llm_respond is not None

    def agent_on(self, channel: str | None) -> bool:
        """The LLM may answer this message: an LLM, the switch on "agent" or "planner", the breaker closed."""
        return self.has_llm and self.mode(channel) in MODES_ON and not self.breaker.open

    # ---------------------------------------------------------------- control
    def control(self) -> dict:
        try:
            return json.loads(self.r.get("vita:control") or "{}")
        except Exception:
            return {}

    def set_control(self, value: dict) -> dict:
        mode = _mode(value.get("mode", "agent"))
        channels = {k: _mode(v) for k, v in (value.get("channels") or {}).items() if _mode(v) in MODES}
        if mode not in MODES:
            raise ValueError("mode must be agent, planner or off")
        ctl = {"mode": mode, "channels": channels, "at": int(time.time())}
        self.r.set("vita:control", json.dumps(ctl))
        return ctl

    def mode(self, channel: str | None) -> str:
        ctl = self.control()
        return _mode((ctl.get("channels") or {}).get(channel or "web") or ctl.get("mode") or "agent")


MODES = ("agent", "planner", "off")
MODES_ON = ("agent", "planner")


def _mode(value) -> str:
    return "off" if value == "rules" else value


def _bind_all(llm, tools: list[dict]):
    """The planner: the model chooses among all its tools, but must call one (reply is a tool too)."""
    try:
        return llm.bind_tools(tools, tool_choice="required")
    except Exception:
        return llm.bind_tools(tools)


def _bind(llm, tool: dict):
    name = tool["function"]["name"]
    try:
        return llm.bind_tools([tool], tool_choice=name)
    except Exception:                                    # a provider without forced tool choice
        return llm.bind_tools([tool])


class VitaEngine:
    def __init__(self, steps: DoctorChat, runtime: VitaRuntime | None):
        self.steps = self.flow = steps
        self.runtime = runtime or VitaRuntime(None, steps.r)       # no LLM: the graph still runs its code steps
        self.pipeline = steps.pipeline                             # GET/DELETE /session use these

    def load(self, sid):
        return self.steps.load(sid)

    def clear(self, sid):
        self.steps.clear(sid)
        try:
            self.runtime.graph.checkpointer.delete_thread(sid)
        except Exception:
            log.exception("conversation state not cleared")

    def session_view(self, sid):
        return self.steps.session_view(sid)

    def state(self, sid) -> dict | None:
        """The conversation's saved graph state (admin: what Vita knows, turn by turn)."""
        snap = self.runtime.graph.get_state({"configurable": {"thread_id": sid}})
        return {k: v for k, v in (snap.values or {}).items() if k in ("messages", "patient", "triage", "stage")} \
            if snap else None

    # ---------------------------------------------------------------- entry
    def handle(self, text: str, session_id: str | None, **kw) -> dict:
        rt, steps = self.runtime, self.steps
        sid, _ = steps.pipeline.sessions.open(session_id)             # the graph's thread id
        ctx = steps.context(kw)
        turn = Turn(steps, text, sid, kw, ctx, rt)
        try:
            res = rt.graph.invoke(dict(TURN_RESET), {"configurable": {"thread_id": sid, "turn": turn},
                                                     "recursion_limit": 40})
            out = res["out"]
        except SelectionError:
            raise                                                    # a stale doctor id: the route answers 422
        except Exception:
            log.exception("Vita graph failed; answering with the fallback")
            out = turn.final or turn.fallback()
            out["engine"] = "fallback"
            rt.traces.record(out.get("session_id") or sid, {"patient": redact(turn.text), "kind": out.get("kind"),
                                                            "engine": "fallback", "reply": (out.get("message") or "")[:700]})
        log.info("vita turn engine=%s tools=%s checks=%s", out.get("engine"), turn.tools_used, turn.checks)
        return steps.finish(out, ctx)
