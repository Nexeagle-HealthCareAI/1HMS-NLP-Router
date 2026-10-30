"""VitaEngine: what POST /v2/agent/chat calls. Same signature and reply contract as DoctorChat.handle, so the
Doctor Dekho app, the website and WhatsApp need no change.

  * LLM configured (VITA_LLM_PROVIDER) and healthy -> the LangGraph graph (graph.py).
  * No LLM, the circuit open after repeated failures, or any error before a tool answered -> DoctorChat (the rules
    engine), exactly as before.
Every turn (either engine) is added to a short conversation memory the LLM reads next time (Redis, 30 min).
"""
from __future__ import annotations

import json
import logging
import threading
import time

from ..application.booking import token_mobile
from ..application.doctor_chat import DoctorChat
from .tools import TOOLS, Turn

log = logging.getLogger("hms_router.vita_engine")

HISTORY_MAX = 12
HISTORY_TTL_S = 1800


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
    """Built once at start-up: the compiled graph, the model with Vita's tools bound, the breaker."""

    def __init__(self, llm, graph, route_service=None, breaker: Breaker | None = None):
        self.llm = llm.bind_tools(TOOLS) if llm is not None else None
        self.graph, self.route_service = graph, route_service
        self.breaker = breaker or Breaker()

    @property
    def active(self) -> bool:
        return self.llm is not None and self.graph is not None


class VitaEngine:
    def __init__(self, flow: DoctorChat, runtime: VitaRuntime | None):
        self.flow, self.runtime = flow, runtime
        self.agent = flow.agent                                 # GET/DELETE /session use these

    def load(self, sid):
        return self.flow.load(sid)

    def clear(self, sid):
        self.flow.clear(sid)
        try:
            self.flow.r.delete(_hkey(sid))
        except Exception:
            pass

    def session_view(self, sid):
        return self.flow.session_view(sid)

    # --------------------------------------------------------------------------------------------- entry
    def handle(self, text: str, session_id: str | None, **kw) -> dict:
        rt = self.runtime
        if rt is None or not rt.active or rt.breaker.open:
            out = self.flow.handle(text, session_id, **kw)
            out.setdefault("engine", "rules")
            self._remember(text, out)
            return out
        turn = Turn(self.flow, text, session_id, kw, _ctx(kw), rt.route_service, rt.llm)
        turn.history = self._history(session_id)
        try:
            res = rt.graph.invoke({}, {"configurable": {"turn": turn}, "recursion_limit": 25})
            out = res["out"]
            if out.get("engine", "").startswith("agent"):
                rt.breaker.ok()
        except Exception:
            log.exception("Vita graph failed; answering with the rules engine")
            rt.breaker.fail()
            out = turn.final
            if out is None:
                fresh = DoctorChat(self.flow.agent, self.flow.directory, self.flow.r,
                                   booking_client=getattr(self.flow.booking, "client", None))
                out = fresh.handle(text, turn.sid, **kw)
                out["engine"] = "rules-fallback"
        if turn.tools_used or turn.checks:
            log.info("vita turn engine=%s tools=%s checks=%s", out.get("engine"), turn.tools_used, turn.checks)
        self._remember(text, out)
        return out

    # --------------------------------------------------------------------------------------------- memory
    def _history(self, sid) -> list[dict]:
        if not sid:
            return []
        try:
            return [json.loads(x) for x in self.flow.r.lrange(_hkey(sid), -HISTORY_MAX, -1)]
        except Exception:
            return []

    def _remember(self, text: str, out: dict) -> None:
        sid = out.get("session_id")
        if not sid:
            return
        try:
            key = _hkey(sid)
            said = "[booking details]" if str((out.get("clarify") or {}).get("kind") or "").startswith("book_") else text
            self.flow.r.rpush(key, json.dumps({"r": "user", "t": said[:500]}),
                              json.dumps({"r": "vita", "t": (out.get("message") or "")[:500]}))
            self.flow.r.ltrim(key, -HISTORY_MAX, -1)
            self.flow.r.expire(key, HISTORY_TTL_S)
        except Exception:
            log.exception("Vita conversation memory not saved")


def _hkey(sid: str) -> str:
    return f"vita:llm:{sid}"


def _ctx(kw: dict) -> dict:
    """The same request context DoctorChat.handle builds."""
    lat, lng = kw.get("lat"), kw.get("lng")
    tok = kw.get("patient_token")
    return {"city": kw.get("city"), "lat": lat, "lng": lng, "hospital_id": kw.get("hospital_id"),
            "located": lat is not None and lng is not None, "client_ip": kw.get("client_ip"),
            "patient_token": tok, "verified_mobile": kw.get("verified_mobile"),
            "token_mobile": token_mobile(tok) if tok else None, "channel": kw.get("channel") or "web",
            "visit_appointment_id": kw.get("visit_appointment_id")}
