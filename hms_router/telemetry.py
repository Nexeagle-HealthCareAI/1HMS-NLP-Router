"""Agent decision telemetry for the CMS "AI Agents" dashboard.

Every /route-symptom decision and every /chat reply becomes one small record that is queued in memory
and posted in batches to the CMS (POST {HMS_CMS_API_URL}/agent-monitor/ingest/*, X-Service-Key). A
background thread does the sending, so a request never waits on it and a CMS outage never fails a
request: when the queue is full the oldest records are dropped and counted.

Privacy: records carry the decision, scores, latency, model version, caller and a keyed hash of the
normalized query, not the patient's words. The text is included (as reviewText, max 300 characters)
only where a person has to look at it for the model to learn: no match, a clarifying question, low
confidence, an emergency flag (to audit false alarms and misses) and "wrong specialist" feedback.
Those become items in the CMS review queue; nothing reaches training data until an admin approves it.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Protocol

log = logging.getLogger("hms_router.telemetry")

REVIEW_TEXT_MAX = 300
DECISIONS, FEEDBACK, MODEL_VERSIONS = "decisions", "feedback", "model-versions"


class Sink(Protocol):
    def send(self, kind: str, items: list[dict]) -> None: ...      # raises on failure


class HttpSink:
    """POSTs {"items": [...]} to {base_url}/agent-monitor/ingest/{kind}."""

    def __init__(self, base_url: str, service_key: str, timeout: float = 5.0):
        self.base = base_url.rstrip("/")
        self.key = service_key
        self.timeout = timeout

    def send(self, kind: str, items: list[dict]) -> None:
        import httpx
        r = httpx.post(f"{self.base}/agent-monitor/ingest/{kind}", json={"items": items},
                       headers={"X-Service-Key": self.key}, timeout=self.timeout)
        r.raise_for_status()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class DecisionRecorder:
    """Thread-safe bounded queue + background flusher. `record()` never blocks and never raises."""

    def __init__(self, sink: Sink | None, hash_key: str = "", *, flush_seconds: float = 5.0,
                 max_batch: int = 200, max_queue: int = 5000):
        self.sink = sink
        self.hash_key = (hash_key or "hms-router").encode()
        self.flush_seconds, self.max_batch = flush_seconds, max_batch
        self._q: dict[str, deque] = {k: deque(maxlen=max_queue) for k in (DECISIONS, FEEDBACK, MODEL_VERSIONS)}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"queued": 0, "sent": 0, "dropped": 0, "failedBatches": 0, "lastError": None, "lastSentAt": None}

    @property
    def enabled(self) -> bool:
        return self.sink is not None

    # ------------------------------------------------------------------ hashing
    def query_hash(self, normalized: str) -> str:
        return hmac.new(self.hash_key, normalized.strip().lower().encode("utf-8"), hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------ queue
    def record(self, kind: str, item: dict) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                q = self._q[kind]
                if len(q) == q.maxlen:
                    self.stats["dropped"] += 1
                q.append(item)
                self.stats["queued"] += 1
        except Exception:                                   # telemetry must never break a request
            log.exception("telemetry record failed")

    def pending(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._q.values())

    def flush(self) -> None:
        """Send everything queued, batch by batch. A failed batch goes back to the front of its queue
        (if there is room) and is retried on the next flush."""
        if not self.enabled:
            return
        for kind in (MODEL_VERSIONS, FEEDBACK, DECISIONS):
            while True:
                with self._lock:
                    q = self._q[kind]
                    batch = [q.popleft() for _ in range(min(self.max_batch, len(q)))]
                if not batch:
                    break
                try:
                    self.sink.send(kind, batch)
                    self.stats["sent"] += len(batch)
                    self.stats["lastSentAt"] = now_iso()
                except Exception as exc:
                    self.stats["failedBatches"] += 1
                    self.stats["lastError"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    log.warning("telemetry batch of %d %s failed: %s", len(batch), kind, exc)
                    with self._lock:
                        q = self._q[kind]
                        room = q.maxlen - len(q)
                        keep = batch[:room]
                        self.stats["dropped"] += len(batch) - len(keep)
                        q.extendleft(reversed(keep))
                    return

    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        def loop():
            while not self._stop.wait(self.flush_seconds):
                self.flush()
        self._thread = threading.Thread(target=loop, name="telemetry-flusher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self.flush()
        except Exception:
            log.exception("final telemetry flush failed")

    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "pending": self.pending(), **self.stats}


def _clip(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    return text[:REVIEW_TEXT_MAX]


# ---------------------------------------------------------------------------- record builders
def route_record(decision, *, recorder: DecisionRecorder, caller: str, latency_ms: float, request_id: str | None,
                 normalized: str, raw_text: str, decisive_min_conf: float) -> dict:
    """One /route-symptom decision. `decision` is a domain.models.RouteDecision."""
    top = decision.candidates[0] if decision.candidates else None
    conf = top.confidence if top else None
    low_conf = conf is not None and conf < decisive_min_conf
    needs_review = decision.noMatch or decision.decision.value != "answer" or low_conf or decision.emergency.flag
    return {
        "decisionId": str(uuid.uuid4()),
        "occurredAt": now_iso(),
        "agent": "route",
        "caller": caller,
        "requestId": request_id,
        "decision": "emergency" if decision.emergency.flag else decision.decision.value,
        "routeDecision": decision.decision.value,
        "topSpecialtyId": top.specialtyId if top and not decision.noMatch else None,
        "candidateIds": decision.specialtyIds[:5],
        "confidence": round(conf, 4) if conf is not None else None,
        "matchScore": round(decision.matchScore, 4),
        "isEmergency": bool(decision.emergency.flag),
        "emergencySource": decision.emergency.source,
        "clarifyKind": decision.clarify.kind if decision.clarify else None,
        "noMatch": bool(decision.noMatch),
        "method": decision.method,
        "usedReasoner": "llm" in (decision.method or ""),
        "latencyMs": round(latency_ms, 1),
        "modelVersion": decision.modelVersion,
        "queryLength": len(raw_text),
        "queryHash": recorder.query_hash(normalized),
        "reviewText": _clip(raw_text) if needs_review else None,
        "notes": "|".join(decision.notes)[:400] if decision.notes else None,
    }


def chat_record(reply: dict, *, recorder: DecisionRecorder, caller: str, latency_ms: float, text: str,
                model_version: str | None) -> dict:
    """One /chat reply (AgentReply.to_dict())."""
    card = reply.get("card") or {}
    outputs = card.get("outputs") or []
    top = outputs[0] if outputs else {}
    kind = reply.get("kind")
    low_conf = bool(card.get("low_confidence"))
    needs_review = kind == "emergency" or low_conf or (kind == "recommendation" and not outputs)
    sid = reply.get("session_id") or ""
    doctors = reply.get("doctors") or {}
    return {
        "decisionId": str(uuid.uuid4()),
        "occurredAt": now_iso(),
        "agent": "chat",
        "caller": caller,
        "decision": kind,
        "routeDecision": kind,
        "topSpecialtyId": None,
        "topSpecialist": top.get("specialist"),
        "candidateIds": [o.get("specialist") for o in outputs if o.get("specialist")][:5],
        "confidence": round(float(top["confidence"]), 4) if top.get("confidence") is not None else None,
        "matchScore": None,
        "isEmergency": kind == "emergency",
        "clarifyKind": "follow_up" if kind == "follow_up" else None,
        "noMatch": kind == "recommendation" and not outputs,
        "method": "agent",
        "usedReasoner": False,
        "latencyMs": round(latency_ms, 1),
        "modelVersion": model_version,
        "queryLength": len(text),
        "queryHash": recorder.query_hash(text),
        "sessionRef": recorder.query_hash("session:" + sid)[:32] if sid else None,
        "turn": reply.get("turn"),
        "degraded": "|".join(reply.get("degraded") or []) or None,
        "reviewText": _clip(text) if needs_review else None,
        # Vita doctor search (VITA-13): filters and counts only, never what the patient typed
        "searchMode": "|".join(doctors.get("mode") or []) or None,
        "doctorsShown": len(doctors.get("items") or []) if doctors else None,
        "doctorSelected": reply.get("selectedDoctorId"),
        "lookupMs": doctors.get("lookupMs"),
        "doctorFallbackUsed": doctors.get("fallbackUsed") if doctors else None,
    }


def feedback_record(kind: str, *, recorder: DecisionRecorder, caller: str, specialist: str | None,
                    suggested_specialist: str | None, symptoms: list[str] | None, phrase: str | None) -> dict:
    """User feedback from /feedback. Only symptoms/phrases, never the full sentence (learning.py rule)."""
    text = phrase or (", ".join(symptoms) if symptoms else None)
    return {
        "feedbackId": str(uuid.uuid4()),
        "occurredAt": now_iso(),
        "agent": "chat",
        "caller": caller,
        "kind": kind,
        "shownSpecialist": specialist,
        "suggestedSpecialist": suggested_specialist,
        "text": _clip(text),
        "queryHash": recorder.query_hash(text) if text else None,
    }


def build_recorder(base_url: str | None, service_key: str | None, hash_key: str | None = None,
                   sink_factory: Callable[[str, str], Sink] = HttpSink) -> DecisionRecorder:
    if base_url and service_key:
        log.info("Agent telemetry ON -> %s/agent-monitor", base_url.rstrip("/"))
        return DecisionRecorder(sink_factory(base_url, service_key), hash_key or service_key)
    log.info("Agent telemetry OFF (HMS_CMS_API_URL / HMS_CMS_SERVICE_KEY not set)")
    return DecisionRecorder(None)


def elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


