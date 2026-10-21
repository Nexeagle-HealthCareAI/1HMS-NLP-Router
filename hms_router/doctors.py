"""Doctor lookups for /chat from 1HMSAPI's MCP server (POST {HMS_DOCTORS_MCP_URL}, X-Service-Key).

When /chat recommends a specialist with confidence, the reply also lists bookable Doctor Dekho doctors
of that specialty (`search_doctors`). The router's specialist names ("Orthopedist", "Cardiologist") and
the API's categories ("Orthopaedic Surgeon (Bone)", "Cardiologist (Heart)") are both reduced to one
specialty key (`specialty_key`), and the category is taken from the live `list_specialties` answer,
so a renamed category on the API side still matches.

Optional and fail-soft: unset settings = off, and any error (timeout, 401/503, a JSON-RPC or tool error,
an unknown specialty) means "no doctors" for that reply, never a failed /chat. Only the specialty and
the caller's city / location are sent - never the patient's words.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

log = logging.getLogger("hms_router.doctors")

PROTOCOL_VERSION = "2025-06-18"
SPECIALTIES_TTL_S = 600.0

# Ordered: the first pattern that matches the normalized name wins, so specific ones ("neurosurg",
# "general surg", "gi surg") come before the broad ones ("neuro", "gastro", "general").
_KEY_PATTERNS: list[tuple[str, str]] = [
    (r"cardiothorac", "cardiothoracic-surgery"),
    (r"neurosurg", "neurosurgery"),
    (r"plastic", "plastic-surgery"),
    (r"vascular", "vascular-surgery"),
    (r"breast", "breast-surgery"),
    (r"\bgi\b.*surg|surgical gastro", "gi-surgery"),
    (r"general surg", "general-surgery"),
    (r"emergency", "emergency-medicine"),
    (r"sports", "sports-medicine"),
    (r"cardi", "cardiology"),
    (r"gastro", "gastroenterology"),
    (r"dermat|\bskin\b", "dermatology"),
    (r"orthop|\bbone\b", "orthopaedics"),
    (r"gyn|obstet", "gynaecology"),
    (r"dent", "dentistry"),
    (r"\bent\b|ear nose", "ent"),
    (r"ophthal|\beye\b", "ophthalmology"),
    (r"neuro", "neurology"),
    (r"psychiat", "psychiatry"),
    (r"urolog", "urology"),
    (r"endocrin|diabet", "endocrinology"),
    (r"pulmon|\blungs?\b", "pulmonology"),
    (r"nephro|kidney", "nephrology"),
    (r"oncol|cancer", "oncology"),
    (r"rheumat", "rheumatology"),
    (r"physio|rehab", "physiotherapy"),
    (r"an(a)?esth", "anaesthesiology"),
    (r"radiol", "radiology"),
    (r"patholog", "pathology"),
    (r"geriat", "geriatrics"),
    (r"p(a)?ediat|\bchild", "paediatrics"),
    (r"general (physician|medicine)|^physician$|internal medicine", "general-medicine"),
]


def specialty_key(name: str | None) -> str | None:
    """'Orthopedist' / 'Orthopaedic Surgeon (Bone)' / 'Orthopaedics' -> 'orthopaedics'. None if unknown
    (e.g. 'Telemedicine consult', 'OPD triage desk')."""
    if not name:
        return None
    text = re.sub(r"[^a-z]+", " ", name.lower()).strip()
    for pattern, key in _KEY_PATTERNS:
        if re.search(pattern, text):
            return key
    return None


class McpError(RuntimeError):
    pass


class McpClient:
    """Minimal MCP client for a stateless Streamable HTTP server: JSON-RPC over POST, answer either as
    application/json or as one SSE `data:` line. Initializes once and re-initializes if the server
    drops the session."""

    def __init__(self, url: str, service_key: str, timeout: float = 2.0, transport=None):
        import httpx
        self.url = url
        self._http = httpx.Client(timeout=timeout, transport=transport, headers={
            "X-Service-Key": service_key,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        })
        self._session: str | None = None
        self._ready = False
        self._ids = 0
        self._lock = threading.Lock()

    def _post(self, payload: dict) -> Any:
        headers = {"MCP-Protocol-Version": PROTOCOL_VERSION}
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        r = self._http.post(self.url, content=json.dumps(payload), headers=headers)
        if r.status_code in (400, 404) and self._session:
            raise _SessionLost()
        if r.status_code >= 400:
            raise McpError(f"HTTP {r.status_code}")
        self._session = r.headers.get("Mcp-Session-Id") or self._session
        if "id" not in payload:                      # notification: 202, no body
            return None
        body = r.text
        if "text/event-stream" in r.headers.get("content-type", ""):
            body = next((ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")), "")
        msg = json.loads(body)
        if "error" in msg:
            raise McpError(f"JSON-RPC {msg['error'].get('code')}: {msg['error'].get('message')}")
        return msg.get("result")

    def _request(self, method: str, params: dict) -> Any:
        self._ids += 1
        return self._post({"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params})

    def _initialize(self) -> None:
        self._session = None
        self._request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                     "clientInfo": {"name": "hms-nlp-router", "version": "2.0.0"}})
        try:                                         # a stateless server may not accept it; tools work regardless
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except McpError:
            pass
        self._ready = True

    def call_tool(self, name: str, arguments: dict) -> Any:
        """The tool's structuredContent (or its JSON text content). Raises McpError on any failure."""
        with self._lock:
            for attempt in (1, 2):
                try:
                    if not self._ready:
                        self._initialize()
                    result = self._request("tools/call", {"name": name, "arguments": arguments}) or {}
                    break
                except _SessionLost:
                    self._ready = False
                    if attempt == 2:
                        raise McpError("session lost twice")
        if result.get("isError"):
            text = " ".join(c.get("text", "") for c in result.get("content") or [])
            raise McpError(f"tool {name} failed: {text[:200]}")
        if result.get("structuredContent") is not None:
            return result["structuredContent"]
        text = next((c.get("text") for c in result.get("content") or [] if c.get("type") == "text"), None)
        return json.loads(text) if text else None


class _SessionLost(Exception):
    pass


class DoctorDirectory:
    """`for_specialist()` never raises; it returns [] whenever it cannot answer."""

    def __init__(self, client: McpClient | None, limit: int = 3):
        self.client = client
        self.limit = max(1, min(int(limit), 10))
        self._categories: dict[str, str] = {}
        self._categories_at = 0.0
        self.stats = {"calls": 0, "errors": 0, "lastError": None}

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _category_for(self, key: str) -> str | None:
        if time.monotonic() - self._categories_at > SPECIALTIES_TTL_S or not self._categories:
            data = self.client.call_tool("list_specialties", {}) or {}
            cats: dict[str, str] = {}
            for s in data.get("specialties") or []:
                if not s.get("doctorCount"):
                    continue
                for label in [s.get("specialtyId"), s.get("name"), *(s.get("aliases") or [])]:
                    k = specialty_key(label)
                    if k and k not in cats:
                        cats[k] = s["specialtyId"]
            self._categories, self._categories_at = cats, time.monotonic()
        return self._categories.get(key)

    def for_specialist(self, specialist: str, *, city: str | None = None, lat: float | None = None,
                       lng: float | None = None) -> dict | None:
        """{"specialist", "specialtyId", "total", "doctors"} for the router's specialist name, or None."""
        if not self.enabled:
            return None
        key = specialty_key(specialist)
        if key is None:
            return None
        self.stats["calls"] += 1
        try:
            category = self._category_for(key)
            if category is None:
                return None
            args: dict[str, Any] = {"specialtyIds": [category], "limit": self.limit}
            if city:
                args["city"] = city
            if lat is not None and lng is not None:
                args.update(lat=lat, lng=lng, sort="distance")
            data = self.client.call_tool("search_doctors", args) or {}
            doctors = data.get("doctors") or []
            if not doctors:
                return None
            return {"specialist": specialist, "specialtyId": category, "total": data.get("total", len(doctors)),
                    "doctors": doctors[: self.limit]}
        except Exception as exc:                       # fail soft: the reply goes out without doctors
            self.stats["errors"] += 1
            self.stats["lastError"] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("doctor lookup failed for %s: %s", specialist, self.stats["lastError"])
            return None


def build_directory(url: str | None, key: str | None, *, limit: int = 3, timeout: float = 2.0) -> DoctorDirectory:
    if not (url and key):
        log.info("Doctor lookups OFF (HMS_DOCTORS_MCP_URL / HMS_DOCTORS_MCP_KEY not set)")
        return DoctorDirectory(None)
    log.info("Doctor lookups ON (%s, limit %d)", url, limit)
    return DoctorDirectory(McpClient(url, key, timeout=timeout), limit=limit)
