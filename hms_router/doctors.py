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
SPECIALTIES_TTL_S = 3600.0

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


class DoctorLookupError(RuntimeError):
    """1HMS could not answer (after one retry). /chat turns this into degraded=["doctor_lookup"]."""


FALLBACK_SPECIALIST = "General Physician"
NAMES_TTL_S = 900.0
SEARCH_TTL_S = 300.0
NEARBY_DEFAULT_KM = 5.0
WIDE_RADIUS_KM = 50.0          # location known but not asked for: rank by distance without hiding far doctors


class DoctorDirectory:
    """The only way /chat reaches 1HMS doctor data (Vita requirements, VITA-1).

    Caches: specialties 1 h, doctor names 15 min, identical searches 5 min. Each MCP call has the client
    timeout (2 s default) and is retried once. `search()` raises DoctorLookupError when 1HMS cannot answer;
    `for_specialist()` and `names()` never raise."""

    def __init__(self, client: McpClient | None, limit: int = 3, clock=time.monotonic):
        self.client = client
        self.limit = max(1, min(int(limit), 10))
        self._clock = clock
        self._categories: dict[str, str] = {}
        self._categories_at = 0.0
        self._names: dict[str, tuple[float, list[dict]]] = {}
        self._searches: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "errors": 0, "retries": 0, "cacheHits": 0, "lastError": None}

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ plumbing
    def _call(self, tool: str, args: dict) -> dict:
        try:
            return self.client.call_tool(tool, args) or {}
        except Exception as first:
            if "HTTP 401" in str(first) or "HTTP 403" in str(first):     # a wrong key will not fix itself
                raise
            self.stats["retries"] += 1
            log.info("MCP %s failed (%s); retrying once", tool, type(first).__name__)
            return self.client.call_tool(tool, args) or {}

    def _failed(self, what: str, exc: Exception) -> None:
        self.stats["errors"] += 1
        self.stats["lastError"] = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("doctor lookup failed (%s): %s", what, self.stats["lastError"])

    def _category_for(self, key: str) -> str | None:
        if self._clock() - self._categories_at > SPECIALTIES_TTL_S or not self._categories:
            data = self._call("list_specialties", {})
            cats: dict[str, str] = {}
            for s in data.get("specialties") or []:
                if not s.get("doctorCount"):
                    continue
                for label in [s.get("specialtyId"), s.get("name"), *(s.get("aliases") or [])]:
                    k = specialty_key(label)
                    if k and k not in cats:
                        cats[k] = s["specialtyId"]
            self._categories, self._categories_at = cats, self._clock()
        return self._categories.get(key)

    # ------------------------------------------------------------------ generic search (five ways to ask)
    def search(self, *, specialty_key: str | None = None, query: str | None = None, degree: str | None = None,
               lat: float | None = None, lng: float | None = None, max_distance_km: float | None = None,
               city: str | None = None, hospital_id: str | None = None, sort: str | None = None,
               limit: int | None = None) -> dict | None:
        """{"doctors", "total", "specialtyId", "args"}; None when the specialty has no bookable doctors on
        1HMS (or nothing to search by). Raises DoctorLookupError when 1HMS cannot answer.
        `doctors` is exactly the MCP list, in 1HMS order - never edited here (VITA-12)."""
        if not self.enabled:
            raise DoctorLookupError("doctor lookups are off")
        self.stats["calls"] += 1
        try:
            args: dict[str, Any] = {"limit": max(1, min(int(limit or self.limit), 10))}
            category = None
            if specialty_key:
                category = self._category_for(specialty_key)
                if category is None:
                    return None
                args["specialtyIds"] = [category]
            if query:
                args["query"] = query
            if degree:
                args["degree"] = degree
            if lat is not None and lng is not None:
                args.update(lat=lat, lng=lng)
                if max_distance_km is not None:
                    args["maxDistanceKm"] = max_distance_km
            if city:
                args["city"] = city
            if hospital_id:
                args["hospitalId"] = hospital_id
            if sort:
                args["sort"] = sort
            if not ({"query", "specialtyIds", "degree", "lat", "hospitalId"} & args.keys()):
                return None                      # the tool needs one real filter; a city alone is not one
            cache_key = json.dumps(args, sort_keys=True)
            with self._lock:
                hit = self._searches.get(cache_key)
            if hit and self._clock() - hit[0] < SEARCH_TTL_S:
                self.stats["cacheHits"] += 1
                data = hit[1]
            else:
                data = self._call("search_doctors", args)
                with self._lock:
                    if len(self._searches) > 500:
                        self._searches.clear()
                    self._searches[cache_key] = (self._clock(), data)
            doctors = list(data.get("doctors") or [])
            return {"doctors": doctors, "total": int(data.get("total", len(doctors)) or 0),
                    "specialtyId": category, "args": args}
        except Exception as exc:
            self._failed("search", exc)
            raise DoctorLookupError(str(exc)) from exc

    def names(self, city: str | None = None) -> list[str]:
        """Names of every live doctor (for spotting a name typed without "dr"). [] on any failure."""
        if not self.enabled:
            return []
        key = (city or "").lower()
        hit = self._names.get(key)
        if hit and self._clock() - hit[0] < NAMES_TTL_S:
            return [d.get("name") or "" for d in hit[1]]
        try:
            data = self._call("list_doctor_names", {"city": city} if city else {})
            docs = list(data.get("doctors") or [])
            self._names[key] = (self._clock(), docs)
            return [d.get("name") or "" for d in docs]
        except Exception as exc:
            self._failed("names", exc)
            return []

    def get_doctor(self, doctor_id: str) -> dict | None:
        """The current public profile, or None when not listed any more. Raises DoctorLookupError."""
        if not self.enabled:
            raise DoctorLookupError("doctor lookups are off")
        try:
            data = self._call("get_doctor", {"doctorId": doctor_id})
        except Exception as exc:
            self._failed("get_doctor", exc)
            raise DoctorLookupError(str(exc)) from exc
        return data.get("doctor") if data.get("found") else None

    # ------------------------------------------------------------------ symptom path
    def for_specialist(self, specialist: str, *, city: str | None = None, lat: float | None = None,
                       lng: float | None = None, fallbacks: list[str] | None = None,
                       hospital_id: str | None = None, raise_errors: bool = False) -> dict | None:
        """{"specialist", "specialtyId", "total", "doctors", "fallbackUsed"} for the router's specialist
        name, trying `fallbacks` in order (VITA-2: SECONDARY, then General Physician) when it has no doctors.
        None when nobody fits. Errors give None too, unless raise_errors."""
        if not self.enabled:
            return None
        located = lat is not None and lng is not None
        for i, name in enumerate([specialist, *(fallbacks or [])]):
            key = specialty_key(name)
            if key is None:
                continue
            try:
                found = self.search(specialty_key=key, city=city, lat=lat if located else None,
                                    lng=lng if located else None,
                                    max_distance_km=WIDE_RADIUS_KM if located else None,
                                    sort="distance" if located else None, hospital_id=hospital_id)
            except DoctorLookupError:
                if raise_errors:
                    raise
                return None
            if found and found["doctors"]:
                return {"specialist": name, "specialtyId": found["specialtyId"], "total": found["total"],
                        "doctors": found["doctors"][: self.limit], "fallbackUsed": i > 0}
        return None


def build_directory(url: str | None, key: str | None, *, limit: int = 3, timeout: float = 2.0) -> DoctorDirectory:
    if not (url and key):
        log.info("Doctor lookups OFF (HMS_DOCTORS_MCP_URL / HMS_DOCTORS_MCP_KEY not set)")
        return DoctorDirectory(None)
    log.info("Doctor lookups ON (%s, limit %d)", url, limit)
    return DoctorDirectory(McpClient(url, key, timeout=timeout), limit=limit)
