"""Patient memory: what Vita remembers about a returning patient, across conversations.

Only for a patient the channel has identified (the WhatsApp sender's verified number, or a signed-in patient's
token); the key is an HMAC of the 10-digit mobile, never the number itself.
  * always (non-health preferences): language, city, last seen;
  * only with the channel's `remember: true` (the patient agreed in the app): who they book for (self / child /
    parent ...), specialties and doctors from past bookings.
Kept 180 days after the last visit. `forget(identity)` (DELETE /v2/agent/patient-memory) removes everything.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time

log = logging.getLogger("hms_router.vita_memory")

TTL_S = 180 * 24 * 3600
_PREFS = ("language", "city")
_CONSENTED = ("who", "specialties", "doctors")


class PatientMemory:
    def __init__(self, redis_client, secret: str | None = None):
        self.r = redis_client
        self.secret = (secret or os.environ.get("HMS_TELEMETRY_HASH_KEY") or os.environ.get("HMS_CMS_SERVICE_KEY")
                       or "vita-patient-memory").encode()

    def key(self, mobile: str | None) -> str | None:
        digits = "".join(ch for ch in (mobile or "") if ch.isdigit())[-10:]
        if len(digits) != 10:
            return None
        return "vita:patient:" + hmac.new(self.secret, digits.encode(), hashlib.sha256).hexdigest()[:32]

    def load(self, mobile: str | None) -> dict:
        k = self.key(mobile)
        if not k:
            return {}
        try:
            raw = self.r.get(k)
            return json.loads(raw) if raw else {}
        except Exception:
            log.exception("patient memory unreadable")
            return {}

    def update(self, mobile: str | None, *, consent: bool, **facts) -> dict:
        """Merge facts in; consented facts are kept only with consent (and dropped when consent is withdrawn)."""
        k = self.key(mobile)
        if not k:
            return {}
        mem = self.load(mobile)
        for f in _PREFS:
            if facts.get(f):
                mem[f] = facts[f]
        if consent:
            if facts.get("who"):
                mem["who"] = facts["who"]
            for f in ("specialties", "doctors"):
                for v in facts.get(f) or []:
                    seen = [x for x in mem.get(f, []) if x != v]
                    mem[f] = (seen + [v])[-5:]
        else:
            for f in _CONSENTED:
                mem.pop(f, None)
        mem["consent"] = bool(consent)
        mem["lastSeen"] = int(time.time())
        try:
            self.r.set(k, json.dumps(mem, ensure_ascii=False), ex=TTL_S)
        except Exception:
            log.exception("patient memory not saved")
        return mem

    def forget(self, mobile: str | None) -> bool:
        k = self.key(mobile)
        return bool(k and self.r.delete(k))
