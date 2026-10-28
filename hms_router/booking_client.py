"""1HMS public API, for booking: WhatsApp OTP, the logged-in patient's profile and family, and the booking.

These are the same public endpoints Doctor Dekho's booking panel uses (X-Api-Key). 1HMS rate-limits them per
caller IP (20 bookings and 8 OTPs a minute); called from the router, every patient would share the router's
IP, so the visitor's own IP is forwarded with 1HMS's trusted-proxy headers (X-Forwarded-Client-Ip + the
shared X-Internal-Proxy-Secret, see 1HMSAPI TrustedProxyIpResolver). Without the secret 1HMS ignores the
forwarded IP, which is safe but shares one budget.

    send_otp(mobile)                -> None                      (raises BookingError with 1HMS's message)
    verify_otp(mobile, code)        -> (access_token, mobile)
    profile(token) / family(token)  -> dict / [dict]             (only with a patient token)
    book(...)                       -> appointment_id            (never retried: a retry could book twice)

Settings: HMS_PUBLIC_API_URL (default: the MCP URL without /mcp), HMS_PUBLIC_API_KEY (optional; needed only to alert
hospitals in an emergency),
HMS_PROXY_FORWARDING_SECRET (1HMSAPI Internal:ProxyForwardingSecret), HMS_BOOKING_TIMEOUT (default 8 s).
The patient token is a credential: it is never logged, never put in telemetry and never echoed back.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("hms_router.booking")


class BookingError(Exception):
    """1HMS said no (message is 1HMS's own, safe to show) or could not be reached (unreachable=True)."""

    def __init__(self, message: str, *, unreachable: bool = False, status: int | None = None):
        super().__init__(message)
        self.unreachable = unreachable
        self.status = status


class HmsPublicClient:
    def __init__(self, base_url: str | None, api_key: str | None, *, proxy_secret: str | None = None,
                 timeout: float = 8.0, transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.proxy_secret = proxy_secret
        self._http = httpx.Client(base_url=self.base_url or "http://unset", timeout=timeout, transport=transport)

    @property
    def enabled(self) -> bool:
        """Booking, OTP, profile and the visit page: 1HMS's public API takes anonymous callers (PublicApiKeyFilter: the
        X-Api-Key header is optional), so the URL is enough. Booking was off on every deployment until 28 Oct 2026
        because the deploy workflow never passed a key."""
        return bool(self.base_url)

    @property
    def identified(self) -> bool:
        """A key was set: needed only to alert hospitals in an emergency (EmergencyTakeoverController)."""
        return bool(self.base_url and self.api_key)

    def _headers(self, client_ip: str | None, token: str | None) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:                                    # an empty X-Api-Key would be refused as an invalid key
            h["X-Api-Key"] = self.api_key
        if self.proxy_secret and client_ip:
            h["X-Internal-Proxy-Secret"] = self.proxy_secret
            h["X-Forwarded-Client-Ip"] = client_ip
        if token:
            h["Authorization"] = f"Bearer {token}"
        return h

    def _call(self, method: str, path: str, *, client_ip: str | None = None, token: str | None = None,
              json: Any = None) -> dict:
        if not self.enabled:
            raise BookingError("Booking is not set up on this server.", unreachable=True)
        try:
            r = self._http.request(method, path, headers=self._headers(client_ip, token), json=json)
        except httpx.HTTPError as exc:
            log.warning("1HMS public %s %s unreachable: %s", method, path, type(exc).__name__)
            raise BookingError("1HMS could not be reached.", unreachable=True) from exc
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code == 429:
            raise BookingError("Too many tries just now. Please wait a minute.", status=429)
        if r.status_code >= 500:
            log.warning("1HMS public %s %s answered %s", method, path, r.status_code)
            raise BookingError("1HMS had a problem.", unreachable=True, status=r.status_code)
        if r.status_code >= 400 or (isinstance(body, dict) and body.get("success") is False):
            msg = (body.get("message") or body.get("Message") or "") if isinstance(body, dict) else ""
            raise BookingError(msg or f"1HMS refused the request ({r.status_code}).", status=r.status_code)
        return body if isinstance(body, dict) else {"items": body}

    # ------------------------------------------------------------------ OTP
    def send_otp(self, mobile: str, *, client_ip: str | None = None) -> None:
        self._call("POST", "/public/patient-auth/otp/send", client_ip=client_ip, json={"mobileNumber": mobile})

    def verify_otp(self, mobile: str, code: str, *, client_ip: str | None = None) -> tuple[str, str]:
        body = self._call("POST", "/public/patient-auth/otp/verify", client_ip=client_ip,
                          json={"mobileNumber": mobile, "otp": code})
        token = body.get("accessToken")
        if not token:
            raise BookingError(body.get("message") or "That code did not match.", status=400)
        return token, body.get("mobile") or mobile

    # ------------------------------------------------------------------ the logged-in patient
    def profile(self, token: str, *, client_ip: str | None = None) -> dict:
        return self._call("GET", "/public/patients/me", client_ip=client_ip, token=token)

    def family(self, token: str, *, client_ip: str | None = None) -> list[dict]:
        return list(self._call("GET", "/public/patients/me/family", client_ip=client_ip, token=token).get("members") or [])

    # ------------------------------------------------------------------ emergency takeover (vision Phase 3)
    def emergency_alert(self, *, lat: float, lng: float, reason: str, capability: str | None, channel: str,
                        session_ref: str | None, client_ip: str | None) -> dict:
        """POST /public/emergency-alerts: the nearest emergency-ready places, alerted (once per chat session per 30 min).
        Returns {eventId, places, alerted, repeated}. The point is used for this lookup only (1HMS keeps distances)."""
        return self._call("POST", "/public/emergency-alerts", client_ip=client_ip, json={
            "latitude": lat, "longitude": lng, "reason": reason, "capability": capability, "channel": channel,
            "sessionRef": session_ref})

    def emergency_places(self, *, lat: float, lng: float, capability: str | None, client_ip: str | None) -> dict:
        """GET /public/emergency-places: the nearest emergency-ready places, open now, nearest first, WITHOUT alerting them
        (no key needed). Returns {places}."""
        from urllib.parse import urlencode
        q = {"lat": lat, "lng": lng, **({"capability": capability} if capability else {})}
        return self._call("GET", f"/public/emergency-places?{urlencode(q)}", client_ip=client_ip)

    def share_emergency_number(self, event_id: str, mobile: str, *, client_ip: str | None = None) -> None:
        """The patient agreed to share their number with the places alerted in this emergency."""
        self._call("POST", f"/public/emergency-alerts/{event_id}/share-number", client_ip=client_ip,
                   json={"mobile": mobile, "consent": True})

    # ------------------------------------------------------------------ the visit (vision Phase 4)
    def visit(self, appointment_id: str, *, client_ip: str | None = None) -> dict:
        """GET /public/appointments/{id}/visit: where and when, the hospital's visit guide, canCheckIn, token and wait."""
        return self._call("GET", f"/public/appointments/{appointment_id}/visit", client_ip=client_ip).get("visit") or {}

    def check_in(self, appointment_id: str, lat: float, lng: float, *, client_ip: str | None = None) -> dict:
        """POST /public/tokens: a queue token for this appointment when the patient is at the hospital (1HMS checks the distance)."""
        return self._call("POST", "/public/tokens", client_ip=client_ip,
                          json={"appointmentId": appointment_id, "latitude": lat, "longitude": lng})

    def my_appointments(self, token: str, *, client_ip: str | None = None) -> list[dict]:
        """GET /public/appointments/mine for a signed-in patient."""
        return list(self._call("GET", "/public/appointments/mine", client_ip=client_ip, token=token).get("appointments") or [])

    # ------------------------------------------------------------------ booking
    def book(self, *, doctor_id: str, patient: dict, preferred_date: str, preferred_time: str | None,
             reason: str | None, token: str | None, client_ip: str | None, language: str | None = None) -> str:
        """POST /public/appointments. Never retried here: a timeout may still have booked."""
        body = self._call("POST", "/public/appointments", client_ip=client_ip, token=token, json={
            "doctorId": doctor_id, "patient": patient, "preferredDate": preferred_date,
            "preferredTime": preferred_time, "reason": reason, "referrerUrl": "vita",
            "language": language})                     # en / hi / hinglish: the WhatsApp confirmation and reminders use it
        appointment_id = body.get("appointmentId")
        if not appointment_id:
            raise BookingError(body.get("message") or "The booking was not accepted.", status=400)
        return str(appointment_id)


def build_public_client(base_url: str | None, api_key: str | None, *, mcp_url: str | None = None,
                        proxy_secret: str | None = None, timeout: float = 8.0) -> HmsPublicClient:
    """The public API lives on the same host as the MCP endpoint unless HMS_PUBLIC_API_URL says otherwise."""
    if not base_url and mcp_url and mcp_url.rstrip("/").endswith("/mcp"):
        base_url = mcp_url.rstrip("/")[: -len("/mcp")]
    if not base_url:
        log.warning("Vita booking OFF: no HMS_PUBLIC_API_URL and no HMS_DOCTORS_MCP_URL to derive it from")
    elif not api_key:
        log.info("Vita booking ON as an anonymous caller (no HMS_PUBLIC_API_KEY): emergency alerts to hospitals stay off")
    return HmsPublicClient(base_url, api_key, proxy_secret=proxy_secret, timeout=timeout)
