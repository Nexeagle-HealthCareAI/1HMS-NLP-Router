"""Vita books the appointment (vision Phase 2), against a fake 1HMS public API.

Guest: day -> (shift) -> who -> name, age, sex -> mobile -> WhatsApp code -> confirm -> booked.
Signed in (patient_token): no OTP; the saved profile and family members are offered.
A channel that verified the number (verified_mobile scope): no OTP.
Never books without "yes"; a refusal offers other days; 1HMS down books nothing; cancel and emergencies win.
"""
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.http.security import hash_key
from hms_router.adapters.settings import ApiClient, ServiceSettings
from hms_router.booking_client import HmsPublicClient
from hms_router.channels.whatsapp import validate
from tests.test_vita_doctor_chat import DOCTORS, URL, Fake1Hms, chat, client, hms, make_directory  # noqa: F401

ANITA = DOCTORS[2]["doctorId"]          # Dr. Anita Sen, Dermatologist, Salt Lake Care, Rs 600
RAKESH = DOCTORS[1]["doctorId"]         # 1HMS refuses bookings for him in the fake below
APPT = "abcdef12-3456-7890-abcd-ef1234567890"


def jwt(mobile: str) -> str:
    part = base64.urlsafe_b64encode(json.dumps({"mobile": mobile, "sessionEpoch": "1"}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{part}.sig"


class FakePublicApi:
    """1HMS /public endpoints the booking uses, recording every call."""

    def __init__(self, *, down=False):
        self.calls: list[tuple[str, str, dict, dict]] = []
        self.down = down

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append((request.method, request.url.path, body, dict(request.headers)))
        if request.headers.get("X-Api-Key") != "pub-key":
            return httpx.Response(401)
        if self.down:
            return httpx.Response(503)
        path = request.url.path
        if path.endswith("/otp/send"):
            if body.get("mobileNumber") == "9999999999":
                return httpx.Response(429, json={"message": "Too many"})
            return httpx.Response(200, json={"success": True, "message": "sent"})
        if path.endswith("/otp/verify"):
            if body.get("otp") == "123456":
                return httpx.Response(200, json={"success": True, "accessToken": jwt(body["mobileNumber"]),
                                                 "mobile": body["mobileNumber"]})
            return httpx.Response(200, json={"success": False, "message": "Invalid OTP"})
        auth = request.headers.get("Authorization", "")
        if path.endswith("/patients/me"):
            if not auth:
                return httpx.Response(401)
            return httpx.Response(200, json={"success": True, "fullName": "Rina Kumari", "age": 32,
                                             "ageUnit": "years", "sex": "Female"})
        if path.endswith("/patients/me/family"):
            return httpx.Response(200, json={"success": True, "members": [
                {"memberId": "11111111-0000-0000-0000-000000000001", "fullName": "Asha Devi", "relation": "Mother",
                 "age": 60, "ageUnit": "years", "sex": "Female"}]})
        if path.endswith("/public/appointments"):
            if body.get("doctorId") == RAKESH:
                return httpx.Response(400, json={"success": False, "message": "Doctor is on leave that day"})
            return httpx.Response(200, json={"success": True, "appointmentId": APPT, "message": "received"})
        return httpx.Response(404)

    def bookings(self):
        return [c for c in self.calls if c[1].endswith("/public/appointments")]


@pytest.fixture
def api(client, hms):
    fake = FakePublicApi()
    client.app.state.booking = HmsPublicClient("http://hms", "pub-key", proxy_secret="proxy-secret",
                                               transport=httpx.MockTransport(fake))
    yield fake
    client.app.state.booking = None


def labels(reply):
    return [o["label"] for o in (reply.get("clarify") or {}).get("options") or []]


def choose_anita(client, **kw):
    lst = chat(client, "skin ka doctor chahiye", city="Kolkata", **kw)
    return chat(client, "Choose", session_id=lst["session_id"], select_doctor_id=ANITA, **kw)


def say(client, prev, text, **kw):
    return chat(client, text, session_id=prev["session_id"], **kw)


# ============================================================================ guest, with OTP
def test_guest_books_with_a_whatsapp_code(client, hms, api):
    sel = choose_anita(client, client_ip="203.0.113.7")
    assert sel["kind"] == "doctor_selected" and sel["clarify"]["kind"] == "book_day"
    assert sel["message"].startswith("Aapne Dr. Anita Sen") and "Kaun sa din" in sel["message"]
    who = say(client, sel, labels(sel)[0])                                   # first working day (one shift)
    assert who["clarify"]["kind"] == "book_who" and labels(who) == ["Mere liye", "Kisi aur ke liye"]
    name = say(client, who, "Mere liye")
    assert name["clarify"]["kind"] == "book_name"
    age = say(client, name, "mera naam rina kumari hai")
    assert age["clarify"]["kind"] == "book_age" and "Rina Kumari" in age["message"]
    sex = say(client, age, "32 saal")
    assert sex["clarify"]["kind"] == "book_sex"
    mobile = say(client, sex, "Mahila")
    assert mobile["clarify"]["kind"] == "book_mobile"
    otp = say(client, mobile, "+91 98765 43210", client_ip="203.0.113.7")
    assert otp["clarify"]["kind"] == "book_otp" and "98xxxxxx10" in otp["message"]
    wrong = say(client, otp, "111111", client_ip="203.0.113.7")
    assert wrong["clarify"]["kind"] == "book_otp" and "match" in wrong["message"]
    confirm = say(client, wrong, "123456", client_ip="203.0.113.7")
    assert confirm["clarify"]["kind"] == "book_confirm"
    assert confirm["patient_session"]["mobile"] == "9876543210"              # the website signs her in
    assert "Rina Kumari (32, F)" in confirm["message"] and "₹600" in confirm["message"]
    assert not api.bookings()                                                # nothing booked before "yes"
    done = say(client, confirm, "Haan, book karo", client_ip="203.0.113.7")
    assert done["kind"] == "booked" and done["booking"]["appointmentId"] == APPT
    assert "ABCDEF12" in done["message"]
    (_, _, body, headers), = api.bookings()
    assert body["doctorId"] == ANITA and body["preferredTime"] == "09:00:00"
    assert body["patient"] == {"fullName": "Rina Kumari", "mobile": "9876543210", "age": 32, "ageUnit": "years",
                               "sex": "Female"}
    assert headers["authorization"].startswith("Bearer ")                    # booked as the verified patient
    assert headers["x-forwarded-client-ip"] == "203.0.113.7" and headers["x-internal-proxy-secret"] == "proxy-secret"


def test_otp_send_refused_says_so(client, hms, api):
    sel = choose_anita(client)
    s = sel
    for text in (labels(sel)[0], "Mere liye", "Rina", "30", "Mahila"):
        s = say(client, s, text)
    r = say(client, s, "9999999999")
    assert r["clarify"]["kind"] == "book_mobile" and "Too many" in r["message"]


# ============================================================================ signed in: no OTP
def test_signed_in_patient_needs_no_code_and_gets_her_family(client, hms, api):
    token = jwt("9123456789")
    sel = choose_anita(client, patient_token=token)
    who = say(client, sel, labels(sel)[0], patient_token=token)
    assert labels(who) == ["Mere liye (Rina Kumari)", "Asha Devi (Mother)", "Kisi aur ke liye"]
    confirm = say(client, who, "Asha Devi (Mother)", patient_token=token)
    assert confirm["clarify"]["kind"] == "book_confirm"                      # all known: no name, age, OTP
    assert "Asha Devi (60, F)" in confirm["message"] and "patient_session" not in confirm
    assert not [c for c in api.calls if "otp" in c[1]]
    done = say(client, confirm, "yes", patient_token=token)
    assert done["kind"] == "booked"
    (_, _, body, headers), = api.bookings()
    assert body["patient"]["fullName"] == "Asha Devi" and body["patient"]["mobile"] == "9123456789"
    assert headers["authorization"] == f"Bearer {token}"


def test_signed_in_self_skips_every_question(client, hms, api):
    token = jwt("9123456789")
    sel = choose_anita(client, patient_token=token)
    who = say(client, sel, labels(sel)[0], patient_token=token)
    confirm = say(client, who, "myself", patient_token=token)
    assert confirm["clarify"]["kind"] == "book_confirm" and "Rina Kumari (32, F)" in confirm["message"]


def test_verified_mobile_needs_the_scope():
    raw = "wa-key"
    settings = ServiceSettings(rate_limit_chat=10_000, api_clients=(
        ApiClient("whatsapp", frozenset({"chat", "verified_mobile"}), hash_key(raw)),
        ApiClient("site", frozenset({"chat"}), hash_key("site-key"))))
    with TestClient(create_app(settings)) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        fake = FakePublicApi()
        c.app.state.booking = HmsPublicClient("http://hms", "pub-key", transport=httpx.MockTransport(fake))

        def run(key):
            def post(**b):
                r = c.post(URL, json=b, headers={"X-Api-Key": key})
                assert r.status_code == 200, r.text
                return r.json()
            lst = post(text="skin ka doctor chahiye", city="Kolkata", verified_mobile="919876543210")
            s = post(text="Choose", session_id=lst["session_id"], select_doctor_id=ANITA, verified_mobile="919876543210")
            for text in (labels(s)[0], "Mere liye", "Rina", "30", "Mahila"):
                s = post(text=text, session_id=s["session_id"], verified_mobile="919876543210")
            return s
        assert run(raw)["clarify"]["kind"] == "book_confirm"                  # WhatsApp: the number is known
        assert run("site-key")["clarify"]["kind"] == "book_mobile"            # anyone else: OTP as usual


# ============================================================================ refusals, outages, exits
def test_refused_booking_offers_other_days_and_never_retries(client, hms, api):
    lst = chat(client, "MS ortho", city="Kolkata")
    s = chat(client, "Choose", session_id=lst["session_id"], select_doctor_id=RAKESH, patient_token=jwt("9123456789"))
    s = say(client, s, labels(s)[0], patient_token=jwt("9123456789"))
    s = say(client, s, "myself", patient_token=jwt("9123456789"))
    out = say(client, s, "yes", patient_token=jwt("9123456789"))
    assert out["clarify"]["kind"] == "book_day" and "on leave" in out["message"]
    assert labels(out)[-1] == "Other doctors" and len(api.bookings()) == 1


def test_hms_down_books_nothing_and_says_so(client, hms):
    fake = FakePublicApi()
    client.app.state.booking = HmsPublicClient("http://hms", "pub-key", transport=httpx.MockTransport(fake))
    try:
        token = jwt("9123456789")
        s = choose_anita(client, patient_token=token)
        s = say(client, s, labels(s)[0], patient_token=token)
        s = say(client, s, "myself", patient_token=token)
        fake.down = True
        out = say(client, s, "yes", patient_token=token)
        assert out["kind"] == "follow_up" and "kuch book nahi hua" in out["message"]
    finally:
        client.app.state.booking = None


def test_cancel_at_any_step(client, hms, api):
    sel = choose_anita(client)
    s = say(client, sel, labels(sel)[0])
    out = say(client, s, "rehne do")
    assert out["kind"] == "info" and "book nahi hua" in out["message"] and not api.bookings()


def test_emergency_mid_booking_takes_over(client, hms, api):
    sel = choose_anita(client)
    s = say(client, sel, labels(sel)[0])
    out = say(client, s, "seene mein tez dard aur saans nahi aa rahi")
    assert out["kind"] == "emergency"
    after = say(client, out, "Rina")                                         # the booking is gone
    assert after["kind"] != "follow_up" or (after.get("clarify") or {}).get("kind") != "book_age"


def test_a_new_search_mid_booking_ends_it(client, hms, api):
    sel = choose_anita(client)
    out = say(client, sel, "dil ka doctor chahiye", city="Kolkata")
    assert out["kind"] == "doctor_list" and "Cardiologist" in out["doctors"]["specialist"]


def test_change_goes_back_to_the_day(client, hms, api):
    token = jwt("9123456789")
    s = choose_anita(client, patient_token=token)
    s = say(client, s, labels(s)[0], patient_token=token)
    s = say(client, s, "myself", patient_token=token)
    out = say(client, s, "Badlo", patient_token=token)
    assert out["clarify"]["kind"] == "book_day"


def test_two_shifts_ask_which(client, hms, api, monkeypatch):
    orig = Fake1Hms._availability

    def two(a):
        out = orig(a)
        for d in out.get("days") or []:
            if d["isAvailable"]:
                d["shifts"].append({"name": "Evening", "start": "17:00", "end": "20:00"})
        return out
    monkeypatch.setattr(Fake1Hms, "_availability", staticmethod(two))
    sel = choose_anita(client)
    shift = say(client, sel, labels(sel)[0])
    assert shift["clarify"]["kind"] == "book_shift" and labels(shift)[:2] == ["Subah 09:00-13:00", "Shaam 17:00-20:00"]
    who = say(client, shift, "shaam")
    assert who["clarify"]["kind"] == "book_who"


def test_without_the_public_api_choosing_is_as_before(client, hms):
    sel = choose_anita(client)
    assert sel["kind"] == "doctor_selected" and sel["clarify"] is None and "Book a slot" in sel["message"]


def test_hindi_booking_is_in_hindi(client, hms, api):
    token = jwt("9123456789")
    lst = chat(client, "मुझे स्किन डॉक्टर चाहिए", city="Kolkata")
    s = chat(client, "Choose", session_id=lst["session_id"], select_doctor_id=ANITA, patient_token=token)
    assert "कौन-सा दिन" in s["message"] and all(ord(c) < 128 or "\u0900" <= c <= "\u097f" for c in labels(s)[1])
    assert "अक्टूबर" in "".join(labels(s)) or "नवंबर" in "".join(labels(s)) or "दिसंबर" in "".join(labels(s)) \
        or any(m in "".join(labels(s)) for m in ("जनवरी", "सितंबर"))
    s = say(client, s, labels(s)[0], patient_token=token)
    assert s["language"] == "hindi" and s["message"] == "अपॉइंटमेंट किसके लिए है?"   # a tapped date chip keeps Hindi
    confirm = say(client, s, "मेरे लिए", patient_token=token)
    assert confirm["clarify"]["kind"] == "book_confirm" and "बुक करूँ?" in confirm["message"]
    assert labels(confirm) == ["हाँ, बुक करो", "बदलो", "रहने दो"]


# ============================================================================ privacy
def test_no_names_numbers_or_codes_reach_telemetry(client, hms, api):
    captured = []

    class Rec:
        enabled = True

        def query_hash(self, text):
            return text

        def record(self, kind, item):
            captured.append(item)
    before = client.app.state.telemetry
    client.app.state.telemetry = Rec()
    try:
        s = choose_anita(client)
        for text in (labels(s)[0], "Mere liye", "Rina Kumari", "32", "Mahila", "9876543210", "123456"):
            s = say(client, s, text)
    finally:
        client.app.state.telemetry = before
    dump = json.dumps(captured)
    assert "Rina" not in dump and "9876543210" not in dump and "123456" not in dump and "eyJ" not in dump


# ============================================================================ WhatsApp
def test_whatsapp_booking_round_trip(client, hms, api):
    wa = {"channel": "whatsapp"}

    def post(**b):
        r = client.post(URL, json={**wa, **b})
        assert r.status_code == 200, r.text
        out = r.json()
        for m in out["whatsapp"]:
            assert validate(m) == [], m
        return out
    lst = post(text="skin ka doctor chahiye", city="Kolkata")
    s = post(session_id=lst["session_id"], reply_id=f"doc:{ANITA}")
    assert s["clarify"]["kind"] == "book_day"
    rows = s["whatsapp"][0]["interactive"]
    first = (rows["action"]["buttons"][0]["reply"]["id"] if rows["type"] == "button"
             else rows["action"]["sections"][0]["rows"][0]["id"])
    s = post(session_id=s["session_id"], reply_id=first)
    assert s["clarify"]["kind"] == "book_who"
    for text in ("Mere liye", "Rina", "30", "Mahila", "9876543210", "123456"):
        s = post(session_id=s["session_id"], text=text)
    assert s["clarify"]["kind"] == "book_confirm" and "patient_session" not in s
    done = post(session_id=s["session_id"], reply_id=s["whatsapp"][0]["interactive"]["action"]["buttons"][0]["reply"]["id"])
    assert done["kind"] == "booked" and [m["type"] for m in done["whatsapp"]] == ["text"]


# ============================================================================ deployment without a key
def test_booking_works_without_an_api_key(client, hms):
    """1HMS's public API takes anonymous callers; the deploy workflow passes no key, so booking must not need one."""
    fake = FakeAnonymousApi()
    client.app.state.booking = HmsPublicClient("http://hms", None, transport=httpx.MockTransport(fake))
    try:
        sel = choose_anita(client)
        assert sel["clarify"]["kind"] == "book_day"
        out = say(client, sel, labels(sel)[0])
        for t in ("Mere liye", "Rina Kumari", "32", "Mahila", "9876543210", "123456", "Haan, book karo"):
            out = say(client, out, t)
        assert out["kind"] == "booked"
        assert fake.sent_key and not any(fake.sent_key)
    finally:
        client.app.state.booking = None


def test_no_key_means_no_hospital_alerts():
    c = HmsPublicClient("http://hms", None)
    assert c.enabled and not c.identified
    from hms_router.application.emergency_help import EmergencyHelp
    assert EmergencyHelp(c).enabled and not EmergencyHelp(c).alerts


class FakeAnonymousApi(FakePublicApi):
    def __init__(self):
        super().__init__()
        self.sent_key = []

    def __call__(self, request):
        self.sent_key.append("x-api-key" in {k.lower() for k in request.headers})
        keyed = httpx.Request(request.method, request.url, headers={**request.headers, "X-Api-Key": "pub-key"},
                              content=request.content)                    # same answers as the keyed fake
        return super().__call__(keyed)
