"""After booking (vision Phase 4): directions, "I've reached", check in from the phone, token status; the booking passes the
patient's language to 1HMS for the WhatsApp confirmation and reminders."""
import json

import httpx
import pytest

from hms_router.application.visit_help import wants
from hms_router.booking_client import HmsPublicClient
from hms_router.channels.whatsapp import validate
from tests.test_vita_booking import APPT, FakePublicApi, choose_anita, jwt, labels, say
from tests.test_vita_doctor_chat import URL, chat, client, hms  # noqa: F401

VISIT = {
    "appointmentId": APPT, "reference": "ABCDEF12", "statusCode": "PRE_APPOINTMENT", "status": "Requested", "confirmed": False,
    "cancelled": False, "completed": False, "doctorName": "Dr. Anita Sen", "hospitalName": "Salt Lake Care", "address": "Sector V",
    "city": "Kolkata", "latitude": 22.58, "longitude": 88.42, "mapUrl": "https://www.google.com/maps/dir/?api=1&destination=22.58,88.42",
    "date": "2026-10-26", "time": "09:00", "isToday": True, "fee": 600,
    "guide": {"landmark": "Opposite City Centre", "entranceNote": "Use Gate 2", "opdLocation": "1st floor, left of the lift",
              "registrationDesk": "Counter 3, ground floor", "paymentNote": "Cash and UPI accepted."},
    "canCheckIn": True, "token": None,
}


class VisitApi(FakePublicApi):
    def __init__(self):
        super().__init__()
        self.visit = json.loads(json.dumps(VISIT))
        self.far = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/visit"):
            self.calls.append((request.method, path, {}, dict(request.headers)))
            if APPT not in path:
                return httpx.Response(404, json={"success": False, "message": "Appointment not found."})
            return httpx.Response(200, json={"success": True, "visit": self.visit})
        if path.endswith("/public/tokens"):
            body = json.loads(request.content)
            self.calls.append((request.method, path, body, dict(request.headers)))
            if self.far:
                return httpx.Response(400, json={"success": False, "message": "You don't appear to be at the hospital yet. Please check in at reception if this seems wrong."})
            self.visit["canCheckIn"] = False
            self.visit["token"] = {"tokenNo": 7, "status": "WAITING", "currentServingTokenNo": 4, "positionInQueue": 3, "estimatedWaitMinutes": 25}
            return httpx.Response(200, json={"success": True, "tokenNo": 7, "status": "WAITING"})
        if path.endswith("/appointments/mine"):
            self.calls.append((request.method, path, {}, dict(request.headers)))
            return httpx.Response(200, json={"success": True, "appointments": [
                {"appointmentId": "00000000-0000-0000-0000-0000000000aa", "apptDate": "2020-01-01T00:00:00", "statusCode": "COMPLETED"},
                {"appointmentId": APPT, "apptDate": "2099-10-26T00:00:00", "startAt": "2099-10-26T09:00:00", "statusCode": "PRE_APPOINTMENT"}]})
        return super().__call__(request)

    def tokens(self):
        return [c for c in self.calls if c[1].endswith("/public/tokens")]


@pytest.fixture
def api(client, hms):
    fake = VisitApi()
    client.app.state.booking = HmsPublicClient("http://hms", "pub-key", proxy_secret="proxy-secret", transport=httpx.MockTransport(fake))
    yield fake
    client.app.state.booking = None


def book(client):
    sel = choose_anita(client)
    out = say(client, sel, labels(sel)[0])
    for t in ("Mere liye", "Rina Kumari", "32", "Mahila", "9876543210", "123456", "Haan, book karo"):
        out = say(client, out, t)
    assert out["kind"] == "booked"
    return out


@pytest.mark.parametrize("text, what", [
    ("main pahunch gayi", "arrival"), ("hospital aa gaya hoon", "arrival"), ("I've reached", "arrival"), ("मैं पहुँच गया", "arrival"),
    ("rasta batao", "directions"), ("how do I get there?", "directions"), ("रास्ता बताओ", "directions"),
    ("mera number kab aayega", "status"), ("how long will it take?", "status"), ("abhi check in karo", "checkin"),
    ("अभी चेक-इन करो", "checkin"), ("bukhar aa gaya hai", None), ("skin ka doctor chahiye", None)])
def test_what_the_patient_asks(text, what):
    assert wants(text) == what


def test_the_booking_tells_1hms_the_patients_language(client, hms, api):
    book(client)
    body = api.bookings()[-1][2]
    assert body["language"] == "hinglish"


def test_directions_after_booking(client, hms, api):
    out = say(client, book(client), "rasta batao")
    assert out["kind"] == "visit" and out["visit_step"] == "directions"
    assert out["visit"]["mapUrl"].endswith("22.58,88.42")
    assert "Pehchaan: Opposite City Centre" in out["message"] and "Gate: Use Gate 2" in out["message"]


def test_arrival_steps_then_check_in_with_the_location(client, hms, api):
    out = say(client, book(client), "main pahunch gayi")
    assert out["visit_step"] == "arrival"
    assert "Counter 3, ground floor" in out["message"] and "ABCDEF12" in out["message"]
    assert "Fees (Rs 600) counter par dein. Cash and UPI accepted." in out["message"]
    assert "1st floor, left of the lift" in out["message"]
    assert [o["label"] for o in out["clarify"]["options"]] == ["Abhi check in karo"]
    done = client.post(URL, json={"session_id": out["session_id"], "text": "location", "lat": 22.5801, "lng": 88.4201,
                                  "client_ip": "203.0.113.5"}).json()
    assert done["visit_step"] == "checked_in"
    assert "Aapka token 7 hai. Abhi 4 chal raha hai. Lagbhag 25 minute." in done["message"]
    (_, _, body, headers), = api.tokens()
    assert body == {"appointmentId": APPT, "latitude": 22.5801, "longitude": 88.4201}
    assert headers["x-forwarded-client-ip"] == "203.0.113.5"


def test_check_in_away_from_the_hospital_says_so(client, hms, api):
    api.far = True
    booked = book(client)
    out = chat(client, "abhi check in karo", session_id=booked["session_id"], lat=22.9, lng=88.9)
    assert out["kind"] == "info" and "abhi hospital nahi pahunche" in out["message"]


def test_check_in_without_a_location_asks_for_it(client, hms, api):
    out = say(client, book(client), "check in karo")
    assert out["clarify"]["kind"] == "location"
    done = client.post(URL, json={"session_id": out["session_id"], "text": "location bhej di", "lat": 22.58, "lng": 88.42}).json()
    assert done["visit_step"] == "checked_in"


def test_token_status(client, hms, api):
    api.visit["canCheckIn"] = False
    api.visit["token"] = {"tokenNo": 12, "currentServingTokenNo": 9, "estimatedWaitMinutes": 15}
    out = say(client, book(client), "how long will it take?")
    assert out["visit_step"] == "status"
    assert out["message"] == "Aapka token 12 hai. Abhi 9 chal raha hai. Lagbhag 15 minute."


def test_the_website_names_the_visit(client, hms, api):
    out = chat(client, "I've reached", visit_appointment_id=APPT)
    assert out["kind"] == "visit" and out["language"] == "english"
    assert out["message"].startswith("Welcome! Here's what to do:")
    assert "Go to Counter 3, ground floor and give your booking ID ABCDEF12." in out["message"]


def test_a_signed_in_patient_is_found_without_a_booking_in_this_chat(client, hms, api):
    out = chat(client, "main hospital pahunch gaya", patient_token=jwt("9876543210"))
    assert out["kind"] == "visit" and out["visit"]["appointmentId"] == APPT


def test_no_known_visit_points_to_my_appointments(client, hms, api):
    out = chat(client, "main pahunch gayi")
    assert out["kind"] == "info" and "Meri appointments" in out["message"]


def test_directions_without_a_visit_is_an_ordinary_question(client, hms, api):
    out = chat(client, "rasta batao", city="Kolkata")
    assert out["kind"] != "visit"


def test_an_emergency_still_comes_first(client, hms, api):
    out = say(client, book(client), "main pahunch gayi par seene mein tez dard aur saans nahi aa rahi")
    assert out["kind"] == "emergency"


def test_whatsapp_gets_the_map_link_and_a_send_location_button(client, hms, api):
    booked = book(client)
    out = chat(client, "main pahunch gayi", session_id=booked["session_id"], channel="whatsapp")
    msgs = out["whatsapp"]
    assert "https://www.google.com/maps/dir/" in msgs[0]["text"]["body"]
    assert msgs[1]["interactive"]["type"] == "location_request_message"
    assert all(validate(m) == [] for m in msgs)
