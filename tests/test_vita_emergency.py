"""Emergency takeover (vision Phase 3): the emergency reply lists the nearest emergency-ready places and alerts them.

Number first (108 / 112) always; places under it from 1HMS (POST /public/emergency-alerts); no location -> ask for it, and
the location shared next gets the places; "Share my number" (WhatsApp sender / signed-in patient) sends the number, with
consent; 1HMS down -> the numbers still go out (degraded). Never a doctor list, never booking.
"""
import httpx
import pytest

from hms_router.booking_client import HmsPublicClient
from hms_router.application.emergency_help import capability_for
from hms_router.channels.whatsapp import validate
from tests.test_vita_booking import FakePublicApi, jwt
from tests.test_vita_doctor_chat import URL, chat, client, hms  # noqa: F401

EVENT = "e0000000-0000-0000-0000-000000000001"
PLACES = [
    {"settingId": "s1", "name": "Salt Lake Care", "kind": "HOSPITAL", "phone": "03340001111", "alternatePhone": None,
     "distanceKm": 1.2, "address": "Sector V", "landmark": "Near City Centre", "city": "Kolkata", "latitude": 22.58,
     "longitude": 88.42, "is24x7": True, "capabilities": ["cardiac", "icu"]},
    {"settingId": "s2", "name": "District Hospital", "kind": "HOSPITAL", "phone": "108", "distanceKm": 4.8,
     "latitude": 22.6, "longitude": 88.45, "is24x7": True, "capabilities": []},
]
CHEST = "seene mein tez dard aur saans nahi aa rahi"


class EmergencyApi(FakePublicApi):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.places = list(PLACES)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/public/emergency-alerts"):
            if request.headers.get("X-Api-Key") != "pub-key":
                return httpx.Response(401)
            if self.down:
                self.calls.append((request.method, path, {}, dict(request.headers)))
                return httpx.Response(503)
            import json
            body = json.loads(request.content)
            self.calls.append((request.method, path, body, dict(request.headers)))
            repeated = sum(1 for c in self.calls if c[1].endswith("/emergency-alerts")) > 1
            return httpx.Response(200, json={"success": True, "eventId": EVENT if self.places else None, "places": self.places,
                                             "alerted": 0 if repeated else len(self.places), "repeated": repeated})
        if path.endswith("/share-number"):
            import json
            self.calls.append((request.method, path, json.loads(request.content), dict(request.headers)))
            return httpx.Response(200, json={"success": True, "count": 2})
        return super().__call__(request)

    def alerts(self):
        return [c for c in self.calls if c[1].endswith("/public/emergency-alerts")]

    def shares(self):
        return [c for c in self.calls if c[1].endswith("/share-number")]


@pytest.fixture
def api(client, hms):
    fake = EmergencyApi()
    client.app.state.booking = HmsPublicClient("http://hms", "pub-key", proxy_secret="proxy-secret",
                                               transport=httpx.MockTransport(fake))
    yield fake
    client.app.state.booking = None


def test_an_emergency_lists_and_alerts_the_nearest_places(client, hms, api):
    out = chat(client, CHEST, lat=22.57, lng=88.41, client_ip="203.0.113.9")
    assert out["kind"] == "emergency" and "doctors" not in out
    lines = out["message"].split("\n")
    assert "108" in lines[0] and "112" in lines[0]                                 # the number stays first
    assert "1. Salt Lake Care (1.2 km) - 03340001111" in out["message"]
    assert "2. District Hospital (4.8 km) - 108" in out["message"]
    assert "bata diya" in out["message"]                                           # Hinglish: "I've told them"
    em = out["emergency"]
    assert em["eventId"] == EVENT and em["alerted"] == 2 and not em["needsLocation"]
    assert em["places"][0]["mapUrl"].endswith("destination=22.58,88.42")
    assert em["canShareNumber"] is False and out.get("clarify") is None           # web guest: the site asks for the number
    (_, _, body, headers), = api.alerts()
    assert body["capability"] == "cardiac" and body["channel"] == "web" and body["sessionRef"] == out["session_id"]
    assert "chest pain" in body["reason"]
    assert headers["x-forwarded-client-ip"] == "203.0.113.9"


def test_no_location_asks_for_it_and_the_shared_location_brings_the_places(client, hms, api):
    out = chat(client, CHEST, city="Kolkata")
    assert out["kind"] == "emergency" and out["emergency"]["needsLocation"] is True
    assert "location" in out["message"].lower() and not api.alerts()
    nxt = client.post(URL, json={"session_id": out["session_id"], "lat": 22.57, "lng": 88.41}).json()
    assert nxt["kind"] == "emergency" and nxt["emergency"]["places"][0]["name"] == "Salt Lake Care"
    assert "112" in nxt["message"].split("\n")[0]
    assert len(api.alerts()) == 1


def test_a_signed_in_patient_shares_their_number_with_one_tap(client, hms, api):
    token = jwt("9876543210")
    out = chat(client, "chest pain and I can't breathe", lat=22.57, lng=88.41, patient_token=token)
    assert out["emergency"]["canShareNumber"] is True
    assert [o["label"] for o in out["clarify"]["options"]] == ["Share my number"]
    done = chat(client, "Share my number", session_id=out["session_id"], patient_token=token)
    assert done["kind"] == "info" and "have your number" in done["message"]
    (_, path, body, _), = api.shares()
    assert path.endswith(f"/public/emergency-alerts/{EVENT}/share-number")
    assert body == {"mobile": "9876543210", "consent": True}


def test_whatsapp_sends_the_places_as_text_and_a_share_button(client, hms, api):
    out = chat(client, CHEST, lat=22.57, lng=88.41, channel="whatsapp")
    msgs = out["whatsapp"]
    assert msgs[0]["type"] == "text" and "Salt Lake Care" in msgs[0]["text"]["body"]
    assert all(validate(m) == [] for m in msgs)
    assert api.alerts()[0][2]["channel"] == "whatsapp"


def test_whatsapp_without_location_asks_with_the_send_location_button(client, hms, api):
    out = chat(client, CHEST, channel="whatsapp")
    kinds = [m.get("interactive", {}).get("type") for m in out["whatsapp"]]
    assert kinds == [None, "location_request_message"]
    assert all(validate(m) == [] for m in out["whatsapp"])


def test_hms_down_still_gives_the_numbers(client, hms, api):
    api.down = True
    out = chat(client, CHEST, lat=22.57, lng=88.41)
    assert out["kind"] == "emergency" and "112" in out["message"]
    assert "emergency_places" in out["degraded"] and out["emergency"]["places"] == []


def test_nothing_nearby_says_call_112(client, hms, api):
    api.places = []
    out = chat(client, CHEST, lat=22.57, lng=88.41)
    assert out["emergency"]["places"] == [] and "112" in out["message"].split("\n")[-1]


def test_booking_off_means_the_emergency_reply_is_unchanged(client, hms):
    out = chat(client, CHEST, lat=22.57, lng=88.41)
    assert out["kind"] == "emergency" and "emergency" not in out


def test_mid_booking_emergency_lists_places_and_ends_the_booking(client, hms, api):
    from tests.test_vita_booking import choose_anita, labels, say
    sel = choose_anita(client, lat=22.57, lng=88.41)
    s = say(client, sel, labels(sel)[0], lat=22.57, lng=88.41)
    out = say(client, s, CHEST, lat=22.57, lng=88.41)
    assert out["kind"] == "emergency" and out["emergency"]["places"]
    assert not api.bookings()


def test_hindi_emergency_speaks_hindi():
    from hms_router.application.emergency_help import EmergencyHelp
    from hms_router.voice import Voice

    class Client:
        enabled = True

        def emergency_alert(self, **kw):
            return {"eventId": EVENT, "places": PLACES, "alerted": 2}

    out = {"message": "अभी 108 पर कॉल करें।", "session_id": "s"}
    st: dict = {}
    EmergencyHelp(Client()).attach(out, st, {"lat": 22.5, "lng": 88.4}, ["chest pain"], "सीने में दर्द", Voice("hindi"))
    assert "सबसे नज़दीकी इमरजेंसी मदद:" in out["message"] and "मैंने उन्हें बता दिया" in out["message"]
    assert st["emergency"]["eventId"] == EVENT


@pytest.mark.parametrize("signs, text, cap", [
    (["chest pain"], "", "cardiac"), ([], "pregnant hai aur bleeding", "maternity"), ([], "baby ko saans nahi", "paediatric"),
    (["unconscious"], "road accident", "trauma"), (["unconscious"], "behosh", None)])
def test_capability_from_what_was_said(signs, text, cap):
    assert capability_for(signs, text) == cap
