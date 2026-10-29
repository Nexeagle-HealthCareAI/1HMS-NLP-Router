"""Home to the clinic: after booking Vita asks how the patient is coming and when they leave, saves that with 1HMS (WhatsApp
"time to leave"), and on the day takes them there: navigation, cab / transit links, ETA, "running late" and arrival for reception."""
import json
from datetime import datetime

import httpx
import pytest

from hms_router.application.journey import km_between, links, mode_of, parse_leave, travel_minutes, wants
from hms_router.booking_client import HmsPublicClient
from tests.test_vita_booking import APPT, choose_anita, labels, say
from tests.test_vita_doctor_chat import chat, client, hms  # noqa: F401
from tests.test_vita_visit import VisitApi


class JourneyApi(VisitApi):
    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/journey") or path.endswith("/journey/progress"):
            body = json.loads(request.content or b"{}")
            self.calls.append((request.method, path, body, dict(request.headers)))
            j = self.visit.get("journey") or {"status": "planned", "shareWithHospital": False}
            if path.endswith("/journey"):
                j.update({k: v for k, v in (("travelMode", body.get("travelMode")), ("shareWithHospital", body.get("shareWithHospital")))
                          if v is not None})
            else:
                j["status"] = body["status"]
                for k in ("etaMinutes", "distanceKm", "lateByMinutes"):
                    if k in body:
                        j[k] = body[k]
            self.visit["journey"] = j
            return httpx.Response(200, json={"success": True, "journey": j})
        return super().__call__(request)

    def journey_calls(self):
        return [c for c in self.calls if "/journey" in c[1]]


@pytest.fixture
def api(client, hms):
    fake = JourneyApi()
    client.app.state.booking = HmsPublicClient("http://hms", "pub-key", proxy_secret="proxy-secret", transport=httpx.MockTransport(fake))
    yield fake
    client.app.state.booking = None


def book(client, **kw):
    sel = choose_anita(client, **kw)
    out = say(client, sel, labels(sel)[0], **kw)
    for t in ("Mere liye", "Rina Kumari", "32", "Mahila", "9876543210", "123456", "Haan, book karo"):
        out = say(client, out, t, **kw)
    assert out["kind"] == "booked"
    return out


# ============================================================================ pieces
@pytest.mark.parametrize("text, mode", [("car se", "drive"), ("bike", "drive"), ("auto / cab", "auto_cab"), ("ola karunga", "auto_cab"),
                                        ("bus se aaunga", "transit"), ("metro", "transit"), ("paidal", "walk"), ("on foot", "walk"),
                                        ("haan", None)])
def test_how_they_come(text, mode):
    assert mode_of(text) == mode


START = datetime(2026, 10, 26, 10, 0)
NOW = datetime(2026, 10, 26, 7, 0)


@pytest.mark.parametrize("text, at", [
    ("8 baje", "08:00"), ("8 30", "08:30"), ("8:30", "08:30"), ("saade 8", "08:30"), ("sava 9", "09:15"), ("paune 9", "08:45"),
    ("subah 7", "07:00"), ("8.15 am", "08:15"), ("abhi", "07:00"), ("20 minute mein", "07:20"), ("aadhe ghante mein", "07:30"),
    ("ek ghante baad", "08:00"), ("9", "09:00")])
def test_leave_times(text, at):
    assert parse_leave(text, START, NOW).strftime("%H:%M") == at


def test_a_bare_hour_is_the_one_before_an_evening_visit():
    assert parse_leave("6 baje", datetime(2026, 10, 26, 19, 0), NOW).strftime("%H:%M") == "18:00"
    assert parse_leave("shaam 5", START, NOW).strftime("%H:%M") == "17:00"


@pytest.mark.parametrize("text", ["pata nahi", "kal", "doctor achhe hain"])
def test_not_a_time(text):
    assert parse_leave(text, START, NOW) is None


def test_travel_estimate():
    km = km_between(22.50, 88.35, 22.58, 88.42)
    assert 11 < km < 12
    assert travel_minutes("drive", km) < travel_minutes("auto_cab", km) < travel_minutes("transit", km) < travel_minutes("walk", km)
    assert travel_minutes("drive", None) is None


@pytest.mark.parametrize("text, what", [
    ("nikal raha hoon", "leaving"), ("ghar se nikal gayi", "leaving"), ("on my way", "leaving"), ("निकल रहा हूँ", "leaving"),
    ("kitni der lagegi", "eta"), ("how far is it", "eta"), ("15 min late hounga", "late"), ("late ho jaunga", "late"),
    ("traffic mein phans gaya", "late"), ("kal raat late bukhar aaya", None), ("rasta batao", None), ("bukhar hai", None)])
def test_what_they_say_on_the_day(text, what):
    assert wants(text) == what


def test_cab_and_transit_links():
    visit = {"hospitalName": "Salt Lake Care", "address": "Sector V", "city": "Kolkata", "latitude": 22.58, "longitude": 88.42}
    cab = {x["id"]: x["url"] for x in links("auto_cab", visit)}
    assert cab["uber"].startswith("https://m.uber.com/ul/?action=setPickup&pickup=my_location&dropoff%5Blatitude%5D=22.58")
    assert "drop_lat=22.58&drop_lng=88.42" in cab["ola"]
    bus = {x["id"]: x["url"] for x in links("transit", visit)}
    assert bus["transit"].endswith("destination=22.58,88.42&travelmode=transit") and "uber" not in bus
    assert [x["id"] for x in links("drive", visit)] == ["navigate"]


# ============================================================================ the plan after booking
def test_booked_then_how_when_and_share(client, hms, api):
    out = book(client)
    assert out["clarify"]["kind"] == "journey_mode" and "kaise aayenge" in out["message"]
    assert labels(out) == ["Car / bike", "Auto / cab", "Bus / train", "Paidal"]
    out = say(client, out, "Auto / cab")
    assert out["clarify"]["kind"] == "journey_leave" and "Kab nikalenge?" in out["message"]
    assert len(labels(out)) == 2                                                     # the suggested time, and 15 min earlier
    out = say(client, out, labels(out)[0])
    assert out["clarify"]["kind"] == "journey_share" and "location nahi" in out["message"]
    out = say(client, out, "Haan, bata do")
    assert out["kind"] == "info" and "WhatsApp par yaad dila dungi" in out["message"]
    (method, path, body, _), = api.journey_calls()
    assert method == "PUT" and path.endswith(f"/{APPT}/journey")
    assert body["travelMode"] == "auto_cab" and body["shareWithHospital"] is True and len(body["leaveAt"]) == 5


def test_a_leave_time_from_the_location(client, hms, api):
    out = book(client, lat=22.50, lng=88.35)
    out = say(client, out, "car se", lat=22.50, lng=88.35)
    assert "minute ka rasta" in out["message"]


def test_a_typed_time_and_no_sharing(client, hms, api):
    out = say(client, say(client, book(client), "bus se"), "saade 8")
    out = say(client, out, "nahi")
    assert "Raste mein hone par" not in out["message"]
    body = api.journey_calls()[0][2]
    assert body == {"travelMode": "transit", "leaveAt": "08:30", "shareWithHospital": False}


def test_skipping_the_plan(client, hms, api):
    out = say(client, book(client), "baad mein")
    assert out["kind"] == "info" and "nikal raha hoon" in out["message"] and not api.journey_calls()


def test_a_new_ask_ends_the_plan(client, hms, api):
    out = say(client, book(client), "dil ka doctor chahiye")
    assert out["kind"] == "doctor_list"


def test_an_emergency_while_planning(client, hms, api):
    out = say(client, book(client), "seene mein tez dard aur saans nahi aa rahi")
    assert out["kind"] == "emergency"


# ============================================================================ on the day
def planned(client, mode="Auto / cab"):
    out = say(client, say(client, book(client), mode), "8 baje")
    return say(client, out, "Haan, bata do")


def test_leaving_now_shows_the_way_and_tells_the_hospital(client, hms, api):
    out = say(client, planned(client), "nikal raha hoon", lat=22.50, lng=88.35)
    assert out["kind"] == "visit" and out["visit_step"] == "navigate"
    assert {x["id"] for x in out["journey"]["links"]} == {"navigate", "uber", "ola"}
    assert out["journey"]["etaMinutes"] and "Pehchaan: Opposite City Centre" in out["message"]
    _, path, body, _ = api.journey_calls()[-1]
    assert path.endswith("/journey/progress") and body["status"] == "on_the_way" and body["etaMinutes"] == out["journey"]["etaMinutes"]
    assert "lat" not in json.dumps(body).lower()                                     # never the location


def test_eta_on_the_way(client, hms, api):
    out = say(client, planned(client), "nikal raha hoon", lat=22.50, lng=88.35)
    out = say(client, out, "kitni der lagegi")
    assert out["visit_step"] == "eta" and "minute" in out["message"] and "km baaki" in out["message"]


def test_running_late_when_shared(client, hms, api):
    out = say(client, planned(client), "20 min late hounga")
    assert out["kind"] == "info" and "20 minute late" in out["message"]
    _, _, body, _ = api.journey_calls()[-1]
    assert body == {"status": "late", "lateByMinutes": 20}


def test_running_late_asks_first_when_not_shared(client, hms, api):
    out = say(client, say(client, say(client, book(client), "car se"), "8 baje"), "nahi")
    out = say(client, out, "late ho jaunga")
    assert out["clarify"]["kind"] == "journey_late"
    n = len(api.journey_calls())
    out = say(client, out, "Haan, bata do")
    assert "15 minute late" in out["message"]
    calls = api.journey_calls()[n:]
    assert calls[0][2] == {"shareWithHospital": True} and calls[1][2] == {"status": "late", "lateByMinutes": 15}


def test_late_and_no_consent_gives_the_number(client, hms, api):
    api.visit["contact"] = "033 4000 1234"
    out = say(client, say(client, say(client, book(client), "car se"), "8 baje"), "nahi")
    out = say(client, say(client, out, "late ho jaunga"), "nahi")
    assert "033 4000 1234" in out["message"]
    assert not [c for c in api.journey_calls() if c[1].endswith("/progress")]


def test_arrival_tells_the_hospital(client, hms, api):
    out = say(client, planned(client), "nikal raha hoon")
    out = say(client, out, "main pahunch gayi")
    assert out["visit_step"] == "arrival"
    assert api.journey_calls()[-1][2] == {"status": "arrived"}


def test_kitna_time_at_the_hospital_is_the_queue(client, hms, api):
    out = say(client, planned(client), "mera number kab aayega")
    assert out["visit_step"] == "status"
