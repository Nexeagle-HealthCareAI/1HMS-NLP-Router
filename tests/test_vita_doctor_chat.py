"""Vita doctor search in /chat (Vita Chat requirements, Part 2: VITA-1 to VITA-13).

A fake 1HMS MCP server with a small doctor table answers search_doctors the way the .NET tool does
(name, specialty, degree, distance, hospital, sort), plus get_doctor and list_doctor_names."""
import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from hms_router.doctor_intent import parse
from hms_router.doctors import DoctorDirectory, DoctorLookupError, McpClient

from tests.test_doctor_directory import API_CATEGORIES

H1, H2 = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
H3 = "33333333-3333-3333-3333-333333333333"
HOSPITALS = {H1: {"hospitalId": H1, "name": "Salt Lake Care", "city": "Kolkata", "lat": 22.58, "lng": 88.41},
             H2: {"hospitalId": H2, "name": "Howrah General", "city": "Kolkata", "lat": 22.59, "lng": 88.31},
             H3: {"hospitalId": H3, "name": "Sadar Hospital", "city": "Purnia", "lat": 25.78, "lng": 87.47}}
ON_LEAVE_TODAY = {"00000000-0000-0000-0000-000000000001"}          # Dr. Neha Sharma
WORKS_ON = {0, 1, 2, 3, 4, 5}                                      # nobody works on Sunday in this fake


def _doc(n, name, cat, degrees, hosp, exp, fee, gender="F"):
    return {"_gender": gender, "doctorId": f"00000000-0000-0000-0000-0000000000{n:02d}", "name": name, "degrees": degrees,
            "specialtyId": cat, "specialtyName": cat, "experienceYears": exp, "consultFee": fee,
            "discountedFee": None, "hospital": HOSPITALS[hosp], "availableToday": True}


DOCTORS = [
    _doc(1, "Dr. Neha Sharma", "Cardiologist (Heart)", "MBBS, MD (Medicine), DM (Cardiology)", H1, 14, 900),
    _doc(2, "Dr. Rakesh Sharma", "Orthopaedic Surgeon (Bone)", "MBBS, MS (Ortho)", H2, 20, 700, "M"),
    _doc(3, "Dr. Anita Sen", "Dermatologist (Skin)", "MBBS, MD (Dermatology)", H1, 9, 600),
    _doc(4, "Dr. Amit Das", "Cardiologist (Heart)", "MBBS, MD (Medicine)", H2, 6, 500, "M"),
    _doc(5, "Dr. Priya Bose", "Dermatologist (Skin)", "MBBS, DNB (Dermatology)", H2, 4, 400),
    _doc(6, "Dr. S. Roy", "General Physician", "MBBS", H1, 25, 300, "M"),
    _doc(7, "Dr. Kunal Ghosh", "Orthopaedic Surgeon (Bone)", "MBBS, DNB (Ortho)", H1, 11, 800, "M"),
]


def _public(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _km(lat1, lng1, lat2, lng2):
    return math.hypot((lat1 - lat2) * 111, (lng1 - lng2) * 111 * math.cos(math.radians(lat1)))


def _name_score(query, name):
    import difflib
    tokens = name.lower().replace(".", "").split()
    words = [w for w in query.lower().replace(".", "").split() if w != "dr"]
    return sum(max(difflib.SequenceMatcher(None, w, t).ratio() for t in tokens) for w in words) / max(len(words), 1)


class Fake1Hms:
    def __init__(self, key="k", fail=0, specialties_without_doctors=("Dentist",)):
        self.key, self.fail, self.empty = key, fail, set(specialties_without_doctors)
        self.calls: list[tuple[str, dict]] = []

    def _search(self, a):
        rows = list(DOCTORS)
        if a.get("specialtyIds"):
            rows = [d for d in rows if d["specialtyId"] in a["specialtyIds"]]
        if a.get("degree"):
            rows = [d for d in rows if a["degree"].lower() in d["degrees"].lower().replace(",", " ").split()]
        if a.get("city"):
            rows = [d for d in rows if d["hospital"]["city"].lower() == a["city"].lower()]
        if a.get("onlyAtHospital"):
            rows = [d for d in rows if d["hospital"]["hospitalId"] == a["hospitalId"]]
        if a.get("gender"):
            rows = [d for d in rows if d["_gender"] == a["gender"][0].upper()]
        if a.get("availableToday"):
            rows = [d for d in rows if d["doctorId"] not in ON_LEAVE_TODAY]
        out = []
        for d in rows:
            d = dict(_public(d), distanceKm=None, sameHospital=None, nameMatchScore=None)
            if a.get("query"):
                d["nameMatchScore"] = round(_name_score(a["query"], d["name"]), 2)
                if d["nameMatchScore"] < 0.72:
                    continue
            if "lat" in a:
                d["distanceKm"] = round(_km(a["lat"], a["lng"], d["hospital"]["lat"], d["hospital"]["lng"]), 1)
                if d["distanceKm"] > a.get("maxDistanceKm", 5):
                    continue
            if a.get("hospitalId"):
                d["sameHospital"] = d["hospital"]["hospitalId"] == a["hospitalId"]
            out.append(d)
        sort = a.get("sort") or "relevance"
        if sort == "distance":
            out.sort(key=lambda d: d["distanceKm"])
        elif sort == "experience":
            out.sort(key=lambda d: -d["experienceYears"])
        elif a.get("query"):
            out.sort(key=lambda d: -d["nameMatchScore"])
        if a.get("hospitalId"):
            out.sort(key=lambda d: not d["sameHospital"])
        off = a.get("offset", 0)
        return {"doctors": out[off: off + a.get("limit", 5)], "total": len(out)}

    @staticmethod
    def _hospitals(a):
        out = []
        for h in HOSPITALS.values():
            if a.get("city") and h["city"].lower() != a["city"].lower():
                continue
            score = None
            if a.get("query"):
                words = [w for w in a["query"].lower().split() if w not in ("hospital", "clinic", "haspatal")]
                import difflib
                targets = h["name"].lower().split() + [h["city"].lower()]
                score = sum(max(difflib.SequenceMatcher(None, w, t).ratio() for t in targets) for w in words) / max(len(words), 1)
                if score < 0.72:
                    continue
            out.append(dict(h, contact=h.get("contact"), nameMatchScore=score,
                            doctorCount=sum(d["hospital"]["hospitalId"] == h["hospitalId"] for d in DOCTORS)))
        out.sort(key=lambda h: -(h["nameMatchScore"] or 0))
        return {"hospitals": out[: a.get("limit", 5)], "total": len(out)}

    @staticmethod
    def _availability(a):
        import datetime as dt
        if not any(d["doctorId"] == a["doctorId"] for d in DOCTORS):
            return {"found": False}
        start = dt.date.fromisoformat(a["fromDate"]) if a.get("fromDate") else \
            (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=330)).date()
        days = []
        for i in range(a.get("days", 7)):
            d = start + dt.timedelta(days=i)
            ok = d.weekday() in WORKS_ON
            days.append({"date": d.isoformat(), "isAvailable": ok,
                         "shifts": [{"name": "Morning", "start": "09:00", "end": "13:00"}] if ok else []})
        nxt = next((x["date"] for x in days if x["isAvailable"]), None)
        return {"found": True, "doctorId": a["doctorId"], "days": days, "nextAvailableDate": nxt,
                "note": "Working days and shift hours only; the hospital confirms the exact appointment time."}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-Service-Key") != self.key:
            return httpx.Response(401)
        msg = json.loads(request.content)
        if "id" not in msg:
            return httpx.Response(202)
        method, params = msg["method"], msg.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fake", "version": "1"}}
        else:
            name, args = params["name"], params.get("arguments") or {}
            self.calls.append((name, args))
            if self.fail:
                self.fail -= 1
                return httpx.Response(503)
            if name == "list_specialties":
                data = {"specialties": [{"specialtyId": c, "name": c, "doctorCount": 0 if c in self.empty else 3,
                                         "aliases": []} for c in API_CATEGORIES]}
            elif name == "search_doctors":
                data = self._search(args)
            elif name == "get_doctor":
                d = next((d for d in DOCTORS if d["doctorId"] == args["doctorId"]), None)
                data = {"found": d is not None, "doctor": _public(d) if d else None}
            elif name == "search_hospitals":
                data = self._hospitals(args)
            elif name == "get_doctor_availability":
                data = self._availability(args)
            elif name == "list_doctor_names":
                data = {"doctors": [{"doctorId": d["doctorId"], "name": d["name"], "city": "Kolkata"} for d in DOCTORS]}
            else:
                data = None
            result = {"structuredContent": data}
        body = "event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def make_directory(server, limit=3):
    return DoctorDirectory(McpClient("http://api/mcp", "k", transport=httpx.MockTransport(server)), limit=limit)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        yield c


@pytest.fixture
def hms(client):
    server = Fake1Hms()
    client.app.state.doctors = make_directory(server)
    yield server
    client.app.state.doctors = DoctorDirectory(None)


URL = "/v2/agent/chat"


def chat(client, text, **kw):
    r = client.post(URL, json={"text": text, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def searches(server):
    return [a for n, a in server.calls if n == "search_doctors"]


# ============================================================================ intent parser (FR-2.1)
@pytest.mark.parametrize("text, modes", [
    ("skin ka doctor chahiye", ["specialty"]),
    ("dil ka doctor", ["specialty"]),
    ("haddi ka doctor", ["specialty"]),
    ("Dr. Sarma se milna hai", ["name"]),
    ("MS ortho", ["specialty", "degree"]),
    ("MD medicine wala doctor", ["specialty", "degree"]),
    ("paas mein koi doctor", ["distance"]),
    ("5 km ke andar", ["distance"]),
    ("MD cardiologist near Salt Lake", ["specialty", "degree", "place", "distance"]),
    ("2 din se bukhar aur khansi", ["symptom"]),
    ("pet mein dard", ["symptom"]),
    ("skin pe daane", ["symptom"]),
])
def test_parser_recognises_the_five_kinds_of_ask(text, modes):
    from hms_router.application.doctor_chat import medical_terms
    assert parse(text, medical_terms=medical_terms).modes == modes


def test_symptom_words_alone_never_skip_triage():
    from hms_router.application.doctor_chat import medical_terms
    for text in ("pet mein dard", "kaan mein dard", "eye pain", "sir mein dard aur chakkar"):
        assert not parse(text, medical_terms=medical_terms).direct, text


def test_parser_reads_radius_and_refinements():
    a = parse("inme se 3 km ke andar kam fees wala")
    assert a.refine and a.near and a.radius_km == 3 and a.sort == "fee"


# ============================================================================ DoctorDirectory (VITA-1)
def test_search_is_cached_for_identical_asks():
    server = Fake1Hms()
    d = make_directory(server)
    d.search(specialty_key="cardiology", city="Kolkata")
    d.search(specialty_key="cardiology", city="Kolkata")
    assert len(searches(server)) == 1 and d.stats["cacheHits"] == 1


def test_one_retry_then_success():
    server = Fake1Hms(fail=1)
    d = make_directory(server)
    assert d.names() and d.stats["retries"] == 1


def test_two_failures_raise_lookup_error():
    server = Fake1Hms(fail=10)
    with pytest.raises(DoctorLookupError):
        make_directory(server).search(specialty_key="cardiology")


def test_symptom_fallback_to_general_physician():
    out = make_directory(Fake1Hms()).for_specialist("Dentist", fallbacks=["General Physician"])
    assert out["fallbackUsed"] and out["specialist"] == "General Physician"


# ============================================================================ /chat journeys


def test_v2_with_lookups_off_says_so(client):
    client.app.state.doctors = DoctorDirectory(None)
    body = chat(client, "skin ka doctor chahiye")
    assert body["kind"] == "doctor_list" and body["degraded"] == ["doctor_lookup"] and body["message"]


def test_specialty_ask_lists_doctors_same_turn(client, hms):            # VITA-3
    body = chat(client, "skin ka doctor chahiye", city="Kolkata")
    assert body["kind"] == "doctor_list" and body["doctors"]["mode"] == ["specialty"]
    assert [d["name"] for d in body["doctors"]["items"]] == ["Dr. Anita Sen", "Dr. Priya Bose"]
    assert body["doctors"]["filters"]["specialtyId"] == "Dermatologist (Skin)"
    assert body["doctors"]["doctors"] == body["doctors"]["items"] and body["doctors"]["source"] == "1hms-mcp"
    assert searches(hms)[-1] == {"limit": 3, "specialtyIds": ["Dermatologist (Skin)"], "city": "Kolkata"}


def test_patient_text_is_never_sent_to_1hms(client, hms):              # NFR-2.6
    chat(client, "mujhe skin ka doctor chahiye please jaldi", city="Kolkata")
    sent = json.dumps(hms.calls)
    assert "mujhe" not in sent and "jaldi" not in sent


def test_misspelt_name_finds_the_doctor(client, hms):                   # VITA-4
    body = chat(client, "dr sarma se milna hai")
    assert "name" in body["doctors"]["mode"]
    assert body["doctors"]["items"][0]["name"].endswith("Sharma")


def test_equal_name_matches_are_told_apart_by_hospital_and_city(client, hms):   # VITA-4 / vision
    body = chat(client, "dr sharma")
    assert body["clarify"]["kind"] == "which_doctor"
    labels = [o["label"] for o in body["clarify"]["options"]]
    assert labels == ["Dr. Neha Sharma · Salt Lake Care, Kolkata", "Dr. Rakesh Sharma · Howrah General, Kolkata"]
    nxt = chat(client, "howrah wale", session_id=body["session_id"])        # typed answer picks one
    assert nxt["kind"] == "doctor_selected" and nxt["doctor"]["name"] == "Dr. Rakesh Sharma"


def test_unknown_name_says_so_and_offers_specialty(client, hms):
    body = chat(client, "dr zzyzxq")
    assert body["kind"] == "doctor_list" and body["doctors"]["items"] == []
    assert "zzyzxq" in body["message"] and body["clarify"]["kind"] == "no_match"


def test_degree_and_specialty_together(client, hms):                    # VITA-5
    body = chat(client, "MS ortho")
    assert [d["name"] for d in body["doctors"]["items"]] == ["Dr. Rakesh Sharma"]
    assert searches(hms)[-1]["degree"] == "MS"


def test_distance_without_location_asks_once_then_uses_city(client, hms):   # VITA-6
    first = chat(client, "paas mein koi cardiologist")
    assert first["kind"] == "follow_up" and first["clarify"]["kind"] == "location"
    assert [o["id"] for o in first["clarify"]["options"]] == ["share_location", "type_area", "use_city"]
    assert searches(hms) == []
    second = chat(client, "Use my city", session_id=first["session_id"], city="Kolkata")
    assert second["kind"] == "doctor_list" and second["doctors"]["filters"]["city"] == "Kolkata"
    assert "city" in second["message"]


def test_distance_with_location_is_nearest_first(client, hms):          # VITA-6
    body = chat(client, "cardiologist near me", lat=22.58, lng=88.40)
    items = body["doctors"]["items"]
    assert [d["name"] for d in items] == ["Dr. Neha Sharma"]           # Howrah is ~9 km away, outside 5 km
    assert items[0]["distanceKm"] is not None and body["doctors"]["filters"]["radiusKm"] == 5
    assert searches(hms)[-1]["sort"] == "distance" and searches(hms)[-1]["maxDistanceKm"] == 5


def test_location_answer_completes_the_pending_search(client, hms):
    first = chat(client, "5 km ke andar dermatologist")
    second = chat(client, "location bhej di", session_id=first["session_id"], lat=22.58, lng=88.40)
    assert second["kind"] == "doctor_list" and [d["name"] for d in second["doctors"]["items"]] == ["Dr. Anita Sen"]


def test_refine_cheaper_keeps_previous_filters(client, hms):            # VITA-7
    first = chat(client, "cardiologist", city="Kolkata")
    assert [d["name"] for d in first["doctors"]["items"]] == ["Dr. Neha Sharma", "Dr. Amit Das"]
    second = chat(client, "inme se kam fees wala", session_id=first["session_id"], city="Kolkata")
    assert [d["name"] for d in second["doctors"]["items"]] == ["Dr. Amit Das", "Dr. Neha Sharma"]
    assert second["doctors"]["filters"]["specialtyId"] == "Cardiologist (Heart)"
    assert second["doctors"]["filters"]["sort"] == "fee"


def test_refine_nearby_after_specialty(client, hms):                    # VITA-7
    first = chat(client, "dermatologist", city="Kolkata")
    second = chat(client, "inme se koi paas wala?", session_id=first["session_id"], lat=22.59, lng=88.31)
    assert [d["name"] for d in second["doctors"]["items"]] == ["Dr. Priya Bose"]
    assert set(second["doctors"]["mode"]) == {"specialty", "distance"}


def test_combined_ask_in_one_turn(client, hms):                         # VITA-7
    body = chat(client, "MD cardiologist near Salt Lake", lat=22.58, lng=88.41)
    assert set(body["doctors"]["mode"]) == {"specialty", "degree", "place", "distance"}
    a = searches(hms)[-1]
    assert a["degree"] == "MD" and a["specialtyIds"] == ["Cardiologist (Heart)"] and "lat" in a


def test_select_doctor_and_session_shows_it(client, hms):               # VITA-8
    first = chat(client, "skin specialist", city="Kolkata")
    pick = first["doctors"]["items"][1]["doctorId"]
    body = chat(client, "Choose", session_id=first["session_id"], select_doctor_id=pick)
    assert body["kind"] == "doctor_selected" and body["doctor"]["doctorId"] == pick
    snap = client.get(f"/v2/agent/session/{first['session_id']}").json()
    assert snap["selectedDoctorId"] == pick


def test_select_doctor_not_shown_is_422(client, hms):                    # VITA-8
    first = chat(client, "skin specialist", city="Kolkata")
    cardio = DOCTORS[0]["doctorId"]
    r = client.post(URL, json={"text": "Choose", "session_id": first["session_id"], "select_doctor_id": cardio})
    assert r.status_code == 422
    r = client.post(URL, json={"text": "Choose", "select_doctor_id": cardio})
    assert r.status_code == 422


@pytest.mark.parametrize("text", ["dr sharma, seene mein dard ho raha hai", "cardiologist chahiye saans nahi aa rahi",
                                  "paas mein doctor, behosh ho gaya"])
def test_emergency_wins_over_a_doctor_ask(client, hms, text):           # VITA-9
    body = chat(client, text, city="Kolkata")
    assert body["kind"] == "emergency" and "doctors" not in body
    assert searches(hms) == []


def test_1hms_down_gives_degraded_reply(client):                        # VITA-10
    client.app.state.doctors = make_directory(Fake1Hms(fail=100))
    body = chat(client, "skin ka doctor", city="Kolkata")
    assert body["kind"] == "doctor_list" and "doctors" not in body
    assert body["degraded"] == ["doctor_lookup"] and body["message"]
    client.app.state.doctors = DoctorDirectory(None)


def _scripted_triage(client):
    from hms_router.vita_graph.engine import VitaRuntime
    from tests.test_vita_graph import ScriptedLLM, say, u
    a = client.app.state
    a.vita_runtime = VitaRuntime(ScriptedLLM(u("symptoms", "high fever for 3 days", enough=True),
                                             say("General Physician ko dikhayein.")), a.redis,
                                 route_service=lambda: a.route)


def _no_llm(client):
    from hms_router.vita_graph.engine import VitaRuntime
    a = client.app.state
    a.vita_runtime = VitaRuntime(None, a.redis, route_service=lambda: a.route)


def test_1hms_down_on_symptom_answer_keeps_the_specialty(client):       # VITA-10
    client.app.state.doctors = make_directory(Fake1Hms(fail=100))
    _scripted_triage(client)
    body = chat(client, "bukhar hai 3 din se bahut tez", city="Kolkata")
    _no_llm(client)
    assert body["kind"] == "recommendation" and body["card"]["outputs"] and "doctors" not in body
    assert "doctor_lookup" in body["degraded"]
    client.app.state.doctors = DoctorDirectory(None)


def test_symptom_recommendation_carries_new_block(client, hms):         # VITA-2
    _scripted_triage(client)
    body = chat(client, "bukhar hai 3 din se bahut tez", city="Kolkata")
    _no_llm(client)
    assert body["kind"] == "recommendation"
    block = body["doctors"]
    assert block["mode"] == ["symptom"] and block["items"] == block["doctors"] and "fallbackUsed" in block


def test_follow_up_turns_never_carry_doctors(client, hms):              # VITA-2
    body = chat(client, "dard hai")
    assert body["kind"] == "follow_up" and "doctors" not in body


def test_hospital_channel_groups_its_doctors_first(client, hms):        # VITA-11
    body = chat(client, "cardiologist", city="Kolkata", hospital_id=H2)
    assert searches(hms)[-1]["hospitalId"] == H2
    groups = body["doctors"]["groups"]
    assert groups[0]["title"] == "Howrah General" and groups[1]["title"] == "Other doctors nearby"
    assert body["doctors"]["items"][0]["name"] == "Dr. Amit Das"


def test_items_are_exactly_what_1hms_returned(client, hms):             # VITA-12
    body = chat(client, "orthopaedic doctor", city="Kolkata")
    expected = hms._search({"specialtyIds": ["Orthopaedic Surgeon (Bone)"], "city": "Kolkata", "limit": 3})["doctors"]
    assert body["doctors"]["items"] == expected


def test_bare_doctor_request_asks_what_kind(client, hms):
    body = chat(client, "doctor chahiye")
    assert body["kind"] == "follow_up" and body["clarify"]["kind"] == "ask_kind" and len(body["clarify"]["options"]) == 5


def test_no_doctors_of_a_specialty_falls_back_to_gp(client, hms):
    body = chat(client, "dentist chahiye", city="Kolkata")
    assert body["doctors"]["fallbackUsed"] and body["doctors"]["items"][0]["name"] == "Dr. S. Roy"


def test_telemetry_fields(client, hms):                                 # VITA-13
    from hms_router.telemetry import chat_record

    class Rec:
        @staticmethod
        def query_hash(t):
            return "h"

    body = chat(client, "skin ka doctor", city="Kolkata")
    rec = chat_record(body, recorder=Rec, caller="t", latency_ms=1, text="skin ka doctor", model_version=None)
    assert rec["searchMode"] == "specialty" and rec["doctorsShown"] == 2 and rec["lookupMs"] is not None
    assert rec["reviewText"] is None


# ============================================================================ Phase 1 gaps (vision, "How Vita talks")


def test_plain_words_first(client, hms):
    body = chat(client, "I need an orthopaedic doctor", city="Kolkata")
    assert "bone doctors (Orthopaedic)" in body["message"]
    hi = chat(client, "mujhe haddi ka doctor chahiye", city="Kolkata")
    assert "haddi ke doctor (Orthopaedic)" in hi["message"]


@pytest.mark.parametrize("text", ["Dr Sharma ka number do", "dr anita sen ka mobile number chahiye"])
def test_never_shares_a_private_number(client, hms, text):
    body = chat(client, text, city="Kolkata")
    assert body["kind"] == "info" and "personal number" in body["message"]
    assert not any(ch.isdigit() for ch in body["message"].replace("98", ""))  # no phone digits made up
    assert "doctors" not in body


def test_front_desk_number_only_from_1hms(client, hms):
    import copy
    from tests import test_vita_doctor_chat as me
    saved = copy.deepcopy(me.HOSPITALS[H1])
    me.HOSPITALS[H1]["contact"] = "033-4000-1234"
    try:
        body = chat(client, "dr anita sen ka number do", city="Kolkata")
        assert "033-4000-1234" in body["message"]
    finally:
        me.HOSPITALS[H1].clear()
        me.HOSPITALS[H1].update(saved)


@pytest.mark.parametrize("text", ["bas dekh raha tha", "no thanks", "thank you"])
def test_just_looking_is_fine(client, hms, text):
    body = chat(client, text)
    assert body["kind"] == "info" and any(w in body["message"] for w in ("No problem", "Koi baat nahi", "welcome"))
    assert searches(hms) == []


def test_greeting_offers_the_five_ways(client, hms):
    body = chat(client, "namaste")
    assert body["clarify"]["kind"] == "ask_kind" and len(body["clarify"]["options"]) == 5


def test_talk_to_a_person(client, hms):
    body = chat(client, "mujhe kisi insaan se baat karni hai")
    assert body["kind"] == "handoff" and body["handoff"]["reason"] == "asked"


def test_repeated_confusion_offers_a_person(client, hms):
    first = chat(client, "doctor chahiye")
    second = chat(client, "koi doctor batao", session_id=first["session_id"])
    assert "human" in [o["id"] for o in second["clarify"]["options"]]


def test_show_more_doctors(client, hms):
    hms_dir = client.app.state.doctors
    hms_dir.limit = 1
    try:
        first = chat(client, "cardiologist", city="Kolkata")
        assert len(first["doctors"]["items"]) == 1 and first["doctors"]["hasMore"]
        more = chat(client, "aur doctor dikhao", session_id=first["session_id"], city="Kolkata")
        assert more["doctors"]["page"] == 2 and more["doctors"]["items"][0]["name"] == "Dr. Amit Das"
        assert not more["doctors"]["hasMore"]
        end = chat(client, "show more", session_id=first["session_id"], city="Kolkata")
        assert end["clarify"]["kind"] == "no_more"
    finally:
        hms_dir.limit = 3


def test_emergency_beats_a_conversation_move(client, hms):
    body = chat(client, "hello, seene mein bahut dard aur saans nahi aa rahi")
    assert body["kind"] == "emergency"


def test_dead_ends_reach_the_review_queue(client, hms):
    from hms_router.telemetry import chat_record

    class Rec:
        @staticmethod
        def query_hash(t):
            return "h"

    body = chat(client, "dr zzyzxq")
    rec = chat_record(body, recorder=Rec, caller="t", latency_ms=1, text="dr zzyzxq", model_version=None)
    assert rec["reviewText"] == "dr zzyzxq" and rec["clarifyKind"] == "no_match"


# ============================================================================ hospital, lady doctor, today, availability
def test_hospital_ask_lists_that_hospitals_doctors(client, hms):
    body = chat(client, "Sadar Hospital Purnia ke doctor")
    assert body["kind"] == "doctor_list" and "hospital" in body["doctors"]["mode"]
    assert body["doctors"]["filters"]["hospital"]["name"] == "Sadar Hospital"
    assert ("search_hospitals", {"limit": 4, "query": "sadar hospital purnia"}) in hms.calls
    a = searches(hms)[-1]
    assert a["hospitalId"] == H3 and a["onlyAtHospital"] is True
    assert "Sadar Hospital" in body["message"]


def test_hospital_and_specialty_together(client, hms):
    body = chat(client, "Howrah General hospital mein cardiologist")
    assert [d["name"] for d in body["doctors"]["items"]] == ["Dr. Amit Das"]


def test_unknown_hospital_says_so(client, hms):
    body = chat(client, "Zzyzx Hospital ke doctor")
    assert body["kind"] == "doctor_list" and body["doctors"]["items"] == [] and "zzyzx" in body["message"].lower()
    assert body["clarify"]["kind"] == "no_match"


def test_lady_doctor_refines_the_list(client, hms):                 # vision: "a lady doctor"
    first = chat(client, "cardiologist", city="Kolkata")
    assert len(first["doctors"]["items"]) == 2
    second = chat(client, "inme se koi lady doctor", session_id=first["session_id"], city="Kolkata")
    assert [d["name"] for d in second["doctors"]["items"]] == ["Dr. Neha Sharma"]
    assert second["doctors"]["filters"]["gender"] == "female" and searches(hms)[-1]["gender"] == "female"


def test_lady_doctor_on_its_own(client, hms):
    body = chat(client, "mahila doctor chahiye", city="Kolkata")
    assert body["kind"] == "doctor_list" and all(d["name"] in ("Dr. Neha Sharma", "Dr. Anita Sen", "Dr. Priya Bose")
                                                 for d in body["doctors"]["items"])


def test_available_today(client, hms):                              # vision: "available today"
    body = chat(client, "aaj available cardiologist", city="Kolkata")
    assert [d["name"] for d in body["doctors"]["items"]] == ["Dr. Amit Das"]       # Dr Neha Sharma is on leave
    assert searches(hms)[-1]["availableToday"] is True


def test_place_without_location_is_the_city(client, hms):           # "MD gynaecologist near Purnia"
    body = chat(client, "skin doctor kolkata mein")
    assert body["kind"] == "doctor_list" and searches(hms)[-1]["city"] == "Kolkata"      # as 1HMS spells it
    assert body["doctors"]["filters"]["city"].lower() == "kolkata" and "Kolkata" in body["message"]


def test_show_more_uses_1hms_offset(client, hms):
    hms_dir = client.app.state.doctors
    hms_dir.limit = 1
    try:
        first = chat(client, "dermatologist", city="Kolkata")
        chat(client, "aur doctor dikhao", session_id=first["session_id"], city="Kolkata")
        assert searches(hms)[-1]["offset"] == 1 and searches(hms)[-1]["limit"] == 1
    finally:
        hms_dir.limit = 3


@pytest.mark.parametrize("text", ["Is Dr Neha Sharma available on Monday?", "dr neha sharma somvar ko baithti hain?"])
def test_availability_on_a_working_day(client, hms, text):         # vision: "is Dr Sharma available Sunday?"
    body = chat(client, text)
    assert body["kind"] == "info" and body["doctor"]["name"] == "Dr. Neha Sharma"
    assert "09:00-13:00" in body["message"]
    assert [o["id"] for o in body["clarify"]["options"]] == [DOCTORS[0]["doctorId"], "no"]
    assert body["availability"]["note"]
    dow = [n for n, a in hms.calls if n == "get_doctor_availability"]
    assert dow and "doctors" not in body                             # answered, no list pushed


def test_availability_on_a_day_off_gives_next_day(client, hms):
    body = chat(client, "is Dr Neha Sharma available Sunday?")
    assert "not available" in body["message"] and "Next working day" in body["message"]


def test_book_offer_can_be_taken_or_declined(client, hms):
    q = chat(client, "is Dr Neha Sharma available Monday?")
    pick = chat(client, "Yes, book", session_id=q["session_id"], select_doctor_id=q["clarify"]["options"][0]["id"])
    assert pick["kind"] == "doctor_selected"
    q2 = chat(client, "is Dr Neha Sharma available Monday?")
    no = chat(client, "No, just checking", session_id=q2["session_id"])
    assert no["kind"] == "info" and "No problem" in no["message"]


def test_parser_hospital_gender_today_when():
    a = parse("Sadar Hospital Purnia ke doctor")
    assert a.hospital_query == "sadar hospital purnia" and a.direct
    assert parse("lady doctor chahiye").gender == "female"
    assert parse("mahila rog ka doctor").specialty_key == "gynaecology" and parse("mahila rog ka doctor").gender is None
    assert parse("aaj available skin doctor").available_today
    assert parse("dr sen kal milenge").when == "+1"
    assert parse("is Dr Sharma available Sunday?").when == "dow:6"
    assert parse("dr sharma kab baithte hain").when == "next"
    assert parse("hospital mein bahut bheed hai").hospital_query is None


def test_new_search_drops_a_removed_filter(client, hms):          # Doctor Dekho: removing a filter chip
    first = chat(client, "lady cardiologist", city="Kolkata")
    assert first["doctors"]["filters"].get("gender") == "female"
    again = chat(client, "cardiologist", session_id=first["session_id"], city="Kolkata", new_search=True)
    assert "gender" not in again["doctors"]["filters"] and len(again["doctors"]["items"]) == 2


@pytest.mark.parametrize("text, gender, key", [("lady skin doctor", "female", "dermatology"),
                                               ("aurat ka doctor", None, "gynaecology"),
                                               ("female dentist", "female", "dentistry")])
def test_gender_with_a_specialty_word(text, gender, key):
    a = parse(text)
    assert a.gender == gender and a.specialty_key == key
