"""Doctor lookups from 1HMSAPI's MCP server (hms_router/doctors.py) (Vita's doctor lists are tested in test_vita_doctor_chat.py)."""
import json

import httpx
import pytest

from hms_router.doctors import DoctorDirectory, McpClient, build_directory, specialty_key

API_CATEGORIES = ["General Physician", "Paediatrician", "Cardiologist (Heart)", "Dermatologist (Skin)",
                  "Orthopaedic Surgeon (Bone)", "Gynaecologist", "Dentist", "ENT Specialist", "Ophthalmologist (Eye)",
                  "Neurologist", "Psychiatrist", "Urologist", "Gastroenterologist",
                  "Endocrinologist (Hormones/Diabetes)", "Pulmonologist (Chest/Lungs)", "Nephrologist (Kidney)",
                  "Oncologist (Cancer)", "Rheumatologist", "Physiotherapist / Rehab", "General Surgeon",
                  "Neurosurgeon", "Plastic Surgeon", "Vascular Surgeon", "Cardiothoracic Surgeon",
                  "Anaesthesiologist", "Radiologist", "Pathologist", "Emergency Medicine Specialist", "Geriatrician",
                  "Sports Medicine Specialist"]


# ----------------------------------------------------------------------------- specialty_key
@pytest.mark.parametrize("router_name, api_label", [
    ("Gastroenterologist", "Gastroenterologist"), ("Orthopedist", "Orthopaedic Surgeon (Bone)"),
    ("Cardiologist", "Cardiologist (Heart)"), ("Pulmonologist", "Pulmonologist (Chest/Lungs)"),
    ("ENT Specialist", "ENT Specialist"), ("Neurologist", "Neurologist"), ("General Physician", "General Physician"),
    ("Pediatrician", "Paediatrician"), ("Gynecology", "Gynaecologist"), ("Neurosurgery", "Neurosurgeon"),
    ("General Surgery", "General Surgeon"), ("Dermatology", "Dermatologist (Skin)"), ("Ophthalmology", "Ophthalmologist (Eye)"),
])
def test_router_and_api_names_share_a_key(router_name, api_label):
    assert specialty_key(router_name) is not None
    assert specialty_key(router_name) == specialty_key(api_label)


def test_api_categories_all_distinct():
    keys = [specialty_key(c) for c in API_CATEGORIES]
    assert None not in keys and len(set(keys)) == len(keys)


@pytest.mark.parametrize("name", ["Telemedicine consult", "OPD triage desk", "", None])
def test_non_specialties_have_no_key(name):
    assert specialty_key(name) is None


# ----------------------------------------------------------------------------- fake MCP server
def doctor(i, category):
    return {"doctorId": f"00000000-0000-0000-0000-00000000000{i}", "name": f"Dr {i}", "specialtyId": category,
            "hospital": {"hospitalId": "11111111-1111-1111-1111-111111111111", "name": "H", "city": "Kolkata"}}


class FakeMcp:
    """Stateless Streamable HTTP MCP server: checks the key, answers as SSE like the .NET SDK."""

    def __init__(self, key="k", fail_status=None, tool_error=False):
        self.key, self.fail_status, self.tool_error = key, fail_status, tool_error
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail_status:
            return httpx.Response(self.fail_status)
        if request.headers.get("X-Service-Key") != self.key:
            return httpx.Response(401)
        msg = json.loads(request.content)
        if "id" not in msg:
            return httpx.Response(202)
        method, params = msg["method"], msg.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fake", "version": "1"}}
        elif method == "tools/call":
            name, args = params["name"], params.get("arguments") or {}
            self.calls.append((name, args))
            if self.tool_error:
                result = {"isError": True, "content": [{"type": "text", "text": "boom"}]}
            elif name == "list_specialties":
                result = {"structuredContent": {"specialties": [
                    {"specialtyId": c, "name": c, "doctorCount": 0 if c == "Dentist" else 4, "aliases": []}
                    for c in API_CATEGORIES]}}
            elif name == "search_doctors":
                cat = args["specialtyIds"][0]
                docs = [doctor(i, cat) for i in range(1, 5)]
                result = {"structuredContent": {"doctors": docs[: args.get("limit", 5)], "total": 4}}
            else:
                result = {"isError": True, "content": [{"type": "text", "text": "unknown tool"}]}
        else:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "no"}})
        body = "event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def directory(server, key="k", limit=3):
    return DoctorDirectory(McpClient("http://api/mcp", key, transport=httpx.MockTransport(server)), limit=limit)


# ----------------------------------------------------------------------------- DoctorDirectory
def test_lookup_maps_specialist_to_api_category():
    server = FakeMcp()
    out = directory(server).for_specialist("Orthopedist", city="Kolkata")
    assert out["specialtyId"] == "Orthopaedic Surgeon (Bone)" and out["total"] == 4 and len(out["doctors"]) == 3
    name, args = server.calls[-1]
    assert name == "search_doctors" and args == {"specialtyIds": ["Orthopaedic Surgeon (Bone)"], "limit": 3, "city": "Kolkata"}


def test_location_sorts_by_distance():
    # a wide radius: the tool's 5 km default would hide every doctor further away on a symptom answer
    server = FakeMcp()
    directory(server).for_specialist("Cardiologist", lat=22.58, lng=88.42)
    assert server.calls[-1][1] == {"specialtyIds": ["Cardiologist (Heart)"], "limit": 3,
                                   "lat": 22.58, "lng": 88.42, "maxDistanceKm": 50.0, "sort": "distance"}


def test_specialties_are_cached():
    server = FakeMcp()
    d = directory(server)
    d.for_specialist("Cardiologist")
    d.for_specialist("Neurologist")
    assert [n for n, _ in server.calls].count("list_specialties") == 1


@pytest.mark.parametrize("specialist", ["Telemedicine consult", "Dentist"])   # no key / no bookable doctors
def test_no_category_means_no_doctors(specialist):
    server = FakeMcp()
    assert directory(server).for_specialist(specialist) is None
    assert "search_doctors" not in [n for n, _ in server.calls]


@pytest.mark.parametrize("server", [FakeMcp(fail_status=503), FakeMcp(fail_status=500), FakeMcp(tool_error=True)])
def test_failures_mean_no_doctors_never_an_exception(server):
    d = directory(server)
    assert d.for_specialist("Cardiologist") is None
    assert d.stats["errors"] == 1 and d.stats["lastError"]


def test_wrong_key_means_no_doctors():
    d = directory(FakeMcp(key="right"), key="wrong")
    assert d.for_specialist("Cardiologist") is None and "401" in d.stats["lastError"]


def test_timeout_means_no_doctors():
    def slow(_request):
        raise httpx.ReadTimeout("slow")
    d = DoctorDirectory(McpClient("http://api/mcp", "k", transport=httpx.MockTransport(slow)))
    assert d.for_specialist("Cardiologist") is None and "ReadTimeout" in d.stats["lastError"]


def test_off_without_settings():
    assert not build_directory(None, None).enabled
    assert not build_directory("http://api/mcp", None).enabled
    assert build_directory(None, None).for_specialist("Cardiologist") is None


