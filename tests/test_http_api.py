"""HTTP contract and security: auth, scopes, rate limits, problem+json, size cap, request ids, readiness,
and the fields Doctor Dekho reads from /route-symptom."""
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.http.security import hash_key
from hms_router.adapters.settings import ServiceSettings

ROUTE_KEY, CHAT_KEY = "route-secret-123", "chat-secret-456"


def keys_env(**extra):
    return {"HMS_API_KEYS": f"doctordekho:route:{hash_key(ROUTE_KEY)},whatsapp:chat|route:{hash_key(CHAT_KEY)}",
            "HMS_RATE_LIMIT_ROUTE": "1000", **extra}


@pytest.fixture(scope="module")
def open_client(bundle):
    with TestClient(create_app(ServiceSettings.from_env({}))) as c:
        yield c


@pytest.fixture(scope="module")
def secured(bundle):
    with TestClient(create_app(ServiceSettings.from_env(keys_env()))) as c:
        yield c


# ----------------------------------------------------------------------------- contract
def test_doctor_dekho_fields_on_alias(open_client):
    body = open_client.post("/route-symptom", json={"query": "dant mein bahut dard hai"}).json()
    for k in ("specialtyIds", "usedDefault", "method", "confidence", "modelVersion"):
        assert k in body
    assert body["specialtyIds"] == ["dentistry"] and body["usedDefault"] is False and body["decision"] == "answer"
    assert body["confidence"] == body["specialties"][0]["confidence"] and body["specialties"][0]["evidence"] is None


def test_v1_full_shape_with_evidence(open_client):
    r = open_client.post("/v1/route-symptom?include=evidence", json={"query": "seene mein dard ho raha hai", "topK": 3},
                         headers={"X-Request-Id": "abc-123"})
    body = r.json()
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc-123" and body["requestId"] == "abc-123"
    assert body["emergency"]["flag"] and body["emergency"]["recommendedSpecialtyIds"][0] == "emergencymedicine"
    assert len(body["specialties"]) == 3 and body["specialties"][0]["evidence"]


def test_clarify_age_round_trip(open_client):
    first = open_client.post("/v1/route-symptom", json={"query": "pet mein dard aur ulti"}).json()
    assert first["decision"] == "clarify" and first["clarify"]["kind"] == "age"
    second = open_client.post("/v1/route-symptom", json={"query": "pet mein dard aur ulti",
                                                          "context": {"ageGroup": "adult"}}).json()
    assert all(s["specialtyId"] != "pediatrics" for s in second["specialties"])


def test_abstain_shape(open_client):
    body = open_client.post("/v1/route-symptom", json={"query": "asdkjh qwe"}).json()
    assert body["decision"] == "abstain" and body["noMatch"] and body["specialtyIds"] == [] and body["method"] is None


def test_taxonomy_ready_model_info(open_client):
    assert len(open_client.get("/v1/taxonomy").json()["specialties"]) == 30
    ready = open_client.get("/ready")
    assert ready.status_code == 200 and ready.json()["ready"] and ready.json()["modelVersion"]
    info = open_client.get("/model-info").json()
    assert info["policy"]["decisive_min_conf"] and info["reasoner"] == "NoReasoner"


# ----------------------------------------------------------------------------- errors
@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": "a" * 1001},
                                     {"query": "x", "topK": 0}, {"query": "x", "topK": 11},
                                     {"query": "bad\x00char"}, {"query": "x", "context": {"ageGroup": "teen"}}])
def test_validation_is_problem_json(open_client, payload):
    r = open_client.post("/v1/route-symptom", json=payload)
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["status"] == 422 and r.json()["title"]


def test_body_size_cap(open_client):
    r = open_client.post("/v1/route-symptom", content=b'{"query": "' + b"a" * 20000 + b'"}',
                         headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.headers["content-type"].startswith("application/problem+json")


def test_security_headers(open_client):
    r = open_client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


# ----------------------------------------------------------------------------- auth + rate limits
def test_missing_and_bad_keys(secured):
    assert secured.post("/v1/route-symptom", json={"query": "dant dard"}).status_code == 401
    r = secured.post("/v1/route-symptom", json={"query": "dant dard"}, headers={"X-Api-Key": "nope"})
    assert r.status_code == 401 and r.json()["detail"] == "Invalid API key"


def test_scopes(secured):
    ok = secured.post("/v1/route-symptom", json={"query": "dant dard"}, headers={"X-Api-Key": ROUTE_KEY})
    assert ok.status_code == 200
    denied = secured.post("/chat", json={"text": "dant dard"}, headers={"X-Api-Key": ROUTE_KEY})
    assert denied.status_code == 403
    assert secured.post("/chat", json={"text": "dant dard"}, headers={"X-Api-Key": CHAT_KEY}).status_code == 200


def test_health_needs_no_key(secured):
    assert secured.get("/health").status_code == 200 and secured.get("/ready").status_code == 200


def test_rate_limit(bundle):
    env = keys_env(HMS_RATE_LIMIT_ROUTE="3")
    with TestClient(create_app(ServiceSettings.from_env(env))) as c:
        codes = [c.post("/v1/route-symptom", json={"query": "dant dard"}, headers={"X-Api-Key": ROUTE_KEY}).status_code
                 for _ in range(4)]
        assert codes[:3] == [200, 200, 200] and codes[3] == 429
        r = c.post("/v1/route-symptom", json={"query": "dant dard"}, headers={"X-Api-Key": ROUTE_KEY})
        assert "retry-after" in {k.lower() for k in r.headers}
        # another client has its own budget
        assert c.post("/v1/route-symptom", json={"query": "dant dard"}, headers={"X-Api-Key": CHAT_KEY}).status_code == 200


@pytest.mark.parametrize("env,msg", [
    ({"HMS_ENV": "production"}, "HMS_API_KEYS"),
    ({"HMS_ENV": "production", "HMS_API_KEYS": f"a:route:{hash_key('k')}"}, "REDIS_URL"),
    ({"HMS_ENV": "production", "HMS_API_KEYS": f"a:route:{hash_key('k')}", "REDIS_URL": "redis://x",
      "HMS_CORS_ORIGINS": "*"}, "CORS"),
    ({"HMS_API_KEYS": "broken"}, "HMS_API_KEYS"),
])
def test_production_refuses_unsafe_settings(env, msg):
    with pytest.raises((RuntimeError, ValueError), match=msg):
        ServiceSettings.from_env(env)
