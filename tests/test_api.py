"""Tests for the api/ layer -- HTTP contract only. These use `api_client`
(see conftest.py), a real FastAPI TestClient wired to a small fixture
classifier instead of the production bundle, so they run fast and don't
depend on what's currently deployed.

If you change RouteResponse's shape or add a new endpoint, add a test here
-- this is the file that pins down what NexEagleWebsite/voice/ can actually
rely on.
"""


class TestHealth:
    def test_reports_ready_once_the_classifier_is_loaded(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "ready": True}


class TestModelInfo:
    def test_returns_whatever_model_meta_json_contains(self, api_client, monkeypatch, tmp_path):
        import api.routes as routes_module
        fake_meta = tmp_path / "model_meta.json"
        fake_meta.write_text('{"modelVersion": "test-1", "lastRetrainedAt": null, "validationMetrics": null}',
                              encoding="utf-8")
        monkeypatch.setattr(routes_module, "MODEL_META_PATH", fake_meta)

        response = api_client.get("/model-info")

        assert response.status_code == 200
        assert response.json()["modelVersion"] == "test-1"

    def test_returns_a_placeholder_when_model_meta_json_is_missing(self, api_client, monkeypatch, tmp_path):
        import api.routes as routes_module
        monkeypatch.setattr(routes_module, "MODEL_META_PATH", tmp_path / "does_not_exist.json")

        response = api_client.get("/model-info")

        assert response.status_code == 200
        assert response.json()["modelVersion"] == "unknown"


class TestRouteSymptom:
    def test_real_query_returns_a_specialist(self, api_client):
        response = api_client.post("/route-symptom", json={
            "query": "dant mein bahut dard ho raha hai kaafi dino se",
        })
        body = response.json()

        assert response.status_code == 200
        assert body["noMatch"] is False
        assert body["method"] == "classifier"
        assert body["raw"]["specialist"] is not None
        assert 0.0 <= body["confidence"] <= 1.0

    def test_gibberish_query_returns_no_match_with_flag_set(self, api_client):
        response = api_client.post("/route-symptom", json={"query": "zxcvbnmlkjhgfdsaqwerty"})
        body = response.json()

        assert response.status_code == 200
        assert body["specialtyIds"] == []
        assert body["noMatch"] is True
        assert body["raw"]["flaggedGibberish"] is True
        assert body["raw"]["message"] == "No matches found."

    def test_empty_query_returns_no_match(self, api_client):
        response = api_client.post("/route-symptom", json={"query": ""})
        assert response.status_code == 200
        assert response.json()["noMatch"] is True

    def test_whitespace_only_query_returns_no_match(self, api_client):
        response = api_client.post("/route-symptom", json={"query": "    "})
        assert response.status_code == 200
        assert response.json()["noMatch"] is True

    def test_query_over_the_length_limit_is_rejected_with_422(self, api_client):
        response = api_client.post("/route-symptom", json={"query": "dard " * 300})  # 1500 chars
        assert response.status_code == 422

    def test_query_at_the_length_limit_is_accepted(self, api_client):
        from api.schemas import MAX_QUERY_LENGTH
        response = api_client.post("/route-symptom", json={"query": "a" * MAX_QUERY_LENGTH})
        assert response.status_code == 200

    def test_missing_query_field_is_rejected_with_422(self, api_client):
        response = api_client.post("/route-symptom", json={})
        assert response.status_code == 422

    def test_specialist_label_maps_to_a_nexeagle_specialty_id(self, api_client, monkeypatch):
        import api.routes as routes_module
        monkeypatch.setattr(
            routes_module, "LABEL_TO_NEXEAGLE_SPECIALTY_ID",
            {"Dentist": "dentistry"},
        )
        response = api_client.post("/route-symptom", json={
            "query": "dant mein bahut dard ho raha hai kaafi dino se",
        })
        assert response.json()["specialtyIds"] == ["dentistry"]

    def test_unmapped_specialist_label_yields_empty_specialty_ids(self, api_client, monkeypatch):
        # If the training data ever introduces a label with no corresponding
        # NexEagleWebsite specialtyId, callers must get an empty list back,
        # not a KeyError or a raw internal label leaking into the response.
        import api.routes as routes_module
        monkeypatch.setattr(routes_module, "LABEL_TO_NEXEAGLE_SPECIALTY_ID", {})
        response = api_client.post("/route-symptom", json={
            "query": "dant mein bahut dard ho raha hai kaafi dino se",
        })
        body = response.json()
        assert body["specialtyIds"] == []
        assert body["raw"]["specialist"] == "Dentist"  # still reported in `raw` for debugging


class TestRateLimit:
    def test_trips_after_30_requests_in_a_minute(self, api_client):
        query = {"query": "sar mein dard hai"}
        for _ in range(30):
            response = api_client.post("/route-symptom", json=query)
            assert response.status_code == 200

        response = api_client.post("/route-symptom", json=query)
        assert response.status_code == 429
