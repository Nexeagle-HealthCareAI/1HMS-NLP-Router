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


class TestRouteSymptomAudio:
    """POST /route-symptom-audio -- transcribe_audio_file() is mocked in
    every test here (see api.routes.transcribe_audio_file) so these never
    need real audio, ffmpeg, or network access; tests/test_speech.py covers
    the transcription logic itself."""

    @staticmethod
    def _upload(api_client, content: bytes = b"pretend-audio-bytes", filename: str = "clip.webm"):
        return api_client.post(
            "/route-symptom-audio",
            files={"audio": (filename, content, "audio/webm")},
        )

    def test_transcribed_query_returns_a_specialist_and_the_transcript(self, api_client, monkeypatch):
        import api.routes as routes_module
        monkeypatch.setattr(
            routes_module, "transcribe_audio_file",
            lambda file_obj: "dant mein bahut dard ho raha hai kaafi dino se",
        )

        response = self._upload(api_client)
        body = response.json()

        assert response.status_code == 200
        assert body["transcript"] == "dant mein bahut dard ho raha hai kaafi dino se"
        assert body["noMatch"] is False
        assert body["raw"]["specialist"] is not None

    def test_unintelligible_audio_returns_no_match_not_an_error(self, api_client, monkeypatch):
        import api.routes as routes_module
        from speech import AudioUnintelligible

        def raise_unintelligible(file_obj):
            raise AudioUnintelligible("Could not understand the audio.")

        monkeypatch.setattr(routes_module, "transcribe_audio_file", raise_unintelligible)

        response = self._upload(api_client)
        body = response.json()

        assert response.status_code == 200
        assert body["transcript"] is None
        assert body["noMatch"] is True
        assert body["raw"]["message"] == "Could not understand the audio."

    def test_bad_audio_format_returns_400(self, api_client, monkeypatch):
        import api.routes as routes_module
        from speech import AudioFormatError

        def raise_format_error(file_obj):
            raise AudioFormatError("Couldn't decode the uploaded file as audio.")

        monkeypatch.setattr(routes_module, "transcribe_audio_file", raise_format_error)

        response = self._upload(api_client)
        assert response.status_code == 400

    def test_speech_service_failure_returns_502(self, api_client, monkeypatch):
        import api.routes as routes_module
        from speech import SpeechServiceError

        def raise_service_error(file_obj):
            raise SpeechServiceError("network down")

        monkeypatch.setattr(routes_module, "transcribe_audio_file", raise_service_error)

        response = self._upload(api_client)
        assert response.status_code == 502

    def test_oversized_upload_is_rejected_with_413(self, api_client):
        from api.schemas import MAX_AUDIO_BYTES
        oversized = b"x" * (MAX_AUDIO_BYTES + 1)

        response = self._upload(api_client, content=oversized)

        assert response.status_code == 413

    def test_upload_at_the_size_limit_is_accepted(self, api_client, monkeypatch):
        import api.routes as routes_module
        from api.schemas import MAX_AUDIO_BYTES
        monkeypatch.setattr(routes_module, "transcribe_audio_file", lambda file_obj: "sar mein dard hai")

        at_limit = b"x" * MAX_AUDIO_BYTES
        response = self._upload(api_client, content=at_limit)

        assert response.status_code == 200

    def test_rate_limit_is_tighter_than_the_text_endpoint(self, api_client, monkeypatch):
        import api.routes as routes_module
        monkeypatch.setattr(routes_module, "transcribe_audio_file", lambda file_obj: "sar mein dard hai")

        for _ in range(10):
            assert self._upload(api_client).status_code == 200

        assert self._upload(api_client).status_code == 429
