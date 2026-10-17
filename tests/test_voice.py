"""Tests for the voice/ layer. `SpeechRecognition` and `indic-transliteration`
are now hard dependencies of api/ too (POST /route-symptom-audio), so they're
in the base requirements.txt CI always installs -- only `PyAudio` (live
microphone capture specifically) stays voice/-only and genuinely optional
(needs the PortAudio system library; see requirements-voice.txt). Tests
below that touch microphone behavior mock `sr.Microphone` entirely, so they
don't actually need PyAudio installed to run.

See tests/test_speech.py for transliteration/transcription tests -- that
logic lives in speech/ now, shared with api/.
"""
import pytest
import requests

from voice.speech_to_text import SPEECH_RECOGNITION_AVAILABLE, is_stop_command


class TestIsStopCommand:
    @pytest.mark.parametrize("text", [
        "stop", "Stop", "  STOP  ", "quit", "exit",
        "ruk jaiye", "ruk jao", "band karo", "band kar do",
    ])
    def test_recognizes_stop_phrases(self, text):
        assert is_stop_command(text) is True

    @pytest.mark.parametrize("text", [
        "dil mein dard hai",
        "mujhe rukna hai kya karu",  # contains "ruk" but isn't a stop phrase
        "stop sign ke paas dard shuru hua",  # contains "stop" but isn't just that
        "",
    ])
    def test_does_not_misfire_on_real_symptom_text(self, text):
        assert is_stop_command(text) is False


class TestSymptomRouterClient:
    @pytest.fixture
    def client(self):
        from voice.api_client import SymptomRouterClient
        return SymptomRouterClient(base_url="http://test-server:5003")

    def test_route_symptom_posts_the_query_and_returns_parsed_json(self, client, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"specialtyIds": ["dentistry"], "noMatch": False}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

        monkeypatch.setattr(requests, "post", fake_post)

        result = client.route_symptom("dant mein dard hai")

        assert captured["url"] == "http://test-server:5003/route-symptom"
        assert captured["json"] == {"query": "dant mein dard hai"}
        assert result == {"specialtyIds": ["dentistry"], "noMatch": False}

    def test_route_symptom_raises_on_http_error_status(self, client, monkeypatch):
        class FakeResponse:
            status_code = 429

            def raise_for_status(self):
                raise requests.HTTPError("429 Too Many Requests")

        monkeypatch.setattr(requests, "post", lambda url, json, timeout: FakeResponse())

        with pytest.raises(requests.HTTPError):
            client.route_symptom("dant mein dard hai")

    def test_health_hits_the_health_endpoint(self, client, monkeypatch):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": "ok", "ready": True}

        def fake_get(url, timeout):
            captured["url"] = url
            return FakeResponse()

        monkeypatch.setattr(requests, "get", fake_get)

        result = client.health()

        assert captured["url"] == "http://test-server:5003/health"
        assert result["status"] == "ok"

    def test_base_url_trailing_slash_is_normalized(self):
        from voice.api_client import SymptomRouterClient
        client = SymptomRouterClient(base_url="http://test-server:5003/")
        assert client.base_url == "http://test-server:5003"


@pytest.mark.skipif(not SPEECH_RECOGNITION_AVAILABLE, reason="SpeechRecognition not installed")
class TestListenFromMicrophone:
    def test_returns_none_when_no_microphone_is_available(self, monkeypatch):
        # Simulates running on a machine/CI runner with no audio device --
        # should degrade gracefully, not crash the interactive loop.
        import voice.speech_to_text as stt

        class RaisingMicrophone:
            def __enter__(self):
                raise OSError("No Default Input Device Available")

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(stt.sr, "Microphone", lambda: RaisingMicrophone())

        assert stt.listen_from_microphone() is None
