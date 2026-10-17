"""Tests for speech/ -- the transcription + transliteration logic shared by
voice/ (live mic) and api/ (uploaded audio, POST /route-symptom-audio).

transcribe_audio_data() is tested with a mocked recognizer (no real speech
API call, no network) -- fast and deterministic. transcribe_audio_file()'s
audio-format-conversion step needs a real `ffmpeg` binary (pydub shells out
to it even for plain WAV -- see _to_wav()'s docstring), which CI doesn't
install (same reasoning as PyAudio: not worth the fragility for something
that isn't part of validating the routing/business logic) -- that one real
round-trip test skips itself via shutil.which("ffmpeg") when it's absent.
"""
import io
import shutil
import struct
import wave

import pytest

from speech.transcription import (
    AudioUnintelligible,
    SpeechServiceError,
    TranscriptionError,
    transcribe_audio_data,
    transcribe_audio_file,
)
from speech.transliteration import TRANSLITERATION_AVAILABLE, devanagari_to_roman

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


class TestDevanagariToRoman:
    def test_converts_devanagari_to_roman_script(self):
        result = devanagari_to_roman("दिल में दर्द है")
        assert result == result.lower()
        assert all(not (0x0900 <= ord(c) <= 0x097F) for c in result), \
            "result should contain no leftover Devanagari characters"

    def test_pure_english_text_passes_through_unchanged(self):
        assert devanagari_to_roman("chest pain and fever") == "chest pain and fever"

    def test_mixed_script_only_converts_the_devanagari_part(self):
        result = devanagari_to_roman("BP high hai aur सर दर्द है")
        assert "BP high hai aur" in result or "bp high hai aur" in result.lower()

    @pytest.mark.skipif(TRANSLITERATION_AVAILABLE, reason="only meaningful when the package is absent")
    def test_returns_original_text_unchanged_if_package_missing(self):
        # Defensive path -- shouldn't be reachable now that indic-transliteration
        # is a hard requirement (requirements.txt), but devanagari_to_roman()
        # degrades gracefully rather than raising if it somehow isn't installed.
        assert devanagari_to_roman("दिल में दर्द है") == "दिल में दर्द है"


class _FakeAudioData:
    """Stands in for sr.AudioData -- transcribe_audio_data() just passes it
    straight to recognizer.recognize_google(), which we mock below, so it
    never needs to be real audio."""


class TestTranscribeAudioData:
    def test_successful_transcription_is_transliterated(self, monkeypatch):
        import speech.transcription as mod

        class FakeRecognizer:
            def recognize_google(self, audio, language):
                return "दिल में दर्द है"

        monkeypatch.setattr(mod.sr, "Recognizer", FakeRecognizer)

        result = transcribe_audio_data(_FakeAudioData())

        assert result == devanagari_to_roman("दिल में दर्द है")

    def test_roman_script_result_passes_through_unchanged(self, monkeypatch):
        import speech.transcription as mod

        class FakeRecognizer:
            def recognize_google(self, audio, language):
                return "chest pain hai"

        monkeypatch.setattr(mod.sr, "Recognizer", FakeRecognizer)

        assert transcribe_audio_data(_FakeAudioData()) == "chest pain hai"

    def test_unintelligible_audio_raises_audio_unintelligible(self, monkeypatch):
        import speech.transcription as mod

        class FakeRecognizer:
            def recognize_google(self, audio, language):
                raise mod.sr.UnknownValueError()

        monkeypatch.setattr(mod.sr, "Recognizer", FakeRecognizer)

        with pytest.raises(AudioUnintelligible):
            transcribe_audio_data(_FakeAudioData())

    def test_service_failure_raises_speech_service_error(self, monkeypatch):
        import speech.transcription as mod

        class FakeRecognizer:
            def recognize_google(self, audio, language):
                raise mod.sr.RequestError("network down")

        monkeypatch.setattr(mod.sr, "Recognizer", FakeRecognizer)

        with pytest.raises(SpeechServiceError):
            transcribe_audio_data(_FakeAudioData())

    def test_raises_transcription_error_if_package_not_available(self, monkeypatch):
        import speech.transcription as mod
        monkeypatch.setattr(mod, "SPEECH_RECOGNITION_AVAILABLE", False)

        with pytest.raises(TranscriptionError):
            transcribe_audio_data(_FakeAudioData())


class TestToWav:
    def test_missing_ffmpeg_binary_is_a_transcription_error_not_a_raw_crash(self, monkeypatch):
        """Regression test: pydub's AudioSegment.from_file() raises a bare
        FileNotFoundError when the `ffmpeg` binary itself isn't on PATH
        (distinct from pydub.exceptions.CouldntDecodeError, which fires
        when ffmpeg runs but rejects the input as not-audio). Without this
        being caught, a missing ffmpeg would leak as an unhandled 500
        through api/routes.py instead of a clean 503."""
        from speech.transcription import _to_wav

        class RaisingAudioSegment:
            @staticmethod
            def from_file(file_obj):
                raise FileNotFoundError("[Errno 2] No such file or directory: 'ffmpeg'")

        monkeypatch.setattr("pydub.AudioSegment", RaisingAudioSegment)

        with pytest.raises(TranscriptionError) as exc_info:
            _to_wav(io.BytesIO(b"irrelevant"))
        assert "ffmpeg" in str(exc_info.value).lower()


class TestTranscribeAudioFile:
    def test_delegates_to_transcribe_audio_data_after_converting_to_wav(self, monkeypatch):
        """Verifies the orchestration (convert -> read as AudioFile -> hand
        off to transcribe_audio_data) without needing real ffmpeg -- mocks
        _to_wav() at the boundary, same idea as mocking recognize_google
        above."""
        import speech.transcription as mod

        fake_wav = _tiny_silent_wav_bytes()
        monkeypatch.setattr(mod, "_to_wav", lambda file_obj: io.BytesIO(fake_wav))
        monkeypatch.setattr(mod, "transcribe_audio_data", lambda audio, language=None: "dant mein dard hai")

        result = transcribe_audio_file(io.BytesIO(b"pretend-audio-bytes"))

        assert result == "dant mein dard hai"


def _tiny_silent_wav_bytes(seconds: float = 0.3, framerate: int = 16000) -> bytes:
    """A minimal valid WAV file (silence) built with the stdlib `wave`
    module -- no ffmpeg needed to CREATE this, only pydub needs ffmpeg to
    READ/convert it."""
    n_frames = int(seconds * framerate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed -- pydub needs it even for WAV (see _to_wav)")
class TestToWavWithRealFfmpeg:
    """Real round-trip through pydub+ffmpeg (no network call -- this only
    checks that format conversion itself works, not the recognize_google
    step, which stays mocked everywhere else in this file to avoid a
    hidden dependency on Google's API actually being reachable)."""

    def test_converts_a_wav_file_to_a_readable_wav_buffer(self):
        from speech.transcription import _to_wav

        result = _to_wav(io.BytesIO(_tiny_silent_wav_bytes()))

        with wave.open(result, "rb") as w:
            assert w.getnframes() > 0
