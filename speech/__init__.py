"""Shared speech (audio-to-text) utilities.

Used by BOTH voice/ (live microphone capture, client-only) and api/
(server-side transcription of uploaded audio, part of the deployed
service) -- neither layer owns this exclusively, so it doesn't live inside
either one. Depends only on `speech_recognition`, `indic-transliteration`,
and `pydub` (+ the `ffmpeg` binary); no FastAPI/mic-capture/HTTP-client
knowledge lives here.
"""
from .transcription import (
    AudioFormatError,
    AudioUnintelligible,
    SpeechServiceError,
    TranscriptionError,
    transcribe_audio_data,
    transcribe_audio_file,
)
from .transliteration import devanagari_to_roman

__all__ = [
    "transcribe_audio_data",
    "transcribe_audio_file",
    "devanagari_to_roman",
    "TranscriptionError",
    "AudioUnintelligible",
    "AudioFormatError",
    "SpeechServiceError",
]
