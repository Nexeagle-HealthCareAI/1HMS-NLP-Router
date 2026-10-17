"""HTTP request/response contracts. Data shapes only -- no behavior."""
from typing import Optional

from pydantic import BaseModel, Field

# Generous headroom over any real symptom description (the longest sample
# query is well under 100 chars) while capping the worst case -- an
# unbounded string field is free amplification for whoever's paying the
# ~50ms/query coverage-gate cost (see nlp_brain.matching), gibberish or not.
MAX_QUERY_LENGTH = 1000


class RouteRequest(BaseModel):
    query: str = Field(max_length=MAX_QUERY_LENGTH)


class RouteResponse(BaseModel):
    specialtyIds: list[str]
    # True when the NLP Brain's gibberish check or cosine-similarity coverage
    # check rejected the input, i.e. there is no specialist to return at all.
    # No default-specialist fallback -- an honest "no match" beats a guess.
    noMatch: bool
    method: Optional[str]
    # Cosine similarity (0-1) between the query and the closest known
    # training example's TF-IDF vector -- NOT a classifier probability.
    confidence: Optional[float]
    modelVersion: Optional[str]
    raw: dict


# Generous headroom over a realistic several-second voice clip (a 10s WAV at
# 16kHz mono is roughly 320KB; compressed webm/opus from a browser's
# MediaRecorder is far smaller) while still bounding the worst case --
# ffmpeg decoding + a speech-recognition API call is real work per request,
# more than the text endpoint's coverage-gate check, so this is deliberately
# tighter than MAX_QUERY_LENGTH's headroom.
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB


class RouteResponseWithTranscript(RouteResponse):
    """POST /route-symptom-audio's response: everything RouteResponse has,
    plus the transcript that was actually routed -- callers need this to
    show the patient what was heard (and let them correct it) since there's
    no other way to know what the server understood from the recording.
    None when transcription didn't produce usable text at all (see
    speech.AudioUnintelligible)."""
    transcript: Optional[str]
