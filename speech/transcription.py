"""Audio -> Roman-script text. Shared by voice/ (live microphone capture
via sr.Microphone()) and api/ (server-side transcription of uploaded audio
via sr.AudioFile()) -- both produce the same sr.AudioData object, so the
actual "call the recognizer, then transliterate" step lives here once
instead of in whichever layer happened to write it first.
"""
import io

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

from .transliteration import devanagari_to_roman

DEFAULT_LANGUAGE = "hi-IN"


class TranscriptionError(Exception):
    """Base class for anything that stops transcription from succeeding.
    Callers should generally catch the specific subclasses below, not this
    directly -- they warrant different responses (see api/routes.py's
    /route-symptom-audio: unintelligible audio is an honest "no match",
    format errors are a 400, service errors are a 502)."""


class AudioUnintelligible(TranscriptionError):
    """The audio was valid but nothing recognizable was said -- silence,
    mumbling, background noise. Not the caller's fault; treat this the
    same way /route-symptom treats unclear text (an honest "couldn't
    understand" result), not as an error response."""


class AudioFormatError(TranscriptionError):
    """The uploaded bytes couldn't be decoded as audio at all -- wrong
    content-type, corrupted upload, empty file. A genuine client error."""


class SpeechServiceError(TranscriptionError):
    """The recognition service itself failed (network issue, quota,
    Google's Web Speech API being unavailable) -- not the caller's fault,
    but no transcript was produced either. Distinct from
    AudioUnintelligible because retrying might just work."""


def transcribe_audio_data(audio: "sr.AudioData", language: str = DEFAULT_LANGUAGE) -> str:
    """Runs Google's Web Speech API (via the SpeechRecognition package) on
    an already-captured AudioData object, then normalizes the result to
    Roman script. Works identically whether `audio` came from a live mic
    (voice/speech_to_text.py) or an uploaded file (transcribe_audio_file()
    below) -- this function doesn't know or care which."""
    if not SPEECH_RECOGNITION_AVAILABLE:
        raise TranscriptionError(
            "The 'SpeechRecognition' package isn't installed (pip install -r requirements.txt)."
        )
    recognizer = sr.Recognizer()
    try:
        raw_text = recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        raise AudioUnintelligible("Could not understand the audio.")
    except sr.RequestError as e:
        raise SpeechServiceError(f"Speech recognition service error: {e}")
    return devanagari_to_roman(raw_text)


def transcribe_audio_file(file_obj, language: str = DEFAULT_LANGUAGE) -> str:
    """Transcribes an uploaded audio recording -- what api/routes.py's
    POST /route-symptom-audio uses. `file_obj` is a file-like object
    opened in binary mode (e.g. FastAPI's UploadFile.file).

    SpeechRecognition can only read WAV/AIFF/FLAC directly; browsers'
    MediaRecorder typically produces webm/opus (or mp4/aac on Safari), so
    this always normalizes to WAV first via pydub+ffmpeg -- callers never
    need to know or care what format the client actually recorded in.
    Requires the `ffmpeg` binary to be on PATH (see Dockerfile)."""
    if not SPEECH_RECOGNITION_AVAILABLE:
        raise TranscriptionError(
            "The 'SpeechRecognition' package isn't installed (pip install -r requirements.txt)."
        )

    wav_buffer = _to_wav(file_obj)
    with sr.AudioFile(wav_buffer) as source:
        recognizer = sr.Recognizer()
        audio = recognizer.record(source)
    return transcribe_audio_data(audio, language=language)


def _to_wav(file_obj) -> io.BytesIO:
    """Converts arbitrary audio (webm/ogg/mp3/m4a/wav/...) to an in-memory
    WAV via pydub+ffmpeg. Always runs the conversion (even if the input
    happens to already be WAV) rather than trying to sniff the format
    first -- ffmpeg reads WAV fine too, and format-sniffing from raw bytes
    is its own source of bugs for marginal benefit on a short voice clip."""
    try:
        from pydub import AudioSegment
        from pydub.exceptions import CouldntDecodeError
    except ImportError:
        raise TranscriptionError(
            "The 'pydub' package isn't installed (pip install -r requirements.txt)."
        )

    try:
        audio_segment = AudioSegment.from_file(file_obj)
    except CouldntDecodeError as e:
        raise AudioFormatError(f"Couldn't decode the uploaded file as audio: {e}")
    except FileNotFoundError:
        # pydub shells out to the `ffmpeg` binary; if it's missing entirely
        # (vs. ffmpeg running but rejecting the input, which raises
        # CouldntDecodeError above) this is an environment/deployment
        # problem -- not something the caller's retry or different input
        # fixes. See Dockerfile for where ffmpeg gets installed.
        raise TranscriptionError(
            "Audio transcription is unavailable on this server (ffmpeg not found)."
        )

    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)
    return buffer
