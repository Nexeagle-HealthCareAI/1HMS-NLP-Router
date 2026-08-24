"""Microphone capture. Single responsibility: turn a live mic feed into an
sr.AudioData object and hand it to speech.transcribe_audio_data() for the
actual transcription -- that step is shared with api/'s uploaded-audio
endpoint (see speech/transcription.py), so it isn't duplicated here.
"""
from speech import (
    AudioUnintelligible,
    SpeechServiceError,
    TranscriptionError,
    transcribe_audio_data,
)

from .config import MIC_PHRASE_TIME_LIMIT, MIC_TIMEOUT, SPEECH_LANGUAGE, STOP_PHRASES

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False


def listen_from_microphone(language: str = SPEECH_LANGUAGE, timeout: float = MIC_TIMEOUT,
                            phrase_time_limit: float = MIC_PHRASE_TIME_LIMIT):
    """Record audio from the default microphone and transcribe it to text,
    normalized to Roman script. Returns the transcribed string, or None if
    nothing could be understood (silence, unclear audio, no network, no
    mic, etc.) -- prints a short status message either way."""
    if not SPEECH_RECOGNITION_AVAILABLE:
        print(
            "Microphone input requires the 'SpeechRecognition' and 'pyaudio' "
            "packages. Install them with:\n    pip install -r requirements-voice.txt"
        )
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print("Listening... speak your symptom now.")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        print("No speech detected before timeout. Try again.")
        return None
    except OSError as e:
        print(f"Microphone not available ({e}).")
        return None

    print("Transcribing...")
    try:
        text = transcribe_audio_data(audio, language=language)
        print(f"You said: {text}")
        return text
    except AudioUnintelligible:
        print("Could not understand the audio. Try again.")
        return None
    except SpeechServiceError as e:
        print(f"{e} Check your internet connection.")
        return None
    except TranscriptionError as e:
        print(str(e))
        return None


def is_stop_command(text: str) -> bool:
    """True if the (already Roman-script) utterance is a stop command, e.g.
    'stop' or 'ruk jaiye'. Matches on the whole normalized phrase so a real
    symptom sentence containing a similar word isn't mistaken for one."""
    normalized = " ".join(str(text).strip().lower().split())
    return normalized in STOP_PHRASES
