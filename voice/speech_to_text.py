"""Microphone capture + transcription. Single responsibility: turn audio
into Roman-script text. Knows nothing about symptom routing."""
from .config import SPEECH_LANGUAGE, STOP_PHRASES
from .transliteration import devanagari_to_roman

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False


def listen_from_microphone(language: str = SPEECH_LANGUAGE, timeout: float = 8.0,
                            phrase_time_limit: float = 12.0):
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
        raw_text = recognizer.recognize_google(audio, language=language)
        text = devanagari_to_roman(raw_text)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio. Try again.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error ({e}). Check your internet connection.")
        return None


def is_stop_command(text: str) -> bool:
    """True if the (already Roman-script) utterance is a stop command, e.g.
    'stop' or 'ruk jaiye'. Matches on the whole normalized phrase so a real
    symptom sentence containing a similar word isn't mistaken for one."""
    normalized = " ".join(str(text).strip().lower().split())
    return normalized in STOP_PHRASES
