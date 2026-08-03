"""
Voice Chatbot Client (Local Brain Version)
==========================================
Listens to the microphone, converts speech to Roman Hinglish script, 
and directly calls the local Model_1_Final brain (bypassing FastAPI).
"""

import re
import joblib

# Import the core logic directly from your Brain file
from Model_1_Final import (
    MODEL_OUT, 
    MATCH_THRESHOLD,
    clean_text, 
    is_gibberish, 
    best_match
)

# --- Audio & Transliteration Setup ---
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False

SPEECH_LANGUAGE = "hi-IN"
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

STOP_PHRASES = {
    "stop", "quit", "exit",
    "ruk jaiye", "ruk jaiye please", "ruk jao", "ruk jayen", "ruko",
    "ruk jaiyega", "band karo", "band kar do", "band kijiye",
}

def devanagari_to_roman(text: str) -> str:
    """Convert any Devanagari-script portion into Roman letters."""
    if not _DEVANAGARI_RE.search(text):
        return text

    if not TRANSLITERATION_AVAILABLE:
        print("Warning: 'indic-transliteration' package isn't installed. Cannot convert to Roman script.")
        return text

    roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    roman = roman.replace(".a", "").replace("'", "")
    roman = " ".join(roman.split())
    return roman.lower()

def is_stop_command(text: str) -> bool:
    normalized = " ".join(str(text).strip().lower().split())
    return normalized in STOP_PHRASES

# --- Local Brain Logic ---
def query_local_brain(symptom_text: str, bundle: dict):
    """Processes the text through the local ML pipeline instead of via HTTP."""
    cleaned = clean_text(symptom_text)
    
    if len(cleaned) < 3:
        print("Result: Input too short. No match found.")
        return

    # 1. Check for gibberish
    if is_gibberish(cleaned):
        print("Result: Gibberish detected. Please describe the symptom clearly.")
        return
        
    # 2. Check coverage (Cosine Similarity)
    ratio, closest = best_match(cleaned, bundle["features"], bundle["texts_matrix"], bundle["texts"])
    
    if ratio < MATCH_THRESHOLD:
        print(f"Result: No confident match found (Similarity: {ratio:.2f}). Please provide more details.")
        return
        
    # 3. Predict Specialist
    feats = bundle["features"].transform([cleaned])
    pred = bundle["model"].predict(feats)[0]
    
    print(f"Predicted specialist: {pred} (Confidence/Similarity: {ratio:.2f})")

# --- Microphone Logic ---
def listen_from_microphone(language: str = SPEECH_LANGUAGE, timeout: float = 8.0, phrase_time_limit: float = 12.0):
    if not SPEECH_RECOGNITION_AVAILABLE:
        print("Requires 'SpeechRecognition' and 'pyaudio'. Install with: pip install SpeechRecognition pyaudio")
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("\nAdjusting for ambient noise... please wait.")
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
        print(f"You said: '{text}'")
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio. Try again.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error ({e}).")
        return None

def interactive_test():
    print("\nLoading AI Brain locally...")
    try:
        # Load the joblib model once when the script starts
        bundle = joblib.load(MODEL_OUT)
        print("Brain loaded successfully!")
    except Exception as e:
        print(f"Failed to load the model bundle: {e}")
        print(f"Make sure {MODEL_OUT} exists in this directory.")
        return

    print("\n=== Try your own queries via microphone ===")
    print("(Say 'stop' or 'ruk jaiye' to quit, Ctrl+C to force-quit)\n")
    
    while True:
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                break
            text = listen_from_microphone()
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

        if text is None:
            continue

        if is_stop_command(text):
            print("Stopping -- mic is now off.")
            break

        # Pass the transcribed text and the loaded model bundle to the local brain function
        query_local_brain(text, bundle)

if __name__ == "__main__":
    interactive_test()