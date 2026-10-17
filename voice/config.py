"""Constants for the Voice-to-Text layer. No nlp_brain import belongs here --
this layer only knows how to turn speech into text and hand it to the API."""
import os

# Overridable via env var so a voice client machine can point at Dev/Prod/
# localhost without a code change.
API_BASE_URL = os.environ.get("NLP_API_BASE_URL", "http://127.0.0.1:5003")

# Language hint passed to the speech recognizer. "hi-IN" gives the most
# accurate transcription for Hindi/Hinglish speech, but always comes back in
# Devanagari script -- speech.devanagari_to_roman() converts it afterward
# (inside speech.transcribe_audio_data(), called from speech_to_text.py) to
# match the Roman-script Hinglish the NLP Brain was trained on.
SPEECH_LANGUAGE = "hi-IN"

# Phrases that tell the mic to stop listening -- matched as an exact match on
# the whole, normalized utterance, so a stray "stop" inside a symptom
# sentence won't accidentally trigger it.
STOP_PHRASES = {
    "stop", "quit", "exit",
    "ruk jaiye", "ruk jaiye please", "ruk jao", "ruk jayen", "ruko",
    "ruk jaiyega", "band karo", "band kar do", "band kijiye",
}
#Changes for the Mic related timings can be changed only 
# How long (seconds) to wait for the user to start speaking before giving up.
# Raise this if users are being cut off before they begin.
MIC_TIMEOUT = 8.0

# Maximum recording length (seconds) once speech has started.
# Raise this if users are being cut off mid-sentence.
MIC_PHRASE_TIME_LIMIT = 12.0
