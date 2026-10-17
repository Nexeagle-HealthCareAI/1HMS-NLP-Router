"""Voice-to-Text layer entry point: mic -> speech-to-text -> HTTP call to the
FastAPI layer -> display result. Deliberately has no idea how symptom
routing works internally -- that's the whole point of the layering.
"""
import requests

from .api_client import SymptomRouterClient
from .speech_to_text import SPEECH_RECOGNITION_AVAILABLE, is_stop_command, listen_from_microphone


def _print_result(response: dict) -> None:
    """Formats a raw /route-symptom JSON response (see api/schemas.py's
    RouteResponse) for a terminal user. Kept separate from interactive()'s
    loop so a future non-terminal voice frontend (e.g. text-to-speech
    output instead of print()) can reuse the request/response plumbing
    below and just swap this one function."""
    if response.get("noMatch"):
        if response.get("raw", {}).get("flaggedGibberish"):
            print("Sorry, that didn't look like a symptom description.")
        else:
            print("Sorry, I couldn't confidently match that to a specialist.")
        return

    specialist = response.get("raw", {}).get("specialist")
    confidence = response.get("confidence")
    print(f"Predicted specialist: {specialist} (confidence: {confidence:.2f})")


def interactive(client: SymptomRouterClient = None) -> None:
    """`python -m voice.cli` -- the voice layer's main loop: listen ->
    transcribe -> (unless it's a stop command) send to the API -> print the
    result -> listen again, until a stop phrase or Ctrl+C. Accepts an
    injected `client` for testing; real usage should just call
    `interactive()` with no arguments and let it build a SymptomRouterClient
    from NLP_API_BASE_URL (see config.py)."""
    client = client or SymptomRouterClient()

    try:
        client.health()
    except requests.RequestException as e:
        print(f"Warning: couldn't reach the NLP API at {client.base_url} ({e}). "
              f"Make sure it's running before continuing.")

    print(
        "\n=== Try your own queries via microphone "
        "(say 'stop' or 'ruk jaiye' to quit, Ctrl+C to force-quit) ==="
    )
    while True:
        if not SPEECH_RECOGNITION_AVAILABLE:
            print(
                "Microphone input requires the 'SpeechRecognition' and 'pyaudio' "
                "packages. Install them with:\n    pip install -r requirements-voice.txt"
            )
            break
        try:
            text = listen_from_microphone()
        except KeyboardInterrupt:
            print("\nStopped.")
            break

        if text is None:
            # Nothing understood -- keep listening, don't stop the mic.
            continue

        if is_stop_command(text):
            print("Stopping -- mic is now off.")
            break

        try:
            response = client.route_symptom(text)
        except requests.RequestException as e:
            print(f"Couldn't reach the NLP API ({e}). Try again.")
            continue

        _print_result(response)


if __name__ == "__main__":
    interactive()
