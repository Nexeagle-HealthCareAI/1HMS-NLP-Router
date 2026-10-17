# voice/ — the Voice-to-Text layer

Microphone capture, followed by an HTTP call into the `api/` layer.
**No import of `nlp_brain` anywhere in this package** — the only way this
layer reaches the NLP Brain is through the FastAPI HTTP contract
(`api_client.SymptomRouterClient`), so it can run on a completely different
machine (one with a microphone, none of the ML dependencies) and be
developed/tested independently of the other layers.

Actual transcription (speech → text) and Devanagari→Roman transliteration
live in [`speech/`](speech.md), not here — that logic is shared with
`api/`'s `POST /route-symptom-audio` (server-side transcription of
*uploaded* audio, for a browser/mobile client that can't run this Python
package directly). `voice/` only owns the microphone-specific part: turning
a live mic feed into an `sr.AudioData` object and handing it to
`speech.transcribe_audio_data()`.

Entry point: `python -m voice.cli`.

## Setup

Voice-layer dependencies are **not all** in `requirements.txt` — install
the client-only ones separately:
```bash
pip install -r requirements-voice.txt
```
`SpeechRecognition` and `indic-transliteration` are in `requirements.txt`
too (the API server needs them for `/route-symptom-audio`); `PyAudio` and
`requests` are `voice/`-only. `PyAudio` needs the PortAudio system library
first on macOS/Linux (see the comment at the top of
`requirements-voice.txt`); Windows usually installs from a prebuilt wheel
with no extra step.

Point the client at the right API instance via an env var (defaults to
`http://127.0.0.1:5003`):
```bash
export NLP_API_BASE_URL=http://151.185.45.77:5003   # Dev
```

## Module map

| File | Responsibility | Touch it when... |
|---|---|---|
| `config.py` | `API_BASE_URL`, `SPEECH_LANGUAGE`, `STOP_PHRASES` | Adding a new stop phrase, or changing the STT language hint. |
| `speech_to_text.py` | `listen_from_microphone()` (mic capture only — transcription itself is `speech.transcribe_audio_data()`), `is_stop_command()` | Changing mic timeout/phrase-limit behavior, or how mic-specific errors (no device, timeout) are handled. |
| `api_client.py` | `SymptomRouterClient` — the ONLY path to the NLP Brain | Changing which API endpoints the voice layer calls, or how HTTP errors are surfaced. |
| `cli.py` | `interactive()` — the mic → transcribe → API → print loop | Changing the interactive session's behavior (what happens on an unclear query, how results are displayed). |

## Why HTTP-only matters

Earlier versions of this layer imported `nlp_brain`'s internals directly
(bypassing the API entirely) or re-implemented the whole training/predict
pipeline inline. Both approaches meant: no independent deployability, no
shared rate limiting or input validation, and two more places to keep in
sync every time the Brain's prediction logic changed. `api_client.py`
exists specifically so this layer only ever depends on a stable HTTP
contract (`api/schemas.py`'s `RouteRequest`/`RouteResponse`) — changing how
prediction works internally should never require a `voice/` code change.

This is also why transcription logic moved into `speech/` rather than
staying in `voice/` when `api/` needed it too: `api/` importing from
`voice/` would create the same kind of backwards dependency, and would drag
`PyAudio` (a live-microphone-only dependency) toward the deployed API image
for no reason. See [docs/speech.md](speech.md).

## Common recipes

**Run the voice client against a local API:**
```bash
# terminal 1
uvicorn api.main:app --port 5003
# terminal 2
pip install -r requirements-voice.txt
python -m voice.cli
```

**Add a new stop phrase:** add it to `config.STOP_PHRASES` (a normalized,
whole-phrase match — see `is_stop_command()`'s docstring for why it's not
a substring match).

**Test without a microphone:** `tests/test_voice.py` covers
`is_stop_command()` and `SymptomRouterClient` (via mocked `requests` calls)
unconditionally, plus `listen_from_microphone()`'s no-mic-available
degradation path (mocked `sr.Microphone`). Transliteration/transcription
tests live in `tests/test_speech.py` instead (see
[docs/testing.md](testing.md)).
