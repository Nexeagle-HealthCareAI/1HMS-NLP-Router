# voice/ — the Voice-to-Text layer

Microphone capture, speech-to-text, and Devanagari→Roman transliteration,
followed by an HTTP call into the `api/` layer. **No import of `nlp_brain`
anywhere in this package** — the only way this layer reaches the NLP Brain
is through the FastAPI HTTP contract (`api_client.SymptomRouterClient`), so
it can run on a completely different machine (one with a microphone, none
of the ML dependencies) and be developed/tested independently of the other
two layers.

Entry point: `python -m voice.cli`.

## Setup

Voice-layer dependencies are **not** in `requirements.txt` (the API server
never needs them) — install them separately:
```bash
pip install -r requirements-voice.txt
```
`PyAudio` needs the PortAudio system library first on macOS/Linux (see the
comment at the top of `requirements-voice.txt`); Windows usually installs
from a prebuilt wheel with no extra step.

Point the client at the right API instance via an env var (defaults to
`http://127.0.0.1:5003`):
```bash
export NLP_API_BASE_URL=http://151.185.45.77:5003   # Dev
```

## Module map

| File | Responsibility | Touch it when... |
|---|---|---|
| `config.py` | `API_BASE_URL`, `SPEECH_LANGUAGE`, `STOP_PHRASES` | Adding a new stop phrase, or changing the STT language hint. |
| `speech_to_text.py` | `listen_from_microphone()`, `is_stop_command()` | Swapping the speech-recognition engine, or changing mic timeout/phrase-limit behavior. |
| `transliteration.py` | `devanagari_to_roman()` | The transcription needs different script normalization. |
| `api_client.py` | `SymptomRouterClient` — the ONLY path to the NLP Brain | Changing which API endpoints the voice layer calls, or how HTTP errors are surfaced. |
| `cli.py` | `interactive()` — the mic → STT → API → print loop | Changing the interactive session's behavior (what happens on an unclear query, how results are displayed). |

## Why HTTP-only matters

Earlier versions of this layer imported `nlp_brain`'s internals directly
(bypassing the API entirely) or re-implemented the whole training/predict
pipeline inline. Both approaches meant: no independent deployability, no
shared rate limiting or input validation, and two more places to keep in
sync every time the Brain's prediction logic changed. `api_client.py`
exists specifically so this layer only ever depends on a stable HTTP
contract (`api/schemas.py`'s `RouteRequest`/`RouteResponse`) — changing how
prediction works internally should never require a `voice/` code change.

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
unconditionally; `devanagari_to_roman()` and `listen_from_microphone()`
tests skip themselves automatically if `requirements-voice.txt` isn't
installed (see [docs/testing.md](testing.md)) — CI doesn't install it, so
those specific tests only run if you have the optional deps locally.
