# speech/ — shared transcription & transliteration

Audio → Roman-script text. Used by **both** `voice/` (live microphone
capture) and `api/` (server-side transcription of uploaded audio via
`POST /route-symptom-audio`) — neither layer owns this exclusively, so it
doesn't live inside either one. See [docs/voice.md](voice.md)'s "Why
HTTP-only matters" section for why this isn't just `api/` importing from
`voice/` (or vice versa).

No FastAPI/mic-capture/HTTP-client knowledge lives here — this package only
knows how to turn an `sr.AudioData` object (or an uploaded file) into
Roman-script text.

## Module map

| File | Responsibility | Touch it when... |
|---|---|---|
| `transliteration.py` | `devanagari_to_roman()` | Changing how Devanagari script gets normalized to match the Roman-script Hinglish the NLP Brain was trained on. |
| `transcription.py` | `transcribe_audio_data()`, `transcribe_audio_file()`, the `TranscriptionError` hierarchy | Changing the STT engine, adding a new audio-format handling case, or changing what counts as which failure mode (see below). |

## The three failure modes

`transcribe_audio_file()`/`transcribe_audio_data()` distinguish three ways
transcription can fail, because they warrant genuinely different responses
(see `api/routes.py`'s `/route-symptom-audio`):

| Exception | Meaning | HTTP status | Retry helps? |
|---|---|---|---|
| `AudioUnintelligible` | Valid audio, nothing recognizable said (silence, mumbling, noise) | `200` with `noMatch: true` — same philosophy as `/route-symptom`'s handling of unclear text | Maybe, with clearer audio |
| `AudioFormatError` | The uploaded bytes aren't decodable as audio at all | `400` | No — client sent something wrong |
| `SpeechServiceError` | The recognition service itself failed (network, quota, `ffmpeg` missing) | `502` (service failure) or `503` (ffmpeg not installed — an environment problem) | Maybe, later |

## Why `ffmpeg` matters

`transcribe_audio_file()` always normalizes incoming audio to WAV via
`pydub` before handing it to `SpeechRecognition` — because
`SpeechRecognition` only reads WAV/AIFF/FLAC natively, and a browser's
`MediaRecorder` typically produces webm/opus (or mp4/aac on Safari), not
WAV. `pydub` shells out to the `ffmpeg` binary for this, **even for
already-WAV input** — there's no format-sniffing shortcut. That means:

- The `ffmpeg` binary must be on `PATH` wherever `transcribe_audio_file()`
  runs. The Dockerfile installs it via `apt-get`.
- If `ffmpeg` is missing, `pydub` raises a bare `FileNotFoundError` (not
  `pydub.exceptions.CouldntDecodeError`, which fires when `ffmpeg` runs but
  rejects the input) — `_to_wav()` catches this specifically and raises
  `TranscriptionError`, not `AudioFormatError`, since it's not the caller's
  fault. See `tests/test_speech.py::TestToWav` — this was a real bug caught
  by testing locally without `ffmpeg` installed, not something designed in
  from the start.

## Common recipes

**Test without `ffmpeg` installed:** everything in `tests/test_speech.py`
mocks the recognizer (`recognize_google`) and, for most tests, `_to_wav()`
itself — no real audio decoding needed. One test
(`TestToWavWithRealFfmpeg`) exercises real `ffmpeg` conversion and
self-skips via `shutil.which("ffmpeg")` if it's absent (true in CI and
possibly your local machine — that's fine, it's not required to validate
routing/business logic).

**Swap the STT engine:** `transcribe_audio_data()` is the one place
`recognizer.recognize_google(...)` gets called — replacing it with a
different engine (paid cloud API, self-hosted Whisper) means changing this
one function; `transcribe_audio_file()` and every caller stay the same.

**Add a new supported input format:** nothing to do — `pydub`+`ffmpeg`
already handles any format `ffmpeg` itself supports (which is nearly
everything). If a specific format is failing, it's likely a `ffmpeg` build
/ codec issue, not something to special-case in this package.
