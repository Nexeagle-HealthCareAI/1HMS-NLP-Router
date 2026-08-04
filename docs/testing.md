# Testing

Tests are **mandatory after any code change** — enforced two ways:

1. **CI gate** (`.github/workflows/deploy-nlp.yml`, Stage 1 — Build): every
   push/PR runs the full test suite. If it fails, the pipeline stops there
   — Docker build, GHCR push, and deploy to Dev/Prod never happen. This is
   the guarantee that actually matters (it can't be skipped or forgotten).
2. **Pre-commit hook** (`.githooks/pre-commit`): runs the fast subset
   locally before a commit is allowed, so failures surface immediately
   instead of at CI. Activate it once per clone:
   ```bash
   git config core.hooksPath .githooks
   ```
   It can be bypassed with `git commit --no-verify` — the CI gate above is
   what actually enforces this, the hook is just faster local feedback.

## Running tests

```bash
pip install -r requirements.txt -r requirements-dev.txt

pytest                          # everything, including integration tests
pytest -m "not integration"     # fast subset only — what the pre-commit hook runs
pytest -m integration           # only the slower real-bundle tests
pytest tests/test_api.py -v     # one file, verbose
pytest -k gibberish              # anything matching "gibberish" in the test name
```

The full suite takes well under a minute. Most of that is fixed overhead
(importing sklearn/pandas/scipy, training the small fixture classifier
once per session) rather than the tests themselves.

## How the suite is organized

| File | Covers |
|---|---|
| `tests/conftest.py` | Shared fixtures — see below. |
| `tests/test_text_utils.py` | `clean_text()`, `is_gibberish()` — pure functions. |
| `tests/test_matching.py` | `normalize_matrix()`, `best_match()` — the coverage gate, on small hand-built matrices. |
| `tests/test_classifier.py` | `SymptomClassifier`: train/predict/save/load, plus `@pytest.mark.integration` tests against the real committed bundle. |
| `tests/test_api.py` | Every `api/` endpoint via a `TestClient`, including `/route-symptom-audio` (transcription mocked), the query length/audio size limits, and both rate limits. |
| `tests/test_voice.py` | `is_stop_command()`, `SymptomRouterClient` (mocked HTTP), `listen_from_microphone()`'s no-mic-available path (mocked). |
| `tests/test_speech.py` | `devanagari_to_roman()`, `transcribe_audio_data()`/`transcribe_audio_file()` (recognizer mocked), plus one real-`ffmpeg` test that self-skips if `ffmpeg` isn't installed. |
| `tests/test_retrain_pipeline.py` | `feedback_to_rows()`, `merge_rows()`, `is_regression()` — the pure decision logic, not the live CMSAPI fetch. |
| `tests/test_performance.py` | Latency regression guard — see below. |

## Fixtures (`tests/conftest.py`)

- **`tiny_classifier`** (session-scoped): a REAL `SymptomClassifier`,
  trained via the actual `.train()` code path on a small (~32 row, 4
  specialist) synthetic dataset defined right in `conftest.py`. Fast
  (trains in well under a second) and independent of whatever the
  production dataset looks like on a given day. Use this for anything that
  needs a working classifier but doesn't specifically need to test the
  real, currently-deployed model.
- **`api_client`**: a FastAPI `TestClient` for `api.main:app`, with
  `SymptomClassifier.load` monkeypatched to return `tiny_classifier`
  instead of reading the real bundle off disk. Resets the rate limiter's
  in-memory counters before and after each test.
- **`requires_real_bundle`** / **`@pytest.mark.integration`**: marks a
  test as needing the actual `symptom_specialist_classifier.joblib` +
  `Hinglish_Symptoms_V28.csv` on disk. Self-skips if they're missing;
  excluded from the pre-commit hook's fast loop via `-m "not integration"`.

## Adding a new test

1. **Bug fix**: write a test that reproduces the bug (should fail against
   the old code), then fix the code until it passes. Put it in the file
   matching the layer you're fixing (see the table above).
2. **New function**: add a test class/function in the matching file. If no
   file matches your module yet, create `tests/test_<module>.py`
   mirroring the pattern in the existing files (one `Test*` class per
   function/behavior grouping, `test_*` methods named after the scenario,
   not the mechanism — `test_gibberish_input_is_rejected`, not
   `test_predict_returns_none`).
3. **Performance-sensitive change**: if you're touching `nlp_brain/matching.py`
   or anything in `SymptomClassifier.predict()`'s hot path, run
   `pytest tests/test_performance.py -m integration -v` and compare the
   printed timing against what's in that file's docstring — it's a
   regression guard specifically because this codebase has already
   regressed here once (see [docs/nlp_brain.md](nlp_brain.md)'s
   performance notes).

## What's deliberately NOT covered

- `voice.speech_to_text.listen_from_microphone()`'s actual audio capture —
  no microphone in CI. The "no mic available" degradation path IS tested
  (mocked `OSError`).
- `speech.transcribe_audio_file()`'s real `ffmpeg` conversion in the
  default test run — needs the `ffmpeg` binary, which CI doesn't install
  (same reasoning as `PyAudio`: not worth the fragility for something
  that's a deployment concern, not routing/business logic). One test
  (`tests/test_speech.py::TestToWavWithRealFfmpeg`) exercises it for real
  and self-skips via `shutil.which("ffmpeg")` if absent.
- Actually calling Google's Web Speech API (`recognize_google`) — every
  test mocks the recognizer. If you need to verify the real API still
  behaves as expected, that's a manual check (`python -m voice.cli`
  against a real microphone), not part of the automated suite.
- `data_pipeline.retrain_pipeline.fetch_live()` — needs a real CMSAPI.
  Test the pure logic it feeds into (`feedback_to_rows`, `merge_rows`,
  `is_regression`) instead, or use `--training-fixture`/`--feedback-fixture`
  for a manual end-to-end dry run.
- Docker/deployment itself — verified by actually watching the CI run
  (`gh run watch`) after a push, not by a test in this suite.
