# nlp_brain/ — the NLP Brain

All ML logic lives here: data loading, feature extraction, model selection,
the gibberish + coverage-gate checks, and prediction. **Zero FastAPI/voice/
HTTP imports** — this package only knows how to turn text into a specialist
prediction (or an honest "not confident enough"). `api/` and `voice/` are
callers of this package, never the other way around.

## Public API

Import from the top of the package (`nlp_brain/__init__.py` curates this —
prefer these over reaching into submodules):

```python
from nlp_brain import SymptomClassifier, PredictionResult, MODEL_OUT, MATCH_THRESHOLD, NO_MATCH_MESSAGE, clean_text, is_gibberish
```

### `SymptomClassifier` — the one class you actually need

| Method | Use case |
|---|---|
| `SymptomClassifier.train(data_path)` | You edited the training CSV, or want to try a new dataset. Returns `(classifier, held_out_metrics)`. |
| `SymptomClassifier.load(path)` | Loading an already-trained bundle — what `api/main.py`'s startup does, and what you'd do in a script/notebook. |
| `classifier.save(path)` | Persisting a trained classifier — the ONLY place the joblib bundle's schema is defined (`.train()`/`.load()`/`data_pipeline/retrain_pipeline.py` all go through this). |
| `classifier.predict(text)` | Getting a prediction. Returns a `PredictionResult(specialist, candidates, match_ratio, closest_known_example, flagged_gibberish, no_match)` — `candidates` is the ordered, deduped shortlist `specialist` was taken from (see `candidates.py` below). |

```python
classifier = SymptomClassifier.load()
result = classifier.predict("dant mein bahut dard hai")
if result.specialist:
    print(result.specialist, result.match_ratio)
elif result.flagged_gibberish:
    print("nonsense input")
else:
    print("didn't recognize this — closest known:", result.closest_known_example)
```

## Module map

| File | Responsibility | Touch it when... |
|---|---|---|
| `config.py` | `DATA_PATH`, `MODEL_OUT`, `MATCH_THRESHOLD`, `CANDIDATE_MARGIN`, `MAX_CANDIDATES`, sample queries | Changing the canonical dataset path, or recalibrating the coverage-gate threshold or candidate margin (see the big comments there — don't change either without re-running the calibration check first). |
| `text_utils.py` | `clean_text()`, `is_gibberish()` — pure functions, no model | Adding a new gibberish-detection rule (keyboard mash, repeated chars, etc.). |
| `features.py` | `build_feature_union()` — the TF-IDF feature extractor | Changing n-gram ranges, adding a new feature type, tuning `min_df`. |
| `matching.py` | `normalize_matrix()`, `best_match()` — the cosine-similarity coverage gate | Changing HOW "is this query similar to something we've seen" is computed. See the module docstring for the perf history here — don't call `sklearn.metrics.pairwise.cosine_similarity()` directly against a raw (non-pre-normalized) corpus matrix inside a per-request path; that regressed `/route-symptom` to ~50ms/request once. |
| `candidates.py` | `ranked_labels()`, `build_candidates()` — expands a single top pick into a close-margin shortlist | Changing how many candidates get surfaced, or how a close runner-up is decided (`CANDIDATE_MARGIN`/`MAX_CANDIDATES` in `config.py`). |
| `training.py` | `load_data()`, `evaluate_candidates()` | Adding a new candidate classifier algorithm to compare (edit the `candidates` dict in `evaluate_candidates`), or changing CSV validation/cleaning rules. |
| `classifier.py` | `SymptomClassifier`, `PredictionResult` | Changing what a prediction returns, or the joblib bundle's schema. |
| `cli.py` | `python -m nlp_brain.cli [train\|predict "<text>"\|interactive]` | You want a script/REPL entry point without spinning up the API. |

## Common recipes

**Retrain and sanity-check locally:**
```bash
python -m nlp_brain.cli train
```
Prints held-out accuracy/macro-F1, a classification report, and runs the
built-in sample + gibberish queries so you can eyeball the result. Does
**not** check for regression against the currently-deployed model — that's
`data_pipeline/retrain_pipeline.py`'s job (see [docs/data_pipeline.md](data_pipeline.md)).

**Debug one query someone reported as misclassified:**
```bash
python -m nlp_brain.cli predict "unclear symptom text here"
```

**Try a batch of hand-written test phrases interactively:**
```bash
python -m nlp_brain.cli
```

**Add a new gibberish rule:** edit `text_utils.is_gibberish()`, then add
cases to `tests/test_text_utils.py::TestIsGibberish` (both the "should
flag" and "should not flag" parametrized lists) before touching the
implementation — see [docs/testing.md](testing.md).

**Recalibrate `MATCH_THRESHOLD`:** don't guess. `nlp_brain/config.py`'s
comment above the constant explains the calibration approach (score real,
novel queries vs. gibberish, pick a value that separates them with
margin) — `data_pipeline/validation_set.csv` can't be used for this (see
its own caveat in [docs/data_pipeline.md](data_pipeline.md)).

## Performance notes

`SymptomClassifier.__init__` precomputes and caches a normalized copy of
the training corpus matrix (`_texts_matrix_normalized`) specifically so
`predict()` doesn't re-normalize the entire multi-thousand-row corpus on
every call — that was a real bug that made `/route-symptom` take ~50ms
server-side before it was fixed (down to ~15ms after). See
`tests/test_performance.py` for the regression guard.
