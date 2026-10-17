# data_pipeline/ — retraining & dataset tooling

Not part of the deployed API image (excluded from the Docker build) and not
one of the three runtime layers — this is an ops/batch component that
produces the artifacts `api/` and `nlp_brain/` consume.

## `retrain_pipeline.py` — the active pipeline

What `.github/workflows/retrain.yml` runs nightly (and on CMS's "Retrain
now" button): fetches the current CMS-editable training set + production
feedback, merges and dedupes them, trains a candidate model via
`nlp_brain`, evaluates it against the frozen `validation_set.csv`, and only
**promotes** (overwrites `Hinglish_Symptoms_V28.csv` + `model_meta.json` +
`symptom_specialist_classifier.joblib`) if it doesn't regress against the
currently-deployed model's recorded metrics. A push to `develop` from this
workflow re-triggers `deploy-nlp.yml`, so promotion = a normal redeploy.

### Two run modes
```bash
# Live -- what retrain.yml actually uses
python retrain_pipeline.py --cmsapi-url "$CMSAPI_BASE_URL" --service-key "$CMSAPI_SERVICE_KEY"

# Dry-run -- for local development, or reproducing a specific past run
python retrain_pipeline.py --training-fixture path/to/training.json --feedback-fixture path/to/feedback.json
```

### The promotion gate

`evaluate()` scores a candidate on `top1Accuracy`, `noMatchRate`, and
`confidentlyWrongRate` against `validation_set.csv`. `is_regression()`
compares those against the currently-deployed model's own recorded
metrics (`model_meta.json`, read via `load_current_meta()`) — a candidate
that's more than `TOLERANCE` (0.01) worse on either accuracy or wrong-rate
is rejected, and **nothing gets written**. This is what keeps a bad batch
of production feedback (or a CMS data-entry mistake) from silently
degrading the live model.

**Known caveat:** `validation_set.csv` currently has ~97% row overlap with
the training data (measured directly), which inflates its accuracy numbers
and makes it a weak signal for genuine generalization. It's still useful as
a *regression* gate (comparing candidate vs. baseline on the same biased
yardstick is apples-to-apples), just not as an absolute quality measure.
Fixing this means reworking `build_validation_set.py`'s split so validation
rows are genuinely held out — a real but separate task from anything
routine retraining touches.

### Feedback weighting

`feedback_to_rows()` weights **corrections** (model suggested X, patient
actually booked Y) at 2x and **silent accepts** (suggested X, booked X) at
0.5x relative to CMS-curated training examples (1x) — see the constants
`WEIGHT_CORRECTION` / `WEIGHT_ACCEPTED` / `WEIGHT_BASE`. Corrections are
unambiguous signal about what *should* have been predicted; silent accepts
are weaker (the patient might just not have looked elsewhere).

## Legacy tooling (not part of the active pipeline)

| File | What it did | Status |
|---|---|---|
| `specialist_seed_data.py` | Hand-curated seed phrase bank, one entry per specialist class | Input to `generate_dataset.py`; not read by anything in the active retrain flow. |
| `generate_dataset.py` | Expanded the seed bank + `Hinglish_Symptoms_Reference.csv` into `Hinglish_Symptoms_Reference_v2.csv` | One-time script that produced an earlier dataset generation. `normalize_for_dedupe()` from this file IS still imported by `retrain_pipeline.py`'s `merge_rows()` — that's the one live dependency. |
| `build_validation_set.py` | Split `Hinglish_Symptoms_Reference_v2.csv` into `validation_set.csv` + `trainable_pool.csv` | One-time script (see its own docstring: "Run ONCE... committed and never regenerated after this"). `validation_set.csv` is still the active frozen validation set (see caveat above); `trainable_pool.csv` isn't read by the active pipeline. |

If you're extending the dataset lineage (e.g. actually fixing the
validation-set leakage), these are the files to start from — but check
whether their logic still matches the current schema
(`text,specialist,type`) and label taxonomy before trusting their output
as-is.

## Common recipes

**Test the merge/promotion logic without a live CMSAPI:** build small
JSON fixtures matching the shapes documented in `retrain_pipeline.py`'s
module docstring, then run with `--training-fixture`/`--feedback-fixture`.

**Change what counts as a regression:** edit `TOLERANCE` or the comparison
logic in `is_regression()` — see `tests/test_retrain_pipeline.py::TestIsRegression`
for the cases to keep passing (and add new ones for whatever you're
changing).

**Debug why a retrain didn't promote:** the script prints both the
candidate's and the currently-promoted model's metrics before deciding —
check the workflow run's logs in `retrain.yml`'s "Run retrain pipeline"
step.
