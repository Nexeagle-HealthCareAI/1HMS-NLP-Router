# -*- coding: utf-8 -*-
"""
Fetches the current CMS-editable training set + production feedback, merges them, trains a
candidate model, evaluates it against the FROZEN validation_set.csv, and only if it doesn't
regress vs. the currently-promoted model's recorded metrics, promotes it — regenerating
../Hinglish_Symptoms_V28.csv, ../model_meta.json, and
../symptom_specialist_classifier.joblib (the artifact the api/ layer actually loads at startup,
via nlp_brain.SymptomClassifier.load()).

If the candidate regresses, nothing is written — this script is safe to run repeatedly with
no effect until there's actually enough good new data to justify a change. The GitHub Actions
workflow that runs this (retrain.yml) commits+pushes only if `git diff` shows changes
afterward, i.e. only on an actual promotion.

Two run modes:
  --cmsapi-url <url> --service-key <key>
      Live mode: fetch training examples + feedback from a running CMSAPI.
  --training-fixture <path.json> --feedback-fixture <path.json>
      Dry-run mode: load the same shapes from local JSON files instead of a live CMSAPI —
      for testing the merge/train/evaluate/promote logic before CMSAPI's endpoints exist,
      or reproducing a specific past run's inputs.

Training-fixture shape:  [{"text": "...", "specialist": "...", "type": "...", "source": "..."}]
Feedback-fixture shape:  [{"query": "...", "predictedSpecialtyId": "...", "method": "...",
                           "confidence": 0.8, "actualBookedSpecialtyId": "..." | null,
                           "wasCorrection": true, "hasBooking": true}]
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make both this file's own directory (for sibling imports like generate_dataset)
# and the repo root (for nlp_brain/specialty_mapping) importable regardless of how
# this module is invoked -- as a script from within data_pipeline/ (retrain.yml's
# working-directory), or imported as data_pipeline.retrain_pipeline from the repo
# root (tests/test_retrain_pipeline.py, run via pytest from the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp_brain import MODEL_OUT, MATCH_THRESHOLD, SymptomClassifier, clean_text  # noqa: E402
from nlp_brain.features import build_feature_union  # noqa: E402
from nlp_brain.matching import best_match, normalize_matrix  # noqa: E402
from nlp_brain.training import evaluate_candidates  # noqa: E402
from specialty_mapping import NEXEAGLE_SPECIALTY_ID_TO_LABEL  # noqa: E402
from generate_dataset import normalize_for_dedupe  # noqa: E402

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
CSV_PATH = REPO_ROOT / "Hinglish_Symptoms_V28.csv"
META_PATH = REPO_ROOT / "model_meta.json"
MODEL_PATH = REPO_ROOT / MODEL_OUT
VALIDATION_PATH = HERE / "validation_set.csv"

# Corrections (the model suggested X, patient actually booked Y) are unambiguous strong
# signal about what SHOULD have been predicted. Silent accepts (suggested X, booked X) are
# weaker — could just mean the patient didn't bother looking elsewhere, not that X was
# necessarily the ideal answer — so they count for less during training.
WEIGHT_CORRECTION = 2.0
WEIGHT_ACCEPTED = 0.5
WEIGHT_BASE = 1.0

# How much worse the candidate is allowed to be and still count as "not a regression" —
# small enough that noise in a modest feedback batch can't accidentally block promotion,
# large enough that a genuinely bad batch of feedback still gets caught.
TOLERANCE = 0.01


def fetch_live(cmsapi_url: str, service_key: str):
    """Live-mode data source: pages through CMSAPI's training-examples and
    feedback-log endpoints. This is what retrain.yml's nightly cron and
    CMS's "Retrain now" button both use. For local development or testing
    the merge/train/evaluate logic without a running CMSAPI, use
    --training-fixture/--feedback-fixture (load_fixture()) instead."""
    import urllib.request

    def get_all_pages(path):
        items, page = [], 1
        while True:
            req = urllib.request.Request(
                f"{cmsapi_url}{path}?page={page}&limit=500",
                headers={"X-Service-Key": service_key},
            )
            with urllib.request.urlopen(req, timeout=30) as res:
                body = json.loads(res.read())
            items.extend(body["data"])
            if page >= body["pagination"]["totalPages"]:
                break
            page += 1
        return items

    training = get_all_pages("/symptom-router/training-examples")
    feedback = get_all_pages("/symptom-router/feedback-log")
    return training, feedback


def load_fixture(path: str):
    """Dry-run data source: loads one JSON file matching the shape either
    fetch_live() call would return a page of. Use --training-fixture and
    --feedback-fixture together (see main()) to test a specific past run's
    inputs, or to develop against CMSAPI endpoints that don't exist yet."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feedback_to_rows(feedback: list[dict]):
    """Converts feedback-log entries into (text, specialist, type, weight) rows. Rows with
    no observed booking carry no usable signal (we don't know if the suggestion was right)
    and are skipped entirely.

    Always trains on actualBookedSpecialtyId, the patient's real choice -- NOT
    predictedSpecialtyId (the router's #1 pick), even for an "accepted" (non-correction) row.
    Since the candidate mechanism (nlp_brain/candidates.py) can surface more than one
    specialist, wasCorrection=False only means the booking landed on SOME candidate the router
    offered (see CMSAPI's SymptomRouterRepository.GetFeedbackLogAsync's acceptableSpecialtyIds)
    -- not necessarily #1. Using predictedSpecialtyId there would silently train the model to
    reinforce a top pick the patient specifically didn't choose, on exactly the close-call
    queries this mechanism exists to handle well."""
    rows = []
    skipped_no_booking = 0
    skipped_unmappable = 0
    for item in feedback:
        if not item.get("hasBooking"):
            skipped_no_booking += 1
            continue

        query = (item.get("query") or "").strip()
        if not query:
            continue

        slug = item.get("actualBookedSpecialtyId")
        if item.get("wasCorrection"):
            row_type = "Production Feedback - Correction"
            weight = WEIGHT_CORRECTION
        else:
            row_type = "Production Feedback - Accepted"
            weight = WEIGHT_ACCEPTED

        label = NEXEAGLE_SPECIALTY_ID_TO_LABEL.get(slug or "")
        if not label:
            skipped_unmappable += 1
            continue

        rows.append((query, label, row_type, weight))

    print(
        f"Feedback: {len(rows)} usable rows "
        f"({skipped_no_booking} skipped, no booking observed; "
        f"{skipped_unmappable} skipped, unmappable specialtyId)"
    )
    return rows


def merge_rows(training_examples: list[dict], feedback_rows: list[tuple]):
    """Dedupe training examples + feedback-derived rows into the final (texts, labels,
    weights, type-tagged-for-CSV) lists. Training examples win ties (same normalized text +
    specialist) over feedback, since DB-curated data is presumed higher quality than raw
    production text."""
    seen = {}
    order = []

    def add(text, specialist, row_type, weight):
        key = (normalize_for_dedupe(text), specialist)
        if key in seen:
            return
        seen[key] = (text, specialist, row_type, weight)
        order.append(key)

    for row in training_examples:
        add(row["text"], row["specialist"], row.get("type") or "", WEIGHT_BASE)
    for text, specialist, row_type, weight in feedback_rows:
        add(text, specialist, row_type, weight)

    texts = [seen[k][0] for k in order]
    labels = [seen[k][1] for k in order]
    types = [seen[k][2] for k in order]
    weights = [seen[k][3] for k in order]
    return texts, labels, types, weights


def load_validation_set():
    """Loads the FROZEN validation_set.csv -- the fixed yardstick every
    candidate model is measured against (see evaluate()/is_regression()).
    "Frozen" means this file should basically never change; if you're
    tempted to edit it to make a candidate pass, that's a sign the
    candidate is the problem, not the validation set."""
    with open(VALIDATION_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["specialist"] for r in rows]


def evaluate(features, clf, train_texts, train_matrix, val_texts, val_labels, threshold: float = MATCH_THRESHOLD):
    """Mirrors nlp_brain.SymptomClassifier.predict()'s two-stage logic (cosine-similarity coverage gate,
    then classifier) against the frozen validation set. `noMatchRate` covers cases the
    coverage gate rejects outright (never reaches the classifier); `confidentlyWrongRate`
    covers cases that passed the gate but got the wrong specialist -- the more costly failure
    mode, since the caller gets a confident-looking wrong answer instead of an honest "no
    match"."""
    train_matrix_normalized = normalize_matrix(train_matrix)

    top1 = wrong = no_match = 0
    n = len(val_texts)
    for text, true_label in zip(val_texts, val_labels):
        cleaned = clean_text(text)
        ratio, _ = best_match(cleaned, features, train_matrix_normalized, train_texts)
        if ratio < threshold:
            no_match += 1
            continue
        pred = clf.predict(features.transform([cleaned]))[0]
        if pred == true_label:
            top1 += 1
        else:
            wrong += 1
    return {
        "top1Accuracy": round(top1 / n, 4),
        "noMatchRate": round(no_match / n, 4),
        "confidentlyWrongRate": round(wrong / n, 4),
    }


def load_current_meta():
    """Reads the CURRENTLY-DEPLOYED model's model_meta.json, to serve as
    is_regression()'s baseline. Returns None on a fresh checkout with no
    model trained yet (first-ever run), which is_regression() treats as
    "nothing to compare against, promote unconditionally"."""
    try:
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def is_regression(candidate: dict, baseline: dict | None) -> bool:
    """The promotion gate: True if `candidate`'s validation metrics are
    worse than `baseline`'s by more than TOLERANCE on either axis. Call
    this after evaluate() and before writing anything to disk (see main())
    -- if this returns True, main() must not call write_promoted_csv()/
    save()/write_meta(), so a bad retrain never overwrites the live model."""
    if baseline is None or not baseline.get("validationMetrics"):
        return False  # nothing to compare against — first-ever run, anything is an improvement
    prev = baseline["validationMetrics"]
    if "top1Accuracy" not in prev or "confidentlyWrongRate" not in prev:
        return False  # baseline used an older/incompatible metrics schema — nothing safe to compare
    if candidate["top1Accuracy"] < prev["top1Accuracy"] - TOLERANCE:
        return True
    if candidate["confidentlyWrongRate"] > prev["confidentlyWrongRate"] + TOLERANCE:
        return True
    return False


def write_promoted_csv(texts, labels, types):
    """Overwrites the canonical training CSV (Hinglish_Symptoms_V28.csv)
    with the merged+deduped rows that just got promoted -- called by
    main() ONLY after is_regression() has said the candidate is safe. This
    is what keeps the committed CSV in sync with what the deployed joblib
    bundle was actually trained on; don't call it speculatively."""
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "specialist", "type"])
        for text, label, row_type in zip(texts, labels, types):
            writer.writerow([text, label, row_type])


def write_meta(metrics: dict, training_row_count: int, validation_row_count: int):
    """Writes model_meta.json for the newly-promoted model -- what
    api/routes.py's /model-info serves, and what the NEXT retrain run's
    load_current_meta() will treat as the baseline to beat."""
    now = datetime.now(timezone.utc)
    meta = {
        "modelVersion": now.strftime("%Y-%m-%d-%H%M%S"),
        "lastRetrainedAt": now.isoformat(),
        "trainingRowCount": training_row_count,
        "validationRowCount": validation_row_count,
        "validationMetrics": metrics,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmsapi-url")
    parser.add_argument("--service-key")
    parser.add_argument("--training-fixture")
    parser.add_argument("--feedback-fixture")
    args = parser.parse_args()

    if args.training_fixture or args.feedback_fixture:
        if not (args.training_fixture and args.feedback_fixture):
            parser.error("--training-fixture and --feedback-fixture must be given together.")
        print(f"Dry-run mode: loading fixtures from {args.training_fixture} / {args.feedback_fixture}\n")
        training_examples = load_fixture(args.training_fixture)
        feedback = load_fixture(args.feedback_fixture)
    elif args.cmsapi_url and args.service_key:
        print(f"Live mode: fetching from {args.cmsapi_url}\n")
        training_examples, feedback = fetch_live(args.cmsapi_url, args.service_key)
    else:
        parser.error("Provide either --cmsapi-url/--service-key or --training-fixture/--feedback-fixture.")
        return

    feedback_rows = feedback_to_rows(feedback)
    texts, labels, types, weights = merge_rows(training_examples, feedback_rows)
    print(f"Merged training set: {len(texts)} rows across {len(set(labels))} classes\n")

    features = build_feature_union()
    X_feats = features.fit_transform(texts)

    print("Cross-validating candidate models on merged training data:")
    best_name, best_clf = evaluate_candidates(X_feats, labels)
    best_clf.fit(X_feats, labels, sample_weight=weights)

    val_texts, val_labels = load_validation_set()
    candidate_metrics = evaluate(features, best_clf, texts, X_feats, val_texts, val_labels)
    print(f"Candidate validation metrics: {candidate_metrics}")

    baseline = load_current_meta()
    if baseline:
        print(f"Currently-promoted metrics:  {baseline.get('validationMetrics')}")

    if is_regression(candidate_metrics, baseline):
        print("\nNOT PROMOTED — candidate regresses vs. the currently-promoted model. No files changed.")
        return

    classifier = SymptomClassifier(
        features=features,
        model=best_clf,
        model_name=best_name,
        classes=sorted(set(labels)),
        texts=texts,
        texts_matrix=X_feats,
    )
    classifier.save(str(MODEL_PATH))
    write_promoted_csv(texts, labels, types)
    write_meta(candidate_metrics, len(texts), len(val_texts))
    print(f"\nPROMOTED — {CSV_PATH.name}, {META_PATH.name}, and {MODEL_PATH.name} updated.")


if __name__ == "__main__":
    main()
