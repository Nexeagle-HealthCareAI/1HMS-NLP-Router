#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
CPU-only: Hinglish symptom -> specialist router.

Builds on the original prototype (TF-IDF char/word n-grams + Logistic
Regression classifier, plus a TF-IDF cosine "nearest known phrase" search)
and adds two things the original didn't handle:

  1. MULTI-SYMPTOM SENTENCES
     "pet mein dard hai aur sar bhi dukh raha hai" names two problems in
     one sentence. This script splits a sentence into symptom segments
     (on commas, "aur"/"and"/"plus"/etc.) and classifies each segment on
     its own, then returns the de-duplicated list of specialists needed.

  2. UNSEEN PARAPHRASES
     A sentence that never appeared in the CSV, but means the same thing
     as something that did (different word order, synonyms, spelling),
     is handled two ways in combination: the classifier generalizes from
     n-gram features, and a cosine-similarity search over the CSV's own
     phrases acts as a fallback when the classifier isn't confident.

  3. DEFAULT / FALLBACK SPECIALIST
     If a segment doesn't clear a confidence bar on EITHER method (e.g.
     "aise hi hai, kuch pata nahi" - no real symptom described), it's
     routed to a default specialist (General Physician) instead of
     forcing a guess.

Dataset: Hinglish_Symptoms_Reference.csv (same folder as this script, or
pass --data /path/to/file.csv)

Run:
    python symptom_router.py                       (interactive, default)
    python symptom_router.py --data /path/to/file.csv
    python symptom_router.py --query "pet mein dard hai aur sar bhi dukh raha hai"
    python symptom_router.py --demo                (built-in sample sentences,
                                                      including multi-symptom
                                                      and unseen paraphrases)
"""
import argparse
import csv
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity

# Was hardcoded to a different machine's path (C:\Users\ASUS\Downloads\...) — defaults to the
# comprehensive 32-class dataset shipped alongside this script. The original 17-class
# Hinglish_Symptoms_Reference.csv lives in the same folder, kept for reference/audit only.
DATA_PATH = Path(__file__).parent / "Hinglish_Symptoms_Reference_v2.csv"

# Specialist used when neither the classifier nor the nearest-phrase
# search is confident enough to commit to a specific specialist.
DEFAULT_SPECIALIST = "General Physician"

# How confident each method has to be before its answer is trusted.
# Both are cosine-similarity-ish scores in [0, 1] (classifier: predicted
# probability of the top class; search: cosine similarity to the closest
# known phrase). Tune these against your own data if predictions look
# too trigger-happy or too conservative.
#
# Retuned against a leakage-free held-out split of the 32-class dataset (train on
# Original/Specialist Term/Keyword Phrase rows, test on Spelling/Semantic Variation rows —
# see data_pipeline/). A sweep over clf_threshold x sim_threshold showed the classifier
# threshold barely moves accuracy or the "confidently wrong" rate in this range (0.22 is
# already close to the model's natural confidence floor for this many classes); the search
# threshold is the load-bearing knob. Raising it from 0.28 to 0.33 cut the confidently-wrong
# rate ~8.8% -> ~7.4% at a cost of only ~0.6pt accuracy and a modest rise in how often the
# router defers to General Physician (1.3% -> 3.6%) — a good trade for a symptom router,
# since "defer to GP" is a safe outcome, unlike a confidently wrong specialist referral.
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.22
SEARCH_SIMILARITY_THRESHOLD = 0.33

# Some datasets label the same specialist under two different names, or
# use spellings that don't match the NMC-standard patient-facing names
# (see Specialist.txt). Treating them as separate classes fragments
# training data and unfairly "penalizes" correct predictions that just
# used a different name. Normalize known duplicates here; this is safe
# to run even if the CSV already uses the canonical names on the right
# (a key that doesn't match anything just does nothing).
LABEL_ALIASES = {
    "Orthopedist": "Orthopaedic Surgeon (Bone)",
    "Orthopedic Surgeon": "Orthopaedic Surgeon (Bone)",
    "Otolaryngologist - ENT": "ENT Specialist",
    "Family Physician / Internist": "General Physician",
    "Cardiologist": "Cardiologist (Heart)",
    "Dermatologist": "Dermatologist (Skin)",
    "Endocrinologist": "Endocrinologist (Hormones/Diabetes)",
    "Gynecologist": "Gynaecologist",
    "Ophthalmologist": "Ophthalmologist (Eye)",
    "Pediatrician": "Paediatrician",
    "Pulmonologist": "Pulmonologist (Chest/Lungs)",
    # Rows that only describe a duration ("kal se ho raha hai") carry no
    # real symptom information; fold them into the same default bucket
    # this script falls back to for low-confidence input.
    "General (Duration)": DEFAULT_SPECIALIST,
}

# Same-organ medical-vs-surgical pairs where whether the case needs medicine or surgery is a
# clinical judgment made AFTER an in-person exam — a patient's own words essentially never
# carry that information, so training the classifier to split them is training it to guess
# something the input can't answer. Merged for THIS router's output only; the underlying
# 32-class dataset (Hinglish_Symptoms_Reference_v2.csv) keeps the finer labels intact for any
# future use where that distinction genuinely is knowable (e.g. after a GP/triage exam).
MODEL_OUTPUT_MERGES = {
    "GI/Surgical Gastroenterologist": "Gastroenterologist",
    "Cardiothoracic Surgeon": "Cardiologist (Heart)",
    "Neurosurgeon": "Neurologist",
}

# Words/punctuation that typically join two separate symptom mentions in
# one sentence. Matched case-insensitively; \b keeps it from matching
# inside other words (e.g. won't split "aurat").
SEGMENT_SPLIT_PATTERN = re.compile(
    r"\s*(?:,|;|/|\bevam\b|\baur\b|\band\b|\bplus\b|\balso\b|\bwith\b|\bsaath hi\b|\bsath hi\b)\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data loading / training (same approach as the original prototype)
# ---------------------------------------------------------------------------

def load_data(path: Path, apply_output_merges: bool = True):
    """Load (text, specialist) pairs from the CSV, skipping malformed rows.

    apply_output_merges: fold the same-organ medical/surgical siblings together (see
    MODEL_OUTPUT_MERGES) — the right default for THIS script's own single-label CLI/demo
    output. Callers with a taxonomy that already distinguishes those siblings (e.g. a
    consumer whose own specialty list keeps neurology/neurosurgery separate) should pass
    False and rely on classify_segment's candidate list to surface both when genuinely
    ambiguous, instead of losing the distinction outright.
    """
    texts, labels = [], []
    skipped = 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{path} appears to be empty.")

        required_cols = {"text", "specialist"}
        missing_cols = required_cols - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"{path} is missing required column(s): {sorted(missing_cols)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("specialist") or "").strip()
            if not text or not label:
                skipped += 1
                continue
            label = LABEL_ALIASES.get(label, label)
            if apply_output_merges:
                label = MODEL_OUTPUT_MERGES.get(label, label)
            texts.append(text)
            labels.append(label)

    if skipped:
        print(f"Note: skipped {skipped} row(s) with missing text/specialist.")

    if not texts:
        raise ValueError(f"No usable rows found in {path}.")

    return texts, labels


def train_classifier(texts, labels):
    # Combine char n-grams (catch typos/spelling variants) with word
    # n-grams (catch word-order / phrase-level patterns). Blending both
    # usually beats either alone.
    vectorizer = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
        )),
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )),
    ])

    X_vec = vectorizer.fit_transform(texts)

    # class_weight="balanced" stops the model from just favoring whichever
    # specialist class has the most examples.
    clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")

    # K-fold cross-validation gives a far more trustworthy accuracy
    # estimate than a single train/test split, especially on a dataset
    # this size. n_splits is capped by the smallest class's example count.
    label_counts = {label: labels.count(label) for label in set(labels)}
    min_class_count = min(label_counts.values())
    n_splits = max(2, min(5, min_class_count))

    try:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_vec, labels, cv=cv)
        print(
            f"{n_splits}-fold cross-validation accuracy: "
            f"{scores.mean():.2f} (+/- {scores.std():.2f})\n"
        )
    except ValueError:
        # Stratified CV needs every class to have >= n_splits examples.
        # A very small class makes that impossible; fall back to plain
        # K-fold so we still get SOME estimate.
        try:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_vec, labels, cv=cv)
            print(
                f"Warning: one or more classes have too few examples for "
                f"stratified CV; used plain {n_splits}-fold CV instead "
                "(less reliable per-class, but still informative overall).\n"
                f"{n_splits}-fold cross-validation accuracy: "
                f"{scores.mean():.2f} (+/- {scores.std():.2f})\n"
            )
        except ValueError as e:
            print(
                f"Warning: cross-validation skipped ({e}). Add more "
                "examples per specialist class for a reliable estimate.\n"
            )

    # Fit the final model on ALL the data (not just one fold) so it has
    # the most information available for real predictions.
    clf.fit(X_vec, labels)

    return vectorizer, clf


def predict(vectorizer, clf, text, top_k=3):
    vec = vectorizer.transform([text])
    probs = clf.predict_proba(vec)[0]
    ranked = sorted(zip(clf.classes_, probs), key=lambda x: -x[1])[:top_k]
    return ranked


def build_search_index(texts, labels, vectorizer):
    """Search index over the same phrase bank: given a raw query,
    return the closest known phrases + their specialist, as a stand-in
    for matching against a real doctor/clinic directory."""
    matrix = vectorizer.transform(texts)
    return matrix, texts, labels


def search(query, vectorizer, index_matrix, index_texts, index_labels, top_k=3):
    qvec = vectorizer.transform([query])
    sims = cosine_similarity(qvec, index_matrix)[0]
    ranked_idx = sims.argsort()[::-1][:top_k]
    return [(index_texts[i], index_labels[i], sims[i]) for i in ranked_idx]


# ---------------------------------------------------------------------------
# New: multi-symptom segmentation + confidence-gated routing
# ---------------------------------------------------------------------------

def split_segments(text):
    """Break a sentence into separate symptom mentions.

    "pet mein dard hai aur sar bhi dukh raha hai" -> two segments.
    A plain single-symptom sentence just comes back as one segment.
    """
    raw_parts = SEGMENT_SPLIT_PATTERN.split(text)
    segments = [p.strip() for p in raw_parts if p and p.strip()]

    # Guard against over-splitting on short fragments/stray punctuation
    # (e.g. a trailing "," with nothing meaningful after it).
    segments = [s for s in segments if len(s.split()) >= 2 or len(s) >= 4]

    if not segments:
        segments = [text.strip()]
    return segments


# If the runner-up option's score is within this margin of the top one, the model isn't
# meaningfully more confident in #1 than #2 — surfacing only #1 manufactures false precision.
# Chasing single-label accuracy on genuinely close calls is a losing game; showing a short,
# honest list is both more useful to the patient and easier to get "right" (true label just
# needs to be somewhere in the list, not uniquely first).
CANDIDATE_MARGIN = 0.12
MAX_CANDIDATES = 3


def build_candidates(primary, classifier_ranked, search_ranked, method):
    """Expand a single primary pick into a short, deduped list of specialists worth showing,
    when the runner-up(s) are close enough that picking only #1 would be overconfident."""
    candidates = [primary]

    def maybe_add(label, top_score, this_score):
        if label not in candidates and len(candidates) < MAX_CANDIDATES:
            if (top_score - this_score) <= CANDIDATE_MARGIN:
                candidates.append(label)

    if method == "classifier":
        top_score = classifier_ranked[0][1]
        for label, score in classifier_ranked[1:]:
            maybe_add(label, top_score, score)
    elif method == "nearest-known-phrase":
        # De-dup the raw phrase hits down to best score per label first.
        best_per_label = {}
        for _, label, score in search_ranked:
            if label not in best_per_label or score > best_per_label[label]:
                best_per_label[label] = score
        ranked_labels = sorted(best_per_label.items(), key=lambda x: -x[1])
        top_score = ranked_labels[0][1]
        for label, score in ranked_labels[1:]:
            maybe_add(label, top_score, score)
        # The classifier's own top guess didn't clear ITS bar, but it's still signal worth a
        # look, especially since search only won because the classifier was under-confident.
        maybe_add(classifier_ranked[0][0], top_score, classifier_ranked[0][1])
    else:  # default (low confidence) — nothing cleared a bar; still surface the closest
        # classifier guess as an explicit low-confidence hint rather than staying silent.
        if classifier_ranked[0][0] != primary and len(candidates) < MAX_CANDIDATES:
            candidates.append(classifier_ranked[0][0])

    return candidates[:MAX_CANDIDATES]


def classify_segment(segment, vectorizer, clf, index_matrix, index_texts, index_labels,
                      clf_threshold=CLASSIFIER_CONFIDENCE_THRESHOLD,
                      sim_threshold=SEARCH_SIMILARITY_THRESHOLD, top_k=3):
    """Route one symptom segment to a specialist (plus close runners-up), or to the default
    specialist if neither method clears its confidence bar."""
    classifier_ranked = predict(vectorizer, clf, segment, top_k=top_k)
    top_label, top_prob = classifier_ranked[0]

    search_ranked = search(segment, vectorizer, index_matrix, index_texts, index_labels, top_k=top_k)
    top_phrase, top_search_label, top_score = search_ranked[0]

    if top_prob >= clf_threshold:
        specialist = top_label
        method = "classifier"
        confidence = float(top_prob)
    elif top_score >= sim_threshold:
        specialist = top_search_label
        method = "nearest-known-phrase"
        confidence = float(top_score)
    else:
        specialist = DEFAULT_SPECIALIST
        method = "default (low confidence)"
        confidence = float(max(top_prob, top_score))

    candidates = build_candidates(specialist, classifier_ranked, search_ranked, method)

    return {
        "segment": segment,
        "specialist": specialist,
        "candidates": candidates,
        "is_ambiguous": len(candidates) > 1,
        "method": method,
        "confidence": confidence,
        "classifier_top": classifier_ranked,
        "search_top": search_ranked,
    }


def classify_sentence(text, vectorizer, clf, index_matrix, index_texts, index_labels):
    """Full pipeline for one input sentence (which may describe several
    symptoms): split -> classify each -> de-duplicate specialists.

    Returns (specialists, per_segment_results):
      specialists       de-duplicated list of specialists worth showing, in the order
                         their symptom was first mentioned — includes each segment's close
                         runners-up (see build_candidates), not just its single top pick
      per_segment_results  one classify_segment() dict per segment, for
                         anyone who wants the detail/confidence behind
                         each recommendation
    """
    segments = split_segments(text)
    per_segment_results = [
        classify_segment(seg, vectorizer, clf, index_matrix, index_texts, index_labels)
        for seg in segments
    ]

    specialists = []
    for result in per_segment_results:
        for candidate in result["candidates"]:
            if candidate not in specialists:
                specialists.append(candidate)

    if not specialists:
        specialists = [DEFAULT_SPECIALIST]

    return specialists, per_segment_results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_sentence_result(text, vectorizer, clf, index_matrix, index_texts, index_labels):
    specialists, per_segment = classify_sentence(
        text, vectorizer, clf, index_matrix, index_texts, index_labels
    )

    def format_result(r):
        if r["is_ambiguous"]:
            others = " / ".join(r["candidates"][1:])
            return (f"-> {r['specialist']}  (close call, also consider: {others})  "
                     f"[{r['method']}, confidence {r['confidence']:.2f}]")
        return f"-> {r['specialist']}  [{r['method']}, confidence {r['confidence']:.2f}]"

    print(f"\nQuery: {text!r}")
    if len(per_segment) > 1:
        print(f"  Detected {len(per_segment)} symptom segment(s):")
        for r in per_segment:
            print(f"    - {r['segment']!r}")
            print(f"        {format_result(r)}")
    else:
        print(f"  {format_result(per_segment[0])}")

    print(f"  Recommended specialist(s): {', '.join(specialists)}")


def run_demo(vectorizer, clf, index_matrix, index_texts, index_labels, queries):
    print("\n--- Symptom routing ---")
    for q in queries:
        print_sentence_result(q, vectorizer, clf, index_matrix, index_texts, index_labels)


def interactive_loop(vectorizer, clf, index_matrix, index_texts, index_labels):
    print("Interactive mode. Type a phrase (single or multi-symptom) and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        print_sentence_result(query, vectorizer, clf, index_matrix, index_texts, index_labels)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the dataset CSV (default: Hinglish_Symptoms_Reference.csv next to this script)",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Custom sentence to route (can be passed multiple times, can name "
        "multiple symptoms in one sentence). If omitted, a built-in set of "
        "sample sentences is used.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive prompt: type a phrase, see results, repeat. "
        "Type 'exit' or 'quit' to stop. This is also the DEFAULT behavior "
        "if no other flag is given.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in sample sentences instead of asking for input.",
    )
    # parse_known_args (not parse_args) so this doesn't crash when run
    # inside Jupyter/IPython, which injects its own extra arguments
    # (e.g. "-f kernel-xxxx.json") that this script doesn't need to know about
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    data_path = args.data

    if not data_path.exists():
        print(f"Missing data file: {data_path}", file=sys.stderr)
        print(
            "Place your dataset at that path, or point to it with "
            "--data \"C:\\path\\to\\Hinglish_Symptoms_Reference.csv\"",
            file=sys.stderr,
        )
        sys.exit(1)

    if data_path.is_dir():
        print(
            f"Error: {data_path} is a folder, not a CSV file. "
            "Pass the full path to the CSV itself, e.g.\n"
            '  --data "C:\\Users\\ASUS\\Downloads\\Hinglish_Symptoms_Reference.csv"',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading dataset from: {data_path.resolve()}\n")

    try:
        texts, labels = load_data(data_path)
    except ValueError as e:
        print(f"Error loading {data_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(
            f"Error: permission denied reading {data_path} ({e}). "
            "Check the file isn't open in another program (e.g. Excel), "
            "and that the path points to the CSV file, not a folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(texts)} examples across {len(set(labels))} specialist classes.\n")

    vectorizer, clf = train_classifier(texts, labels)
    index_matrix, index_texts, index_labels = build_search_index(texts, labels, vectorizer)

    # Decide what to run, in order of priority:
    #   1. --query "..."   -> route exactly those sentences, then exit
    #   2. --demo          -> run the built-in sample sentences
    #   3. nothing passed  -> DEFAULT: ask the user for input interactively
    if args.query:
        run_demo(vectorizer, clf, index_matrix, index_texts, index_labels, args.query)
    elif args.demo:
        sample_queries = [
            # single symptom, in-dataset style
            "pet mai bahut dard ho raha hai",
            # multi-symptom, in ONE sentence -> should return 2 specialists
            "pet mein dard hai aur sar bhi dukh raha hai",
            "mujhe chest mein dard ho raha hai, saath hi ghutno mein bhi dikkat hai",
            "aankhon mein jalan ho rahi hai and daant mein bhi dard hai",
            # paraphrase NOT literally in the CSV, but same meaning as a
            # psychiatrist-type entry
            "aajkal mera mood bahut off rehta hai, neend bhi theek se nahi aa rahi",
            # vague / no real symptom described -> should fall back to default
            "bas thoda ajeeb sa lag raha hai, kuch samajh nahi aa raha",
            # newer, finer-grained classes (added when the taxonomy grew from 17 to 32)
            "pairo ki nasein phool kar bahar dikhne lagi hain, khade rehne se dard hota hai",
            "MRI karwani hai ghutne ka, doctor ne bola hai",
            "gaanth dheere dheere badh rahi hai aur ab dard bhi hone laga hai",
            "mere dada ji ko chalte waqt balance nahi banta",
        ]
        run_demo(vectorizer, clf, index_matrix, index_texts, index_labels, sample_queries)
    else:
        # default: always take input from the user (input() prompt),
        # whether or not --interactive was explicitly passed
        interactive_loop(vectorizer, clf, index_matrix, index_texts, index_labels)


if __name__ == "__main__":
    main()


# In[ ]:




