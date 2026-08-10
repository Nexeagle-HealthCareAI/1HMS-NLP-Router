"""Candidate ranking: when the classifier's top pick has a close runner-up,
show both instead of forcing an overconfident single guess."""
import numpy as np


def ranked_labels(model, feats) -> list:
    """Every class `model` knows, ranked most-confident-first for one
    query's already-vectorized feature row. Prefers predict_proba
    (LogisticRegression, ComplementNB) since it's already a 0-1 distribution
    that sums to 1; LinearSVC has no predict_proba, so its decision_function
    margins go through softmax to land in the same comparable range instead
    -- not a true probability, but consistent enough for CANDIDATE_MARGIN to
    mean roughly the same thing regardless of which model type cross-
    validation picked (see nlp_brain.training.evaluate_candidates)."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(feats)[0]
    else:
        raw = model.decision_function(feats)[0]
        exp = np.exp(raw - raw.max())
        scores = exp / exp.sum()
    return sorted(zip(model.classes_, scores), key=lambda pair: -pair[1])


def build_candidates(ranked: list, margin: float, max_candidates: int) -> list:
    """Expand the top-ranked label into a short, deduped list of specialists
    worth showing, when runner-up(s) are within `margin` of the top score --
    close enough that returning only the #1 pick would be overconfident.
    `ranked` must already be sorted most-confident-first (see
    ranked_labels()). Always returns at least [ranked[0][0]], and never more
    than `max_candidates` entries."""
    top_score = ranked[0][1]
    candidates = [ranked[0][0]]
    for label, score in ranked[1:]:
        if len(candidates) >= max_candidates:
            break
        if top_score - score <= margin:
            candidates.append(label)
    return candidates


def merge_candidates(segment_candidates: list, max_candidates: int) -> list:
    """Combines several segments' own candidate lists (classifier.predict()
    calls this once per multi-symptom query, one list per segment -- see
    segmentation.split_segments()) into one final, deduped shortlist, in
    first-mention order, re-capped at `max_candidates`.

    Each segment's own list already respects `max_candidates` (build_candidates()
    guarantees that), but simply concatenating N segments' capped lists can
    still exceed it -- e.g. two 3-candidate segments combine to as many as 6
    unique labels before this trims it back down. This is the one place that
    invariant is actually guaranteed for the final, user-facing result; a
    real 3-segment query once returned 5 candidates (cap 3) before this
    existed -- see tests/test_candidates.py."""
    merged: list = []
    for candidates in segment_candidates:
        for label in candidates:
            if label not in merged:
                merged.append(label)
    return merged[:max_candidates]
