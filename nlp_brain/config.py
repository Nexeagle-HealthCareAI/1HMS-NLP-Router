'''
"""Constants for the NLP Brain layer. No FastAPI/voice/HTTP imports belong here."""

DATA_PATH = "Hinglish_Symptoms_V28.csv"
MODEL_OUT = "symptom_specialist_classifier.joblib"
RANDOM_STATE = 42

# Minimum cosine similarity (0-1) between a user's input and the closest known
# training example (both as TF-IDF vectors) before we trust the model's
# prediction. Below this, we ask the user for more information instead of
# guessing.
#
# NOTE on calibration: data_pipeline/validation_set.csv can't be used to pick
# this -- ~97% of its rows are exact-text duplicates of rows already in the
# training CSV, so every query scores a trivial 1.0 against it. This value
# was instead picked from realistic hand-written Hinglish symptom queries NOT
# present in the training data (which scored ~0.29-0.87) vs. gibberish/
# keyboard-mash strings (~0.0-0.19) -- 0.25 clears all the former with margin
# while rejecting the latter. It does NOT reliably reject coherent but
# off-topic Hinglish text (e.g. "aaj cricket match kab hai" scored ~0.26) --
# cosine similarity over TF-IDF is a lexical/character overlap signal, not a
# semantic one, so some shared function words ("mein", "hai", "kab") are
# enough to clear this bar. Tightening it to filter those out would also
# start rejecting real symptom queries (the lowest-scoring genuine one seen
# was ~0.29) -- a real fix needs a semantic signal, not just a higher
# threshold. Re-verify this still holds when retraining against a new corpus
# (see nlp_brain.cli's calibration check).
MATCH_THRESHOLD = 0.25

# How much worse (in ranked-confidence score) a runner-up specialist is
# allowed to be than the top pick and still get surfaced alongside it,
# instead of showing only the single top guess. Ported from the original
# prototype's calibrated value (Model_1_Doctor_Dekho.py, deleted in 25df289
# during the O(N^2) perf rewrite that also dropped this mechanism) which
# tuned it against a single LogisticRegression model's predict_proba output.
# nlp_brain.candidates.ranked_labels() now also has to handle LinearSVC
# (no predict_proba -- falls back to a softmax of decision_function), so
# this margin's exact calibration should be re-checked if cross-validation
# starts consistently picking LinearSVC over the probability-based models
# (see nlp_brain.training.evaluate_candidates).
CANDIDATE_MARGIN = 0.12

# Hard cap on how many specialists a single prediction can surface, even if
# more than this many fall within CANDIDATE_MARGIN of the top score --
# keeps an off-calibration margin from ever turning one query into "here's
# half our specialist list."
MAX_CANDIDATES = 5

NO_MATCH_MESSAGE = "No matches found."

# Sample queries to sanity-check a freshly trained model on.
SAMPLE_QUERIES = [
    "Dil mein bahut zor se dard ho raha hai aur saans phool rahi hai",
    "Bachche ko bukhar hai aur khaana nahi kha raha",
    "Skin par laal daane ho gaye hain aur khujli ho rahi hai",
    "Dant mein bahut dard hai, thanda garam nahi sehta",
]

GIBBERISH_SAMPLE_QUERIES = ["jgjhkj", "asfgtqwafazfyiur", "zxcvbnmlkjhgfdsaqwerty"]
'''
"""Constants for the NLP Brain layer. No FastAPI/voice/HTTP imports belong here."""

DATA_PATH = "Hinglish_Symptoms_V28.csv"
MODEL_OUT = "symptom_specialist_classifier.joblib"
RANDOM_STATE = 42

# Minimum cosine similarity (0-1) between a user's input and the closest known
# training example (both as TF-IDF vectors) before we trust the model's
# prediction. Below this, we ask the user for more information instead of
# guessing.
#
# NOTE on calibration: data_pipeline/validation_set.csv can't be used to pick
# this -- ~97% of its rows are exact-text duplicates of rows already in the
# training CSV, so every query scores a trivial 1.0 against it. This value
# was instead picked from realistic hand-written Hinglish symptom queries NOT
# present in the training data (which scored ~0.29-0.87) vs. gibberish/
# keyboard-mash strings (~0.0-0.19) -- 0.25 clears all the former with margin
# while rejecting the latter. It does NOT reliably reject coherent but
# off-topic Hinglish text (e.g. "aaj cricket match kab hai" scored ~0.26) --
# cosine similarity over TF-IDF is a lexical/character overlap signal, not a
# semantic one, so some shared function words ("mein", "hai", "kab") are
# enough to clear this bar. Tightening it to filter those out would also
# start rejecting real symptom queries (the lowest-scoring genuine one seen
# was ~0.29) -- a real fix needs a semantic signal, not just a higher
# threshold. Re-verify this still holds when retraining against a new corpus
# (see nlp_brain.cli's calibration check).
MATCH_THRESHOLD = 0.25

# How much worse (in ranked-confidence score) a runner-up specialist is
# allowed to be than the top pick and still get surfaced alongside it,
# instead of showing only the single top guess. Ported from the original
# prototype's calibrated value (Model_1_Doctor_Dekho.py, deleted in 25df289
# during the O(N^2) perf rewrite that also dropped this mechanism) which
# tuned it against a single LogisticRegression model's predict_proba output.
# nlp_brain.candidates.ranked_labels() now also has to handle LinearSVC
# (no predict_proba -- falls back to a softmax of decision_function), so
# this margin's exact calibration should be re-checked if cross-validation
# starts consistently picking LinearSVC over the probability-based models
# (see nlp_brain.training.evaluate_candidates).
CANDIDATE_MARGIN = 0.12

# Hard cap on how many specialists a single prediction can surface, even if
# more than this many fall within CANDIDATE_MARGIN of the top score --
# keeps an off-calibration margin from ever turning one query into "here's
# half our specialist list."
MAX_CANDIDATES = 5

NO_MATCH_MESSAGE = "No matches found."

# Sample queries to sanity-check a freshly trained model on.
SAMPLE_QUERIES = [
    "Dil mein bahut zor se dard ho raha hai aur saans phool rahi hai",
    "Bachche ko bukhar hai aur khaana nahi kha raha",
    "Skin par laal daane ho gaye hain aur khujli ho rahi hai",
    "Dant mein bahut dard hai, thanda garam nahi sehta",
]

GIBBERISH_SAMPLE_QUERIES = ["jgjhkj", "asfgtqwafazfyiur", "zxcvbnmlkjhgfdsaqwerty"]

# --- Groq-based classification (GroqSymptomClassifier) ---------------------
# Independent of the sklearn pipeline's constants above -- kept separate
# rather than reusing MATCH_THRESHOLD/CANDIDATE_MARGIN/MAX_CANDIDATES, since
# those were calibrated against TF-IDF cosine similarity and this pipeline's
# confidence signal is an LLM's self-reported score instead; the numbers
# aren't interchangeable even though the roles look similar.

# Groq's currently recommended model as of this writing. llama-3.3-70b-versatile
# (the previous default) was decommissioned by Groq on 2026-08-16 -- check
# https://console.groq.com/docs/deprecations before assuming this is still
# current if this code has sat untouched for a while.
GROQ_MODEL = "openai/gpt-oss-120b"

# Minimum self-reported confidence (0-1) from the Groq model before its
# top-ranked specialist is trusted. Below this (or on an outright API
# failure), GroqSymptomClassifier falls back to DEFAULT_SPECIALIST rather
# than reporting no match -- there's no gibberish-check failure here to
# explain an empty result the way there was in the sklearn pipeline.
GROQ_CONFIDENCE_THRESHOLD = 0.5

# How many specialists to return per query, ranked highest-confidence first.
# Unlike sklearn's CANDIDATE_MARGIN-based build_candidates(), this is not a
# "how close is close enough" cutoff -- Groq always returns up to this many,
# regardless of how far apart their scores are.
GROQ_MAX_CANDIDATES = 3

# Safe fallback recommendation, guaranteed to appear in every non-gibberish
# result -- either topping up the model's confident picks to fill
# GROQ_MAX_CANDIDATES, or standing alone when confidence is too low or the
# API call fails outright. Never applied to gibberish input.
DEFAULT_SPECIALIST = "General Physician"

GROQ_MAX_RETRIES = 2

