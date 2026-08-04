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

NO_MATCH_MESSAGE = "No matches found."

# Sample queries to sanity-check a freshly trained model on.
SAMPLE_QUERIES = [
    "Dil mein bahut zor se dard ho raha hai aur saans phool rahi hai",
    "Bachche ko bukhar hai aur khaana nahi kha raha",
    "Skin par laal daane ho gaye hain aur khujli ho rahi hai",
    "Dant mein bahut dard hai, thanda garam nahi sehta",
]

GIBBERISH_SAMPLE_QUERIES = ["jgjhkj", "asfgtqwafazfyiur", "zxcvbnmlkjhgfdsaqwerty"]
