#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Hinglish Symptom -> Specialist Classifier (with microphone input)
====================================================================
Trains a text classifier that maps a Hinglish (Hindi-English code-mixed,
Roman script) symptom description to the medical specialist it should be
routed to.

Dataset: Hinglish_Symptoms_Reference_V4_.csv
Columns: text, specialist, type
(31 specialists, ~3.1k rows: originals + spelling/keyword/phrase variations)

Approach
--------
- Char-ngram TF-IDF (robust to the spelling variations present in the data,
  e.g. "mein"/"me", "doctor"/"daktar"/"docter") combined with word-ngram
  TF-IDF (captures medical keywords/terms).
- Compares Linear SVM vs Logistic Regression vs Complement Naive Bayes
  via stratified cross-validation, picks the best, then reports held-out
  test performance.
- Saves the trained pipeline + label list to disk for reuse.

Microphone input
-----------------
The interactive test loop listens to the microphone instead of reading
typed input. Speech is transcribed with the `SpeechRecognition` library
(Google's free Web Speech API under the hood, needs internet).

Because the model was trained on Roman-script Hinglish text, the raw
transcription is normalized to Roman script before being shown or fed to
the classifier:

  - Google's recognizer is asked for Hindi ("hi-IN"), since that gives the
    most accurate transcription for Hindi/Hinglish speech -- but it always
    returns the text in Devanagari script (e.g. "आंखों से धुंधला").
  - `devanagari_to_roman()` below converts that Devanagari output into
    Roman letters (e.g. "aankhon se dhundhala") using the
    `indic-transliteration` library's ITRANS scheme, then lowercases and
    tidies it up so it looks like the Hinglish spellings in the training
    data.
  - English words spoken in the same sentence are already returned in
    Roman script by the recognizer and pass through untouched.

Requirements for microphone support (install before running):
    pip install SpeechRecognition pyaudio indic-transliteration

On Linux you may also need the PortAudio system library first, e.g.:
    sudo apt-get install portaudio19-dev

Note on the raw CSV
--------------------
One row in the V4 file (the "Jyada chalne par sans phool jaati hai..."
Cardiologist row) has an unescaped comma inside the free-text field, which
breaks the default C parser ("Expected 3 fields, saw 4"). load_data() below
reads the file with the more tolerant Python engine and skips any row that
still can't be parsed, so training is not interrupted by the one bad row.
"""

import re
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False

DATA_PATH = "Hinglish_Symptoms_Reference_V4_.csv"
MODEL_OUT = "symptom_specialist_classifier.joblib"
RANDOM_STATE = 42

# Language hint passed to the speech recognizer. "hi-IN" gives the most
# accurate transcription for Hindi/Hinglish speech, but always comes back
# in Devanagari script -- devanagari_to_roman() converts it afterwards.
SPEECH_LANGUAGE = "hi-IN"

# Any character in this Unicode block means the string contains Devanagari.
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Phrases that tell the mic to stop listening -- checked (after Devanagari
# is already converted to Roman script) as an exact match on the whole,
# normalized utterance, so a stray "stop" inside a symptom sentence won't
# accidentally trigger it.
STOP_PHRASES = {
    "stop", "quit", "exit",
    "ruk jaiye", "ruk jaiye please", "ruk jao", "ruk jayen", "ruko",
    "ruk jaiyega", "band karo", "band kar do", "band kijiye",
}

# Sample queries to sanity-check the trained model on (edit/add your own).
SAMPLE_QUERIES = [
    "Dil mein bahut zor se dard ho raha hai aur saans phool rahi hai",
    "Bachche ko bukhar hai aur khaana nahi kha raha",
    "Skin par laal daane ho gaye hain aur khujli ho rahi hai",
    "Dant mein bahut dard hai, thanda garam nahi sehta",
    "Aankhon se dhundla dikhayi de raha hai kaafi din se",
    "Ghutno mein dard hai, sidhi chadhne mein takleef hoti hai",
    "Bahut udaas aur akela mehsoos hota hai kaafi dino se",
    "asdkjqwlkej",  # gibberish check
]


class TextSelector(BaseEstimator, TransformerMixin):
    """Pass a text column straight through a FeatureUnion step."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def clean_text(s: str) -> str:
    """Light normalization: collapse whitespace, strip. Case and punctuation
    are left mostly intact -- they carry signal here (e.g. 'BP', 'CT scan')."""
    return " ".join(str(s).split())


def devanagari_to_roman(text: str) -> str:
    """Convert any Devanagari-script portion of `text` into Roman letters,
    leaving already-Roman words (English, or Hindi typed phonetically)
    untouched.

    Uses the ITRANS scheme from `indic-transliteration`, then does a light
    cleanup pass (lowercase, drop ITRANS's special punctuation like the
    apostrophe used for schwa-deletion markers) so the result reads like
    the Hinglish spellings used in the training data, e.g.:
        "आंखों से धुंधला दिखायी दे रहा है" -> "aankhon se dhundhala dikhayi de raha hai"

    If the text has no Devanagari characters at all (e.g. the recognizer
    returned pure English), it is returned unchanged. If the
    `indic-transliteration` package isn't installed, the original text is
    returned unchanged and a one-time warning is printed.
    """
    if not _DEVANAGARI_RE.search(text):
        return text

    if not TRANSLITERATION_AVAILABLE:
        print(
            "Note: got Devanagari-script speech but the 'indic-transliteration' "
            "package isn't installed, so it can't be converted to Roman script. "
            "Install it with:\n    pip install indic-transliteration"
        )
        return text

    roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    roman = roman.replace(".a", "").replace("'", "")
    roman = " ".join(roman.split())
    return roman.lower()


def is_stop_command(text: str) -> bool:
    """True if the (already Roman-script) utterance is a stop command,
    e.g. 'stop' or 'ruk jaiye'. Matches on the whole normalized phrase
    so a real symptom sentence that happens to contain a similar word
    isn't mistaken for a stop command."""
    normalized = " ".join(str(text).strip().lower().split())
    return normalized in STOP_PHRASES


def is_gibberish(text: str) -> bool:
    """Heuristic check for gibberish / nonsensical input.

    Flags text that is empty, too short, has almost no vowels, contains a
    long run of repeated characters, or has a long run of consonants with
    no vowel in between -- all signs of random keyboard mashing rather than
    an actual (Hinglish) symptom description. This is intentionally a
    simple, dependency-free heuristic rather than a model, so it runs
    instantly before we bother the classifier."""
    text = clean_text(text)

    if len(text) < 3:
        return True

    letters_only = re.sub(r"[^a-zA-Z]", "", text)
    if len(letters_only) < 3:
        return True

    # Almost no vowels at all -> unlikely to be real words.
    vowels = sum(1 for c in letters_only.lower() if c in "aeiou")
    vowel_ratio = vowels / len(letters_only)
    if vowel_ratio < 0.15:
        return True

    # Same character repeated 4+ times in a row (e.g. "aaaaa", "asdfff").
    if re.search(r"(.)\1{3,}", text):
        return True

    # A run of 6+ consonants with no vowel in between (e.g. "qwrtplk").
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", text.lower()):
        return True

    if len(text.split()) == 0:
        return True

    return False


def load_data(path: str) -> pd.DataFrame:
    # The Python engine (vs. the default C engine) tolerates the one
    # malformed row in the V4 CSV (an unescaped comma inside the text
    # field); on_bad_lines="skip" drops it instead of erroring out.
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    required_cols = {"text", "specialist", "type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected column(s): {missing}")

    df["text"] = df["text"].map(clean_text)
    df["specialist"] = df["specialist"].map(lambda s: str(s).strip())

    before = len(df)
    df = df.drop_duplicates(subset=["text", "specialist"]).reset_index(drop=True)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate/empty rows ({before} -> {after}).")

    # Drop classes with too few examples to stratify/split (need >= 2 for
    # a stratified split, and ideally more for CV folds).
    counts = df["specialist"].value_counts()
    too_small = counts[counts < 2].index.tolist()
    if too_small:
        print(f"Dropping classes with <2 examples: {too_small}")
        df = df[~df["specialist"].isin(too_small)].reset_index(drop=True)

    return df


def build_feature_union() -> FeatureUnion:
    """Combine word-level and char-level TF-IDF. Char n-grams give
    robustness to the spelling variations in this dataset; word n-grams
    capture medical terms and multi-word keywords."""
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )),
    ])


def evaluate_candidates(X_train_feats, y_train):
    """5-fold stratified CV over a few candidate classifiers; returns the
    name of the best one by mean macro-F1."""
    # Number of CV folds can't exceed the smallest class's example count.
    min_class_count = pd.Series(y_train).value_counts().min()
    n_splits = max(2, min(5, min_class_count))

    candidates = {
        "LinearSVC": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=5.0, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "ComplementNB": ComplementNB(),
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, clf in candidates.items():
        scores = cross_val_score(
            clf, X_train_feats, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        results[name] = scores
        print(f"{name:20s}  macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    best_name = max(results, key=lambda k: results[k].mean())
    print(f"\nBest model: {best_name}")
    return best_name, candidates[best_name]


def main():
    df = load_data(DATA_PATH)
    print(f"Final dataset: {len(df)} rows, {df['specialist'].nunique()} specialists\n")

    X = df["text"].values
    y = df["specialist"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    features = build_feature_union()
    X_train_feats = features.fit_transform(X_train)
    X_test_feats = features.transform(X_test)

    print("Cross-validating candidate models on training data:")
    best_name, best_clf = evaluate_candidates(X_train_feats, y_train)

    # Refit best model on full training set, evaluate on held-out test set
    best_clf.fit(X_train_feats, y_train)
    y_pred = best_clf.predict(X_test_feats)

    print("\n=== Held-out test set performance ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro-F1:  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit feature extractor + model on ALL data for the final deployed pipeline
    final_features = build_feature_union()
    X_all_feats = final_features.fit_transform(X)
    final_clf = type(best_clf)(**best_clf.get_params())
    final_clf.fit(X_all_feats, y)

    pipeline_bundle = {
        "features": final_features,
        "model": final_clf,
        "model_name": best_name,
        "classes": sorted(df["specialist"].unique().tolist()),
    }
    joblib.dump(pipeline_bundle, MODEL_OUT)
    print(f"\nSaved trained pipeline to {MODEL_OUT}")

    print("\n=== Sample query predictions ===")
    for query in SAMPLE_QUERIES:
        if is_gibberish(query):
            print(f"{query!r:70s} -> Wrong input. Try again")
            continue
        feats = final_features.transform([clean_text(query)])
        pred = final_clf.predict(feats)[0]
        print(f"{query!r:70s} -> {pred}")


def predict(text: str, bundle_path: str = MODEL_OUT):
    """Convenience function to load the saved model and predict a specialist
    for a new piece of Hinglish symptom text. Returns None (and prints a
    message) for gibberish input instead of generating a prediction."""
    if is_gibberish(text):
        print("Wrong input. Try again")
        return None

    bundle = joblib.load(bundle_path)
    feats = bundle["features"].transform([clean_text(text)])
    pred = bundle["model"].predict(feats)[0]
    return pred


def listen_from_microphone(language: str = SPEECH_LANGUAGE,
                            timeout: float = 8.0,
                            phrase_time_limit: float = 12.0):
    """Record audio from the default microphone and transcribe it to text,
    normalized to Roman script.

    Returns the transcribed (Roman-script) string, or None if nothing
    could be understood (silence, unclear audio, no network, etc.). Prints
    a short status message either way so the console output stays plain
    text.
    """
    if not SPEECH_RECOGNITION_AVAILABLE:
        print(
            "Microphone input requires the 'SpeechRecognition' and 'pyaudio' "
            "packages. Install them with:\n"
            "    pip install SpeechRecognition pyaudio"
        )
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print("Listening... speak your symptom now.")
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )
    except sr.WaitTimeoutError:
        print("No speech detected before timeout. Try again.")
        return None
    except OSError as e:
        print(f"Microphone not available ({e}).")
        return None

    print("Transcribing...")
    try:
        raw_text = recognizer.recognize_google(audio, language=language)
        text = devanagari_to_roman(raw_text)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio. Try again.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error ({e}). Check your internet connection.")
        return None


def interactive_test(bundle_path: str = MODEL_OUT):
    """Listen to the microphone for symptom descriptions and print the
    predicted specialist as text, one query at a time, looping for as
    many queries as you like.

    The mic keeps listening across multiple queries and only stops when:
      - You say a stop phrase (e.g. 'stop' or 'ruk jaiye'), or
      - You press Ctrl+C.

    Unclear audio or gibberish input does NOT stop the mic -- it just
    asks again. Getting a prediction also does NOT stop the mic -- it
    goes right back to listening for your next symptom.
    """
    print(
        "\n=== Try your own queries via microphone "
        "(say 'stop' or 'ruk jaiye' to quit, Ctrl+C to force-quit) ==="
    )
    while True:
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                print(
                    "Microphone input requires the 'SpeechRecognition' and 'pyaudio' "
                    "packages. Install them with:\n"
                    "    pip install SpeechRecognition pyaudio"
                )
                break
            else:
                text = listen_from_microphone()
        except KeyboardInterrupt:
            print("\nStopped.")
            break

        if text is None:
            # Nothing understood -- keep listening, don't stop the mic.
            continue

        if is_stop_command(text):
            print("Stopping -- mic is now off.")
            break

        if is_gibberish(text):
            print("Wrong input. Try again")
            continue

        print(f"Predicted specialist: {predict(text, bundle_path)}")


if __name__ == "__main__":
    main()
    interactive_test()


# In[ ]:


# In[ ]:




