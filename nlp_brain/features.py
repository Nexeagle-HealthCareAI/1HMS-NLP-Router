"""Feature engineering: turns cleaned text into the vectors the classifier and
the coverage-gate matcher both operate on."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_feature_union() -> FeatureUnion:
    """Returns a fresh, UNFITTED FeatureUnion -- call this once per training
    run (SymptomClassifier.train() calls it twice: once for the held-out CV
    split, once again for the final all-data fit) and call `.fit_transform()`
    /`.transform()` on the result; never share one fitted instance's
    vocabulary across two different datasets. Combines word-level and
    char-level TF-IDF: char n-grams give robustness to the spelling
    variations in this dataset (e.g. "mein"/"me", "doctor"/"daktar"/"docter");
    word n-grams capture medical terms and multi-word keywords."""
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
