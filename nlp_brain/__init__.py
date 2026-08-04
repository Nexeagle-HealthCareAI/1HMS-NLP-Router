"""NLP Brain layer: Hinglish symptom -> specialist classification.

Zero FastAPI/voice/HTTP knowledge lives here -- this package only knows how
to turn text into a specialist prediction (or an honest "not confident
enough"). The api/ and voice/ layers are callers of this package, never the
other way around.
"""
from .classifier import PredictionResult, SymptomClassifier
from .config import MATCH_THRESHOLD, MODEL_OUT, NO_MATCH_MESSAGE
from .text_utils import clean_text, is_gibberish

__all__ = [
    "SymptomClassifier",
    "PredictionResult",
    "MODEL_OUT",
    "MATCH_THRESHOLD",
    "NO_MATCH_MESSAGE",
    "clean_text",
    "is_gibberish",
]
