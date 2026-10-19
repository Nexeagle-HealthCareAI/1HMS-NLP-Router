"""FastAPI layer: HTTP contract + orchestration only.

Depends on nlp_brain.SymptomClassifier as its one abstraction for "predict a
specialist" -- no TF-IDF/gibberish/coverage-gate logic is reimplemented
here. Nothing outside this package (voice/, data_pipeline/) should import
from it; it's the outermost layer.
"""
