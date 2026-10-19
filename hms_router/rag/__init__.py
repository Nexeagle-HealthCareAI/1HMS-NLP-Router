"""RAG layer for /route-symptom: retrieve labelled example sentences, score specialties, fuse.

Separate from the cluster retriever /chat uses (specialist_rag.SpecialistRetriever), which is untouched.
"""
from .classifier_scorer import ClassifierScorer, EmergencyModelScore
from .corpus import LabelledExample, load_examples
from .encoder import TextEncoder
from .example_retriever import EvidenceVoteScorer, ExampleRetriever
from .fusion import WeightedFusion
from .model_bundle import MODEL_PATH, ModelBundle
from .protocols import Retriever, Scorer

__all__ = ["ClassifierScorer", "EmergencyModelScore", "LabelledExample", "load_examples", "TextEncoder",
           "EvidenceVoteScorer", "ExampleRetriever", "WeightedFusion", "MODEL_PATH", "ModelBundle", "Retriever",
           "Scorer"]
