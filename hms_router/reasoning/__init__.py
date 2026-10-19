"""Reasoning layer: explain, ask, and (optionally) let a fenced LLM choose among retrieved candidates."""
from .clarify import ClarifyingQuestions
from .explain import Explainer
from .llm_reasoner import LlmReasoner
from .protocols import NoReasoner, Reasoner
from .verifier import Verdict, Verifier

__all__ = ["ClarifyingQuestions", "Explainer", "LlmReasoner", "NoReasoner", "Reasoner", "Verdict", "Verifier"]
