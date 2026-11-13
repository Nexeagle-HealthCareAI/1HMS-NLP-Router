"""LLM adapter: providers behind the complete(system, user) port, with timeout and circuit breaker."""
from .factory import build_llm_completion, redact
from .guard import CircuitOpen, GuardedCompletion

__all__ = ["build_llm_completion", "redact", "CircuitOpen", "GuardedCompletion"]
