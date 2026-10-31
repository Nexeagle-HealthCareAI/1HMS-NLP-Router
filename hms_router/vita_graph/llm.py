"""The tool-calling chat model behind Vita (Groq by default).

    VITA_LLM_PROVIDER=groq        GROQ_API_KEY   (pip install langchain-groq)
    VITA_LLM_MODEL                default openai/gpt-oss-120b (tool calling, fast on Groq)
    VITA_LLM_TIMEOUT              seconds per model call (default 6)
Unset provider = no LLM: Vita runs on its rules engine exactly as before.
Patient text is redacted (phone numbers, emails, "my name is ...") before it leaves the server.
"""
from __future__ import annotations

import logging

log = logging.getLogger("hms_router.vita_llm")

DEFAULT_MODEL = "openai/gpt-oss-120b"


def build_vita_llm(provider: str | None, model: str | None = None, timeout: float = 6.0):
    """A LangChain chat model with .bind_tools, or None when Vita should run without an LLM."""
    if not provider:
        return None
    provider = provider.strip().lower()
    try:
        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model or DEFAULT_MODEL, temperature=0.2, timeout=timeout, max_retries=1)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=0.2, timeout=timeout, max_retries=1)
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=0.2, timeout=timeout, max_retries=1)
    except Exception:
        log.exception("Vita LLM (%s) could not be built; Vita runs on its rules engine", provider)
        return None
    log.error("unknown VITA_LLM_PROVIDER %r; Vita runs on its rules engine", provider)
    return None
