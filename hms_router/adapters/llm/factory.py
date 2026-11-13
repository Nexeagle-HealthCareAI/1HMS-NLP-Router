"""Build the LLM port (`complete(system, user) -> str`) from settings, via LangChain chat models.

OFF by default. Turning it on sends patient text to an external provider: record the consent and
data-processing decision first (see the design doc's open decisions). Names, phone numbers and emails
are redacted before sending.

    HMS_LLM_PROVIDER=groq       pip install langchain-groq       (GROQ_API_KEY)
    HMS_LLM_PROVIDER=anthropic  pip install langchain-anthropic  (ANTHROPIC_API_KEY)
    HMS_LLM_PROVIDER=openai     pip install langchain-openai     (OPENAI_API_KEY; also OpenAI-compatible hosts
                                                                  via OPENAI_BASE_URL, e.g. a self-hosted model)
"""
from __future__ import annotations

import re
from typing import Callable

from ..settings import ServiceSettings
from .guard import GuardedCompletion

_PHONE = re.compile(r"(?<!\d)(\+?\d[\d\s-]{8,}\d)(?!\d)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_NAME_INTRO = re.compile(r"\b(my name is|mera naam|naam hai|i am|main)\s+[A-Z][a-z]+(\s[A-Z][a-z]+)?", re.I)


def redact(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _NAME_INTRO.sub(lambda m: m.group(1) + " [name]", text)


def _chat_model(provider: str, model: str | None, timeout: float):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "openai/gpt-oss-20b", temperature=0, timeout=timeout, max_retries=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not model:
            raise RuntimeError("HMS_LLM_MODEL is required for the anthropic provider")
        return ChatAnthropic(model=model, temperature=0, timeout=timeout, max_retries=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not model:
            raise RuntimeError("HMS_LLM_MODEL is required for the openai provider")
        return ChatOpenAI(model=model, temperature=0, timeout=timeout, max_retries=0)
    raise RuntimeError(f"unknown HMS_LLM_PROVIDER '{provider}'")


def build_llm_completion(settings: ServiceSettings) -> Callable[[str, str], str] | None:
    if not settings.llm_provider:
        return None
    from langchain_core.messages import HumanMessage, SystemMessage
    model = _chat_model(settings.llm_provider, settings.llm_model, settings.llm_timeout)

    def complete(system: str, user: str) -> str:
        reply = model.invoke([SystemMessage(content=system), HumanMessage(content=redact(user))])
        return reply.content if isinstance(reply.content, str) else str(reply.content)

    return GuardedCompletion(complete, timeout=settings.llm_timeout)
