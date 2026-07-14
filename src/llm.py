"""LLM factory: a single, environment-configured ChatOpenAI instance.

Supports any OpenAI-compatible API (model, key, base URL, and
temperature all come from environment variables).
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return the shared ChatOpenAI instance, creating it on first use."""
    global _llm
    if _llm is None:
        settings = get_settings()
        kwargs: dict = {
            "model": settings.model_name,
            "temperature": settings.temperature,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_api_base:
            kwargs["base_url"] = settings.openai_api_base
        _llm = ChatOpenAI(**kwargs)
        logger.info(
            "LLM initialized: model=%s temperature=%s base=%s",
            settings.model_name,
            settings.temperature,
            settings.openai_api_base or "default",
        )
    return _llm
