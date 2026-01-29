"""Utility for auto-detecting available LLM providers and selecting default models.

This module checks for available API keys and returns appropriate default models
based on what's configured in the environment.
"""

import os
from typing import Tuple


def detect_available_provider() -> str | None:
    """
    Detect which LLM provider has an API key configured.

    Checks in order of preference:
    1. Anthropic (ANTHROPIC_API_KEY)
    2. OpenAI (OPENAI_API_KEY)
    3. Google (GOOGLE_API_KEY or GEMINI_API_KEY)

    Returns:
        Provider name ("anthropic", "openai", "google") or None if none found
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "google"
    if os.getenv("CEREBRAS_API_KEY"):
        return "cerebras"
    return None


def get_default_models() -> Tuple[str, str]:
    """
    Get default generation and grading models based on available API keys.

    Returns:
        Tuple of (generation_model, grading_model)

    Raises:
        EnvironmentError: If no API key is found
    """
    from synkro.models import Anthropic, Cerebras, Google, OpenAI

    provider = detect_available_provider()

    if provider == "anthropic":
        return (Anthropic.CLAUDE_35_HAIKU, Anthropic.CLAUDE_35_SONNET)
    elif provider == "openai":
        return (OpenAI.GPT_4O_MINI, OpenAI.GPT_4O)
    elif provider == "google":
        return (Google.GEMINI_2_FLASH, Google.GEMINI_2_FLASH)
    elif provider == "cerebras":
        return (Cerebras.LLAMA_33_70B, Cerebras.GPT_OSS_120B)
    else:
        raise EnvironmentError(
            "No LLM API key found. Please set one of:\n"
            "  - ANTHROPIC_API_KEY (for Claude)\n"
            "  - OPENAI_API_KEY (for GPT)\n"
            "  - GOOGLE_API_KEY or GEMINI_API_KEY (for Gemini)\n"
            "  - CEREBRAS_API_KEY (for Cerebras)"
        )


def get_default_model() -> str:
    """Get default model for general use."""
    gen_model, _ = get_default_models()
    return gen_model


def get_default_grading_model() -> str:
    """Get default model for grading/verification."""
    _, grade_model = get_default_models()
    return grade_model


def get_provider_info() -> dict:
    """
    Get information about available providers.

    Returns:
        Dict with provider status and recommended models
    """
    from synkro.models import Anthropic, Cerebras, Google, OpenAI

    info = {
        "anthropic": {
            "available": bool(os.getenv("ANTHROPIC_API_KEY")),
            "env_var": "ANTHROPIC_API_KEY",
            "generation_model": Anthropic.CLAUDE_35_HAIKU,
            "grading_model": Anthropic.CLAUDE_35_SONNET,
        },
        "openai": {
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "env_var": "OPENAI_API_KEY",
            "generation_model": OpenAI.GPT_4O_MINI,
            "grading_model": OpenAI.GPT_4O,
        },
        "google": {
            "available": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "env_var": "GOOGLE_API_KEY or GEMINI_API_KEY",
            "generation_model": Google.GEMINI_2_FLASH,
            "grading_model": Google.GEMINI_2_FLASH,
        },
        "cerebras": {
            "available": bool(os.getenv("CEREBRAS_API_KEY")),
            "env_var": "CEREBRAS_API_KEY",
            "generation_model": Cerebras.LLAMA_33_70B,
            "grading_model": Cerebras.GPT_OSS_120B,
        },
    }

    info["active_provider"] = detect_available_provider()
    return info


__all__ = [
    "detect_available_provider",
    "get_default_models",
    "get_default_model",
    "get_default_grading_model",
    "get_provider_info",
]
