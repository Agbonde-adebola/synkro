"""Utility for auto-detecting available LLM providers and selecting default models.

This module checks for available API keys and returns appropriate default models
based on what's configured in the environment. Automatically loads .env files
from common locations.
"""

import os
from pathlib import Path
from typing import Tuple

# Track if we've already tried loading env files
_env_loaded = False


def _load_env_files() -> None:
    """
    Auto-load .env files from common locations.

    Checks (in order):
    1. Current working directory (.env)
    2. Parent directories up to 3 levels
    3. User's home directory (~/.env)

    Uses python-dotenv if available, otherwise manually parses.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True

    # Possible .env locations
    locations = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path.cwd().parent.parent / ".env",
        Path.home() / ".env",
    ]

    # Also check for examples/.env pattern (common in test runs)
    examples_env = Path.cwd() / "examples" / ".env"
    if examples_env.exists():
        locations.insert(0, examples_env)

    # Try to use python-dotenv if available
    try:
        from dotenv import load_dotenv

        for loc in locations:
            if loc.exists():
                load_dotenv(loc, override=False)
                break
    except ImportError:
        # Manual fallback if dotenv not installed
        for loc in locations:
            if loc.exists():
                _parse_env_file(loc)
                break


def _parse_env_file(path: Path) -> None:
    """Manually parse a .env file and set environment variables."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key and key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass  # Silently ignore parse errors


def detect_available_provider() -> str | None:
    """
    Detect which LLM provider has an API key configured.

    Automatically loads .env files from common locations before checking.

    Checks in order of preference:
    1. Anthropic (ANTHROPIC_API_KEY)
    2. OpenAI (OPENAI_API_KEY)
    3. Google (GOOGLE_API_KEY or GEMINI_API_KEY)
    4. Cerebras (CEREBRAS_API_KEY)

    Returns:
        Provider name ("anthropic", "openai", "google", "cerebras") or None if none found
    """
    # Auto-load .env files on first call
    _load_env_files()

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

    Automatically loads .env files and detects the first available provider.
    If no provider is found, returns OpenAI models as fallback (will fail
    gracefully at API call time with a clear error).

    Returns:
        Tuple of (generation_model, grading_model)
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
        return (Cerebras.GPT_OSS_120B, Cerebras.GPT_OSS_120B)
    else:
        # No API key found - return OpenAI as fallback
        # This will fail gracefully at API call time with a clear error from LiteLLM
        return (OpenAI.GPT_4O_MINI, OpenAI.GPT_4O)


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

    Automatically loads .env files before checking.

    Returns:
        Dict with provider status and recommended models
    """
    from synkro.models import Anthropic, Cerebras, Google, OpenAI

    # Auto-load .env files
    _load_env_files()

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
            "generation_model": Cerebras.GPT_OSS_120B,
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
