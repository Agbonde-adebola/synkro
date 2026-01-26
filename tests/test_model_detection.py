"""Test 17: Tests for LLM client model detection."""

import os
from unittest.mock import patch

import pytest

from synkro.utils.model_detection import (
    detect_available_provider,
    get_default_model,
    get_default_models,
    get_provider_info,
)


def test_detect_provider_anthropic():
    """detect_available_provider returns anthropic when key is set."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
        assert detect_available_provider() == "anthropic"


def test_detect_provider_openai():
    """detect_available_provider returns openai when key is set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        assert detect_available_provider() == "openai"


def test_detect_provider_google_api_key():
    """detect_available_provider returns google when GOOGLE_API_KEY is set."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
        assert detect_available_provider() == "google"


def test_detect_provider_gemini_api_key():
    """detect_available_provider returns google when GEMINI_API_KEY is set."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
        assert detect_available_provider() == "google"


def test_detect_provider_priority():
    """detect_available_provider prioritizes anthropic over openai."""
    env = {"ANTHROPIC_API_KEY": "anthropic-key", "OPENAI_API_KEY": "openai-key"}
    with patch.dict(os.environ, env, clear=True):
        assert detect_available_provider() == "anthropic"


def test_detect_provider_none():
    """detect_available_provider returns None when no key is set."""
    with patch.dict(os.environ, {}, clear=True):
        assert detect_available_provider() is None


def test_get_default_models_anthropic():
    """get_default_models returns anthropic models when anthropic key is set."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
        gen, grade = get_default_models()
        assert "claude" in str(gen).lower() or "anthropic" in str(gen).lower()


def test_get_default_models_openai():
    """get_default_models returns openai models when openai key is set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        gen, grade = get_default_models()
        assert "gpt" in str(gen).lower() or "openai" in str(gen).lower()


def test_get_default_models_google():
    """get_default_models returns google models when google key is set."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
        gen, grade = get_default_models()
        assert "gemini" in str(gen).lower() or "google" in str(gen).lower()


def test_get_default_models_no_key_raises():
    """get_default_models raises EnvironmentError when no key is set."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            get_default_models()
        assert "API key" in str(exc_info.value)


def test_get_default_model():
    """get_default_model returns generation model."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        model = get_default_model()
        assert model is not None


def test_get_provider_info_structure():
    """get_provider_info returns expected structure."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        info = get_provider_info()
        assert "anthropic" in info
        assert "openai" in info
        assert "google" in info
        assert "active_provider" in info


def test_get_provider_info_available_flag():
    """get_provider_info marks provider as available when key is set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        info = get_provider_info()
        assert info["openai"]["available"] is True
        assert info["anthropic"]["available"] is False


def test_get_provider_info_active_provider():
    """get_provider_info includes active_provider field."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
        info = get_provider_info()
        assert info["active_provider"] == "google"
