"""Test 10: Tests for error handling and friendly messages."""

from synkro.errors import (
    APIKeyError,
    FileNotFoundError,
    ModelNotFoundError,
    PolicyTooShortError,
    RateLimitError,
    SynkroError,
)


def test_synkro_error_base_class():
    """SynkroError stores message and suggestion."""
    error = SynkroError("Test error", "Try this fix")
    assert error.message == "Test error"
    assert error.suggestion == "Try this fix"
    assert str(error) == "Test error"


def test_synkro_error_without_suggestion():
    """SynkroError works without suggestion."""
    error = SynkroError("Just an error")
    assert error.message == "Just an error"
    assert error.suggestion == ""


def test_api_key_error_openai():
    """APIKeyError for OpenAI includes correct env var."""
    error = APIKeyError("openai")
    assert "OPENAI_API_KEY" in error.suggestion
    assert "platform.openai.com" in error.suggestion


def test_api_key_error_anthropic():
    """APIKeyError for Anthropic includes correct env var."""
    error = APIKeyError("anthropic")
    assert "ANTHROPIC_API_KEY" in error.suggestion
    assert "console.anthropic.com" in error.suggestion


def test_api_key_error_google():
    """APIKeyError for Google includes correct env var."""
    error = APIKeyError("google")
    assert "GEMINI_API_KEY" in error.suggestion
    assert "aistudio.google.com" in error.suggestion


def test_api_key_error_case_insensitive():
    """APIKeyError provider matching is case insensitive."""
    error_lower = APIKeyError("openai")
    error_upper = APIKeyError("OpenAI")
    assert "OPENAI_API_KEY" in error_lower.suggestion
    assert "OPENAI_API_KEY" in error_upper.suggestion


def test_file_not_found_error_basic():
    """FileNotFoundError includes file path in message."""
    error = FileNotFoundError("/path/to/file.txt", None)
    assert "/path/to/file.txt" in error.message
    assert "Could not find" in error.message


def test_file_not_found_error_with_suggestions():
    """FileNotFoundError shows similar files if provided."""
    error = FileNotFoundError("/path/to/polcy.txt", ["policy.txt", "policies.txt"])
    assert "Did you mean" in error.suggestion
    assert "policy.txt" in error.suggestion


def test_rate_limit_error_basic():
    """RateLimitError includes provider in message."""
    error = RateLimitError("OpenAI")
    assert "OpenAI" in error.message
    assert "Rate limited" in error.message


def test_rate_limit_error_with_retry():
    """RateLimitError shows retry time if provided."""
    error = RateLimitError("Google", retry_after=30)
    assert "30s" in error.message or "30" in error.message


def test_rate_limit_error_suggestion():
    """RateLimitError includes helpful suggestion."""
    error = RateLimitError("OpenAI")
    assert "traces" in error.suggestion or "provider" in error.suggestion


def test_policy_too_short_error():
    """PolicyTooShortError includes word count in message."""
    error = PolicyTooShortError(word_count=5)
    assert "5 words" in error.message
    assert "too short" in error.message.lower()


def test_policy_too_short_error_suggestion():
    """PolicyTooShortError includes example in suggestion."""
    error = PolicyTooShortError(word_count=3)
    assert "50+" in error.suggestion or "Minimum" in error.suggestion


def test_model_not_found_error():
    """ModelNotFoundError includes model name in message."""
    error = ModelNotFoundError("gpt-5-turbo")
    assert "gpt-5-turbo" in error.message
    assert "not found" in error.message.lower()


def test_model_not_found_error_suggestion():
    """ModelNotFoundError lists available models."""
    error = ModelNotFoundError("invalid-model")
    assert "OpenAI" in error.suggestion or "GPT" in error.suggestion
    assert "Anthropic" in error.suggestion or "Claude" in error.suggestion
