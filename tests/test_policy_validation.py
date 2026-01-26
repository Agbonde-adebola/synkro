"""Test 6: Tests for Policy validation and minimum word requirements."""

import pytest

from synkro.core.policy import MIN_POLICY_WORDS, Policy
from synkro.errors import PolicyTooShortError


def test_policy_word_count_calculated_correctly():
    """Word count is calculated from whitespace-separated tokens."""
    policy = Policy(text="This is a five word policy")
    assert policy.word_count == 6


def test_policy_char_count_calculated_correctly():
    """Character count includes all characters."""
    policy = Policy(text="Hello")
    assert policy.char_count == 5


def test_policy_validates_minimum_words():
    """Policy with sufficient words passes validation."""
    long_text = " ".join(["word"] * (MIN_POLICY_WORDS + 5))
    policy = Policy(text=long_text)
    policy.validate_length()  # Should not raise


def test_policy_too_short_raises_error():
    """Policy below minimum words raises PolicyTooShortError."""
    short_text = "Too short"
    policy = Policy(text=short_text)
    with pytest.raises(PolicyTooShortError) as exc_info:
        policy.validate_length()
    assert str(policy.word_count) in exc_info.value.message


def test_policy_too_short_error_has_suggestion():
    """PolicyTooShortError includes helpful suggestion."""
    error = PolicyTooShortError(word_count=5)
    assert "5 words" in error.message
    assert "50+" in error.suggestion or "Minimum" in error.suggestion


def test_policy_str_representation():
    """String representation shows source and word count."""
    policy = Policy(text="A simple policy document here", source="test.txt")
    assert "test.txt" in str(policy)
    assert "words=" in str(policy)


def test_policy_inline_source():
    """Policy without source shows 'inline' in representation."""
    policy = Policy(text="Inline policy text here")
    assert "inline" in str(policy)


def test_min_policy_words_constant():
    """MIN_POLICY_WORDS constant is set to reasonable value."""
    assert MIN_POLICY_WORDS >= 5
    assert MIN_POLICY_WORDS <= 100
