"""Test 4: Tests for parse_policy_complexity() in synkro/parsers.py"""

from synkro.parsers import (
    DEFAULT_COMPLEXITY_LEVEL,
    DEFAULT_COMPLEXITY_VARIABLE_COUNT,
    DEFAULT_RECOMMENDED_TURNS,
    parse_policy_complexity,
)


def test_parse_valid_complexity():
    """Parse valid complexity response with all fields."""
    response = '{"variable_count": 5, "complexity_level": "complex", "recommended_turns": 4, "reasoning": "Many interrelated rules"}'
    result = parse_policy_complexity(response)
    assert result.variable_count == 5
    assert result.complexity_level == "complex"
    assert result.recommended_turns == 4
    assert result.reasoning == "Many interrelated rules"


def test_parse_simple_complexity():
    """Parse simple complexity level."""
    response = '{"variable_count": 1, "complexity_level": "simple", "recommended_turns": 1, "reasoning": "Single rule"}'
    result = parse_policy_complexity(response)
    assert result.complexity_level == "simple"
    assert result.recommended_turns == 1


def test_parse_missing_variable_count_uses_default():
    """Missing variable_count defaults to 2."""
    response = '{"complexity_level": "conditional", "recommended_turns": 3}'
    result = parse_policy_complexity(response)
    assert result.variable_count == DEFAULT_COMPLEXITY_VARIABLE_COUNT


def test_parse_missing_complexity_level_uses_default():
    """Missing complexity_level defaults to conditional."""
    response = '{"variable_count": 3, "recommended_turns": 2}'
    result = parse_policy_complexity(response)
    assert result.complexity_level == DEFAULT_COMPLEXITY_LEVEL


def test_parse_invalid_json_returns_defaults():
    """Invalid JSON returns all default values."""
    response = "This is not valid JSON"
    result = parse_policy_complexity(response)
    assert result.variable_count == DEFAULT_COMPLEXITY_VARIABLE_COUNT
    assert result.complexity_level == DEFAULT_COMPLEXITY_LEVEL
    assert result.recommended_turns == DEFAULT_RECOMMENDED_TURNS


def test_parse_empty_response_returns_defaults():
    """Empty response returns all default values."""
    response = ""
    result = parse_policy_complexity(response)
    assert result.variable_count == DEFAULT_COMPLEXITY_VARIABLE_COUNT
    assert result.complexity_level == DEFAULT_COMPLEXITY_LEVEL
