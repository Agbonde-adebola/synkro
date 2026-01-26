"""Test 9: Tests for complexity derivation logic."""

import pytest
from pydantic import ValidationError

from synkro.schemas import PolicyComplexity


def test_simple_complexity_valid():
    """Simple complexity level is valid."""
    complexity = PolicyComplexity(
        variable_count=1,
        complexity_level="simple",
        recommended_turns=1,
        reasoning="Single rule policy",
    )
    assert complexity.complexity_level == "simple"
    assert complexity.variable_count == 1


def test_conditional_complexity_valid():
    """Conditional complexity level is valid."""
    complexity = PolicyComplexity(
        variable_count=3,
        complexity_level="conditional",
        recommended_turns=3,
        reasoning="Multiple conditions",
    )
    assert complexity.complexity_level == "conditional"
    assert complexity.variable_count == 3


def test_complex_complexity_valid():
    """Complex complexity level is valid."""
    complexity = PolicyComplexity(
        variable_count=5,
        complexity_level="complex",
        recommended_turns=5,
        reasoning="Many interrelated rules",
    )
    assert complexity.complexity_level == "complex"
    assert complexity.variable_count == 5


def test_invalid_complexity_level_rejected():
    """Invalid complexity level raises ValidationError."""
    with pytest.raises(ValidationError):
        PolicyComplexity(
            variable_count=2,
            complexity_level="medium",  # Invalid level
            recommended_turns=2,
            reasoning="Test",
        )


def test_recommended_turns_minimum():
    """Recommended turns must be at least 1."""
    with pytest.raises(ValidationError):
        PolicyComplexity(
            variable_count=1,
            complexity_level="simple",
            recommended_turns=0,  # Below minimum
            reasoning="Test",
        )


def test_recommended_turns_maximum():
    """Recommended turns cannot exceed 6."""
    with pytest.raises(ValidationError):
        PolicyComplexity(
            variable_count=5,
            complexity_level="complex",
            recommended_turns=7,  # Above maximum
            reasoning="Test",
        )


def test_recommended_turns_boundary_values():
    """Boundary values 1 and 6 are valid for recommended_turns."""
    min_turns = PolicyComplexity(
        variable_count=1,
        complexity_level="simple",
        recommended_turns=1,
        reasoning="Minimum turns",
    )
    assert min_turns.recommended_turns == 1

    max_turns = PolicyComplexity(
        variable_count=5,
        complexity_level="complex",
        recommended_turns=6,
        reasoning="Maximum turns",
    )
    assert max_turns.recommended_turns == 6


def test_complexity_serialization():
    """PolicyComplexity can be serialized to dict and back."""
    original = PolicyComplexity(
        variable_count=3,
        complexity_level="conditional",
        recommended_turns=3,
        reasoning="Test serialization",
    )
    data = original.model_dump()
    restored = PolicyComplexity.model_validate(data)
    assert restored.variable_count == original.variable_count
    assert restored.complexity_level == original.complexity_level
    assert restored.recommended_turns == original.recommended_turns
    assert restored.reasoning == original.reasoning


def test_complexity_levels_literal():
    """Only 'simple', 'conditional', 'complex' are valid complexity levels."""
    valid_levels = ["simple", "conditional", "complex"]
    for level in valid_levels:
        complexity = PolicyComplexity(
            variable_count=2,
            complexity_level=level,
            recommended_turns=2,
            reasoning=f"Testing {level}",
        )
        assert complexity.complexity_level == level
