"""Test 5: Tests for parse_policy_plan() with distribution in synkro/parsers.py"""

from synkro.parsers import DEFAULT_PLAN_CATEGORIES, parse_policy_plan


def test_parse_valid_plan_with_categories():
    """Parse valid plan with multiple categories."""
    response = '{"categories": [{"name": "Happy Path", "description": "Success cases", "traces": 5}, {"name": "Edge Cases", "description": "Boundary conditions", "traces": 3}], "reasoning": "Good coverage"}'
    result = parse_policy_plan(response, target_traces=10)
    assert len(result.categories) == 2
    assert result.categories[0].name == "Happy Path"
    assert result.categories[0].traces == 5
    assert result.categories[1].name == "Edge Cases"
    assert result.reasoning == "Good coverage"


def test_parse_plan_missing_description():
    """Parse plan with missing category descriptions uses default description."""
    # Need at least 2 categories for PolicyPlan validation (min_length=2)
    response = '{"categories": [{"name": "TestCat", "traces": 3}, {"name": "Other", "traces": 2}], "reasoning": "Test"}'
    result = parse_policy_plan(response, target_traces=5)
    assert len(result.categories) == 2
    assert result.categories[0].name == "TestCat"
    assert result.categories[0].description == "General scenarios"
    assert result.categories[1].name == "Other"


def test_parse_plan_missing_traces_uses_default():
    """Parse plan with missing traces count uses target_traces/3."""
    response = '{"categories": [{"name": "Cat1", "description": "Desc"}], "reasoning": "Test"}'
    result = parse_policy_plan(response, target_traces=9)
    assert result.categories[0].traces == 3  # 9 // 3


def test_parse_invalid_json_returns_default_plan():
    """Invalid JSON returns default 3-category plan."""
    response = "Not valid JSON"
    result = parse_policy_plan(response, target_traces=9)
    assert len(result.categories) == 3
    assert result.categories[0].name == DEFAULT_PLAN_CATEGORIES[0]
    assert result.categories[1].name == DEFAULT_PLAN_CATEGORIES[1]
    assert result.categories[2].name == DEFAULT_PLAN_CATEGORIES[2]


def test_parse_empty_categories_returns_default():
    """Empty categories array returns default plan."""
    response = '{"categories": [], "reasoning": "Empty"}'
    result = parse_policy_plan(response, target_traces=6)
    assert len(result.categories) == 3  # Default plan


def test_parse_plan_distributes_traces_correctly():
    """Default plan distributes traces evenly with remainder to last category."""
    response = "invalid"
    result = parse_policy_plan(response, target_traces=10)
    # 10 // 3 = 3, remainder = 1
    # First two get 3 each, last gets 3 + 1 = 4
    assert result.categories[0].traces == 3
    assert result.categories[1].traces == 3
    assert result.categories[2].traces == 4
