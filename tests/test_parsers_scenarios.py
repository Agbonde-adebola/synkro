"""Test 2: Tests for parse_scenarios() graceful fallback in synkro/parsers.py"""

from synkro.parsers import parse_scenarios


def test_parse_valid_json_array():
    """Parse valid JSON array of scenarios."""
    response = '[{"scenario": "Test case 1", "context": "Background 1"}, {"scenario": "Test case 2", "context": "Background 2"}]'
    result = parse_scenarios(response, expected_count=2)
    assert len(result) == 2
    assert result[0].scenario == "Test case 1"
    assert result[0].context == "Background 1"
    assert result[1].scenario == "Test case 2"


def test_parse_with_alternate_field_names():
    """Parse scenarios with alternate field names (description/background)."""
    response = '[{"description": "Alternate name", "background": "Alternate context"}]'
    result = parse_scenarios(response, expected_count=1)
    assert len(result) == 1
    assert result[0].scenario == "Alternate name"
    assert result[0].context == "Alternate context"


def test_parse_limits_to_expected_count():
    """Parse only up to expected_count scenarios."""
    response = '[{"scenario": "1", "context": "c1"}, {"scenario": "2", "context": "c2"}, {"scenario": "3", "context": "c3"}]'
    result = parse_scenarios(response, expected_count=2)
    assert len(result) == 2
    assert result[0].scenario == "1"
    assert result[1].scenario == "2"


def test_parse_invalid_json_returns_fallback():
    """Return fallback scenarios when JSON is invalid."""
    response = "This is not valid JSON at all"
    result = parse_scenarios(response, expected_count=3)
    assert len(result) == 3
    # Fallback scenarios have default prefix
    assert "scenario" in result[0].scenario.lower() or "policy" in result[0].scenario.lower()


def test_parse_empty_response_returns_fallback():
    """Return fallback scenarios for empty response."""
    response = ""
    result = parse_scenarios(response, expected_count=2)
    assert len(result) == 2


def test_parse_with_missing_fields():
    """Handle scenarios with missing fields gracefully."""
    response = '[{"scenario": "Only scenario, no context"}]'
    result = parse_scenarios(response, expected_count=1)
    assert len(result) == 1
    assert result[0].scenario == "Only scenario, no context"
    assert result[0].context == ""  # Empty string for missing field
