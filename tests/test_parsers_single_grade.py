"""Test 3: Tests for parse_single_grade() validation in synkro/parsers.py"""

from synkro.parsers import parse_single_grade


def test_parse_valid_grade_all_fields():
    """Parse valid grade with all fields."""
    response = '{"pass": true, "policy_violations": [], "missing_citations": [], "incomplete_reasoning": [], "vague_recommendations": [], "feedback": "Good response"}'
    result = parse_single_grade(response)
    assert result is not None
    assert result.passed is True
    assert result.feedback == "Good response"
    assert result.policy_violations == []


def test_parse_failed_grade_with_issues():
    """Parse failed grade with violation details."""
    response = '{"pass": false, "policy_violations": ["Violated rule R001"], "missing_citations": ["R002"], "feedback": "Needs improvement"}'
    result = parse_single_grade(response)
    assert result is not None
    assert result.passed is False
    assert "Violated rule R001" in result.policy_violations
    assert "R002" in result.missing_citations


def test_parse_grade_missing_optional_fields():
    """Parse grade with missing optional fields uses defaults."""
    response = '{"pass": true}'
    result = parse_single_grade(response)
    assert result is not None
    assert result.passed is True
    assert result.policy_violations == []
    assert result.feedback == ""


def test_parse_invalid_json_returns_none():
    """Return None for invalid JSON."""
    response = "Not valid JSON"
    result = parse_single_grade(response)
    assert result is None


def test_parse_empty_response_returns_none():
    """Return None for empty response."""
    response = ""
    result = parse_single_grade(response)
    assert result is None


def test_parse_grade_with_extra_fields_ignored():
    """Extra unexpected fields are ignored."""
    response = '{"pass": true, "feedback": "OK", "unexpected_field": "ignored", "another": 123}'
    result = parse_single_grade(response)
    assert result is not None
    assert result.passed is True
    assert result.feedback == "OK"
