"""Test 1: Tests for extract_json() edge cases in synkro/parsers.py"""

import json

from synkro.parsers import extract_json


def test_extract_simple_array():
    """Extract a simple JSON array from text."""
    content = "Here is the data: [1, 2, 3] and more text"
    result = extract_json(content, "[")
    assert result == "[1, 2, 3]"
    assert json.loads(result) == [1, 2, 3]


def test_extract_nested_objects():
    """Extract nested JSON objects."""
    content = 'Data: {"outer": {"inner": {"deep": "value"}}}'
    result = extract_json(content, "{")
    assert result is not None
    parsed = json.loads(result)
    assert parsed["outer"]["inner"]["deep"] == "value"


def test_extract_with_escaped_quotes():
    """Extract JSON with escaped quotes in strings."""
    content = r'Data: {"text": "He said \"hello\""}'
    result = extract_json(content, "{")
    assert result is not None
    parsed = json.loads(result)
    assert parsed["text"] == 'He said "hello"'


def test_extract_no_json_returns_none():
    """Return None when no JSON is found."""
    content = "Just plain text without any JSON"
    result = extract_json(content, "[")
    assert result is None


def test_extract_with_brackets_in_strings():
    """Handle brackets inside string values correctly."""
    content = '{"text": "array [1,2] and object {a:b}"}'
    result = extract_json(content, "{")
    assert result is not None
    parsed = json.loads(result)
    assert "[1,2]" in parsed["text"]
