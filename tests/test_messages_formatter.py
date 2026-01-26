"""Test 11: Tests for MessagesFormatter output structure."""

import json

from synkro.formatters.messages import MessagesFormatter
from synkro.types.core import GradeResult, Message, Scenario, Trace


def _make_trace(
    system: str = "System prompt",
    user: str = "User message",
    assistant: str = "Assistant response",
    category: str = "Test",
    passed: bool = True,
) -> Trace:
    """Helper to create a trace for testing."""
    return Trace(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=user),
            Message(role="assistant", content=assistant),
        ],
        scenario=Scenario(description="Test scenario", context="Test context", category=category),
        grade=GradeResult(passed=passed, issues=[], feedback=""),
    )


def test_format_returns_list_of_dicts():
    """format() returns a list of dictionaries."""
    formatter = MessagesFormatter()
    traces = [_make_trace()]
    result = formatter.format(traces)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], dict)


def test_format_has_messages_key():
    """Each formatted example has 'messages' key."""
    formatter = MessagesFormatter()
    traces = [_make_trace()]
    result = formatter.format(traces)
    assert "messages" in result[0]


def test_format_messages_structure():
    """Messages have correct role and content structure."""
    formatter = MessagesFormatter()
    traces = [_make_trace(system="Sys", user="Usr", assistant="Asst")]
    result = formatter.format(traces)
    messages = result[0]["messages"]

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Sys"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Usr"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Asst"


def test_format_without_metadata():
    """Without include_metadata, no metadata key is present."""
    formatter = MessagesFormatter(include_metadata=False)
    traces = [_make_trace()]
    result = formatter.format(traces)
    assert "metadata" not in result[0]


def test_format_with_metadata():
    """With include_metadata=True, metadata is included."""
    formatter = MessagesFormatter(include_metadata=True)
    traces = [_make_trace(category="TestCategory")]
    result = formatter.format(traces)

    assert "metadata" in result[0]
    assert result[0]["metadata"]["category"] == "TestCategory"
    assert "scenario" in result[0]["metadata"]
    assert "grade" in result[0]["metadata"]


def test_format_multiple_traces():
    """format() handles multiple traces correctly."""
    formatter = MessagesFormatter()
    traces = [_make_trace(user=f"User {i}") for i in range(3)]
    result = formatter.format(traces)

    assert len(result) == 3
    assert result[0]["messages"][1]["content"] == "User 0"
    assert result[1]["messages"][1]["content"] == "User 1"
    assert result[2]["messages"][1]["content"] == "User 2"


def test_format_empty_traces():
    """format() returns empty list for empty input."""
    formatter = MessagesFormatter()
    result = formatter.format([])
    assert result == []


def test_to_jsonl_format():
    """to_jsonl() returns valid JSONL string."""
    formatter = MessagesFormatter()
    traces = [_make_trace(), _make_trace()]
    jsonl = formatter.to_jsonl(traces)

    lines = jsonl.strip().split("\n")
    assert len(lines) == 2

    for line in lines:
        parsed = json.loads(line)
        assert "messages" in parsed


def test_to_jsonl_pretty_print():
    """to_jsonl with pretty_print adds indentation."""
    formatter = MessagesFormatter()
    traces = [_make_trace()]
    jsonl = formatter.to_jsonl(traces, pretty_print=True)

    # Pretty printed JSON should contain newlines within the object
    assert "\n" in jsonl
    # Should still be valid JSON
    parsed = json.loads(jsonl)
    assert "messages" in parsed
