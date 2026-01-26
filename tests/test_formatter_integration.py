"""Test 14: Tests for multiple formatter types integration."""

import json

from synkro.formatters import (
    ChatMLFormatter,
    LangfuseFormatter,
    LangSmithFormatter,
    MessagesFormatter,
    QAFormatter,
)
from synkro.types.core import GradeResult, Message, Scenario, Trace


def _make_trace() -> Trace:
    """Helper to create a trace for testing."""
    return Trace(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="2+2 equals 4."),
        ],
        scenario=Scenario(description="Math question", context="Basic arithmetic", category="Math"),
        grade=GradeResult(passed=True, issues=[], feedback="Correct"),
    )


def test_messages_formatter_output():
    """MessagesFormatter produces messages format."""
    formatter = MessagesFormatter()
    result = formatter.format([_make_trace()])
    assert len(result) == 1
    assert "messages" in result[0]
    assert len(result[0]["messages"]) == 3


def test_qa_formatter_output():
    """QAFormatter produces question/answer format."""
    formatter = QAFormatter()
    result = formatter.format([_make_trace()])
    assert len(result) == 1
    # QA format should have question and answer or similar fields
    example = result[0]
    # Check for common QA format keys
    assert any(k in example for k in ["question", "input", "prompt", "user"])


def test_chatml_formatter_output():
    """ChatMLFormatter produces ChatML format."""
    formatter = ChatMLFormatter()
    result = formatter.format([_make_trace()])
    assert len(result) == 1
    example = result[0]
    # ChatML format typically has text or prompt field
    assert any(k in example for k in ["text", "prompt", "messages"])


def test_langsmith_formatter_output():
    """LangSmithFormatter produces LangSmith compatible format."""
    formatter = LangSmithFormatter()
    result = formatter.format([_make_trace()])
    assert len(result) == 1
    example = result[0]
    # LangSmith format has inputs/outputs structure
    assert "inputs" in example or "input" in example or "messages" in example


def test_langfuse_formatter_output():
    """LangfuseFormatter produces Langfuse compatible format."""
    formatter = LangfuseFormatter()
    result = formatter.format([_make_trace()])
    assert len(result) == 1
    example = result[0]
    # Langfuse format structure check
    assert isinstance(example, dict)


def test_all_formatters_produce_valid_json():
    """All formatters produce JSON-serializable output."""
    trace = _make_trace()
    formatters = [
        MessagesFormatter(),
        QAFormatter(),
        ChatMLFormatter(),
        LangSmithFormatter(),
        LangfuseFormatter(),
    ]

    for formatter in formatters:
        result = formatter.format([trace])
        # Should be JSON serializable without error
        json_str = json.dumps(result)
        assert len(json_str) > 0


def test_all_formatters_handle_empty_input():
    """All formatters handle empty trace list."""
    formatters = [
        MessagesFormatter(),
        QAFormatter(),
        ChatMLFormatter(),
        LangSmithFormatter(),
        LangfuseFormatter(),
    ]

    for formatter in formatters:
        result = formatter.format([])
        assert result == []


def test_all_formatters_have_to_jsonl():
    """All formatters have to_jsonl method."""
    trace = _make_trace()
    formatters = [
        MessagesFormatter(),
        QAFormatter(),
        ChatMLFormatter(),
        LangSmithFormatter(),
        LangfuseFormatter(),
    ]

    for formatter in formatters:
        jsonl = formatter.to_jsonl([trace])
        assert isinstance(jsonl, str)
        assert len(jsonl) > 0


def test_formatters_preserve_content():
    """Formatters preserve the original message content somewhere in output."""
    trace = _make_trace()
    formatters = [
        MessagesFormatter(),
        QAFormatter(),
        ChatMLFormatter(),
    ]

    for formatter in formatters:
        jsonl = formatter.to_jsonl([trace])
        # The user message content should appear somewhere
        assert "2+2" in jsonl or "What is" in jsonl
