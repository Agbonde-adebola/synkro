"""Test 12: Tests for MessagesFormatter save to JSONL functionality."""

import json
import tempfile
from pathlib import Path

from synkro.formatters.messages import MessagesFormatter
from synkro.types.core import GradeResult, Message, Scenario, Trace


def _make_trace(user: str = "User message", assistant: str = "Response") -> Trace:
    """Helper to create a trace for testing."""
    return Trace(
        messages=[
            Message(role="system", content="System"),
            Message(role="user", content=user),
            Message(role="assistant", content=assistant),
        ],
        scenario=Scenario(description="Test", context="Context", category="Cat"),
        grade=GradeResult(passed=True, issues=[], feedback=""),
    )


def test_save_creates_file():
    """save() creates a file at the specified path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.jsonl"
        formatter = MessagesFormatter()
        formatter.save([_make_trace()], path)
        assert path.exists()


def test_save_writes_valid_jsonl():
    """save() writes valid JSONL with one JSON object per line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.jsonl"
        formatter = MessagesFormatter()
        formatter.save([_make_trace(), _make_trace()], path)

        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line.strip())
            assert "messages" in parsed


def test_save_preserves_message_content():
    """save() preserves message content correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.jsonl"
        formatter = MessagesFormatter()
        formatter.save([_make_trace(user="Hello world", assistant="Hi there")], path)

        with open(path) as f:
            data = json.loads(f.readline())

        assert data["messages"][1]["content"] == "Hello world"
        assert data["messages"][2]["content"] == "Hi there"


def test_save_with_string_path():
    """save() accepts string path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path_str = str(Path(tmpdir) / "output.jsonl")
        formatter = MessagesFormatter()
        formatter.save([_make_trace()], path_str)
        assert Path(path_str).exists()


def test_save_pretty_print():
    """save() with pretty_print creates multi-line JSON entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.jsonl"
        formatter = MessagesFormatter()
        formatter.save([_make_trace()], path, pretty_print=True)

        content = path.read_text()
        # Pretty printed should have multiple lines for a single object
        # followed by blank line separator
        assert content.count("\n") > 1


def test_save_empty_traces():
    """save() with empty traces creates empty file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.jsonl"
        formatter = MessagesFormatter()
        formatter.save([], path)

        assert path.exists()
        assert path.read_text() == ""


def test_save_with_metadata():
    """save() with include_metadata includes metadata in output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.jsonl"
        formatter = MessagesFormatter(include_metadata=True)
        formatter.save([_make_trace()], path)

        with open(path) as f:
            data = json.loads(f.readline())

        assert "metadata" in data
        assert "category" in data["metadata"]


def test_save_overwrites_existing_file():
    """save() overwrites existing file content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.jsonl"
        formatter = MessagesFormatter()

        # Write initial content
        formatter.save([_make_trace(user="First")], path)

        # Overwrite with new content
        formatter.save([_make_trace(user="Second")], path)

        with open(path) as f:
            data = json.loads(f.readline())

        assert data["messages"][1]["content"] == "Second"
