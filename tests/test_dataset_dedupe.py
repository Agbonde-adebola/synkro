"""Test 13: Tests for Dataset deduplication logic."""

import pytest

from synkro.core.dataset import Dataset
from synkro.types.core import GradeResult, Message, Scenario, Trace


def _make_trace(user: str = "User msg", assistant: str = "Response") -> Trace:
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


def test_dedupe_exact_removes_user_duplicates():
    """Exact dedupe removes traces with duplicate user messages."""
    dataset = Dataset(
        traces=[
            _make_trace(user="Same question", assistant="Response 1"),
            _make_trace(user="Same question", assistant="Response 2"),
            _make_trace(user="Different question", assistant="Response 3"),
        ]
    )
    deduped = dataset.dedupe(method="exact", field="user")
    assert len(deduped) == 2


def test_dedupe_exact_removes_assistant_duplicates():
    """Exact dedupe on assistant field removes duplicate responses."""
    dataset = Dataset(
        traces=[
            _make_trace(user="Question 1", assistant="Same response"),
            _make_trace(user="Question 2", assistant="Same response"),
            _make_trace(user="Question 3", assistant="Different response"),
        ]
    )
    deduped = dataset.dedupe(method="exact", field="assistant")
    assert len(deduped) == 2


def test_dedupe_exact_both_fields():
    """Exact dedupe on both fields requires both to match."""
    dataset = Dataset(
        traces=[
            _make_trace(user="Q1", assistant="A1"),
            _make_trace(user="Q1", assistant="A1"),  # Exact duplicate
            _make_trace(user="Q1", assistant="A2"),  # Different assistant
            _make_trace(user="Q2", assistant="A1"),  # Different user
        ]
    )
    deduped = dataset.dedupe(method="exact", field="both")
    assert len(deduped) == 3


def test_dedupe_keeps_first_occurrence():
    """Dedupe keeps the first occurrence of duplicates."""
    traces = [
        _make_trace(user="Same", assistant="First"),
        _make_trace(user="Same", assistant="Second"),
    ]
    dataset = Dataset(traces=traces)
    deduped = dataset.dedupe(method="exact", field="user")

    assert len(deduped) == 1
    assert deduped[0].assistant_message == "First"


def test_dedupe_empty_dataset():
    """Dedupe on empty dataset returns empty dataset."""
    dataset = Dataset(traces=[])
    deduped = dataset.dedupe(method="exact")
    assert len(deduped) == 0


def test_dedupe_no_duplicates():
    """Dedupe with no duplicates returns all traces."""
    dataset = Dataset(
        traces=[
            _make_trace(user="Q1", assistant="A1"),
            _make_trace(user="Q2", assistant="A2"),
            _make_trace(user="Q3", assistant="A3"),
        ]
    )
    deduped = dataset.dedupe(method="exact", field="user")
    assert len(deduped) == 3


def test_dedupe_invalid_method_raises():
    """Invalid dedupe method raises ValueError."""
    dataset = Dataset(traces=[_make_trace()])
    with pytest.raises(ValueError) as exc_info:
        dataset.dedupe(method="invalid")
    assert "invalid" in str(exc_info.value).lower()


def test_dedupe_returns_new_dataset():
    """Dedupe returns a new Dataset instance."""
    original = Dataset(traces=[_make_trace(), _make_trace()])
    deduped = original.dedupe(method="exact")
    assert original is not deduped


def test_dedupe_default_field_is_user():
    """Default field for dedupe is 'user'."""
    dataset = Dataset(
        traces=[
            _make_trace(user="Same", assistant="A1"),
            _make_trace(user="Same", assistant="A2"),
        ]
    )
    # Default field should be 'user'
    deduped = dataset.dedupe(method="exact")
    assert len(deduped) == 1
