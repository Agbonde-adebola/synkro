"""Test 7: Tests for Dataset filtering combinations."""

from synkro.core.dataset import Dataset
from synkro.types.core import GradeResult, Message, Scenario, Trace


def _make_trace(
    user_msg: str = "Test user message",
    assistant_msg: str = "Test assistant response",
    category: str | None = None,
    passed: bool = True,
) -> Trace:
    """Helper to create a trace for testing."""
    return Trace(
        messages=[
            Message(role="system", content="System prompt"),
            Message(role="user", content=user_msg),
            Message(role="assistant", content=assistant_msg),
        ],
        scenario=Scenario(description="Test scenario", context="Test context", category=category),
        grade=GradeResult(passed=passed, issues=[] if passed else ["Failed"]),
    )


def test_filter_by_passed_true():
    """Filter returns only passing traces."""
    dataset = Dataset(
        traces=[
            _make_trace(passed=True),
            _make_trace(passed=False),
            _make_trace(passed=True),
        ]
    )
    filtered = dataset.filter(passed=True)
    assert len(filtered) == 2
    assert all(t.grade.passed for t in filtered)


def test_filter_by_passed_false():
    """Filter returns only failing traces."""
    dataset = Dataset(
        traces=[
            _make_trace(passed=True),
            _make_trace(passed=False),
            _make_trace(passed=False),
        ]
    )
    filtered = dataset.filter(passed=False)
    assert len(filtered) == 2
    assert all(not t.grade.passed for t in filtered)


def test_filter_by_category():
    """Filter returns only traces from specified category."""
    dataset = Dataset(
        traces=[
            _make_trace(category="Happy Path"),
            _make_trace(category="Edge Cases"),
            _make_trace(category="Happy Path"),
        ]
    )
    filtered = dataset.filter(category="Happy Path")
    assert len(filtered) == 2
    assert all(t.scenario.category == "Happy Path" for t in filtered)


def test_filter_by_min_length():
    """Filter returns traces with assistant message at least min_length."""
    dataset = Dataset(
        traces=[
            _make_trace(assistant_msg="short"),
            _make_trace(assistant_msg="This is a much longer response message"),
            _make_trace(assistant_msg="medium length"),
        ]
    )
    filtered = dataset.filter(min_length=20)
    assert len(filtered) == 1
    assert len(filtered[0].assistant_message) >= 20


def test_filter_combined_passed_and_category():
    """Filter combines multiple criteria with AND logic."""
    dataset = Dataset(
        traces=[
            _make_trace(category="Happy Path", passed=True),
            _make_trace(category="Happy Path", passed=False),
            _make_trace(category="Edge Cases", passed=True),
        ]
    )
    filtered = dataset.filter(passed=True, category="Happy Path")
    assert len(filtered) == 1
    assert filtered[0].grade.passed
    assert filtered[0].scenario.category == "Happy Path"


def test_filter_combined_all_criteria():
    """Filter with all three criteria applied."""
    dataset = Dataset(
        traces=[
            _make_trace(category="Cat1", passed=True, assistant_msg="Long enough response"),
            _make_trace(category="Cat1", passed=True, assistant_msg="Short"),
            _make_trace(category="Cat1", passed=False, assistant_msg="Long enough response"),
            _make_trace(category="Cat2", passed=True, assistant_msg="Long enough response"),
        ]
    )
    filtered = dataset.filter(passed=True, category="Cat1", min_length=10)
    assert len(filtered) == 1


def test_filter_no_matches_returns_empty():
    """Filter with no matches returns empty dataset."""
    dataset = Dataset(
        traces=[
            _make_trace(category="Cat1"),
            _make_trace(category="Cat2"),
        ]
    )
    filtered = dataset.filter(category="Nonexistent")
    assert len(filtered) == 0


def test_filter_empty_dataset():
    """Filter on empty dataset returns empty dataset."""
    dataset = Dataset(traces=[])
    filtered = dataset.filter(passed=True)
    assert len(filtered) == 0


def test_filter_returns_new_dataset():
    """Filter returns a new Dataset instance, not modified original."""
    traces = [_make_trace(passed=True), _make_trace(passed=False)]
    dataset = Dataset(traces=traces)
    filtered = dataset.filter(passed=True)
    assert len(dataset) == 2
    assert len(filtered) == 1
    assert dataset is not filtered
