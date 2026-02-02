"""Types for SQL verification.

Follows LangSmith/Langfuse conventions for evaluation results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Verification verdict (categorical value)."""

    PASS = "pass"
    VIOLATION = "violation"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class Result:
    """Result from a single verifier.

    Follows LangSmith/Langfuse evaluation result conventions:
    - name: Evaluator identifier (LangSmith: key, Langfuse: name)
    - score: Numeric score 0-1 (confidence)
    - value: Categorical result (pass/violation/skipped/error)
    - comment: Explanation of the result
    - metadata: Additional context (params, ground_truth, etc.)
    """

    name: str
    score: float
    value: Verdict
    comment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if value is PASS."""
        return self.value == Verdict.PASS


@dataclass
class Report:
    """Aggregated report from all verifiers."""

    trace_id: str
    response: str
    results: list[Result] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no violations."""
        return all(r.value != Verdict.VIOLATION for r in self.results)

    @property
    def violations(self) -> list[Result]:
        """List of violation results."""
        return [r for r in self.results if r.value == Verdict.VIOLATION]

    @property
    def passed_count(self) -> int:
        """Number of passed verifiers."""
        return sum(1 for r in self.results if r.value == Verdict.PASS)

    @property
    def skipped_count(self) -> int:
        """Number of skipped verifiers."""
        return sum(1 for r in self.results if r.value == Verdict.SKIPPED)
