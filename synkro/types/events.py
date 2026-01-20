"""Event types for streaming APIs.

These types define the events yielded by streaming generators, enabling
real-time progress updates and incremental result delivery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from synkro.types.core import Trace
    from synkro.types.coverage import CoverageReport
    from synkro.types.logic_map import GoldenScenario, Rule
    from synkro.types.metrics import PhaseMetrics


# Event type literals
EventType = Literal[
    "progress",
    "rule_found",
    "scenario_generated",
    "trace_generated",
    "trace_verified",
    "trace_refined",
    "refinement_started",
    "coverage_calculated",
    "complete",
    "error",
]


@dataclass
class Event:
    """Base event for streaming APIs.

    All streaming events inherit from this base class, providing
    a consistent interface for event handling.

    Examples:
        >>> async for event in synkro.extract_rules_stream(policy):
        ...     match event.type:
        ...         case "progress":
        ...             print(f"Progress: {event.message}")
        ...         case "rule_found":
        ...             print(f"Found: {event.rule.rule_id}")
        ...         case "complete":
        ...             logic_map = event.result.logic_map
    """

    type: EventType
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "timestamp": self.timestamp,
        }


@dataclass
class ProgressEvent(Event):
    """Progress update during an operation.

    Emitted periodically during long-running operations to provide
    visibility into progress.

    Attributes:
        phase: Current phase name
        message: Human-readable progress message
        progress: Progress value (0.0 to 1.0)
        completed: Number of items completed
        total: Total number of items
    """

    type: Literal["progress"] = "progress"
    phase: str = ""
    message: str = ""
    progress: float = 0.0
    completed: int = 0
    total: int = 0

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "phase": self.phase,
                "message": self.message,
                "progress": self.progress,
                "completed": self.completed,
                "total": self.total,
            }
        )
        return data


@dataclass
class RuleFoundEvent(Event):
    """Emitted when a rule is extracted.

    Fired during the extraction phase as each rule is identified
    in the policy document.

    Attributes:
        rule: The extracted Rule object
        index: Index of this rule (0-based)
    """

    type: Literal["rule_found"] = "rule_found"
    rule: "Rule | None" = None
    index: int = 0

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "rule": self.rule.model_dump() if self.rule else None,
                "index": self.index,
            }
        )
        return data


@dataclass
class ScenarioGeneratedEvent(Event):
    """Emitted when a scenario is generated.

    Fired during the scenario generation phase as each scenario
    is created.

    Attributes:
        scenario: The generated GoldenScenario object
        index: Index of this scenario (0-based)
    """

    type: Literal["scenario_generated"] = "scenario_generated"
    scenario: "GoldenScenario | None" = None
    index: int = 0

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "scenario": self.scenario.model_dump() if self.scenario else None,
                "index": self.index,
            }
        )
        return data


@dataclass
class TraceGeneratedEvent(Event):
    """Emitted when a trace is synthesized.

    Fired during trace synthesis as each conversation trace
    is generated.

    Attributes:
        trace: The generated Trace object
        index: Index of this trace (0-based)
    """

    type: Literal["trace_generated"] = "trace_generated"
    trace: "Trace | None" = None
    index: int = 0

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "trace": self.trace.model_dump() if self.trace else None,
                "index": self.index,
            }
        )
        return data


@dataclass
class TraceVerifiedEvent(Event):
    """Emitted when a trace is verified.

    Fired during verification as each trace is checked against
    the Logic Map.

    Attributes:
        trace: The verified Trace object
        index: Index of this trace (0-based)
        passed: Whether the trace passed verification
        issues: List of issues found (if any)
    """

    type: Literal["trace_verified"] = "trace_verified"
    trace: "Trace | None" = None
    index: int = 0
    passed: bool = False
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "trace": self.trace.model_dump() if self.trace else None,
                "index": self.index,
                "passed": self.passed,
                "issues": self.issues,
            }
        )
        return data


@dataclass
class RefinementStartedEvent(Event):
    """Emitted when refinement begins for failed traces.

    Indicates that the pipeline is starting to refine traces
    that failed verification.

    Attributes:
        count: Number of traces to be refined
        iteration: Current refinement iteration (1-based)
        max_iterations: Maximum refinement iterations allowed
    """

    type: Literal["refinement_started"] = "refinement_started"
    count: int = 0
    iteration: int = 1
    max_iterations: int = 3

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "count": self.count,
                "iteration": self.iteration,
                "max_iterations": self.max_iterations,
            }
        )
        return data


@dataclass
class TraceRefinedEvent(Event):
    """Emitted when a trace is refined.

    Fired after a failed trace has been successfully refined
    and re-verified.

    Attributes:
        trace: The refined Trace object
        index: Index of this trace (0-based)
        original_issues: Issues that were fixed
        passed: Whether the refined trace now passes
    """

    type: Literal["trace_refined"] = "trace_refined"
    trace: "Trace | None" = None
    index: int = 0
    original_issues: list[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "trace": self.trace.model_dump() if self.trace else None,
                "index": self.index,
                "original_issues": self.original_issues,
                "passed": self.passed,
            }
        )
        return data


@dataclass
class CoverageCalculatedEvent(Event):
    """Emitted when coverage is calculated.

    Fired after scenarios are tagged and coverage metrics
    are computed.

    Attributes:
        coverage: The CoverageReport object
    """

    type: Literal["coverage_calculated"] = "coverage_calculated"
    coverage: "CoverageReport | None" = None

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "coverage": self.coverage.model_dump() if self.coverage else None,
            }
        )
        return data


@dataclass
class CompleteEvent(Event):
    """Emitted when an operation completes.

    Fired at the end of a streaming operation with the final
    result object.

    Attributes:
        result: The final result object (type depends on operation)
        metrics: Phase metrics for the completed operation
    """

    type: Literal["complete"] = "complete"
    result: Any = None
    metrics: "PhaseMetrics | None" = None

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "result": self.result.to_dict()
                if hasattr(self.result, "to_dict")
                else str(self.result),
                "metrics": self.metrics.to_dict() if self.metrics else None,
            }
        )
        return data


@dataclass
class ErrorEvent(Event):
    """Emitted when an error occurs.

    Fired when an operation fails with an exception.

    Attributes:
        error: The exception that occurred
        message: Human-readable error message
        phase: Phase where the error occurred
    """

    type: Literal["error"] = "error"
    error: Exception | None = None
    message: str = ""
    phase: str = ""

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "error": str(self.error) if self.error else None,
                "message": self.message,
                "phase": self.phase,
            }
        )
        return data


# Type alias for all event types
StreamEvent = (
    Event
    | ProgressEvent
    | RuleFoundEvent
    | ScenarioGeneratedEvent
    | TraceGeneratedEvent
    | TraceVerifiedEvent
    | RefinementStartedEvent
    | TraceRefinedEvent
    | CoverageCalculatedEvent
    | CompleteEvent
    | ErrorEvent
)


__all__ = [
    "EventType",
    "Event",
    "ProgressEvent",
    "RuleFoundEvent",
    "ScenarioGeneratedEvent",
    "TraceGeneratedEvent",
    "TraceVerifiedEvent",
    "RefinementStartedEvent",
    "TraceRefinedEvent",
    "CoverageCalculatedEvent",
    "CompleteEvent",
    "ErrorEvent",
    "StreamEvent",
]
