"""Pipeline state types for tracking progress through generation stages.

These types enable real-time visibility into pipeline execution and support
both callback-based and query-based progress monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from synkro.types.metrics import Metrics

if TYPE_CHECKING:
    from synkro.types.core import Trace
    from synkro.types.logic_map import GoldenScenario, LogicMap


class PipelinePhase(str, Enum):
    """Phases of the generation pipeline."""

    IDLE = "idle"
    PLANNING = "planning"
    EXTRACTION = "extraction"
    SCENARIOS = "scenarios"
    COVERAGE = "coverage"
    HITL = "hitl"
    TRACES = "traces"
    VERIFICATION = "verification"
    COMPLETE = "complete"
    ERROR = "error"

    @property
    def display_name(self) -> str:
        """Human-readable name for the phase."""
        names = {
            "idle": "Idle",
            "planning": "Planning",
            "extraction": "Extracting Rules",
            "scenarios": "Generating Scenarios",
            "coverage": "Analyzing Coverage",
            "hitl": "Editing (HITL)",
            "traces": "Synthesizing Traces",
            "verification": "Verifying Traces",
            "complete": "Complete",
            "error": "Error",
        }
        return names.get(self.value, self.value.title())

    @property
    def weight(self) -> float:
        """Weight for overall progress calculation (0.0 to 1.0)."""
        weights = {
            "idle": 0.0,
            "planning": 0.05,
            "extraction": 0.15,
            "scenarios": 0.20,
            "coverage": 0.05,
            "hitl": 0.05,
            "traces": 0.35,
            "verification": 0.15,
            "complete": 1.0,
            "error": 0.0,
        }
        return weights.get(self.value, 0.0)


@dataclass
class PipelineState:
    """Current state of the generation pipeline.

    Tracks progress through pipeline phases, accumulated metrics,
    and intermediate artifacts. Designed for both push (callbacks)
    and pull (querying) based progress monitoring.

    Examples:
        >>> state = PipelineState()
        >>> state.transition_to(PipelinePhase.EXTRACTION)
        >>> state.update_progress(0.5, "Processing rule 3 of 6")
        >>> print(f"Overall: {state.total_progress:.0%}")
        >>> print(f"Phase: {state.current_phase.display_name}")
    """

    current_phase: PipelinePhase = PipelinePhase.IDLE
    phase_progress: float = 0.0  # 0.0 to 1.0 within current phase
    phase_message: str = ""
    metrics: Metrics = field(default_factory=Metrics)

    # Artifacts populated as phases complete
    logic_map: "LogicMap | None" = None
    scenarios: list["GoldenScenario"] | None = None
    traces: list["Trace"] | None = None

    # Error information
    error: Exception | None = None
    error_message: str = ""

    # History of phase transitions
    phase_history: list[tuple[PipelinePhase, float]] = field(default_factory=list)

    @property
    def total_progress(self) -> float:
        """Weighted overall progress (0.0 to 1.0).

        Calculates progress based on completed phases plus
        current phase progress weighted by phase importance.
        """
        if self.current_phase == PipelinePhase.COMPLETE:
            return 1.0
        if self.current_phase == PipelinePhase.ERROR:
            return self._completed_weight()

        # Sum completed phase weights
        completed = self._completed_weight()

        # Add weighted current phase progress
        current_weight = self.current_phase.weight
        prev_weight = self._previous_phase_cumulative_weight()

        # Current phase contribution
        phase_contribution = (current_weight - prev_weight) * self.phase_progress

        return min(completed + phase_contribution, 1.0)

    def _completed_weight(self) -> float:
        """Calculate cumulative weight of completed phases."""
        phases_order = [
            PipelinePhase.PLANNING,
            PipelinePhase.EXTRACTION,
            PipelinePhase.SCENARIOS,
            PipelinePhase.COVERAGE,
            PipelinePhase.HITL,
            PipelinePhase.TRACES,
            PipelinePhase.VERIFICATION,
        ]

        total = 0.0
        for phase in phases_order:
            if phase.value in [p[0].value for p in self.phase_history]:
                total = phase.weight
            if phase == self.current_phase:
                break

        return total

    def _previous_phase_cumulative_weight(self) -> float:
        """Get cumulative weight before current phase."""
        phases_order = [
            PipelinePhase.IDLE,
            PipelinePhase.PLANNING,
            PipelinePhase.EXTRACTION,
            PipelinePhase.SCENARIOS,
            PipelinePhase.COVERAGE,
            PipelinePhase.HITL,
            PipelinePhase.TRACES,
            PipelinePhase.VERIFICATION,
        ]

        prev_weight = 0.0
        for phase in phases_order:
            if phase == self.current_phase:
                break
            prev_weight = phase.weight

        return prev_weight

    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.current_phase not in (
            PipelinePhase.IDLE,
            PipelinePhase.COMPLETE,
            PipelinePhase.ERROR,
        )

    @property
    def is_complete(self) -> bool:
        """Check if pipeline has completed (successfully or with error)."""
        return self.current_phase in (PipelinePhase.COMPLETE, PipelinePhase.ERROR)

    @property
    def is_error(self) -> bool:
        """Check if pipeline ended with an error."""
        return self.current_phase == PipelinePhase.ERROR

    def transition_to(self, phase: PipelinePhase, message: str = "") -> None:
        """Transition to a new phase.

        Args:
            phase: The phase to transition to
            message: Optional message for the transition
        """
        # Record the transition
        import time

        self.phase_history.append((self.current_phase, time.time()))

        self.current_phase = phase
        self.phase_progress = 0.0
        self.phase_message = message or phase.display_name

        # Start metrics tracking for this phase
        if phase.value not in ("idle", "complete", "error"):
            self.metrics.start_phase(phase.value)

    def update_progress(self, progress: float, message: str = "") -> None:
        """Update progress within the current phase.

        Args:
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message
        """
        self.phase_progress = min(max(progress, 0.0), 1.0)
        if message:
            self.phase_message = message

    def complete_phase(self) -> None:
        """Mark the current phase as complete."""
        self.phase_progress = 1.0

        # End metrics tracking for this phase
        if self.current_phase.value in self.metrics.phases:
            phase_metrics = self.metrics.phases[self.current_phase.value]
            phase_metrics.complete()

    def set_error(self, error: Exception, message: str = "") -> None:
        """Set error state.

        Args:
            error: The exception that occurred
            message: Optional error message
        """
        self.error = error
        self.error_message = message or str(error)
        self.transition_to(PipelinePhase.ERROR, self.error_message)

    def set_artifact(
        self,
        logic_map: "LogicMap | None" = None,
        scenarios: list["GoldenScenario"] | None = None,
        traces: list["Trace"] | None = None,
    ) -> None:
        """Set pipeline artifacts.

        Args:
            logic_map: Extracted Logic Map
            scenarios: Generated scenarios
            traces: Generated traces
        """
        if logic_map is not None:
            self.logic_map = logic_map
        if scenarios is not None:
            self.scenarios = scenarios
        if traces is not None:
            self.traces = traces

    def format_status(self) -> str:
        """Format current status as a string.

        Returns:
            Status string like "[Extracting Rules] 50% - Processing rule 3..."
        """
        phase_name = self.current_phase.display_name
        progress_pct = int(self.phase_progress * 100)
        overall_pct = int(self.total_progress * 100)

        if self.phase_message:
            return (
                f"[{phase_name}] {progress_pct}% - {self.phase_message} (Overall: {overall_pct}%)"
            )
        return f"[{phase_name}] {progress_pct}% (Overall: {overall_pct}%)"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "current_phase": self.current_phase.value,
            "phase_progress": self.phase_progress,
            "phase_message": self.phase_message,
            "total_progress": self.total_progress,
            "is_running": self.is_running,
            "is_complete": self.is_complete,
            "is_error": self.is_error,
            "error_message": self.error_message,
            "metrics": self.metrics.to_dict(),
            "artifacts": {
                "has_logic_map": self.logic_map is not None,
                "scenario_count": len(self.scenarios) if self.scenarios else 0,
                "trace_count": len(self.traces) if self.traces else 0,
            },
        }


__all__ = ["PipelinePhase", "PipelineState"]
