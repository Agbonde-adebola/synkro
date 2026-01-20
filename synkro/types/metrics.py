"""Metrics types for tracking costs, calls, and timing across pipeline phases.

These types provide per-phase breakdown of costs and enable tracking
throughout the generation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.llm.client import LLM


@dataclass
class PhaseMetrics:
    """Metrics for a single phase of the pipeline.

    Tracks cost, call count, timing, and model used for phases like:
    - extraction: Logic Map extraction
    - scenarios: Scenario generation
    - traces: Response generation
    - verification: Trace verification and refinement
    - hitl: Human-in-the-loop editing

    Examples:
        >>> metrics = PhaseMetrics(phase="extraction")
        >>> metrics.add_call(0.002, "gpt-4o")
        >>> print(f"Cost: ${metrics.cost:.4f}")
    """

    phase: str
    cost: float = 0.0
    calls: int = 0
    duration_seconds: float = 0.0
    model: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def add_call(self, cost: float = 0.0, model: str = "") -> None:
        """Record an LLM call with its cost."""
        self.calls += 1
        self.cost += cost
        if model and not self.model:
            self.model = model

    def start(self) -> None:
        """Mark the phase as started."""
        self.started_at = datetime.now()

    def complete(self) -> None:
        """Mark the phase as completed and calculate duration."""
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase,
            "cost": self.cost,
            "calls": self.calls,
            "duration_seconds": self.duration_seconds,
            "model": self.model,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhaseMetrics":
        """Create from dictionary."""
        return cls(
            phase=data["phase"],
            cost=data.get("cost", 0.0),
            calls=data.get("calls", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            model=data.get("model", ""),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
        )


@dataclass
class Metrics:
    """Unified metrics tracking with per-phase breakdown.

    Accumulates metrics across all pipeline phases and provides
    aggregated totals and formatted summaries.

    Examples:
        >>> metrics = Metrics()
        >>> metrics.start_phase("extraction", model="gpt-4o")
        >>> # ... do extraction ...
        >>> metrics.end_phase("extraction", cost=0.05, calls=3)
        >>> print(f"Total: ${metrics.total_cost:.2f}")
        >>> print(metrics.format_summary())
    """

    phases: dict[str, PhaseMetrics] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Total cost across all phases."""
        return sum(p.cost for p in self.phases.values())

    @property
    def total_calls(self) -> int:
        """Total LLM calls across all phases."""
        return sum(p.calls for p in self.phases.values())

    @property
    def total_duration(self) -> float:
        """Total duration in seconds across all phases."""
        return sum(p.duration_seconds for p in self.phases.values())

    @property
    def breakdown(self) -> dict[str, float]:
        """Cost breakdown by phase."""
        return {phase: m.cost for phase, m in self.phases.items()}

    @property
    def calls_breakdown(self) -> dict[str, int]:
        """Call count breakdown by phase."""
        return {phase: m.calls for phase, m in self.phases.items()}

    def start_phase(self, phase: str, model: str = "") -> PhaseMetrics:
        """Start tracking a new phase.

        Args:
            phase: Phase name (e.g., "extraction", "scenarios")
            model: Model being used for this phase

        Returns:
            PhaseMetrics object for the phase
        """
        if phase not in self.phases:
            self.phases[phase] = PhaseMetrics(phase=phase, model=model)
        self.phases[phase].start()
        if model:
            self.phases[phase].model = model
        return self.phases[phase]

    def end_phase(self, phase: str, cost: float = 0.0, calls: int = 0) -> None:
        """Complete a phase and record final metrics.

        Args:
            phase: Phase name
            cost: Total cost for this phase
            calls: Total calls made during this phase
        """
        if phase in self.phases:
            self.phases[phase].cost = cost
            self.phases[phase].calls = calls
            self.phases[phase].complete()

    def add_call(self, phase: str, cost: float = 0.0, model: str = "") -> None:
        """Record an LLM call for a specific phase.

        Args:
            phase: Phase name
            cost: Cost of this call
            model: Model used
        """
        if phase not in self.phases:
            self.phases[phase] = PhaseMetrics(phase=phase)
        self.phases[phase].add_call(cost, model)

    def get_phase(self, phase: str) -> PhaseMetrics | None:
        """Get metrics for a specific phase."""
        return self.phases.get(phase)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "phases": {name: p.to_dict() for name, p in self.phases.items()},
            "total_cost": self.total_cost,
            "total_calls": self.total_calls,
            "total_duration": self.total_duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Metrics":
        """Create from dictionary."""
        metrics = cls()
        for name, phase_data in data.get("phases", {}).items():
            metrics.phases[name] = PhaseMetrics.from_dict(phase_data)
        return metrics

    def format_summary(self) -> str:
        """Format a one-line summary.

        Returns:
            String like "Cost: $0.42 | Calls: 91 | Time: 2m 34s"
        """
        duration = self.total_duration
        if duration >= 60:
            time_str = f"{int(duration // 60)}m {int(duration % 60)}s"
        else:
            time_str = f"{duration:.1f}s"

        return f"Cost: ${self.total_cost:.4f} | Calls: {self.total_calls} | Time: {time_str}"

    def format_table(self) -> str:
        """Format a detailed breakdown table.

        Returns:
            Multi-line string with phase-by-phase breakdown
        """
        lines = ["Phase Breakdown:"]
        lines.append("-" * 50)
        lines.append(f"{'Phase':<15} {'Cost':>10} {'Calls':>8} {'Time':>10}")
        lines.append("-" * 50)

        for phase, m in self.phases.items():
            duration = m.duration_seconds
            if duration >= 60:
                time_str = f"{int(duration // 60)}m {int(duration % 60)}s"
            else:
                time_str = f"{duration:.1f}s"
            lines.append(f"{phase:<15} ${m.cost:>9.4f} {m.calls:>8} {time_str:>10}")

        lines.append("-" * 50)
        lines.append(f"{'Total':<15} ${self.total_cost:>9.4f} {self.total_calls:>8}")

        return "\n".join(lines)


class TrackedLLM:
    """Wrapper that tracks LLM calls to an external Metrics object.

    Use this to track costs across multiple LLM clients into a unified
    Metrics instance.

    Examples:
        >>> metrics = Metrics()
        >>> llm = LLM(model=OpenAI.GPT_4O)
        >>> tracked = TrackedLLM(llm, metrics, phase="extraction")
        >>> # All calls now track to metrics under "extraction" phase
        >>> result = await tracked.generate("Hello")
    """

    def __init__(self, llm: "LLM", metrics: Metrics, phase: str):
        """Initialize tracked wrapper.

        Args:
            llm: Underlying LLM client
            metrics: Metrics object to track to
            phase: Phase name for tracking
        """
        self._llm = llm
        self._metrics = metrics
        self._phase = phase
        self._initial_cost = llm.total_cost
        self._initial_calls = llm.call_count

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text and track metrics."""
        result = await self._llm.generate(prompt, system)
        self._track()
        return result

    async def generate_structured(self, prompt: str, response_model, system: str | None = None):
        """Generate structured response and track metrics."""
        result = await self._llm.generate_structured(prompt, response_model, system)
        self._track()
        return result

    async def generate_chat(self, messages: list[dict], response_model=None):
        """Generate chat response and track metrics."""
        result = await self._llm.generate_chat(messages, response_model)
        self._track()
        return result

    async def generate_batch(self, prompts: list[str], system: str | None = None) -> list[str]:
        """Generate batch responses and track metrics."""
        result = await self._llm.generate_batch(prompts, system)
        self._track()
        return result

    def _track(self) -> None:
        """Update metrics with delta from initial state."""
        cost_delta = self._llm.total_cost - self._initial_cost
        self._llm.call_count - self._initial_calls
        self._metrics.add_call(self._phase, cost_delta, self._llm.model)
        self._initial_cost = self._llm.total_cost
        self._initial_calls = self._llm.call_count

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._llm.model

    @property
    def total_cost(self) -> float:
        """Get total cost from underlying LLM."""
        return self._llm.total_cost

    @property
    def call_count(self) -> int:
        """Get call count from underlying LLM."""
        return self._llm.call_count


__all__ = ["PhaseMetrics", "Metrics", "TrackedLLM"]
