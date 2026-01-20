"""Session class for stateful multi-step dataset generation.

The Session class tracks state across operations, accumulates metrics,
supports persistence, and provides both sync and async interfaces.

Examples:
    >>> # Create and use a session
    >>> session = synkro.Session()
    >>> await session.extract_rules(policy)
    >>> await session.generate_scenarios(count=50)
    >>> await session.synthesize_traces()
    >>> await session.verify_traces()
    >>> dataset = session.to_dataset()

    >>> # Session persistence
    >>> session.save("session.json")
    >>> restored = synkro.Session.load("session.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from synkro.core.dataset import Dataset
from synkro.core.policy import Policy
from synkro.types.events import Event
from synkro.types.metrics import Metrics
from synkro.types.results import (
    ExtractionResult,
    ScenariosResult,
    TracesResult,
    VerificationResult,
)
from synkro.utils.model_detection import get_default_models

if TYPE_CHECKING:
    from synkro.types.core import Trace
    from synkro.types.coverage import CoverageReport
    from synkro.types.logic_map import GoldenScenario, LogicMap


@dataclass
class SessionSnapshot:
    """Snapshot of session state for undo/history."""

    logic_map_json: str | None = None
    scenarios_json: str | None = None
    traces_json: str | None = None
    description: str = ""


@dataclass
class Session:
    """
    Stateful session for multi-step dataset generation.

    Tracks artifacts, metrics, and history across operations.
    Designed for agentic workflows where state persists between calls.

    Attributes:
        policy: Current policy being processed
        logic_map: Extracted Logic Map
        scenarios: Generated scenarios
        traces: Generated traces
        verified_traces: Verified and refined traces
        coverage_report: Coverage analysis report
        metrics: Accumulated metrics across all operations

    Examples:
        >>> # Basic workflow
        >>> session = Session()
        >>> await session.extract_rules("Expenses over $50 need approval")
        >>> print(session.logic_map)  # Access extracted rules
        >>> await session.generate_scenarios(count=20)
        >>> await session.synthesize_traces()
        >>> await session.verify_traces()
        >>> dataset = session.to_dataset()

        >>> # Edit rules with natural language
        >>> await session.edit_rules("add rule: overtime needs director approval")
        >>> print(session.metrics)  # See accumulated costs

        >>> # Persistence
        >>> session.save("my_session.json")
        >>> restored = Session.load("my_session.json")
    """

    # Current state
    policy: Policy | None = None
    logic_map: "LogicMap | None" = None
    scenarios: list["GoldenScenario"] | None = None
    distribution: dict[str, int] | None = None
    traces: list["Trace"] | None = None
    verified_traces: list["Trace"] | None = None
    coverage_report: "CoverageReport | None" = None

    # Accumulated metrics
    metrics: Metrics = field(default_factory=Metrics)

    # History for undo
    history: list[SessionSnapshot] = field(default_factory=list)

    # Configuration (auto-detected if not specified)
    model: str | None = None
    grading_model: str | None = None
    base_url: str | None = None

    def __post_init__(self):
        """Auto-detect models if not specified."""
        if self.model is None or self.grading_model is None:
            try:
                gen_model, grade_model = get_default_models()
                if self.model is None:
                    self.model = gen_model
                if self.grading_model is None:
                    self.grading_model = grade_model
            except EnvironmentError:
                # No API keys found, will fail later when used
                pass

    async def extract_rules(
        self,
        policy: str | Policy,
        model: str | None = None,
    ) -> ExtractionResult:
        """
        Extract rules from a policy document.

        Stores the Logic Map in the session for subsequent operations.

        Args:
            policy: Policy text or Policy object
            model: Optional model override

        Returns:
            ExtractionResult with Logic Map and metrics
        """
        from synkro.api import extract_rules_async

        if isinstance(policy, str):
            policy = Policy(text=policy)

        self.policy = policy
        self._save_snapshot("Before extraction")

        result = await extract_rules_async(
            policy,
            model=model or self.grading_model,
            base_url=self.base_url,
        )

        self.logic_map = result.logic_map

        # Accumulate metrics
        if result.metrics:
            self.metrics.phases["extraction"] = result.metrics

        return result

    async def edit_rules(self, instruction: str) -> tuple["LogicMap", str]:
        """
        Edit rules using natural language.

        Uses the stored Logic Map and applies the instruction.

        Args:
            instruction: Natural language edit instruction

        Returns:
            Tuple of (updated LogicMap, summary string)

        Raises:
            ValueError: If no Logic Map is available
        """
        from synkro.interactive.standalone import edit_rules_async

        if self.logic_map is None:
            raise ValueError("No Logic Map available. Call extract_rules first.")
        if self.policy is None:
            raise ValueError("No policy available. Call extract_rules first.")

        self._save_snapshot(f"Before edit: {instruction[:30]}...")

        new_logic_map, summary = await edit_rules_async(
            self.logic_map,
            instruction,
            self.policy.text,
            model=self.grading_model,
            base_url=self.base_url,
        )

        self.logic_map = new_logic_map
        return new_logic_map, summary

    async def generate_scenarios(
        self,
        count: int = 20,
        model: str | None = None,
    ) -> ScenariosResult:
        """
        Generate scenarios from the stored Logic Map.

        Args:
            count: Number of scenarios to generate
            model: Optional model override

        Returns:
            ScenariosResult with scenarios and distribution

        Raises:
            ValueError: If no Logic Map is available
        """
        from synkro.api import generate_scenarios_async

        if self.logic_map is None:
            raise ValueError("No Logic Map available. Call extract_rules first.")
        if self.policy is None:
            raise ValueError("No policy available. Call extract_rules first.")

        self._save_snapshot("Before scenario generation")

        result = await generate_scenarios_async(
            self.policy,
            logic_map=self.logic_map,
            count=count,
            model=model or self.model,
            base_url=self.base_url,
        )

        self.scenarios = result.scenarios
        self.distribution = result.distribution
        self.coverage_report = result.coverage_report

        # Accumulate metrics
        if result.metrics:
            self.metrics.phases["scenarios"] = result.metrics

        return result

    async def edit_scenarios(
        self, instruction: str
    ) -> tuple[list["GoldenScenario"], dict[str, int], str]:
        """
        Edit scenarios using natural language.

        Args:
            instruction: Natural language edit instruction

        Returns:
            Tuple of (updated scenarios, distribution, summary)

        Raises:
            ValueError: If no scenarios are available
        """
        from synkro.interactive.standalone import edit_scenarios_async

        if self.scenarios is None:
            raise ValueError("No scenarios available. Call generate_scenarios first.")
        if self.logic_map is None:
            raise ValueError("No Logic Map available.")
        if self.policy is None:
            raise ValueError("No policy available.")

        self._save_snapshot(f"Before edit: {instruction[:30]}...")

        new_scenarios, new_dist, summary = await edit_scenarios_async(
            self.scenarios,
            instruction,
            self.policy.text,
            self.logic_map,
            distribution=self.distribution,
            model=self.grading_model,
            base_url=self.base_url,
        )

        self.scenarios = new_scenarios
        self.distribution = new_dist
        return new_scenarios, new_dist, summary

    async def synthesize_traces(
        self,
        turns: int = 1,
        model: str | None = None,
    ) -> TracesResult:
        """
        Synthesize traces from the stored scenarios.

        Args:
            turns: Number of conversation turns
            model: Optional model override

        Returns:
            TracesResult with generated traces

        Raises:
            ValueError: If no scenarios are available
        """
        from synkro.api import synthesize_traces_async

        if self.scenarios is None:
            raise ValueError("No scenarios available. Call generate_scenarios first.")
        if self.logic_map is None:
            raise ValueError("No Logic Map available.")
        if self.policy is None:
            raise ValueError("No policy available.")

        self._save_snapshot("Before trace synthesis")

        result = await synthesize_traces_async(
            self.policy,
            scenarios=self.scenarios,
            logic_map=self.logic_map,
            turns=turns,
            model=model or self.model,
            base_url=self.base_url,
        )

        self.traces = result.traces

        # Accumulate metrics
        if result.metrics:
            self.metrics.phases["traces"] = result.metrics

        return result

    async def verify_traces(
        self,
        max_iterations: int = 3,
        model: str | None = None,
    ) -> VerificationResult:
        """
        Verify and refine traces.

        Args:
            max_iterations: Maximum refinement attempts
            model: Optional model override

        Returns:
            VerificationResult with verified traces and pass rate

        Raises:
            ValueError: If no traces are available
        """
        from synkro.api import verify_traces_async

        if self.traces is None:
            raise ValueError("No traces available. Call synthesize_traces first.")
        if self.logic_map is None:
            raise ValueError("No Logic Map available.")
        if self.scenarios is None:
            raise ValueError("No scenarios available.")
        if self.policy is None:
            raise ValueError("No policy available.")

        self._save_snapshot("Before verification")

        result = await verify_traces_async(
            self.policy,
            traces=self.traces,
            logic_map=self.logic_map,
            scenarios=self.scenarios,
            max_iterations=max_iterations,
            model=model or self.grading_model,
            base_url=self.base_url,
        )

        self.verified_traces = result.verified_traces

        # Accumulate metrics
        if result.metrics:
            self.metrics.phases["verification"] = result.metrics

        return result

    # =========================================================================
    # STREAMING METHODS
    # =========================================================================

    async def extract_rules_stream(
        self,
        policy: str | Policy,
        model: str | None = None,
    ) -> AsyncIterator[Event]:
        """
        Extract rules with streaming events.

        Yields events during extraction and updates session state on completion.
        """
        from synkro.api import extract_rules_stream

        if isinstance(policy, str):
            policy = Policy(text=policy)

        self.policy = policy
        self._save_snapshot("Before extraction")

        async for event in extract_rules_stream(
            policy,
            model=model or self.grading_model,
            base_url=self.base_url,
        ):
            yield event
            if event.type == "complete":
                self.logic_map = event.result.logic_map
                if event.metrics:
                    self.metrics.phases["extraction"] = event.metrics

    async def generate_scenarios_stream(
        self,
        count: int = 20,
        model: str | None = None,
    ) -> AsyncIterator[Event]:
        """
        Generate scenarios with streaming events.
        """
        from synkro.api import generate_scenarios_stream

        if self.logic_map is None:
            raise ValueError("No Logic Map available. Call extract_rules first.")
        if self.policy is None:
            raise ValueError("No policy available.")

        self._save_snapshot("Before scenario generation")

        async for event in generate_scenarios_stream(
            self.policy,
            logic_map=self.logic_map,
            count=count,
            model=model or self.model,
            base_url=self.base_url,
        ):
            yield event
            if event.type == "complete":
                self.scenarios = event.result.scenarios
                self.distribution = event.result.distribution
                if event.metrics:
                    self.metrics.phases["scenarios"] = event.metrics

    async def synthesize_traces_stream(
        self,
        turns: int = 1,
        model: str | None = None,
    ) -> AsyncIterator[Event]:
        """
        Synthesize traces with streaming events.
        """
        from synkro.api import synthesize_traces_stream

        if self.scenarios is None:
            raise ValueError("No scenarios available.")
        if self.logic_map is None:
            raise ValueError("No Logic Map available.")
        if self.policy is None:
            raise ValueError("No policy available.")

        self._save_snapshot("Before trace synthesis")

        async for event in synthesize_traces_stream(
            self.policy,
            scenarios=self.scenarios,
            logic_map=self.logic_map,
            turns=turns,
            model=model or self.model,
            base_url=self.base_url,
        ):
            yield event
            if event.type == "complete":
                self.traces = event.result.traces
                if event.metrics:
                    self.metrics.phases["traces"] = event.metrics

    async def verify_traces_stream(
        self,
        max_iterations: int = 3,
        model: str | None = None,
    ) -> AsyncIterator[Event]:
        """
        Verify traces with streaming events.
        """
        from synkro.api import verify_traces_stream

        if self.traces is None:
            raise ValueError("No traces available.")
        if self.logic_map is None:
            raise ValueError("No Logic Map available.")
        if self.scenarios is None:
            raise ValueError("No scenarios available.")
        if self.policy is None:
            raise ValueError("No policy available.")

        self._save_snapshot("Before verification")

        async for event in verify_traces_stream(
            self.policy,
            traces=self.traces,
            logic_map=self.logic_map,
            scenarios=self.scenarios,
            max_iterations=max_iterations,
            model=model or self.grading_model,
            base_url=self.base_url,
        ):
            yield event
            if event.type == "complete":
                self.verified_traces = event.result.verified_traces
                if event.metrics:
                    self.metrics.phases["verification"] = event.metrics

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _save_snapshot(self, description: str) -> None:
        """Save current state to history."""
        snapshot = SessionSnapshot(
            logic_map_json=self.logic_map.model_dump_json() if self.logic_map else None,
            scenarios_json=json.dumps([s.model_dump() for s in self.scenarios])
            if self.scenarios
            else None,
            traces_json=json.dumps([t.model_dump() for t in self.traces]) if self.traces else None,
            description=description,
        )
        self.history.append(snapshot)

        # Keep only last 10 snapshots
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def undo(self) -> bool:
        """
        Restore the previous state.

        Returns:
            True if undo was successful, False if no history available
        """
        if len(self.history) < 2:
            return False

        # Remove current state
        self.history.pop()

        # Restore previous state
        snapshot = self.history[-1]
        self._restore_snapshot(snapshot)
        return True

    def _restore_snapshot(self, snapshot: SessionSnapshot) -> None:
        """Restore state from a snapshot."""
        from synkro.types.core import Trace
        from synkro.types.logic_map import GoldenScenario, LogicMap

        if snapshot.logic_map_json:
            self.logic_map = LogicMap.model_validate_json(snapshot.logic_map_json)
        if snapshot.scenarios_json:
            data = json.loads(snapshot.scenarios_json)
            self.scenarios = [GoldenScenario.model_validate(s) for s in data]
        if snapshot.traces_json:
            data = json.loads(snapshot.traces_json)
            self.traces = [Trace.model_validate(t) for t in data]

    def reset(self) -> None:
        """Reset session to initial state."""
        self.policy = None
        self.logic_map = None
        self.scenarios = None
        self.distribution = None
        self.traces = None
        self.verified_traces = None
        self.coverage_report = None
        self.metrics = Metrics()
        self.history = []

    def to_dataset(self) -> Dataset:
        """
        Convert session state to a Dataset.

        Uses verified traces if available, otherwise uses traces.

        Returns:
            Dataset containing the traces

        Raises:
            ValueError: If no traces are available
        """
        traces = self.verified_traces or self.traces
        if traces is None:
            raise ValueError("No traces available. Run the pipeline first.")

        return Dataset(traces=traces)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """
        Save session state to a JSON file.

        Args:
            path: File path for saving
        """
        path = Path(path)

        data = {
            "policy": self.policy.text if self.policy else None,
            "logic_map": self.logic_map.model_dump() if self.logic_map else None,
            "scenarios": [s.model_dump() for s in self.scenarios] if self.scenarios else None,
            "distribution": self.distribution,
            "traces": [t.model_dump() for t in self.traces] if self.traces else None,
            "verified_traces": [t.model_dump() for t in self.verified_traces]
            if self.verified_traces
            else None,
            "metrics": self.metrics.to_dict(),
            "model": self.model,
            "grading_model": self.grading_model,
            "base_url": self.base_url,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        """
        Load session state from a JSON file.

        Args:
            path: File path to load from

        Returns:
            Restored Session instance
        """
        from synkro.types.core import Trace
        from synkro.types.logic_map import GoldenScenario, LogicMap

        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        session = cls()

        if data.get("policy"):
            session.policy = Policy(text=data["policy"])

        if data.get("logic_map"):
            session.logic_map = LogicMap.model_validate(data["logic_map"])

        if data.get("scenarios"):
            session.scenarios = [GoldenScenario.model_validate(s) for s in data["scenarios"]]

        if data.get("distribution"):
            session.distribution = data["distribution"]

        if data.get("traces"):
            session.traces = [Trace.model_validate(t) for t in data["traces"]]

        if data.get("verified_traces"):
            session.verified_traces = [Trace.model_validate(t) for t in data["verified_traces"]]

        if data.get("metrics"):
            session.metrics = Metrics.from_dict(data["metrics"])

        # Load saved model preferences, or auto-detect
        session.model = data.get("model")
        session.grading_model = data.get("grading_model")
        session.base_url = data.get("base_url")

        # Auto-detect if not specified in saved data
        if session.model is None or session.grading_model is None:
            try:
                gen_model, grade_model = get_default_models()
                if session.model is None:
                    session.model = gen_model
                if session.grading_model is None:
                    session.grading_model = grade_model
            except EnvironmentError:
                pass

        return session

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def format_status(self) -> str:
        """
        Format current session status.

        Returns:
            Multi-line status string
        """
        lines = ["Session Status:"]
        lines.append("=" * 40)

        # Policy
        if self.policy:
            lines.append(f"Policy: {self.policy.text[:50]}...")
        else:
            lines.append("Policy: Not set")

        # Logic Map
        if self.logic_map:
            lines.append(f"Logic Map: {len(self.logic_map.rules)} rules")
        else:
            lines.append("Logic Map: Not extracted")

        # Scenarios
        if self.scenarios:
            lines.append(f"Scenarios: {len(self.scenarios)}")
            if self.distribution:
                dist_str = ", ".join(f"{k}: {v}" for k, v in self.distribution.items())
                lines.append(f"  Distribution: {dist_str}")
        else:
            lines.append("Scenarios: Not generated")

        # Traces
        if self.traces:
            lines.append(f"Traces: {len(self.traces)}")
        else:
            lines.append("Traces: Not synthesized")

        # Verified
        if self.verified_traces:
            passed = sum(1 for t in self.verified_traces if t.grade and t.grade.passed)
            lines.append(f"Verified: {passed}/{len(self.verified_traces)} passed")
        else:
            lines.append("Verified: Not verified")

        # Metrics
        lines.append("-" * 40)
        lines.append(self.metrics.format_summary())

        return "\n".join(lines)

    def to_agent_context(self) -> str:
        """
        Format session state for LLM reasoning context.

        Returns:
            Compact representation suitable for LLM context
        """
        parts = []

        parts.append("Session State:")

        if self.policy:
            parts.append(f"- Policy: {self.policy.text[:100]}...")

        if self.logic_map:
            parts.append(f"- Rules: {len(self.logic_map.rules)}")

        if self.scenarios:
            parts.append(f"- Scenarios: {len(self.scenarios)}")

        if self.traces:
            parts.append(f"- Traces: {len(self.traces)}")

        if self.verified_traces:
            passed = sum(1 for t in self.verified_traces if t.grade and t.grade.passed)
            parts.append(f"- Verified: {passed}/{len(self.verified_traces)} passed")

        parts.append(f"- Cost: ${self.metrics.total_cost:.4f}")

        return "\n".join(parts)


__all__ = ["Session", "SessionSnapshot"]
