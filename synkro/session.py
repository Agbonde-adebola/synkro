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
    from synkro.types.coverage import CoverageReport, SubCategoryTaxonomy
    from synkro.types.logic_map import GoldenScenario, LogicMap
    from synkro.types.tool import ToolDefinition


def _fix_trace_grade(trace_dict: dict) -> dict:
    """Fix invalid grade data before Pydantic validation.

    Some traces have dynamically-created grade objects that serialize
    to empty dicts. This ensures they're set to None for proper validation.
    """
    if "grade" in trace_dict:
        grade = trace_dict["grade"]
        # Empty dict or dict without required 'passed' field -> None
        if not grade or (isinstance(grade, dict) and "passed" not in grade):
            trace_dict["grade"] = None
    return trace_dict


def _serialize_trace(trace) -> dict:
    """Serialize a trace, handling dynamic grade objects.

    Dynamic grades (created with type()) don't serialize properly via model_dump().
    This extracts the grade attributes manually if needed.
    """
    data = trace.model_dump()

    # Fix dynamic grade objects that serialize to empty dicts
    if trace.grade is not None:
        # Check if grade has passed attribute (dynamic or real)
        if hasattr(trace.grade, "passed"):
            data["grade"] = {
                "passed": trace.grade.passed,
                "issues": getattr(trace.grade, "issues", []),
                "feedback": getattr(trace.grade, "feedback", ""),
            }
        elif isinstance(data.get("grade"), dict) and "passed" not in data["grade"]:
            data["grade"] = None

    return data


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
    dataset_type: str = "conversation"  # conversation, instruction, evaluation, tool_call
    tools: list["ToolDefinition"] | None = None  # Required for TOOL_CALL dataset type
    skip_grading: bool = False  # Skip verification phase for faster generation
    concurrency: int = 50  # Max concurrent LLM calls for I/O parallelism

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
        await self._persist()

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
        from synkro.llm.client import LLM

        if self.logic_map is None:
            raise ValueError("No Logic Map available. Call extract_rules first.")
        if self.policy is None:
            raise ValueError("No policy available. Call extract_rules first.")

        self._save_snapshot(f"Before edit: {instruction[:30]}...")

        # Create LLM and track costs
        llm = LLM(model=self.grading_model, base_url=self.base_url, temperature=0.3)
        initial_cost = llm.total_cost

        new_logic_map, summary = await edit_rules_async(
            self.logic_map,
            instruction,
            self.policy.text,
            model=self.grading_model,
            base_url=self.base_url,
            llm=llm,
        )

        # Track HITL costs
        self.metrics.add_call("hitl", llm.total_cost - initial_cost, str(self.grading_model))

        self.logic_map = new_logic_map
        await self._persist()
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
        await self._persist()

        # Accumulate metrics
        if result.metrics:
            self.metrics.phases["scenarios"] = result.metrics

        return result

    async def generate_taxonomy(
        self,
        model: str | None = None,
    ) -> "SubCategoryTaxonomy":
        """
        Generate a sub-category taxonomy for coverage tracking.

        Extracts a hierarchical taxonomy of testable sub-categories from
        the policy, enabling fine-grained coverage analysis.

        Args:
            model: Optional model override (uses grading_model by default)

        Returns:
            SubCategoryTaxonomy with extracted sub-categories

        Raises:
            ValueError: If no Logic Map is available

        Examples:
            >>> await session.extract_rules(policy)
            >>> await session.generate_taxonomy()
            >>> print(f"Extracted {len(session.taxonomy.sub_categories)} sub-categories")
        """
        from synkro.coverage.taxonomy_extractor import TaxonomyExtractor
        from synkro.generation.planner import Planner
        from synkro.llm.client import LLM

        if self.logic_map is None:
            raise ValueError("No Logic Map available. Call extract_rules first.")
        if self.policy is None:
            raise ValueError("No policy available.")

        # Create planner to get categories
        planner_llm = LLM(
            model=model or self.grading_model,
            base_url=self.base_url,
            temperature=0.3,
        )
        planner = Planner(llm=planner_llm)
        plan = await planner.plan(self.policy, target_traces=20, analyze_turns=False)

        # Extract taxonomy
        extractor = TaxonomyExtractor(
            llm=LLM(model=model or self.grading_model, base_url=self.base_url, temperature=0.3)
        )
        taxonomy = await extractor.extract(
            policy_text=self.policy.text,
            logic_map=self.logic_map,
            categories=plan.categories,
        )

        self._taxonomy = taxonomy  # type: ignore[attr-defined]
        await self._persist()
        return taxonomy

    @property
    def taxonomy(self) -> "SubCategoryTaxonomy | None":
        """Get the current taxonomy (None if not generated)."""
        return getattr(self, "_taxonomy", None)

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
        from synkro.llm.client import LLM

        if self.scenarios is None:
            raise ValueError("No scenarios available. Call generate_scenarios first.")
        if self.logic_map is None:
            raise ValueError("No Logic Map available.")
        if self.policy is None:
            raise ValueError("No policy available.")

        self._save_snapshot(f"Before edit: {instruction[:30]}...")

        # Create LLM and track costs
        llm = LLM(model=self.grading_model, base_url=self.base_url, temperature=0.3)
        initial_cost = llm.total_cost

        new_scenarios, new_dist, summary = await edit_scenarios_async(
            self.scenarios,
            instruction,
            self.policy.text,
            self.logic_map,
            distribution=self.distribution,
            model=self.grading_model,
            base_url=self.base_url,
            llm=llm,
        )

        # Track HITL costs
        self.metrics.add_call("hitl", llm.total_cost - initial_cost, str(self.grading_model))

        self.scenarios = new_scenarios
        self.distribution = new_dist
        await self._persist()
        return new_scenarios, new_dist, summary

    async def synthesize_traces(
        self,
        turns: int | None = None,
        model: str | None = None,
        concurrency: int | None = None,
    ) -> TracesResult:
        """
        Synthesize traces from the stored scenarios.

        Args:
            turns: Number of conversation turns (forced to 1 for instruction/evaluation)
            model: Optional model override
            concurrency: Max concurrent LLM calls (default: session.concurrency or 50)

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

        # Force turns=1 for instruction and evaluation (matching Generator behavior)
        if self.dataset_type in ("instruction", "evaluation"):
            turns = 1
        elif turns is None:
            turns = 1  # Default

        self._save_snapshot("Before trace synthesis")

        result = await synthesize_traces_async(
            self.policy,
            scenarios=self.scenarios,
            logic_map=self.logic_map,
            turns=turns,
            model=model or self.model,
            base_url=self.base_url,
            dataset_type=self.dataset_type,
            tools=self.tools,
            concurrency=concurrency or self.concurrency,
        )

        self.traces = result.traces
        await self._persist()

        # Accumulate metrics
        if result.metrics:
            self.metrics.phases["traces"] = result.metrics

        return result

    async def verify_traces(
        self,
        max_iterations: int = 3,
        model: str | None = None,
        concurrency: int | None = None,
    ) -> VerificationResult:
        """
        Verify and refine traces.

        Args:
            max_iterations: Maximum refinement attempts
            model: Optional model override
            concurrency: Max concurrent LLM calls (default: session.concurrency or 50)

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
            dataset_type=self.dataset_type,
            tools=self.tools,
            concurrency=concurrency or self.concurrency,
        )

        self.verified_traces = result.verified_traces
        await self._persist()

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

    async def undo(self) -> str:
        """
        Restore the previous state.

        Returns:
            Description of what was restored, or message if no history

        Examples:
            >>> await session.undo()
            "Restored: Before edit: add rule..."
        """
        if len(self.history) < 2:
            return "No history to undo."

        # Remove current state
        self.history.pop()

        # Restore previous state
        snapshot = self.history[-1]
        self._restore_snapshot(snapshot)
        await self._persist()
        return f"Restored: {snapshot.description}"

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
        Note: If session was lazily loaded, call ensure_loaded() first.

        Returns:
            Dataset containing the traces

        Raises:
            ValueError: If no traces are available
        """
        traces = self.verified_traces or self.traces
        if traces is None:
            # Check if lazy loading might help
            if hasattr(self, "_lazy_loaded") and not self._lazy_loaded:  # type: ignore[attr-defined]
                raise ValueError(
                    "No traces available. Session was lazily loaded - "
                    "call 'await session.ensure_loaded()' first, or use "
                    "'Session.load_from_db(id, full=True)'."
                )
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

    # =========================================================================
    # DATABASE PERSISTENCE
    # =========================================================================

    @classmethod
    async def create(
        cls,
        policy: str | None = None,
        session_id: str | None = None,
        db_url: str | None = None,
        dataset_type: str = "conversation",
        tools: list["ToolDefinition"] | None = None,
        skip_grading: bool = False,
        concurrency: int = 50,
    ) -> "Session":
        """Create a database-backed session.

        Creates a new session with automatic persistence to SQLite (default) or
        PostgreSQL. State is saved automatically after each operation.

        Args:
            policy: Optional policy text to initialize with
            session_id: Optional custom session ID (8-char random ID if not provided)
            db_url: Database URL. Defaults to SQLite at ~/.synkro/sessions.db.
                    For Postgres: "postgresql://user:pass@host/db"
            dataset_type: Type of dataset (conversation, instruction, evaluation, tool_call)
            tools: List of ToolDefinition for TOOL_CALL dataset type (required for tool_call)
            skip_grading: Skip verification phase for faster generation (default: False)
            concurrency: Max concurrent LLM calls for I/O parallelism (default: 50)

        Returns:
            A new Session instance with database persistence enabled.

        Examples:
            >>> session = await Session.create(policy="Expenses over $50 need approval")
            >>> session = await Session.create(policy="...", session_id="exp001")
            >>> session = await Session.create(db_url="postgresql://localhost/synkro")
            >>> session = await Session.create(policy="...", dataset_type="tool_call", tools=[...])
            >>> session = await Session.create(policy="...", skip_grading=True)
            >>> session = await Session.create(policy="...", concurrency=100)  # Higher parallelism
        """
        # Validate dataset_type
        valid_types = {"conversation", "instruction", "evaluation", "tool_call"}
        if dataset_type not in valid_types:
            raise ValueError(
                f"Invalid dataset_type: {dataset_type}. Must be one of: {', '.join(valid_types)}"
            )

        # TOOL_CALL requires tools (matching Generator validation)
        if dataset_type == "tool_call" and not tools:
            raise ValueError("TOOL_CALL dataset type requires tools parameter")

        from synkro.storage import Storage

        store = Storage(db_url)
        sid = await store.create_session(
            session_id=session_id,
            policy_text=policy or "",
        )

        session = cls()
        session._storage = store  # type: ignore[attr-defined]
        session._session_id = sid  # type: ignore[attr-defined]
        session.dataset_type = dataset_type
        session.tools = tools
        session.skip_grading = skip_grading
        session.concurrency = concurrency

        if policy:
            session.policy = Policy(text=policy)
            await session._persist()

        return session

    @classmethod
    async def load_from_db(
        cls,
        session_id: str,
        db_url: str | None = None,
        full: bool = False,
    ) -> "Session":
        """Load a session from the database (lazy loading by default).

        Args:
            session_id: The session ID to load.
            db_url: Database URL. Defaults to SQLite at ~/.synkro/sessions.db.
            full: If True, load all data immediately. If False (default), load
                  rules/scenarios/traces lazily on first access.

        Returns:
            The loaded Session instance.

        Raises:
            ValueError: If session not found.

        Examples:
            >>> session = await Session.load_from_db("exp001")  # Fast, lazy
            >>> session = await Session.load_from_db("exp001", full=True)  # Load all
        """
        from synkro.storage import Storage

        store = Storage(db_url)

        if full:
            # Load everything at once
            data = await store.load_session_full(session_id)
            if not data:
                raise ValueError(f"Session not found: {session_id}")
            session = cls._from_data(data)
        else:
            # Lazy loading - just metadata
            data = await store.load_session(session_id)
            if not data:
                raise ValueError(f"Session not found: {session_id}")
            session = cls._from_metadata(data)
            session._lazy_loaded = False  # type: ignore[attr-defined]

        session._storage = store  # type: ignore[attr-defined]
        session._session_id = session_id  # type: ignore[attr-defined]
        return session

    async def ensure_loaded(self) -> "Session":
        """Load all data from DB if using lazy loading.

        Call this before using show_*() methods or accessing data on a
        lazily-loaded session.

        Returns:
            Self for chaining.

        Examples:
            >>> session = await Session.load_from_db("my-session")
            >>> await session.ensure_loaded()
            >>> print(session.show_rules())
        """
        if not hasattr(self, "_lazy_loaded") or self._lazy_loaded:  # type: ignore[attr-defined]
            return self
        if not hasattr(self, "_storage") or not hasattr(self, "_session_id"):
            return self

        # Load rules if needed
        if self.logic_map is None:
            rules_data = await self._storage.get_rules(self._session_id)  # type: ignore[attr-defined]
            if rules_data:
                self._load_rules(rules_data)

        # Load scenarios if needed
        if self.scenarios is None:
            scenarios_data = await self._storage.get_scenarios(self._session_id)  # type: ignore[attr-defined]
            if scenarios_data:
                self._load_scenarios(scenarios_data)

        # Load traces if needed
        if self.traces is None and self.verified_traces is None:
            traces_data = await self._storage.get_traces(self._session_id)  # type: ignore[attr-defined]
            if traces_data:
                self._load_traces(traces_data)

        self._lazy_loaded = True  # type: ignore[attr-defined]
        return self

    @classmethod
    def _from_metadata(cls, data: dict) -> "Session":
        """Create session from metadata only (for lazy loading)."""
        session = cls()

        if data.get("policy_text"):
            session.policy = Policy(text=data["policy_text"])

        session.model = data.get("model")
        session.grading_model = data.get("grading_model")
        session.base_url = data.get("base_url")
        session.dataset_type = data.get("dataset_type", "conversation")

        return session

    def _load_rules(self, rules_data: list[dict]) -> None:
        """Load rules into logic_map from data."""
        from synkro.types.logic_map import LogicMap, Rule, RuleCategory

        rules = []
        for r in rules_data:
            category = r.get("category")
            if category and isinstance(category, str):
                try:
                    category = RuleCategory(category)
                except ValueError:
                    category = RuleCategory.CORE
            rules.append(
                Rule(
                    rule_id=r["rule_id"],
                    text=r["text"],
                    condition=r.get("condition", ""),
                    action=r.get("action", ""),
                    category=category or RuleCategory.CORE,
                    dependencies=r.get("dependencies", []),
                )
            )
        all_rule_ids = {r.rule_id for r in rules}
        root_rules = [
            r.rule_id
            for r in rules
            if not r.dependencies or not any(d in all_rule_ids for d in r.dependencies)
        ]
        self.logic_map = LogicMap(rules=rules, root_rules=root_rules)

    def _load_scenarios(self, scenarios_data: list[dict]) -> None:
        """Load scenarios from data."""
        from synkro.types.logic_map import GoldenScenario, ScenarioType

        scenarios = []
        for s in scenarios_data:
            scenario_type = s.get("type", "positive")
            if isinstance(scenario_type, str):
                try:
                    scenario_type = ScenarioType(scenario_type)
                except ValueError:
                    scenario_type = ScenarioType.POSITIVE
            scenarios.append(
                GoldenScenario(
                    description=s.get("description", ""),
                    context=s.get("context", ""),
                    category=s.get("category", ""),
                    scenario_type=scenario_type,
                    target_rule_ids=s.get("rules_tested", []),
                    expected_outcome=s.get("expected_outcome", ""),
                )
            )
        self.scenarios = scenarios

        # Compute distribution
        dist: dict[str, int] = {}
        for s in scenarios:
            t = s.scenario_type.value if hasattr(s.scenario_type, "value") else str(s.scenario_type)
            dist[t] = dist.get(t, 0) + 1
        self.distribution = dist

    def _load_traces(self, traces_data: list[dict]) -> None:
        """Load traces from data."""
        from synkro.types.core import GradeResult, Message, Trace

        traces = []
        for t in traces_data:
            messages = [
                Message(role=m["role"], content=m["content"]) for m in t.get("messages", [])
            ]
            grade = None
            if t.get("grade_passed") is not None:
                grade = GradeResult(
                    passed=t["grade_passed"],
                    feedback=t.get("grade_feedback", ""),
                )
            traces.append(
                Trace(
                    messages=messages,
                    grade=grade,
                    rules_applied=t.get("rules_applied", []),
                    rules_excluded=t.get("rules_violated", []),
                    reasoning_chain=t.get("reasoning_chain"),
                )
            )
        # If any are graded, put them in verified_traces
        if any(t.grade for t in traces):
            self.verified_traces = traces
        else:
            self.traces = traces

    @classmethod
    async def list_sessions(cls, db_url: str | None = None, limit: int = 50) -> list[dict]:
        """List all saved sessions.

        Args:
            db_url: Database URL. Defaults to SQLite at ~/.synkro/sessions.db.
            limit: Maximum number of sessions to return.

        Returns:
            List of session metadata dicts with session_id, created_at, updated_at.

        Examples:
            >>> sessions = await Session.list_sessions()
            >>> for s in sessions:
            ...     print(f"{s['session_id']} - {s['updated_at']}")
        """
        from synkro.storage import Storage

        store = Storage(db_url)
        return await store.list_sessions(limit=limit)

    async def delete(self) -> bool:
        """Delete this session from the database.

        Returns:
            True if deleted, False if not found or not using database storage.

        Examples:
            >>> await session.delete()
            True
        """
        if not hasattr(self, "_storage") or not hasattr(self, "_session_id"):
            return False
        return await self._storage.delete_session(self._session_id)

    @property
    def session_id(self) -> str | None:
        """Get the session ID (None for in-memory sessions)."""
        return getattr(self, "_session_id", None)

    async def _persist(self) -> None:
        """Save current state to database (if using database storage)."""
        if not hasattr(self, "_storage"):
            return

        sid = self._session_id  # type: ignore[attr-defined]

        # Update session metadata
        await self._storage.update_session(
            sid,
            model=self.model,
            grading_model=self.grading_model,
        )

        # Save rules if we have a logic map
        if self.logic_map:
            rules = [
                {
                    "rule_id": r.rule_id,
                    "text": r.text,
                    "condition": r.condition,
                    "action": r.action,
                    "category": r.category.value if r.category else None,
                    "dependencies": r.dependencies,
                }
                for r in self.logic_map.rules
            ]
            await self._storage.save_rules(sid, rules)

        # Save scenarios
        if self.scenarios:
            scenarios = [
                {
                    "scenario_id": f"S{i+1:03d}",
                    "type": s.scenario_type.value
                    if hasattr(s.scenario_type, "value")
                    else s.scenario_type,
                    "category": s.category,
                    "description": s.description,
                    "context": s.context,
                    "expected_outcome": s.expected_outcome,
                    "rules_tested": s.target_rule_ids,
                }
                for i, s in enumerate(self.scenarios)
            ]
            await self._storage.save_scenarios(sid, scenarios)

        # Save traces (verified traces take priority)
        traces_to_save = self.verified_traces if self.verified_traces else self.traces
        if traces_to_save:
            traces = [
                {
                    "messages": [{"role": m.role, "content": m.content} for m in t.messages],
                    "grade_passed": t.grade.passed if t.grade else None,
                    "grade_feedback": t.grade.feedback if t.grade else None,
                    "rules_applied": t.rules_applied or [],
                    "rules_violated": t.rules_excluded or [],
                    "reasoning_chain": t.reasoning_chain,
                }
                for t in traces_to_save
            ]
            await self._storage.save_traces(sid, traces)

        # Save taxonomy
        taxonomy = getattr(self, "_taxonomy", None)
        if taxonomy and taxonomy.sub_categories:
            tax_list = [
                {
                    "category": sc.parent_category,
                    "subcategory": sc.name,
                    "description": sc.description,
                    "scenario_count": 0,  # Will be computed from scenarios
                }
                for sc in taxonomy.sub_categories
            ]
            await self._storage.save_taxonomy(sid, tax_list)

    def _to_data(self) -> dict:
        """Serialize session state to dict for storage (legacy, used by _from_data)."""
        taxonomy = getattr(self, "_taxonomy", None)
        return {
            "policy_text": self.policy.text if self.policy else None,
            "logic_map": self.logic_map.model_dump() if self.logic_map else None,
            "scenarios": [s.model_dump() for s in self.scenarios] if self.scenarios else None,
            "distribution": self.distribution,
            "coverage": self.coverage_report.model_dump() if self.coverage_report else None,
            "taxonomy": taxonomy.model_dump() if taxonomy else None,
            "traces": [_serialize_trace(t) for t in self.traces] if self.traces else None,
            "verified_traces": (
                [_serialize_trace(t) for t in self.verified_traces]
                if self.verified_traces
                else None
            ),
            "model": self.model,
            "grading_model": self.grading_model,
            "dataset_type": self.dataset_type,
        }

    @classmethod
    def _from_data(cls, data: dict) -> "Session":
        """Deserialize session state from normalized storage format (full load)."""
        session = cls._from_metadata(data)

        # Load rules/scenarios/traces using helper methods
        if data.get("rules"):
            session._load_rules(data["rules"])
        if data.get("scenarios"):
            session._load_scenarios(data["scenarios"])
        if data.get("traces"):
            session._load_traces(data["traces"])

        return session

    # =========================================================================
    # SHOW METHODS - Human-readable state inspection
    # =========================================================================

    def show_rules(self, limit: int | None = None) -> str:
        """Show extracted rules in a readable format.

        Args:
            limit: Max rules to show (None = all)

        Returns:
            Formatted string of rules

        Examples:
            >>> print(session.show_rules())
            >>> print(session.show_rules(limit=5))
        """
        if not self.logic_map:
            return "No rules extracted yet. Call extract_rules() first."

        lines = [f"Rules ({len(self.logic_map.rules)} total):", ""]
        rules = self.logic_map.rules[:limit] if limit else self.logic_map.rules
        for r in rules:
            lines.append(f"  {r.rule_id}: {r.text[:80]}{'...' if len(r.text) > 80 else ''}")
        if limit and len(self.logic_map.rules) > limit:
            lines.append(f"  ... and {len(self.logic_map.rules) - limit} more")
        return "\n".join(lines)

    def show_scenarios(self, filter: str | None = None, limit: int | None = None) -> str:
        """Show scenarios in a readable format.

        Args:
            filter: Filter by rule ID, scenario type, or text match
            limit: Max scenarios to show (None = all matching)

        Returns:
            Formatted string of scenarios

        Examples:
            >>> print(session.show_scenarios())
            >>> print(session.show_scenarios(filter="R005"))
            >>> print(session.show_scenarios(filter="edge_case"))
        """
        if not self.scenarios:
            return "No scenarios generated yet. Call generate_scenarios() first."

        filtered = self.scenarios
        if filter:
            filter_lower = filter.lower()
            filtered = [
                s
                for s in self.scenarios
                if filter_lower in s.description.lower()
                or filter_lower in (s.scenario_type.value if s.scenario_type else "")
                or any(filter.upper() in rid for rid in (s.target_rule_ids or []))
            ]

        if not filtered:
            return f"No scenarios matching '{filter}'"

        lines = [f"Scenarios ({len(filtered)} {'matching' if filter else 'total'}):", ""]
        scenarios = filtered[:limit] if limit else filtered
        for i, s in enumerate(scenarios, 1):
            stype = f"[{s.scenario_type.value}]" if s.scenario_type else ""
            rules = f" ({', '.join(s.target_rule_ids)})" if s.target_rule_ids else ""
            lines.append(f"  S{i:02d}: {s.description[:55]}... {stype}{rules}")
        if limit and len(filtered) > limit:
            lines.append(f"  ... and {len(filtered) - limit} more")
        return "\n".join(lines)

    def show_distribution(self) -> str:
        """Show scenario distribution.

        Returns:
            Formatted distribution string

        Examples:
            >>> print(session.show_distribution())
        """
        if not self.distribution:
            return "No distribution yet. Call generate_scenarios() first."

        lines = ["Distribution:", ""]
        total = sum(self.distribution.values())
        for stype, count in self.distribution.items():
            pct = 100 * count / total if total > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"  {stype:12} {count:3} ({pct:5.1f}%) {bar}")
        return "\n".join(lines)

    def show_coverage(self) -> str:
        """Show coverage report.

        Returns:
            Formatted coverage string

        Examples:
            >>> print(session.show_coverage())
        """
        if not self.coverage_report:
            return "No coverage report yet."

        cr = self.coverage_report
        lines = [
            "Coverage Report:",
            "",
            f"  Total scenarios: {cr.total_scenarios}",
            f"  Sub-categories: {cr.total_sub_categories}",
            f"  Covered: {cr.covered_count} ({100*cr.covered_count/cr.total_sub_categories:.0f}%)"
            if cr.total_sub_categories > 0
            else "  Covered: 0",
            f"  Partial: {cr.partial_count}",
            f"  Gaps: {cr.gap_count}",
        ]
        return "\n".join(lines)

    def show_taxonomy(self, limit: int | None = None) -> str:
        """Show taxonomy sub-categories.

        Args:
            limit: Max sub-categories to show

        Returns:
            Formatted taxonomy string

        Examples:
            >>> print(session.show_taxonomy())
        """
        taxonomy = getattr(self, "_taxonomy", None)
        if not taxonomy:
            return "No taxonomy yet. Call generate_taxonomy() first."

        lines = [f"Taxonomy ({len(taxonomy.sub_categories)} sub-categories):", ""]

        # Group by parent category
        by_cat: dict[str, list] = {}
        for sc in taxonomy.sub_categories:
            cat = sc.parent_category
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(sc)

        count = 0
        for cat, scs in by_cat.items():
            lines.append(f"  [{cat}]")
            for sc in scs:
                if limit and count >= limit:
                    remaining = len(taxonomy.sub_categories) - count
                    lines.append(f"  ... and {remaining} more")
                    return "\n".join(lines)
                lines.append(f"    {sc.id}: {sc.name}")
                count += 1
        return "\n".join(lines)

    def show_failed(self) -> str:
        """Show failed traces with issues.

        Returns:
            Formatted string of failed traces

        Examples:
            >>> print(session.show_failed())
        """
        traces = self.verified_traces or self.traces
        if not traces:
            return "No traces yet. Call synthesize_traces() first."

        failed = [t for t in traces if t.grade and not t.grade.passed]
        if not failed:
            return "No failed traces! All passed."

        lines = [f"Failed Traces ({len(failed)}):", ""]
        for t in failed:
            lines.append(f"  - {t.scenario.description[:50]}...")
            if t.grade and t.grade.issues:
                for issue in t.grade.issues[:2]:
                    lines.append(f"    → {issue[:60]}...")
        return "\n".join(lines)

    def show_passed(self) -> str:
        """Show pass rate summary.

        Returns:
            Formatted pass rate string

        Examples:
            >>> print(session.show_passed())
        """
        traces = self.verified_traces or self.traces
        if not traces:
            return "No traces yet."

        passed = sum(1 for t in traces if t.grade and t.grade.passed)
        total = len(traces)
        pct = 100 * passed / total if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        return f"Pass Rate: {passed}/{total} ({pct:.1f}%) {bar}"

    def status(self) -> str:
        """Show pipeline status - what's done vs pending.

        Returns:
            Formatted status string

        Examples:
            >>> print(session.status())
            Rules: ✓ (17) | Scenarios: ✓ (30) | Traces: ✗ | Verified: ✗
        """
        parts = []

        # Rules
        if self.logic_map:
            parts.append(f"Rules: ✓ ({len(self.logic_map.rules)})")
        else:
            parts.append("Rules: ✗")

        # Taxonomy
        taxonomy = getattr(self, "_taxonomy", None)
        if taxonomy:
            parts.append(f"Taxonomy: ✓ ({len(taxonomy.sub_categories)})")
        else:
            parts.append("Taxonomy: ✗")

        # Scenarios
        if self.scenarios:
            parts.append(f"Scenarios: ✓ ({len(self.scenarios)})")
        else:
            parts.append("Scenarios: ✗")

        # Traces
        if self.traces:
            parts.append(f"Traces: ✓ ({len(self.traces)})")
        else:
            parts.append("Traces: ✗")

        # Verified
        if self.verified_traces:
            passed = sum(1 for t in self.verified_traces if t.grade and t.grade.passed)
            parts.append(f"Verified: ✓ ({passed}/{len(self.verified_traces)})")
        else:
            parts.append("Verified: ✗")

        # Cost (if any)
        if self.metrics.total_cost > 0:
            parts.append(f"Cost: ${self.metrics.total_cost:.4f}")

        return " | ".join(parts)

    def show_cost(self) -> str:
        """Show cost breakdown by pipeline phase.

        Returns:
            Formatted cost table with per-phase breakdown

        Examples:
            >>> print(session.show_cost())
            Cost Breakdown:
            --------------------------------------------------
            Phase             Cost      Calls       Time
            --------------------------------------------------
            extraction       $0.0012        3       2.1s
            scenarios        $0.0089       12       8.4s
            traces           $0.2341       30      45.2s
            verification     $0.1823       30      38.1s
            --------------------------------------------------
            Total            $0.4265       75
        """
        if not self.metrics.phases:
            return "No cost data yet. Run pipeline steps first."

        return self.metrics.format_table()

    def show_cost_summary(self) -> str:
        """Show one-line cost summary.

        Returns:
            Summary like "Cost: $0.42 | Calls: 91 | Time: 2m 34s"

        Examples:
            >>> print(session.show_cost_summary())
            Cost: $0.4265 | Calls: 75 | Time: 1m 34s
        """
        if not self.metrics.phases:
            return "No cost data yet."

        return self.metrics.format_summary()

    def show_trace(self, index: int) -> str:
        """Show a specific trace in detail.

        Args:
            index: Trace index (0-based)

        Returns:
            Formatted trace details

        Examples:
            >>> print(session.show_trace(0))
            >>> print(session.show_trace(3))
        """
        traces = self.verified_traces or self.traces
        if not traces:
            return "No traces yet."

        if index < 0 or index >= len(traces):
            return f"Invalid index. Valid range: 0-{len(traces) - 1}"

        t = traces[index]
        lines = [f"Trace #{index}:", ""]

        # Scenario
        lines.append(f"Scenario: {t.scenario.description[:70]}...")
        if hasattr(t.scenario, "scenario_type") and t.scenario.scenario_type:
            lines.append(f"Type: {t.scenario.scenario_type}")

        # Grade
        if t.grade:
            status = "✓ PASSED" if t.grade.passed else "✗ FAILED"
            lines.append(f"Grade: {status}")
            if t.grade.issues:
                lines.append("Issues:")
                for issue in t.grade.issues:
                    lines.append(f"  - {issue[:70]}...")
        else:
            lines.append("Grade: Not graded")

        # Messages preview
        lines.append("")
        lines.append("Conversation:")
        for m in t.messages[:4]:  # First 4 messages
            role = m.role.upper()
            content = (m.content or "")[:60]
            lines.append(f"  [{role}] {content}...")
        if len(t.messages) > 4:
            lines.append(f"  ... ({len(t.messages) - 4} more messages)")

        return "\n".join(lines)

    async def refine_trace(self, index: int, feedback: str) -> str:
        """Refine a specific trace with feedback.

        Args:
            index: Trace index (0-based)
            feedback: Natural language feedback for improvement

        Returns:
            Summary of changes

        Examples:
            >>> await session.refine_trace(3, "mention the receipt requirement")
        """
        from synkro.quality.golden_refiner import GoldenRefiner
        from synkro.quality.verifier import GoldenVerifier

        traces = self.verified_traces or self.traces
        if not traces:
            raise ValueError("No traces yet.")

        if index < 0 or index >= len(traces):
            raise ValueError(f"Invalid index. Valid range: 0-{len(traces) - 1}")

        if not self.logic_map or not self.scenarios:
            raise ValueError("Missing logic_map or scenarios.")

        trace = traces[index]
        scenario = self.scenarios[index] if index < len(self.scenarios) else None
        if not scenario:
            raise ValueError("Cannot find matching scenario for trace.")

        # Create a mock verification result with the feedback
        from synkro.llm.client import LLM
        from synkro.types.core import GradeResult

        mock_result = type(
            "VerificationResult",
            (),
            {"passed": False, "issues": [feedback], "feedback": feedback},
        )()

        # Create LLMs and track costs
        refiner_llm = LLM(model=self.grading_model or self.model, temperature=0.5)
        verifier_llm = LLM(model=self.grading_model or self.model, temperature=0.3)
        initial_cost = 0.0

        # Refine
        refiner = GoldenRefiner(llm=refiner_llm)
        refined = await refiner.refine(trace, self.logic_map, scenario, mock_result)
        initial_cost += refiner_llm.total_cost

        # Re-verify
        verifier = GoldenVerifier(llm=verifier_llm)
        result = await verifier.verify(refined, self.logic_map, scenario, self.policy.text)
        initial_cost += verifier_llm.total_cost

        # Track HITL costs
        self.metrics.add_call("hitl", initial_cost, str(self.grading_model or self.model))

        # Update grade
        refined.grade = GradeResult(
            passed=result.passed,
            issues=result.issues or [],
            feedback=result.feedback or "",
        )

        # Replace in list
        if self.verified_traces:
            self.verified_traces[index] = refined
        elif self.traces:
            self.traces[index] = refined

        await self._persist()

        status = "now passes" if result.passed else "still has issues"
        return f"Trace #{index} refined - {status}"

    # =========================================================================
    # DONE - Complete pipeline in one call
    # =========================================================================

    async def done(
        self,
        output: str | None = None,
        count: int | None = None,
        concurrency: int | None = None,
    ) -> "Dataset":
        """Complete the pipeline: synthesize, verify, and optionally export.

        One-liner to finish everything after rules/scenarios are ready.
        Respects skip_grading setting from Session.create().

        Args:
            output: Optional output file path (e.g., "traces.jsonl")
            count: Optional scenario count if scenarios not yet generated
            concurrency: Max concurrent LLM calls (default: session.concurrency or 50)

        Returns:
            The final Dataset

        Examples:
            >>> dataset = await session.done()
            >>> dataset = await session.done(output="my_traces.jsonl")
            >>> dataset = await session.done(count=50, output="traces.jsonl")
            >>> dataset = await session.done(concurrency=100)  # Higher parallelism
        """
        # Use provided concurrency or session default
        effective_concurrency = concurrency or self.concurrency

        # Generate scenarios if needed
        if not self.scenarios and count:
            await self.generate_scenarios(count=count)
        elif not self.scenarios:
            raise ValueError("No scenarios. Call generate_scenarios() or pass count=N")

        # Synthesize traces
        if not self.traces:
            await self.synthesize_traces(concurrency=effective_concurrency)

        # Verify traces (skip if skip_grading=True, matching Generator behavior)
        if not self.skip_grading and not self.verified_traces:
            await self.verify_traces(concurrency=effective_concurrency)

        # Create dataset
        dataset = self.to_dataset()

        # Save if output specified
        if output:
            dataset.save(output)

        return dataset

    # =========================================================================
    # ONE-LINER REFINEMENT METHODS
    # =========================================================================

    async def refine_rules(self, feedback: str) -> str:
        """Refine rules with natural language and auto-persist.

        A convenience wrapper around edit_rules() that returns just the summary
        and automatically persists changes.

        Args:
            feedback: Natural language instruction for editing rules.

        Returns:
            Summary of changes made.

        Examples:
            >>> await session.refine_rules("add rule for overtime approval")
            >>> await session.refine_rules("remove R005")
            >>> await session.refine_rules("merge R002 and R003")
        """
        _, summary = await self.edit_rules(feedback)
        await self._persist()
        return summary

    async def refine_scenarios(self, feedback: str) -> str:
        """Refine scenarios with natural language and auto-persist.

        A convenience wrapper around edit_scenarios() that returns just the summary
        and automatically persists changes.

        Args:
            feedback: Natural language instruction for editing scenarios.

        Returns:
            Summary of changes made.

        Examples:
            >>> await session.refine_scenarios("add 5 edge cases")
            >>> await session.refine_scenarios("delete S3")
            >>> await session.refine_scenarios("more negative cases for R002")
        """
        _, _, summary = await self.edit_scenarios(feedback)
        await self._persist()
        return summary

    async def refine_taxonomy(self, feedback: str) -> str:
        """Refine taxonomy with natural language and auto-persist.

        Modifies the sub-category taxonomy used for coverage tracking.

        Args:
            feedback: Natural language instruction for editing taxonomy.

        Returns:
            Summary of changes made.

        Raises:
            ValueError: If no taxonomy is available.

        Examples:
            >>> await session.refine_taxonomy("add category for travel expenses")
            >>> await session.refine_taxonomy("rename SC001 to Approval Thresholds")
        """
        from synkro.interactive.taxonomy_editor import TaxonomyEditor
        from synkro.llm.client import LLM

        if not hasattr(self, "_taxonomy") or getattr(self, "_taxonomy", None) is None:
            raise ValueError("No taxonomy. Call generate_scenarios with coverage first.")

        # Create LLM and track costs
        llm = LLM(model=self.grading_model, temperature=0.1)
        initial_cost = llm.total_cost

        editor = TaxonomyEditor(llm=llm)
        new_taxonomy, summary = await editor.refine(
            self._taxonomy,  # type: ignore[attr-defined]
            feedback,
            self.policy.text,
            self.logic_map,
        )

        # Track HITL costs
        self.metrics.add_call("hitl", llm.total_cost - initial_cost, str(self.grading_model))

        self._taxonomy = new_taxonomy  # type: ignore[attr-defined]
        await self._persist()
        return summary


__all__ = ["Session", "SessionSnapshot"]
