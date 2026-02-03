"""Policy refiner - generate golden traces from violations.

Pulls violations from a ViolationStore and generates compliant traces
using the GoldenResponseGenerator.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from synkro.core.dataset import Dataset
from synkro.generation.golden_responses import GoldenResponseGenerator
from synkro.llm.client import LLM
from synkro.remediation.types import Violation
from synkro.types.core import Trace
from synkro.types.logic_map import GoldenScenario, LogicMap, ScenarioType

if TYPE_CHECKING:
    from synkro.remediation.store import ViolationStore


class PolicyRefiner:
    """Generate golden traces from policy violations.

    Takes violations from a ViolationStore and generates compliant traces
    using the existing GoldenResponseGenerator.

    Example:
        >>> from synkro import ingest, PolicyDetector, PolicyRefiner, ViolationStore
        >>>
        >>> # Detect violations
        >>> config = ingest("./policy.pdf")
        >>> detector = PolicyDetector(config.logic_map)
        >>> violations = await detector.detect(bad_traces)
        >>>
        >>> # Store violations
        >>> store = ViolationStore()
        >>> store.add_batch(violations)
        >>>
        >>> # Generate golden traces
        >>> refiner = PolicyRefiner(config.logic_map)
        >>> golden = await refiner.refine(store, traces_per_violation=3)
        >>> golden.save("./golden_traces.jsonl")
    """

    def __init__(
        self,
        logic_map: LogicMap,
        llm: LLM | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ):
        """Initialize the refiner.

        Args:
            logic_map: Structured policy (LogicMap from ingest())
            llm: LLM client (creates one if not provided)
            model: Model to use if creating LLM (auto-detected if not specified)
            temperature: Sampling temperature for generation
        """
        self.logic_map = logic_map
        if llm is not None:
            self.llm = llm
        elif model is not None:
            self.llm = LLM(model=model, temperature=temperature)
        else:
            self.llm = LLM(temperature=temperature)  # Use default model
        self._generator = GoldenResponseGenerator(llm=self.llm)

    def _format_policy(self) -> str:
        """Format LogicMap rules as policy text."""
        lines = []
        for rule in self.logic_map.rules:
            lines.append(f"{rule.rule_id}: {rule.text}")
            if rule.condition:
                lines.append(f"  Condition: {rule.condition}")
            if rule.action:
                lines.append(f"  Action: {rule.action}")
            lines.append("")
        return "\n".join(lines)

    def _violation_to_scenario(self, violation: Violation) -> GoldenScenario:
        """Convert a Violation to a GoldenScenario for trace generation.

        Args:
            violation: The violation to convert

        Returns:
            GoldenScenario suitable for GoldenResponseGenerator
        """
        # Determine scenario type based on expected outcome
        # For remediation, we always want positive (compliant) traces
        scenario_type = ScenarioType.POSITIVE

        # Get category from first violated rule
        category = ""
        if violation.rules_violated:
            rule_id = violation.rules_violated[0]
            rule = self.logic_map.get_rule(rule_id)
            if rule:
                category = rule.category.value

        return GoldenScenario(
            description=violation.user_intent or self._extract_user_intent(violation.trace),
            context=violation.context or self._extract_context(violation.trace),
            category=category,
            scenario_type=scenario_type,
            target_rule_ids=violation.rules_violated,
            expected_outcome=violation.expected_outcome,
            sub_category_ids=[],
        )

    def _extract_user_intent(self, trace: list[dict]) -> str:
        """Extract user intent from trace messages."""
        for msg in trace:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _extract_context(self, trace: list[dict]) -> str:
        """Extract context from trace (system message or conversation)."""
        for msg in trace:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return ""

    async def refine_single(
        self,
        violation: Violation,
        count: int = 1,
        turns: int = 1,
    ) -> list[Trace]:
        """Generate golden traces from a single violation.

        Args:
            violation: The violation to remediate
            count: Number of golden traces to generate
            turns: Conversation turns per trace

        Returns:
            List of compliant Trace objects
        """
        scenario = self._violation_to_scenario(violation)
        traces = []

        for _ in range(count):
            trace = await self._generator.generate_single(
                policy_text=self._format_policy(),
                logic_map=self.logic_map,
                scenario=scenario,
                target_turns=turns,
            )
            traces.append(trace)

        return traces

    async def refine(
        self,
        store: ViolationStore,
        traces_per_violation: int = 1,
        turns: int = 1,
        concurrency: int = 20,
    ) -> Dataset:
        """Generate golden traces from all violations in store.

        Args:
            store: ViolationStore containing violations to remediate
            traces_per_violation: Number of golden traces per violation
            turns: Conversation turns per trace
            concurrency: Max parallel generation calls

        Returns:
            Dataset containing golden traces
        """
        violations = store.get_all()

        if not violations:
            return Dataset(traces=[])

        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_limit(violation: Violation) -> list[Trace]:
            async with semaphore:
                return await self.refine_single(violation, count=traces_per_violation, turns=turns)

        tasks = [generate_with_limit(v) for v in violations]
        results = await asyncio.gather(*tasks)

        # Flatten list of lists
        all_traces = [trace for traces in results for trace in traces]

        return Dataset(traces=all_traces)

    async def refine_violations(
        self,
        violations: list[Violation],
        traces_per_violation: int = 1,
        turns: int = 1,
        concurrency: int = 20,
    ) -> Dataset:
        """Generate golden traces from a list of violations.

        Alternative API that takes violations directly instead of a store.

        Args:
            violations: List of violations to remediate
            traces_per_violation: Number of golden traces per violation
            turns: Conversation turns per trace
            concurrency: Max parallel generation calls

        Returns:
            Dataset containing golden traces
        """
        if not violations:
            return Dataset(traces=[])

        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_limit(violation: Violation) -> list[Trace]:
            async with semaphore:
                return await self.refine_single(violation, count=traces_per_violation, turns=turns)

        tasks = [generate_with_limit(v) for v in violations]
        results = await asyncio.gather(*tasks)

        # Flatten list of lists
        all_traces = [trace for traces in results for trace in traces]

        return Dataset(traces=all_traces)
