"""Pipeline phases for generation.

Each phase is a self-contained, testable unit that handles one step
of the generation pipeline.
"""

import asyncio
from asyncio import Semaphore
from typing import TYPE_CHECKING

from synkro.core.policy import Policy
from synkro.generation.planner import Planner
from synkro.generation.responses import ResponseGenerator
from synkro.generation.scenarios import ScenarioGenerator
from synkro.quality.grader import Grader
from synkro.quality.refiner import Refiner
from synkro.types.core import Plan, Scenario, Trace

if TYPE_CHECKING:
    from synkro.generation.golden_responses import GoldenResponseGenerator
    from synkro.generation.golden_scenarios import GoldenScenarioGenerator
    from synkro.generation.golden_tool_responses import GoldenToolCallResponseGenerator
    from synkro.generation.logic_extractor import LogicExtractor
    from synkro.generation.tool_responses import ToolCallResponseGenerator
    from synkro.quality.golden_refiner import GoldenRefiner
    from synkro.quality.verifier import TraceVerifier
    from synkro.types.logic_map import GoldenScenario, LogicMap


class PlanPhase:
    """
    Planning phase - analyzes policy and creates category distribution.

    This phase uses a stronger model to understand the policy and
    determine optimal scenario distribution. When analyze_turns is True,
    also performs complexity analysis to determine recommended turns.
    """

    async def execute(
        self,
        policy: Policy,
        traces: int,
        planner: Planner,
        analyze_turns: bool = True,
    ) -> Plan:
        """
        Execute the planning phase.

        Args:
            policy: The policy to analyze
            traces: Target number of traces
            planner: Planner component to use
            analyze_turns: Whether to analyze complexity for turn recommendations

        Returns:
            Plan with categories, trace distribution, and turn recommendations
        """
        return await planner.plan(policy.text, traces, analyze_turns=analyze_turns)


class ScenarioPhase:
    """
    Scenario generation phase - creates scenarios for each category.

    Runs in parallel across categories for efficiency.
    """

    async def execute(
        self,
        policy: Policy,
        plan: Plan,
        generator: ScenarioGenerator,
        semaphore: Semaphore,
    ) -> list[Scenario]:
        """
        Execute scenario generation.

        Args:
            policy: The policy text
            plan: Plan with categories
            generator: ScenarioGenerator component
            semaphore: Semaphore for rate limiting

        Returns:
            List of all generated scenarios
        """

        async def limited_generate(category):
            async with semaphore:
                return await generator.generate(policy.text, category.count, category=category)

        tasks = [limited_generate(cat) for cat in plan.categories]
        results = await asyncio.gather(*tasks)

        # Flatten results
        return [scenario for batch in results for scenario in batch]


class ResponsePhase:
    """
    Response generation phase - creates responses for each scenario.

    Runs fully parallel with semaphore control. Supports both single-turn
    and multi-turn trace generation based on target_turns.
    """

    async def execute(
        self,
        policy: Policy,
        scenarios: list[Scenario],
        generator: ResponseGenerator,
        semaphore: Semaphore,
        target_turns: int = 1,
    ) -> list[Trace]:
        """
        Execute response generation.

        Args:
            policy: The policy text
            scenarios: List of scenarios to respond to
            generator: ResponseGenerator component
            semaphore: Semaphore for rate limiting
            target_turns: Number of conversation turns (1 for single-turn)

        Returns:
            List of traces with generated responses
        """

        async def limited_generate(scenario):
            async with semaphore:
                return await generator._generate_single(policy.text, scenario, target_turns)

        tasks = [limited_generate(s) for s in scenarios]
        return await asyncio.gather(*tasks)


class GradingPhase:
    """
    Grading and refinement phase - evaluates and improves responses.

    Includes the refinement loop for failed traces.
    """

    async def execute(
        self,
        policy: Policy,
        traces: list[Trace],
        grader: Grader,
        refiner: Refiner,
        max_iterations: int,
        semaphore: Semaphore,
    ) -> tuple[list[Trace], float]:
        """
        Execute grading and refinement.

        Args:
            policy: The policy text
            traces: List of traces to grade
            grader: Grader component
            refiner: Refiner component
            max_iterations: Maximum refinement iterations
            semaphore: Semaphore for rate limiting

        Returns:
            Tuple of (graded traces, pass rate percentage)
        """

        async def limited_grade(trace):
            async with semaphore:
                return await grader.grade(trace, policy.text)

        async def limited_refine(trace, grade):
            async with semaphore:
                return await refiner.refine(trace, grade, policy.text)

        # Initial grading
        grade_tasks = [limited_grade(t) for t in traces]
        grades = await asyncio.gather(*grade_tasks)

        # Attach grades
        final_traces = list(traces)
        for trace, grade in zip(final_traces, grades):
            trace.grade = grade

        # Refinement loop
        for iteration in range(1, max_iterations):
            failed_indices = [i for i, t in enumerate(final_traces) if not t.grade.passed]

            if not failed_indices:
                break

            # Refine failed traces
            refine_tasks = [
                limited_refine(final_traces[i], final_traces[i].grade) for i in failed_indices
            ]
            refined_traces = await asyncio.gather(*refine_tasks)

            # Preserve original scenarios and update traces
            for idx, refined in zip(failed_indices, refined_traces):
                refined.scenario = final_traces[idx].scenario
                final_traces[idx] = refined

            # Re-grade refined traces
            regrade_tasks = [limited_grade(final_traces[i]) for i in failed_indices]
            new_grades = await asyncio.gather(*regrade_tasks)

            for idx, grade in zip(failed_indices, new_grades):
                final_traces[idx].grade = grade

        # Calculate pass rate
        passed_count = sum(1 for t in final_traces if t.grade and t.grade.passed)
        pass_rate = (passed_count / len(final_traces) * 100) if final_traces else 0

        return final_traces, pass_rate


class ToolCallResponsePhase:
    """
    Tool call response generation phase - creates traces with proper tool calling format.

    Uses ToolCallResponseGenerator to produce traces with:
    - System message with tool descriptions
    - User message
    - Assistant message with tool_calls (or direct response)
    - Tool response messages
    - Final assistant message

    Supports multi-turn tool calling sequences with follow-up questions.
    """

    async def execute(
        self,
        policy: Policy,
        scenarios: list[Scenario],
        generator: "ToolCallResponseGenerator",
        semaphore: Semaphore,
        target_turns: int = 1,
    ) -> list[Trace]:
        """
        Execute tool call response generation.

        Args:
            policy: The policy/guidelines text
            scenarios: List of scenarios to respond to
            generator: ToolCallResponseGenerator component
            semaphore: Semaphore for rate limiting
            target_turns: Number of conversation turns (1 for single-turn)

        Returns:
            List of traces with proper tool calling format
        """

        async def limited_generate(scenario):
            async with semaphore:
                return await generator.generate_single(
                    policy.text, scenario, target_turns=target_turns
                )

        tasks = [limited_generate(s) for s in scenarios]
        return await asyncio.gather(*tasks)


# =============================================================================
# GOLDEN TRACE PHASES
# =============================================================================


class LogicExtractionPhase:
    """
    Logic Extraction phase (The Cartographer) - extracts rules as DAG.

    This is Stage 1 of the Golden Trace pipeline.
    """

    async def execute(
        self,
        policy: Policy,
        extractor: "LogicExtractor",
    ) -> "LogicMap":
        """
        Execute logic extraction.

        Args:
            policy: The policy to analyze
            extractor: LogicExtractor component

        Returns:
            LogicMap with extracted rules as DAG
        """
        return await extractor.extract(policy.text)


class GoldenScenarioPhase:
    """
    Golden Scenario phase (The Adversary) - generates typed scenarios.

    Distributes scenarios across types:
    - 35% positive (happy path)
    - 30% negative (violations)
    - 25% edge_case (boundaries)
    - 10% irrelevant (out of scope)

    This is Stage 2 of the Golden Trace pipeline.
    """

    async def execute(
        self,
        policy: Policy,
        logic_map: "LogicMap",
        plan: Plan,
        generator: "GoldenScenarioGenerator",
        semaphore: Semaphore,
    ) -> tuple[list["GoldenScenario"], dict[str, int]]:
        """
        Execute golden scenario generation.

        Args:
            policy: The policy text
            logic_map: The extracted Logic Map
            plan: Plan with categories
            generator: GoldenScenarioGenerator component
            semaphore: Semaphore for rate limiting

        Returns:
            Tuple of (scenarios, type distribution dict)
        """

        async def limited_generate(category):
            async with semaphore:
                return await generator.generate(policy.text, logic_map, category, category.count)

        tasks = [limited_generate(cat) for cat in plan.categories]
        results = await asyncio.gather(*tasks)

        # Flatten scenarios
        all_scenarios = [s for batch in results for s in batch]

        # Calculate distribution
        distribution = {
            "positive": 0,
            "negative": 0,
            "edge_case": 0,
            "irrelevant": 0,
        }
        for s in all_scenarios:
            distribution[s.scenario_type.value] += 1

        return all_scenarios, distribution


class GoldenTracePhase:
    """
    Golden Trace phase (The Thinker) - generates traces with grounded reasoning.

    Produces traces with:
    - Chain-of-thought reasoning with rule citations
    - Exclusionary reasoning (why rules don't apply)
    - DAG-compliant dependency order

    This is Stage 3 of the Golden Trace pipeline for CONVERSATION/INSTRUCTION.
    """

    async def execute(
        self,
        policy: Policy,
        logic_map: "LogicMap",
        scenarios: list["GoldenScenario"],
        generator: "GoldenResponseGenerator",
        semaphore: Semaphore,
        target_turns: int = 1,
    ) -> list[Trace]:
        """
        Execute golden trace generation.

        Args:
            policy: The policy text
            logic_map: The extracted Logic Map
            scenarios: List of golden scenarios
            generator: GoldenResponseGenerator component
            semaphore: Semaphore for rate limiting
            target_turns: Number of conversation turns

        Returns:
            List of traces with grounded reasoning
        """

        async def limited_generate(scenario):
            async with semaphore:
                return await generator.generate_single(
                    policy.text, logic_map, scenario, target_turns
                )

        tasks = [limited_generate(s) for s in scenarios]
        return await asyncio.gather(*tasks)


class GoldenToolCallPhase:
    """
    Golden Tool Call phase (The Thinker for Tools) - generates tool traces.

    Produces tool call traces with:
    - Rule citations for tool selection decisions
    - Grounded reasoning linking rules to tool usage
    - OpenAI function calling format

    This is Stage 3 of the Golden Trace pipeline for TOOL_CALL.
    """

    async def execute(
        self,
        policy: Policy,
        logic_map: "LogicMap",
        scenarios: list["GoldenScenario"],
        generator: "GoldenToolCallResponseGenerator",
        semaphore: Semaphore,
        target_turns: int = 1,
    ) -> list[Trace]:
        """
        Execute golden tool call trace generation.

        Args:
            policy: The policy text
            logic_map: The extracted Logic Map
            scenarios: List of golden scenarios
            generator: GoldenToolCallResponseGenerator component
            semaphore: Semaphore for rate limiting
            target_turns: Number of conversation turns

        Returns:
            List of traces with tool calling format
        """

        async def limited_generate(scenario):
            async with semaphore:
                return await generator.generate_single(
                    policy.text, logic_map, scenario, target_turns
                )

        tasks = [limited_generate(s) for s in scenarios]
        return await asyncio.gather(*tasks)


class VerificationPhase:
    """
    Verification phase (The Auditor) - verifies traces against Logic Map.

    Checks:
    - No skipped rules
    - No hallucinated rules
    - No contradictions
    - DAG compliance

    This is Stage 4 of the Golden Trace pipeline.
    """

    async def execute(
        self,
        policy: Policy,
        logic_map: "LogicMap",
        scenarios: list["GoldenScenario"],
        traces: list[Trace],
        verifier: "TraceVerifier",
        refiner: "GoldenRefiner",
        max_iterations: int,
        semaphore: Semaphore,
    ) -> tuple[list[Trace], float]:
        """
        Execute verification and refinement.

        Args:
            policy: The policy text
            logic_map: The Logic Map (ground truth)
            scenarios: The golden scenarios (for verification context)
            traces: List of traces to verify
            verifier: TraceVerifier component
            refiner: GoldenRefiner component
            max_iterations: Maximum refinement iterations
            semaphore: Semaphore for rate limiting

        Returns:
            Tuple of (verified traces, pass rate percentage)
        """

        async def limited_verify(trace, scenario):
            async with semaphore:
                verification, grade = await verifier.verify_and_grade(trace, logic_map, scenario)
                return verification, grade

        async def limited_refine(trace, scenario, verification):
            async with semaphore:
                return await refiner.refine(trace, logic_map, scenario, verification)

        # Create scenario lookup by matching trace.scenario.description
        scenario_lookup = {s.description: s for s in scenarios}

        # Initial verification
        verify_tasks = []
        for trace in traces:
            # Find matching scenario
            scenario = scenario_lookup.get(trace.scenario.description)
            if not scenario:
                # Create a minimal GoldenScenario from the trace scenario
                from synkro.types.logic_map import GoldenScenario, ScenarioType

                scenario = GoldenScenario(
                    description=trace.scenario.description,
                    context=trace.scenario.context or "",
                    category=trace.scenario.category or "",
                    scenario_type=ScenarioType.POSITIVE,
                    target_rule_ids=[],
                    expected_outcome="",
                )
            verify_tasks.append(limited_verify(trace, scenario))

        results = await asyncio.gather(*verify_tasks)

        # Single pass: attach grades, track verifications, and find initial failures
        final_traces = list(traces)
        verifications = []
        failed_indices = []
        for i, (verification, grade) in enumerate(results):
            final_traces[i].grade = grade
            verifications.append(verification)
            if not verification.passed:
                failed_indices.append(i)

        # Helper to get or create scenario for a trace (avoids repeated fallback creation)
        def get_scenario_for_trace(trace_idx: int) -> "GoldenScenario":
            trace = final_traces[trace_idx]
            scenario = scenario_lookup.get(trace.scenario.description)
            if scenario:
                return scenario
            # Create minimal fallback (only if not in lookup)
            from synkro.types.logic_map import GoldenScenario, ScenarioType

            return GoldenScenario(
                description=trace.scenario.description,
                context=trace.scenario.context or "",
                category=trace.scenario.category or "",
                scenario_type=ScenarioType.POSITIVE,
                target_rule_ids=[],
                expected_outcome="",
            )

        # Refinement loop
        for iteration in range(1, max_iterations):
            if not failed_indices:
                break

            # Refine failed traces
            refine_tasks = [
                limited_refine(final_traces[i], get_scenario_for_trace(i), verifications[i])
                for i in failed_indices
            ]
            refined_traces = await asyncio.gather(*refine_tasks)

            # Update traces
            for idx, refined in zip(failed_indices, refined_traces):
                refined.scenario = final_traces[idx].scenario
                final_traces[idx] = refined

            # Re-verify
            reverify_tasks = [
                limited_verify(final_traces[i], get_scenario_for_trace(i)) for i in failed_indices
            ]
            new_results = await asyncio.gather(*reverify_tasks)

            # Update results and find new failures for next iteration
            prev_failed = failed_indices
            failed_indices = []
            for idx, (verification, grade) in zip(prev_failed, new_results):
                final_traces[idx].grade = grade
                verifications[idx] = verification
                if not verification.passed:
                    failed_indices.append(idx)

        # Calculate pass rate
        passed_count = sum(1 for v in verifications if v.passed)
        pass_rate = (passed_count / len(verifications) * 100) if verifications else 0

        return final_traces, pass_rate


__all__ = [
    "PlanPhase",
    "ScenarioPhase",
    "ResponsePhase",
    "GradingPhase",
    "ToolCallResponsePhase",
    # Golden Trace phases
    "LogicExtractionPhase",
    "GoldenScenarioPhase",
    "GoldenTracePhase",
    "GoldenToolCallPhase",
    "VerificationPhase",
]
