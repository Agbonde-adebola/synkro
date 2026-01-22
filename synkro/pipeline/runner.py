"""Pipeline runner that orchestrates all phases.

Uses the Golden Trace 4-stage pipeline for all dataset types:
1. Logic Extraction (The Cartographer)
2. Scenario Synthesis (The Adversary)
3. Trace Synthesis (The Thinker)
4. Verification (The Auditor)
"""

from __future__ import annotations

import asyncio
from datetime import datetime

# Type hints for HITL components (imported dynamically to avoid circular imports)
from typing import TYPE_CHECKING, Callable

from synkro.core.checkpoint import CheckpointManager, hash_policy
from synkro.core.dataset import Dataset
from synkro.core.policy import Policy
from synkro.factory import ComponentFactory
from synkro.pipeline.phases import (
    GoldenScenarioPhase,
    GoldenToolCallPhase,
    GoldenTracePhase,
    LogicExtractionPhase,
    PlanPhase,
    VerificationPhase,
)
from synkro.reporting import ProgressReporter
from synkro.types.logic_map import LogicMap
from synkro.types.metrics import Metrics, PhaseMetrics
from synkro.types.results import (
    ExtractionResult,
    PipelineResult,
    TracesResult,
    VerificationResult,
)
from synkro.types.results import (
    ScenariosResult as ScenariosResultNew,
)
from synkro.types.state import PipelinePhase, PipelineState

if TYPE_CHECKING:
    from synkro.ingestion import PolicyConfig
    from synkro.interactive.logic_map_editor import LogicMapEditor
    from synkro.interactive.scenario_editor import ScenarioEditor
    from synkro.types.core import Plan
    from synkro.types.coverage import CoverageReport, SubCategoryTaxonomy
    from synkro.types.logic_map import GoldenScenario

# Type for state callback
StateCallback = Callable[[PipelineState], None]


class GenerationResult:
    """
    Result of the generation pipeline.

    Provides access to both the dataset and internal artifacts like the Logic Map.

    Examples:
        >>> result = await pipeline.run(policy, traces=50, ...)
        >>> dataset = result.dataset
        >>> logic_map = result.logic_map  # Inspect extracted rules
        >>> coverage = result.coverage_report  # View coverage metrics
    """

    def __init__(
        self,
        dataset: "Dataset",
        logic_map: LogicMap | None = None,
        scenarios: list | None = None,
        distribution: dict[str, int] | None = None,
        coverage_report: "CoverageReport | None" = None,
    ):
        self.dataset = dataset
        self.logic_map = logic_map
        self.scenarios = scenarios or []
        self.distribution = distribution or {}
        self.coverage_report = coverage_report

    # Allow unpacking: dataset, logic_map = result
    def __iter__(self):
        return iter((self.dataset, self.logic_map))

    # Allow direct Dataset access for backwards compatibility
    def __getattr__(self, name):
        # Delegate to dataset for backwards compatibility
        return getattr(self.dataset, name)


class ScenariosResult:
    """
    Result of scenario-only generation for eval datasets.

    Contains scenarios with ground truth labels but no synthetic responses.
    Use with synkro.grade() to evaluate your own model's outputs.

    Examples:
        >>> result = synkro.generate_scenarios(policy, count=100)
        >>> for scenario in result.scenarios:
        ...     response = my_model(scenario.user_message)
        ...     grade = synkro.grade(response, scenario, policy)
    """

    def __init__(
        self,
        scenarios: list,
        logic_map: LogicMap,
        distribution: dict[str, int],
    ):
        from synkro.types.core import EvalScenario

        # Convert GoldenScenarios to EvalScenarios for public API
        self.scenarios: list[EvalScenario] = [
            EvalScenario(
                user_message=s.description,
                expected_outcome=s.expected_outcome,
                target_rule_ids=s.target_rule_ids,
                scenario_type=s.scenario_type.value
                if hasattr(s.scenario_type, "value")
                else s.scenario_type,
                category=s.category,
                context=s.context,
            )
            for s in scenarios
        ]
        self.logic_map = logic_map
        self.distribution = distribution

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    def save(self, path: str | None = None, format: str = "langsmith") -> "ScenariosResult":
        """
        Save scenarios to a JSONL file.

        Args:
            path: Output file path (auto-generates if not provided)
            format: Output format - "langsmith", "langfuse", or "qa"

        Returns:
            Self for method chaining

        Example:
            >>> result.save()  # Auto-names: synkro_eval_2024-01-15_1430.jsonl
            >>> result.save("my_eval.jsonl")
            >>> result.save(format="langfuse")
        """
        import json
        from datetime import datetime
        from pathlib import Path

        from rich.console import Console

        console = Console()

        # Auto-generate filename if not provided
        if path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            path = f"synkro_eval_{timestamp}.jsonl"

        path = Path(path)

        examples = []
        for scenario in self.scenarios:
            if format == "langsmith":
                example = {
                    "inputs": {
                        "question": scenario.user_message,
                        "context": scenario.context or "",
                    },
                    "outputs": {
                        "answer": scenario.expected_outcome or "",
                        "expected_outcome": scenario.expected_outcome or "",
                    },
                    "metadata": {
                        "ground_truth_rules": scenario.target_rule_ids or [],
                        "difficulty": scenario.scenario_type or "unknown",
                        "category": scenario.category or "",
                    },
                }
            elif format == "langfuse":
                example = {
                    "input": {
                        "question": scenario.user_message,
                        "context": scenario.context or "",
                    },
                    "expectedOutput": {
                        "answer": scenario.expected_outcome or "",
                        "expected_outcome": scenario.expected_outcome or "",
                    },
                    "metadata": {
                        "ground_truth_rules": scenario.target_rule_ids or [],
                        "difficulty": scenario.scenario_type or "unknown",
                        "category": scenario.category or "",
                    },
                }
            elif format == "qa":
                example = {
                    "question": scenario.user_message,
                    "answer": scenario.expected_outcome or "",
                    "context": scenario.context or "",
                    "expected_outcome": scenario.expected_outcome or "",
                    "ground_truth_rules": scenario.target_rule_ids or [],
                    "difficulty": scenario.scenario_type or "unknown",
                    "category": scenario.category or "",
                }
            else:
                raise ValueError(f"Unknown format: {format}. Use 'langsmith', 'langfuse', or 'qa'")

            examples.append(example)

        with open(path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        file_size = path.stat().st_size
        size_str = (
            f"{file_size / 1024:.1f} KB"
            if file_size < 1024 * 1024
            else f"{file_size / 1024 / 1024:.1f} MB"
        )
        console.print(f"[green]📁 Saved:[/green] {path} ({size_str})")

        return self


class GenerationPipeline:
    """
    Orchestrates the Golden Trace generation pipeline.

    All dataset types (CONVERSATION, INSTRUCTION, TOOL_CALL) use the unified 4-stage pipeline:
    - Stage 1: Logic Extraction - Extract rules as DAG
    - Stage 2: Scenario Synthesis - Generate typed scenarios (positive, negative, edge_case, irrelevant)
    - Stage 3: Trace Synthesis - Produce grounded reasoning with rule citations
    - Stage 4: Verification - Cross-reference against Logic Map

    Examples:
        >>> pipeline = GenerationPipeline(factory, reporter, workers=10)
        >>> dataset = await pipeline.run(policy, traces=50)
    """

    def __init__(
        self,
        factory: ComponentFactory,
        reporter: ProgressReporter,
        workers: int,
        max_iterations: int = 1,
        skip_grading: bool = False,
        checkpoint_manager: CheckpointManager | None = None,
        enable_hitl: bool = False,
        hitl_editor: "LogicMapEditor | None" = None,
        scenario_editor: "ScenarioEditor | None" = None,
        skip_coverage: bool = False,
        state_callback: StateCallback | None = None,
        metrics: Metrics | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            factory: ComponentFactory for creating pipeline components
            reporter: ProgressReporter for reporting progress
            workers: Number of concurrent workers (API calls)
            max_iterations: Maximum refinement iterations
            skip_grading: Whether to skip the verification phase
            checkpoint_manager: Optional checkpoint manager for resumable generation
            enable_hitl: Whether to enable Human-in-the-Loop editing (rules + scenarios)
            hitl_editor: Optional LogicMapEditor for HITL sessions
            scenario_editor: Optional ScenarioEditor for scenario editing
            skip_coverage: Whether to skip coverage tracking (faster generation)
            state_callback: Optional callback for state updates
            metrics: Optional Metrics object for centralized cost tracking
        """
        self.factory = factory
        self.reporter = reporter
        self.workers = workers
        self.max_iterations = max_iterations
        self.skip_grading = skip_grading
        self.skip_coverage = skip_coverage
        self.checkpoint_manager = checkpoint_manager
        self.enable_hitl = enable_hitl
        self.hitl_editor = hitl_editor
        self.scenario_editor = scenario_editor
        self._state_callback = state_callback
        self._metrics = metrics or Metrics()

        # Pipeline state for tracking progress
        self._state = PipelineState(metrics=self._metrics)

        # Golden Trace phases
        self.plan_phase = PlanPhase()
        self.logic_extraction_phase = LogicExtractionPhase()
        self.golden_scenario_phase = GoldenScenarioPhase()
        self.golden_trace_phase = GoldenTracePhase()
        self.golden_tool_call_phase = GoldenToolCallPhase()
        self.verification_phase = VerificationPhase()

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    def _transition_phase(self, phase: PipelinePhase, message: str = "") -> None:
        """Transition to a new pipeline phase and notify callbacks."""
        self._state.transition_to(phase, message)
        if self._state_callback:
            self._state_callback(self._state)

    def _update_progress(self, progress: float, message: str = "") -> None:
        """Update progress within current phase and notify callbacks."""
        self._state.update_progress(progress, message)
        if self._state_callback:
            self._state_callback(self._state)

    def _complete_phase(self) -> None:
        """Mark current phase as complete and notify callbacks."""
        self._state.complete_phase()
        if self._state_callback:
            self._state_callback(self._state)

    async def run(
        self,
        policy: Policy,
        traces: int,
        model: str,
        dataset_type: str,
        turns: int | str = "auto",
        return_result: bool = False,
        policy_config: "PolicyConfig | None" = None,
    ) -> Dataset | GenerationResult:
        """
        Run the Golden Trace generation pipeline.

        All dataset types use the same 4-stage pipeline, with Stage 3
        branching based on whether TOOL_CALL is needed.

        Args:
            policy: The policy to generate from
            traces: Target number of traces
            model: Model name (for reporting)
            dataset_type: Dataset type (conversation, instruction, evaluation, tool_call)
            turns: Conversation turns per trace. Use int for fixed turns, or "auto"
                for policy complexity-driven turns
            return_result: If True, return GenerationResult with logic_map access
            policy_config: Pre-computed PolicyConfig from synkro.ingest(). When provided,
                skips planning and logic extraction phases for faster generation.

        Returns:
            Dataset (default) or GenerationResult if return_result=True
        """
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.workers)

        # Reset state for new run
        self._state = PipelineState(metrics=self._metrics)

        # Check if this is a tool_call dataset
        is_tool_call = dataset_type == "tool_call"

        # Checkpointing setup
        cm = self.checkpoint_manager
        policy_hash = hash_policy(policy.text) if cm else ""
        resuming = False

        # Check for existing checkpoint
        if cm and cm.has_checkpoint():
            if cm.matches_config(policy_hash, traces, dataset_type):
                resuming = True
                from rich.console import Console

                Console().print(f"[cyan]🔄 Resuming from checkpoint (stage: {cm.stage})[/cyan]")
            else:
                cm.clear()  # Config mismatch, start fresh

        # Report start
        self.reporter.on_start(traces, model, dataset_type)
        self._transition_phase(PipelinePhase.PLANNING, "Analyzing policy...")

        # Set up live cost tracking if reporter supports it
        if hasattr(self.reporter, "set_cost_source"):
            self.reporter.set_cost_source(
                lambda: self.factory.generation_llm.total_cost + self.factory.grading_llm.total_cost
            )

        # Create components via factory (only what's needed)
        golden_scenario_gen = self.factory.create_golden_scenario_generator()
        verifier = self.factory.create_verifier()
        golden_refiner = self.factory.create_golden_refiner()

        # Create appropriate trace generator based on dataset type
        if is_tool_call and self.factory.has_tools:
            golden_tool_call_gen = self.factory.create_golden_tool_call_generator()
        else:
            golden_response_gen = self.factory.create_golden_response_generator()

        # =====================================================================
        # FAST PATH: Use pre-computed PolicyConfig (skips planning + extraction)
        # =====================================================================
        if policy_config is not None:
            from synkro.types.core import Plan

            # Route through reporter to avoid stacking when Live display is active
            if hasattr(self.reporter, "_display") and self.reporter._display:
                self.reporter._display.add_event("FAST: Using pre-computed Logic Map")

            # Use pre-computed logic map and categories
            logic_map = policy_config.logic_map
            plan = Plan(
                categories=policy_config.categories,
                reasoning="Loaded from pre-computed PolicyConfig",
                recommended_turns=policy_config.complexity.recommended_turns,
                complexity_level=policy_config.complexity.level,
            )

            # Determine target turns
            if isinstance(turns, int):
                target_turns = turns
            else:
                target_turns = policy_config.complexity.recommended_turns

            self.reporter.on_plan_complete(plan)
            self.reporter.on_logic_map_complete(logic_map)
            self._state.set_artifact(logic_map=logic_map)
            self._complete_phase()

        else:
            # =====================================================================
            # STANDARD PATH: Run planning and extraction phases
            # =====================================================================
            planner = self.factory.create_planner()
            logic_extractor = self.factory.create_logic_extractor()

            # Phase 0: Planning (for category distribution)
            analyze_turns = turns == "auto"
            plan = await self.plan_phase.execute(
                policy, traces, planner, analyze_turns=analyze_turns
            )
            self.reporter.on_plan_complete(plan)
            self._complete_phase()

            # Determine target turns
            if isinstance(turns, int):
                target_turns = turns
            else:
                target_turns = plan.recommended_turns

            # =====================================================================
            # STAGE 1: Logic Extraction (The Cartographer)
            # =====================================================================
            self._transition_phase(PipelinePhase.EXTRACTION, "Extracting rules...")

            if resuming and cm and cm.stage in ("logic_map", "scenarios", "traces", "complete"):
                logic_map = cm.get_logic_map()
                from rich.console import Console

                Console().print("[dim]📂 Loaded Logic Map from checkpoint[/dim]")
            else:
                with self.reporter.spinner("Extracting rules..."):
                    logic_map = await self.logic_extraction_phase.execute(policy, logic_extractor)
                if cm:
                    cm.save_logic_map(logic_map, policy_hash, traces, dataset_type)

            self.reporter.on_logic_map_complete(logic_map)
            self._state.set_artifact(logic_map=logic_map)
            self._complete_phase()

        # Reset grading LLM call counter after setup phases
        # (planner and logic extractor use grading_llm but aren't "grading" calls)
        self.factory.grading_llm.reset_call_count()

        # =====================================================================
        # STAGE 2: Scenario Synthesis (The Adversary)
        # =====================================================================
        self._transition_phase(PipelinePhase.SCENARIOS, "Generating scenarios...")

        # Track scenario generation calls
        scenario_calls_start = self.factory.generation_llm.call_count

        if resuming and cm and cm.stage in ("scenarios", "traces", "complete"):
            golden_scenarios = cm.get_scenarios()
            distribution = cm.load().scenario_distribution
            from rich.console import Console

            Console().print(
                f"[dim]📂 Loaded {len(golden_scenarios)} scenarios from checkpoint[/dim]"
            )
        else:
            with self.reporter.spinner("Generating scenarios..."):
                golden_scenarios, distribution = await self.golden_scenario_phase.execute(
                    policy, logic_map, plan, golden_scenario_gen, semaphore
                )
            if cm:
                cm.save_scenarios(golden_scenarios, distribution)

        scenario_calls = self.factory.generation_llm.call_count - scenario_calls_start
        self.reporter.on_golden_scenarios_complete(golden_scenarios, distribution)
        self._state.set_artifact(scenarios=golden_scenarios)
        self._complete_phase()

        # =====================================================================
        # COVERAGE TRACKING: Extract taxonomy and calculate coverage
        # =====================================================================
        coverage_report = None
        taxonomy = None
        coverage_calls_start = self.factory.generation_llm.call_count

        if self.skip_coverage:
            # Skip coverage tracking for faster generation
            pass
        else:
            self._transition_phase(PipelinePhase.COVERAGE, "Analyzing coverage...")
            try:
                taxonomy_extractor = self.factory.create_taxonomy_extractor()
                scenario_tagger = self.factory.create_scenario_tagger()
                coverage_calculator = self.factory.create_coverage_calculator()

                # Extract sub-category taxonomy from policy
                # Handle both Category objects and strings
                category_names = [
                    cat.name if hasattr(cat, "name") else str(cat) for cat in plan.categories
                ]
                with self.reporter.spinner("Extracting coverage taxonomy..."):
                    taxonomy = await taxonomy_extractor.extract(
                        policy.text,
                        logic_map,
                        category_names,
                    )

                if taxonomy and taxonomy.sub_categories:
                    self.reporter.on_taxonomy_extracted(taxonomy)

                    # Tag scenarios with sub-category IDs
                    with self.reporter.spinner("Tagging scenarios..."):
                        golden_scenarios = await scenario_tagger.tag(
                            golden_scenarios,
                            taxonomy,
                            logic_map,
                        )

                    # Calculate coverage report
                    with self.reporter.spinner("Calculating coverage..."):
                        coverage_report = await coverage_calculator.calculate(
                            golden_scenarios,
                            taxonomy,
                            generate_suggestions=True,
                        )

                    # Always report coverage (HITL will also display it in session)
                    self.reporter.on_coverage_calculated(coverage_report)

                self._complete_phase()
            except Exception as e:
                # Coverage tracking is optional - don't fail the whole pipeline
                # But log the error for debugging
                import sys

                print(f"[Coverage tracking error: {e}]", file=sys.stderr)
                self._complete_phase()

        coverage_calls = self.factory.generation_llm.call_count - coverage_calls_start

        # =====================================================================
        # HUMAN-IN-THE-LOOP: Unified Session (Turns + Rules + Scenarios)
        # =====================================================================
        # Track HITL calls separately (all HITL components use grading_llm)
        hitl_calls = 0
        if self.enable_hitl and self.hitl_editor:
            # Update phase to show we're ready for user input
            self._transition_phase(PipelinePhase.HITL, "Awaiting feedback")
            hitl_calls_start = self.factory.generation_llm.call_count
            (
                logic_map,
                golden_scenarios,
                distribution,
                target_turns,
                coverage_report,
            ) = await self._run_hitl_session(
                logic_map,
                golden_scenarios,
                distribution,
                policy,
                plan,
                target_turns,
                coverage_report=coverage_report,
                taxonomy=taxonomy,
            )
            hitl_calls = self.factory.generation_llm.call_count - hitl_calls_start
            # Reset grading_llm after HITL so only verification calls count as "grading"
            # (LogicMapEditor and ScenarioEditor use grading_llm but aren't grading)
            self.factory.grading_llm.reset_call_count()
            self._state.set_artifact(logic_map=logic_map, scenarios=golden_scenarios)
            self._complete_phase()

        # =====================================================================
        # STAGE 3: Trace Synthesis (The Thinker)
        # =====================================================================
        self._transition_phase(PipelinePhase.TRACES, "Synthesizing traces...")

        # Track response generation calls
        response_calls_start = self.factory.generation_llm.call_count

        if resuming and cm and cm.stage in ("traces", "complete"):
            # Resume from checkpoint - get already completed traces
            existing_traces = cm.get_traces()
            pending_indices = cm.get_pending_scenario_indices(len(golden_scenarios))

            if pending_indices:
                from rich.console import Console

                Console().print(
                    f"[dim]📂 Resuming: {len(existing_traces)} done, {len(pending_indices)} pending[/dim]"
                )

                # Generate only pending scenarios
                pending_scenarios = [golden_scenarios[i] for i in pending_indices]

                with self.reporter.spinner("Generating responses..."):
                    if is_tool_call and self.factory.has_tools:
                        new_traces = await self.golden_tool_call_phase.execute(
                            policy,
                            logic_map,
                            pending_scenarios,
                            golden_tool_call_gen,
                            semaphore,
                            target_turns,
                        )
                    else:
                        new_traces = await self.golden_trace_phase.execute(
                            policy,
                            logic_map,
                            pending_scenarios,
                            golden_response_gen,
                            semaphore,
                            target_turns,
                        )

                # Save new traces to checkpoint
                if cm:
                    cm.save_traces_batch(list(new_traces), pending_indices)

                all_traces = existing_traces + list(new_traces)
            else:
                all_traces = existing_traces
        else:
            with self.reporter.spinner("Generating responses..."):
                if is_tool_call and self.factory.has_tools:
                    all_traces = await self.golden_tool_call_phase.execute(
                        policy,
                        logic_map,
                        golden_scenarios,
                        golden_tool_call_gen,
                        semaphore,
                        target_turns,
                    )
                else:
                    all_traces = await self.golden_trace_phase.execute(
                        policy,
                        logic_map,
                        golden_scenarios,
                        golden_response_gen,
                        semaphore,
                        target_turns,
                    )

            # Save all traces to checkpoint
            if cm:
                cm.save_traces_batch(list(all_traces), list(range(len(all_traces))))

        response_calls = self.factory.generation_llm.call_count - response_calls_start
        self.reporter.on_responses_complete(list(all_traces))
        self._state.set_artifact(traces=list(all_traces))
        self._complete_phase()

        # =====================================================================
        # STAGE 4: Verification (The Auditor)
        # =====================================================================
        self._transition_phase(PipelinePhase.VERIFICATION, "Verifying traces...")

        pass_rate: float | None = None
        # Track refinement calls (GoldenRefiner uses generation_llm during verification)
        refinement_calls_start = self.factory.generation_llm.call_count

        if resuming and cm and cm.stage == "complete":
            final_traces = cm.get_verified_traces()
            passed_count = sum(1 for t in final_traces if t.grade and t.grade.passed)
            pass_rate = (passed_count / len(final_traces) * 100) if final_traces else 0
            from rich.console import Console

            Console().print(
                f"[dim]📂 Loaded {len(final_traces)} verified traces from checkpoint[/dim]"
            )
        elif self.skip_grading:
            final_traces = list(all_traces)
            self.reporter.on_grading_skipped()
        else:
            with self.reporter.spinner("Verifying responses..."):
                final_traces, pass_rate = await self.verification_phase.execute(
                    policy,
                    logic_map,
                    golden_scenarios,
                    list(all_traces),
                    verifier,
                    golden_refiner,
                    self.max_iterations,
                    semaphore,
                )
            if cm:
                cm.save_verified_traces(final_traces)

            self.reporter.on_grading_complete(final_traces, pass_rate)

        # Calculate refinement calls (generation_llm calls during verification phase)
        refinement_calls = self.factory.generation_llm.call_count - refinement_calls_start
        self._complete_phase()

        # Transition to complete state
        self._transition_phase(PipelinePhase.COMPLETE, "Pipeline complete")

        # Report completion with cost tracking
        elapsed = (datetime.now() - start_time).total_seconds()
        total_cost = self.factory.generation_llm.total_cost + self.factory.grading_llm.total_cost
        self.reporter.on_complete(
            len(final_traces),
            elapsed,
            pass_rate,
            total_cost=total_cost,
            generation_calls=self.factory.generation_llm.call_count,
            grading_calls=self.factory.grading_llm.call_count,
            scenario_calls=scenario_calls,
            response_calls=response_calls,
            refinement_calls=refinement_calls,
            hitl_calls=hitl_calls,
            coverage_calls=coverage_calls,
        )

        dataset = Dataset(traces=final_traces)

        if return_result:
            # Build stage-specific results for PipelineResult
            extraction_result = ExtractionResult(
                logic_map=logic_map,
                metrics=self._metrics.get_phase("extraction") or PhaseMetrics(phase="extraction"),
            )

            scenarios_result = ScenariosResultNew(
                scenarios=golden_scenarios,
                logic_map=logic_map,
                distribution=distribution,
                coverage_report=coverage_report,
                metrics=self._metrics.get_phase("scenarios") or PhaseMetrics(phase="scenarios"),
            )

            traces_result = TracesResult(
                traces=list(all_traces),
                logic_map=logic_map,
                scenarios=golden_scenarios,
                metrics=self._metrics.get_phase("traces") or PhaseMetrics(phase="traces"),
            )

            verification_result = VerificationResult(
                verified_traces=final_traces,
                pass_rate=pass_rate or 0.0,
                refinement_count=refinement_calls,
                refinement_history=[],
                metrics=self._metrics.get_phase("verification")
                or PhaseMetrics(phase="verification"),
            )

            # Return PipelineResult with all stage results
            return PipelineResult(
                dataset=dataset,
                metrics=self._metrics,
                extraction=extraction_result,
                scenarios=scenarios_result,
                traces=traces_result,
                verification=verification_result if not self.skip_grading else None,
            )

        return dataset

    async def run_scenarios_only(
        self,
        policy: Policy,
        count: int,
        model: str,
    ) -> ScenariosResult:
        """
        Run stages 0-2 only, returning scenarios without generating responses.

        This is the eval-focused pipeline that produces test scenarios with
        ground truth labels but no synthetic responses.

        Args:
            policy: The policy to generate scenarios from
            count: Target number of scenarios
            model: Model name (for reporting)

        Returns:
            ScenariosResult with scenarios, logic_map, and distribution
        """
        from datetime import datetime

        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.workers)

        # Report start (using a simplified message)
        self.reporter.on_start(count, model, "scenarios")

        # Set up live cost tracking if reporter supports it
        if hasattr(self.reporter, "set_cost_source"):
            self.reporter.set_cost_source(
                lambda: self.factory.generation_llm.total_cost + self.factory.grading_llm.total_cost
            )

        # Create components via factory
        planner = self.factory.create_planner()
        logic_extractor = self.factory.create_logic_extractor()
        golden_scenario_gen = self.factory.create_golden_scenario_generator()

        # Phase 0: Planning (for category distribution)
        plan = await self.plan_phase.execute(policy, count, planner, analyze_turns=False)
        self.reporter.on_plan_complete(plan)

        # =====================================================================
        # STAGE 1: Logic Extraction (The Cartographer)
        # =====================================================================
        with self.reporter.spinner("Extracting rules..."):
            logic_map = await self.logic_extraction_phase.execute(policy, logic_extractor)

        self.reporter.on_logic_map_complete(logic_map)

        # =====================================================================
        # STAGE 2: Scenario Synthesis (The Adversary)
        # =====================================================================
        with self.reporter.spinner("Generating scenarios..."):
            golden_scenarios, distribution = await self.golden_scenario_phase.execute(
                policy, logic_map, plan, golden_scenario_gen, semaphore
            )

        self.reporter.on_golden_scenarios_complete(golden_scenarios, distribution)

        # =====================================================================
        # HUMAN-IN-THE-LOOP (optional)
        # =====================================================================
        # Track HITL calls separately
        hitl_calls = 0
        scenario_calls = self.factory.generation_llm.call_count
        if self.enable_hitl and self.hitl_editor:
            hitl_calls_start = self.factory.generation_llm.call_count
            logic_map, golden_scenarios, distribution, _, _ = await self._run_hitl_session(
                logic_map, golden_scenarios, distribution, policy, plan, 1
            )
            hitl_calls = self.factory.generation_llm.call_count - hitl_calls_start

        # Report completion
        elapsed = (datetime.now() - start_time).total_seconds()
        # Include both LLM costs (grading_llm is used by planner, logic extractor, HITL editors)
        total_cost = self.factory.generation_llm.total_cost + self.factory.grading_llm.total_cost

        self.reporter.on_complete(
            len(golden_scenarios),
            elapsed,
            pass_rate=None,
            total_cost=total_cost,
            generation_calls=self.factory.generation_llm.call_count,
            grading_calls=0,  # No verification phase in scenarios_only
            scenario_calls=scenario_calls,
            response_calls=0,
            refinement_calls=0,
            hitl_calls=hitl_calls,
        )

        return ScenariosResult(
            scenarios=golden_scenarios,
            logic_map=logic_map,
            distribution=distribution,
        )

    async def _run_hitl_session(
        self,
        logic_map: LogicMap,
        scenarios: list["GoldenScenario"],
        distribution: dict[str, int],
        policy: Policy,
        plan: "Plan",
        initial_turns: int,
        coverage_report: "CoverageReport | None" = None,
        taxonomy: "SubCategoryTaxonomy | None" = None,
    ) -> tuple[LogicMap, list["GoldenScenario"], dict[str, int], int, "CoverageReport | None"]:
        """
        Run unified HITL session for turns, Logic Map, scenario, and coverage editing.

        Args:
            logic_map: The extracted Logic Map to edit
            scenarios: The generated scenarios to edit
            distribution: The scenario type distribution
            policy: The policy document (for context in refinements)
            plan: The generation plan (for complexity info)
            initial_turns: Initial target turns setting
            coverage_report: Coverage report (for display and increase commands)
            taxonomy: Sub-category taxonomy (for coverage improvement)

        Returns:
            Tuple of (modified LogicMap, modified scenarios, modified distribution, confirmed target_turns, updated coverage_report)
        """
        from synkro.interactive.hitl_session import HITLSession
        from synkro.interactive.intent_classifier import HITLIntentClassifier
        from synkro.interactive.rich_ui import InteractivePrompt, LogicMapDisplay

        session = HITLSession(original_logic_map=logic_map)
        session.set_scenarios(scenarios, distribution)

        # Get LiveProgressDisplay from reporter if available (for compact HITL mode)
        live_display = None
        if hasattr(self.reporter, "display") and self.reporter.display is not None:
            live_display = self.reporter.display
            # Enter HITL mode (pause live updates)
            live_display.enter_hitl_mode()

        # If no live_display available, create one for HITL
        if live_display is None:
            from synkro.interactive.live_display import LiveProgressDisplay

            live_display = LiveProgressDisplay()

        display = LogicMapDisplay(live_display=live_display)
        prompt = InteractivePrompt()
        classifier = HITLIntentClassifier(llm=self.factory.grading_llm)

        current_turns = initial_turns
        turns_history: list[int] = []  # For undo support
        complexity_level = plan.complexity_level if plan else "medium"

        # Always use live_display for HITL - it has proper screen clearing
        def get_hitl_input() -> str:
            return live_display.hitl_get_input(
                session.current_logic_map,
                session.current_scenarios or [],  # Pass empty list if None
                coverage_report,
                current_turns,
                complexity_level=complexity_level,
            )

        while True:
            feedback = get_hitl_input()

            # Handle explicit commands first (no LLM needed)
            if feedback.lower() == "done":
                break

            if feedback.lower() in ("exit", "quit"):
                import sys

                live_display.stop()
                self.console.print("[dim]Exiting...[/dim]")
                sys.exit(0)

            if feedback.lower() == "undo":
                # Undo turns, rules, or scenarios
                if session.can_undo or turns_history:
                    restored_map, restored_scenarios, restored_dist = session.undo()
                    if turns_history:
                        current_turns = turns_history.pop()
                    # State will be re-rendered on next loop iteration
                else:
                    display.show_error("Nothing to undo")
                continue

            if feedback.lower() == "reset":
                session.reset()
                current_turns = initial_turns
                turns_history.clear()
                # State will be re-rendered on next loop iteration
                continue

            if feedback.lower() == "help":
                prompt.show_unified_instructions()
                continue

            if feedback.lower().startswith("show ") or feedback.lower().startswith("find "):
                # Use enhanced command handling (supports show rules, show scenarios,
                # show gaps, show coverage, find, and single item views)
                handled = display.handle_show_command(
                    feedback,
                    session.current_logic_map,
                    session.current_scenarios,
                    coverage_report,
                )
                if not handled:
                    # Fallback to basic show for specific items
                    target = (
                        feedback.split()[1].strip().upper() if len(feedback.split()) > 1 else ""
                    )
                    if target.startswith("S") and session.current_scenarios:
                        display.display_scenario(target, session.current_scenarios)
                    elif target.startswith("R"):
                        display.display_rule(target, session.current_logic_map)
                continue

            # Empty input
            if not feedback:
                continue

            # Classify intent via LLM (wrap in try/except to handle errors gracefully)
            try:
                scenario_count = len(session.current_scenarios) if session.current_scenarios else 0
                history = session.get_history_for_prompt()

                # Build coverage summary for classifier context
                coverage_summary = "Not available"
                if coverage_report:
                    coverage_summary = (
                        f"{coverage_report.overall_coverage_percent:.0f}% overall, "
                        f"{coverage_report.covered_count} covered, "
                        f"{coverage_report.partial_count} partial, "
                        f"{coverage_report.uncovered_count} uncovered, "
                        f"{len(coverage_report.gaps)} gaps"
                    )

                with display.spinner("Processing..."):
                    intent = await classifier.classify(
                        feedback,
                        current_turns,
                        plan.complexity_level,
                        len(session.current_logic_map.rules),
                        scenario_count=scenario_count,
                        conversation_history=history,
                        coverage_summary=coverage_summary,
                    )
            except Exception as e:
                display.show_error(f"Failed to process: {e}")
                continue

            if intent.intent_type == "turns" and intent.target_turns is not None:
                # Handle turns change
                turns_history.append(current_turns)
                current_turns = intent.target_turns
                reasoning = intent.turns_reasoning or "User preference"
                summary = f"Set to {current_turns} turns ({reasoning})"
                session.record_feedback(feedback, "turns", summary)
                display.show_success(summary)

            elif intent.intent_type == "rules" and intent.rule_feedback:
                # Handle rule change
                try:
                    with display.spinner("Updating rules..."):
                        new_map, changes_summary = await self.hitl_editor.refine(
                            session.current_logic_map,
                            intent.rule_feedback,
                            policy.text,
                            conversation_history=history,
                        )

                        # Validate the refinement
                        is_valid, issues = self.hitl_editor.validate_refinement(
                            session.current_logic_map,
                            new_map,
                        )

                    if is_valid:
                        display.display_diff(session.current_logic_map, new_map)
                        session.apply_change(intent.rule_feedback, new_map)
                        display.show_success(changes_summary)
                        session.record_feedback(feedback, "rules", changes_summary)
                    else:
                        display.show_error(f"Invalid refinement: {', '.join(issues)}")

                except Exception as e:
                    display.show_error(f"Failed to apply refinement: {e}")

            elif intent.intent_type == "scenarios" and self.scenario_editor:
                # Handle scenario change
                try:
                    scenario_feedback = intent.scenario_feedback or feedback
                    with display.spinner("Updating scenarios..."):
                        (
                            new_scenarios,
                            new_dist,
                            changes_summary,
                        ) = await self.scenario_editor.refine(
                            session.current_scenarios or [],
                            session.current_distribution or {},
                            scenario_feedback,
                            policy.text,
                            session.current_logic_map,
                            conversation_history=history,
                        )

                        # Validate the scenarios
                        is_valid, issues = self.scenario_editor.validate_scenarios(
                            new_scenarios,
                            session.current_logic_map,
                        )

                    if is_valid:
                        if session.current_scenarios:
                            display.display_scenario_diff(session.current_scenarios, new_scenarios)
                        session.apply_scenario_change(scenario_feedback, new_scenarios, new_dist)
                        display.show_success(changes_summary)
                        session.record_feedback(feedback, "scenarios", changes_summary)
                    else:
                        display.show_error(f"Invalid scenario edit: {', '.join(issues)}")

                except Exception as e:
                    display.show_error(f"Failed to apply scenario edit: {e}")

            elif intent.intent_type == "scenarios" and not self.scenario_editor:
                display.show_error("Scenario editor not available")

            elif intent.intent_type == "coverage":
                # Handle coverage improvement commands
                if not coverage_report or not taxonomy:
                    display.show_error("Coverage data not available")
                    continue

                if intent.coverage_operation in ("increase", "target"):
                    try:
                        coverage_improver = self.factory.create_coverage_improver()
                        coverage_calculator = self.factory.create_coverage_calculator()

                        # Track scenarios generated
                        generated_scenarios: list = []
                        old_coverage = coverage_report

                        def on_scenario_generated(scenario: GoldenScenario) -> None:
                            """Callback to update session state as scenarios are generated."""
                            generated_scenarios.append(scenario)

                            # Update session state incrementally
                            current = session.current_scenarios or []
                            session.current_scenarios = current + [scenario]

                            t = scenario.scenario_type.value
                            dist = session.current_distribution or {}
                            dist[t] = dist.get(t, 0) + 1
                            session.current_distribution = dist

                        # Snapshot coverage for diff display
                        if live_display:
                            live_display.snapshot_coverage()

                        # Use spinner for progress - stays on same panel
                        # Use improve_from_intent() to skip redundant command parsing LLM call
                        with display.spinner("Improving coverage..."):
                            new_scenarios = await coverage_improver.improve_from_intent(
                                operation=intent.coverage_operation,
                                target_percent=intent.coverage_target_percent,
                                target_sub_category=intent.coverage_target_sub_category,
                                coverage_report=coverage_report,
                                taxonomy=taxonomy,
                                logic_map=session.current_logic_map,
                                policy_text=policy.text,
                                existing_scenarios=session.current_scenarios,
                                on_scenario_generated=on_scenario_generated,
                            )

                        if new_scenarios:
                            # Session already updated incrementally, just record history
                            all_scenarios = session.current_scenarios or []
                            new_dist = session.current_distribution or {}
                            session.apply_scenario_change(feedback, all_scenarios, new_dist)

                            # Recalculate coverage (skip suggestions LLM call - not needed for live updates)
                            with display.spinner("Recalculating coverage..."):
                                coverage_report = await coverage_calculator.calculate(
                                    all_scenarios,
                                    taxonomy,
                                    generate_suggestions=False,
                                )

                            # Update display state with new coverage and scenarios
                            if live_display:
                                live_display._state.scenarios_count = len(all_scenarios)
                                live_display._state.positive_count = new_dist.get("positive", 0)
                                live_display._state.negative_count = new_dist.get("negative", 0)
                                live_display._state.edge_count = new_dist.get("edge_case", 0)
                                live_display.update_coverage(coverage_report)

                            display.show_success(f"Added {len(new_scenarios)} coverage scenarios")
                            self.reporter.on_coverage_improved(
                                old_coverage,
                                coverage_report,
                                len(new_scenarios),
                            )
                            session.record_feedback(
                                feedback, "coverage", f"Added {len(new_scenarios)} scenarios"
                            )
                        else:
                            # Set error to prefill in input field (user can delete)
                            if live_display:
                                live_display.set_error(
                                    "Error: No scenarios generated for coverage improvement"
                                )

                    except Exception as e:
                        # Set error to prefill in input field (user can delete)
                        if live_display:
                            live_display.set_error(f"Error: {e}")

            elif intent.intent_type == "taxonomy":
                # Handle taxonomy (category/sub-category) management
                if not taxonomy:
                    display.show_error("Taxonomy data not available")
                    continue

                if intent.taxonomy_operation == "view":
                    # Show categories detail view
                    if live_display:
                        live_display.enter_detail_view("categories_detail")
                    continue

                # For add/modify/delete operations, use TaxonomyEditor
                if intent.taxonomy_operation in (
                    "add_category",
                    "add_subcategory",
                    "modify",
                    "delete",
                ):
                    try:
                        taxonomy_editor = self.factory.create_taxonomy_editor()
                        taxonomy_feedback = intent.taxonomy_feedback or feedback

                        # Snapshot for diff display
                        if live_display:
                            live_display.snapshot_coverage()

                        with display.spinner("Updating taxonomy..."):
                            new_taxonomy, changes_summary = await taxonomy_editor.refine(
                                taxonomy,
                                taxonomy_feedback,
                                policy.text,
                                session.current_logic_map,
                            )

                            # Validate the refinement
                            is_valid, issues = taxonomy_editor.validate_refinement(
                                taxonomy, new_taxonomy
                            )

                        if is_valid:
                            # Update taxonomy
                            taxonomy = new_taxonomy

                            # Recalculate coverage with new taxonomy
                            coverage_calculator = self.factory.create_coverage_calculator()
                            with display.spinner("Recalculating coverage..."):
                                coverage_report = await coverage_calculator.calculate(
                                    session.current_scenarios or [],
                                    taxonomy,
                                    generate_suggestions=True,
                                )

                            # Update display
                            if live_display:
                                live_display.update_coverage(coverage_report)

                            display.show_success(changes_summary)
                            session.record_feedback(feedback, "taxonomy", changes_summary)
                        else:
                            display.show_error(f"Invalid taxonomy change: {', '.join(issues)}")

                    except Exception as e:
                        if live_display:
                            live_display.set_error(f"Error: {e}")

            elif (
                intent.intent_type == "compound"
                and intent.rule_feedback
                and intent.scenario_feedback
            ):
                # Handle compound intent: rules first, then scenarios
                try:
                    # Step 1: Apply rule changes
                    with display.spinner("Updating rules..."):
                        new_map, rule_summary = await self.hitl_editor.refine(
                            session.current_logic_map,
                            intent.rule_feedback,
                            policy.text,
                            conversation_history=history,
                        )

                        is_valid, issues = self.hitl_editor.validate_refinement(
                            session.current_logic_map,
                            new_map,
                        )

                    if not is_valid:
                        display.show_error(f"Invalid rule change: {', '.join(issues)}")
                        continue

                    # Show rule diff and apply
                    display.display_diff(session.current_logic_map, new_map)
                    session.apply_change(intent.rule_feedback, new_map)
                    display.show_success(rule_summary)
                    session.record_feedback(feedback, "rules", rule_summary)

                    # Step 2: Apply scenario changes (using updated logic map)
                    # Get updated history after rule change
                    updated_history = session.get_history_for_prompt()
                    if self.scenario_editor:
                        with display.spinner("Updating scenarios..."):
                            (
                                new_scenarios,
                                new_dist,
                                scenario_summary,
                            ) = await self.scenario_editor.refine(
                                session.current_scenarios or [],
                                session.current_distribution or {},
                                intent.scenario_feedback,
                                policy.text,
                                session.current_logic_map,  # Now has the new rules
                                conversation_history=updated_history,
                            )

                            is_valid, issues = self.scenario_editor.validate_scenarios(
                                new_scenarios,
                                session.current_logic_map,
                            )

                        if is_valid:
                            if session.current_scenarios:
                                display.display_scenario_diff(
                                    session.current_scenarios, new_scenarios
                                )
                            session.apply_scenario_change(
                                intent.scenario_feedback, new_scenarios, new_dist
                            )
                            display.show_success(scenario_summary)
                            session.record_feedback(feedback, "scenarios", scenario_summary)
                        else:
                            display.show_error(f"Invalid scenario edit: {', '.join(issues)}")
                    else:
                        display.show_error("Scenario editor not available for compound operation")

                except Exception as e:
                    display.show_error(f"Failed to apply compound change: {e}")

            elif intent.intent_type == "unclear":
                display.show_error("Could not understand feedback. Try 'help' for examples.")

        # Exit HITL mode (resume live display seamlessly)
        if live_display:
            live_display.exit_hitl_mode()

        return (
            session.current_logic_map,
            session.current_scenarios or scenarios,
            session.current_distribution or distribution,
            current_turns,
            coverage_report,
        )


__all__ = ["GenerationPipeline", "GenerationResult", "ScenariosResult"]
