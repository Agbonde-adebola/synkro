"""Pipeline runner that orchestrates all phases.

Uses the Golden Trace 4-stage pipeline for all dataset types:
1. Logic Extraction (The Cartographer)
2. Scenario Synthesis (The Adversary)
3. Trace Synthesis (The Thinker)
4. Verification (The Auditor)
"""

import asyncio
from datetime import datetime

from synkro.core.policy import Policy
from synkro.core.dataset import Dataset
from synkro.core.checkpoint import CheckpointManager, hash_policy
from synkro.factory import ComponentFactory
from synkro.reporting import ProgressReporter
from synkro.pipeline.phases import (
    PlanPhase,
    LogicExtractionPhase,
    GoldenScenarioPhase,
    GoldenTracePhase,
    GoldenToolCallPhase,
    VerificationPhase,
)
from synkro.types.logic_map import LogicMap

# Type hints for HITL components (imported dynamically to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from synkro.interactive.logic_map_editor import LogicMapEditor
    from synkro.interactive.scenario_editor import ScenarioEditor
    from synkro.types.core import Plan
    from synkro.types.logic_map import GoldenScenario


class GenerationResult:
    """
    Result of the generation pipeline.

    Provides access to both the dataset and internal artifacts like the Logic Map.

    Examples:
        >>> result = await pipeline.run(policy, traces=50, ...)
        >>> dataset = result.dataset
        >>> logic_map = result.logic_map  # Inspect extracted rules
    """

    def __init__(
        self,
        dataset: "Dataset",
        logic_map: LogicMap | None = None,
        scenarios: list | None = None,
        distribution: dict[str, int] | None = None,
    ):
        self.dataset = dataset
        self.logic_map = logic_map
        self.scenarios = scenarios or []
        self.distribution = distribution or {}

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
                scenario_type=s.scenario_type.value if hasattr(s.scenario_type, 'value') else s.scenario_type,
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
        """
        self.factory = factory
        self.reporter = reporter
        self.workers = workers
        self.max_iterations = max_iterations
        self.skip_grading = skip_grading
        self.checkpoint_manager = checkpoint_manager
        self.enable_hitl = enable_hitl
        self.hitl_editor = hitl_editor
        self.scenario_editor = scenario_editor

        # Golden Trace phases
        self.plan_phase = PlanPhase()
        self.logic_extraction_phase = LogicExtractionPhase()
        self.golden_scenario_phase = GoldenScenarioPhase()
        self.golden_trace_phase = GoldenTracePhase()
        self.golden_tool_call_phase = GoldenToolCallPhase()
        self.verification_phase = VerificationPhase()

    async def run(
        self,
        policy: Policy,
        traces: int,
        model: str,
        dataset_type: str,
        turns: int | str = "auto",
        return_result: bool = False,
    ) -> Dataset | GenerationResult:
        """
        Run the Golden Trace generation pipeline.

        All dataset types use the same 4-stage pipeline, with Stage 3
        branching based on whether TOOL_CALL is needed.

        Args:
            policy: The policy to generate from
            traces: Target number of traces
            model: Model name (for reporting)
            dataset_type: Dataset type (sft, qa, tool_call)
            turns: Conversation turns per trace. Use int for fixed turns, or "auto"
                for policy complexity-driven turns
            return_result: If True, return GenerationResult with logic_map access

        Returns:
            Dataset (default) or GenerationResult if return_result=True
        """
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.workers)

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

        # Create components via factory
        planner = self.factory.create_planner()
        logic_extractor = self.factory.create_logic_extractor()
        golden_scenario_gen = self.factory.create_golden_scenario_generator()
        verifier = self.factory.create_verifier()
        golden_refiner = self.factory.create_golden_refiner()

        # Create appropriate trace generator based on dataset type
        if is_tool_call and self.factory.has_tools:
            golden_tool_call_gen = self.factory.create_golden_tool_call_generator()
        else:
            golden_response_gen = self.factory.create_golden_response_generator()

        # Phase 0: Planning (for category distribution)
        analyze_turns = turns == "auto"
        plan = await self.plan_phase.execute(policy, traces, planner, analyze_turns=analyze_turns)
        self.reporter.on_plan_complete(plan)

        # Determine target turns
        if isinstance(turns, int):
            target_turns = turns
        else:
            target_turns = plan.recommended_turns

        # =====================================================================
        # STAGE 1: Logic Extraction (The Cartographer)
        # =====================================================================
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

        # Reset grading LLM call counter after setup phases
        # (planner and logic extractor use grading_llm but aren't "grading" calls)
        self.factory.grading_llm.reset_tracking()

        # =====================================================================
        # STAGE 2: Scenario Synthesis (The Adversary)
        # =====================================================================
        # Track scenario generation calls
        scenario_calls_start = self.factory.generation_llm.call_count

        if resuming and cm and cm.stage in ("scenarios", "traces", "complete"):
            golden_scenarios = cm.get_scenarios()
            distribution = cm.load().scenario_distribution
            from rich.console import Console
            Console().print(f"[dim]📂 Loaded {len(golden_scenarios)} scenarios from checkpoint[/dim]")
        else:
            with self.reporter.spinner("Generating scenarios..."):
                golden_scenarios, distribution = await self.golden_scenario_phase.execute(
                    policy, logic_map, plan, golden_scenario_gen, semaphore
                )
            if cm:
                cm.save_scenarios(golden_scenarios, distribution)

        scenario_calls = self.factory.generation_llm.call_count - scenario_calls_start
        self.reporter.on_golden_scenarios_complete(golden_scenarios, distribution)

        # =====================================================================
        # HUMAN-IN-THE-LOOP: Unified Session (Turns + Rules + Scenarios)
        # =====================================================================
        if self.enable_hitl and self.hitl_editor:
            logic_map, golden_scenarios, distribution, target_turns = await self._run_hitl_session(
                logic_map, golden_scenarios, distribution, policy, plan, target_turns
            )

        # =====================================================================
        # STAGE 3: Trace Synthesis (The Thinker)
        # =====================================================================
        # Track response generation calls
        response_calls_start = self.factory.generation_llm.call_count

        if resuming and cm and cm.stage in ("traces", "complete"):
            # Resume from checkpoint - get already completed traces
            existing_traces = cm.get_traces()
            pending_indices = cm.get_pending_scenario_indices(len(golden_scenarios))

            if pending_indices:
                from rich.console import Console
                Console().print(f"[dim]📂 Resuming: {len(existing_traces)} done, {len(pending_indices)} pending[/dim]")

                # Generate only pending scenarios
                pending_scenarios = [golden_scenarios[i] for i in pending_indices]

                with self.reporter.spinner("Generating responses..."):
                    if is_tool_call and self.factory.has_tools:
                        new_traces = await self.golden_tool_call_phase.execute(
                            policy, logic_map, pending_scenarios, golden_tool_call_gen, semaphore, target_turns
                        )
                    else:
                        new_traces = await self.golden_trace_phase.execute(
                            policy, logic_map, pending_scenarios, golden_response_gen, semaphore, target_turns
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
                        policy, logic_map, golden_scenarios, golden_tool_call_gen, semaphore, target_turns
                    )
                else:
                    all_traces = await self.golden_trace_phase.execute(
                        policy, logic_map, golden_scenarios, golden_response_gen, semaphore, target_turns
                    )

            # Save all traces to checkpoint
            if cm:
                cm.save_traces_batch(list(all_traces), list(range(len(all_traces))))

        response_calls = self.factory.generation_llm.call_count - response_calls_start
        self.reporter.on_responses_complete(list(all_traces))

        # =====================================================================
        # STAGE 4: Verification (The Auditor)
        # =====================================================================
        pass_rate: float | None = None

        if resuming and cm and cm.stage == "complete":
            final_traces = cm.get_verified_traces()
            passed_count = sum(1 for t in final_traces if t.grade and t.grade.passed)
            pass_rate = (passed_count / len(final_traces) * 100) if final_traces else 0
            from rich.console import Console
            Console().print(f"[dim]📂 Loaded {len(final_traces)} verified traces from checkpoint[/dim]")
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

        # Report completion with cost tracking
        elapsed = (datetime.now() - start_time).total_seconds()
        total_cost = (
            self.factory.generation_llm.total_cost +
            self.factory.grading_llm.total_cost
        )
        self.reporter.on_complete(
            len(final_traces),
            elapsed,
            pass_rate,
            total_cost=total_cost,
            generation_calls=self.factory.generation_llm.call_count,
            grading_calls=self.factory.grading_llm.call_count,
            scenario_calls=scenario_calls,
            response_calls=response_calls,
        )

        dataset = Dataset(traces=final_traces)

        if return_result:
            return GenerationResult(
                dataset=dataset,
                logic_map=logic_map,
                scenarios=golden_scenarios,
                distribution=distribution,
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
        if self.enable_hitl and self.hitl_editor:
            logic_map, golden_scenarios, distribution, _ = await self._run_hitl_session(
                logic_map, golden_scenarios, distribution, policy, plan, 1
            )

        # Report completion
        elapsed = (datetime.now() - start_time).total_seconds()
        total_cost = self.factory.generation_llm.total_cost

        self.reporter.on_complete(
            len(golden_scenarios),
            elapsed,
            pass_rate=None,
            total_cost=total_cost,
            generation_calls=self.factory.generation_llm.call_count,
            grading_calls=0,
            scenario_calls=self.factory.generation_llm.call_count,
            response_calls=0,
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
    ) -> tuple[LogicMap, list["GoldenScenario"], dict[str, int], int]:
        """
        Run unified HITL session for turns, Logic Map, and scenario editing.

        Args:
            logic_map: The extracted Logic Map to edit
            scenarios: The generated scenarios to edit
            distribution: The scenario type distribution
            policy: The policy document (for context in refinements)
            plan: The generation plan (for complexity info)
            initial_turns: Initial target turns setting

        Returns:
            Tuple of (modified LogicMap, modified scenarios, modified distribution, confirmed target_turns)
        """
        from synkro.interactive.hitl_session import HITLSession
        from synkro.interactive.rich_ui import LogicMapDisplay, InteractivePrompt
        from synkro.interactive.intent_classifier import HITLIntentClassifier

        session = HITLSession(original_logic_map=logic_map)
        session.set_scenarios(scenarios, distribution)

        display = LogicMapDisplay()
        prompt = InteractivePrompt()
        classifier = HITLIntentClassifier(llm=self.factory.generation_llm)

        current_turns = initial_turns
        turns_history: list[int] = []  # For undo support

        # Show initial state (includes session details with instructions)
        display.display_full_session_state(
            plan,
            session.current_logic_map,
            current_turns,
            session.current_scenarios,
            session.current_distribution,
        )

        while True:
            feedback = prompt.get_feedback().strip()

            # Handle explicit commands first (no LLM needed)
            if feedback.lower() == "done":
                break

            if feedback.lower() == "undo":
                # Undo turns, rules, or scenarios
                if session.can_undo or turns_history:
                    restored_map, restored_scenarios, restored_dist = session.undo()
                    if turns_history:
                        current_turns = turns_history.pop()
                    display.show_success("Reverted to previous state")
                    display.display_full_session_state(
                        plan,
                        session.current_logic_map,
                        current_turns,
                        session.current_scenarios,
                        session.current_distribution,
                    )
                else:
                    display.show_error("Nothing to undo")
                continue

            if feedback.lower() == "reset":
                session.reset()
                current_turns = initial_turns
                turns_history.clear()
                display.show_success("Reset to original state")
                display.display_full_session_state(
                    plan,
                    session.current_logic_map,
                    current_turns,
                    session.current_scenarios,
                    session.current_distribution,
                )
                continue

            if feedback.lower() == "help":
                prompt.show_unified_instructions()
                continue

            if feedback.lower().startswith("show "):
                target = feedback[5:].strip().upper()
                if target.startswith("S") and target[1:].isdigit():
                    # Show scenario
                    if session.current_scenarios:
                        display.display_scenario(target, session.current_scenarios)
                else:
                    # Show rule
                    display.display_rule(target, session.current_logic_map)
                continue

            # Empty input
            if not feedback:
                continue

            # Classify intent via LLM
            scenario_count = len(session.current_scenarios) if session.current_scenarios else 0
            history = session.get_history_for_prompt()
            with display.spinner("Processing..."):
                intent = await classifier.classify(
                    feedback,
                    current_turns,
                    plan.complexity_level,
                    len(session.current_logic_map.rules),
                    scenario_count=scenario_count,
                    conversation_history=history,
                )

            if intent.intent_type == "turns" and intent.target_turns is not None:
                # Handle turns change
                turns_history.append(current_turns)
                current_turns = intent.target_turns
                reasoning = intent.turns_reasoning or "User preference"
                summary = f"Set to {current_turns} turns ({reasoning})"
                display.show_success(summary)
                session.record_feedback(feedback, "turns", summary)
                display.display_full_session_state(
                    plan,
                    session.current_logic_map,
                    current_turns,
                    session.current_scenarios,
                    session.current_distribution,
                )

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
                        new_scenarios, new_dist, changes_summary = await self.scenario_editor.refine(
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

            elif intent.intent_type == "compound" and intent.rule_feedback and intent.scenario_feedback:
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
                            new_scenarios, new_dist, scenario_summary = await self.scenario_editor.refine(
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
                                display.display_scenario_diff(session.current_scenarios, new_scenarios)
                            session.apply_scenario_change(intent.scenario_feedback, new_scenarios, new_dist)
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

        # Final summary
        display.console.print(
            f"\n[green]✅ Session complete[/green] - "
            f"{session.rule_change_count} rule change(s), "
            f"{session.scenario_change_count} scenario change(s), "
            f"{current_turns} turns"
        )

        return (
            session.current_logic_map,
            session.current_scenarios or scenarios,
            session.current_distribution or distribution,
            current_turns,
        )


__all__ = ["GenerationPipeline", "GenerationResult", "ScenariosResult"]
