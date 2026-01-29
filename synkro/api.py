"""Step-by-step API functions for granular control over dataset generation.

These functions provide individual access to each pipeline stage, enabling:
- Inspection and editing of intermediate results
- Custom workflows with selective stage execution
- Integration with external systems at each stage

Also provides streaming variants that yield events for real-time progress.

Examples:
    >>> # Step-by-step usage
    >>> extraction = await synkro.extract_rules(policy)
    >>> scenarios = await synkro.generate_scenarios(policy, extraction.logic_map)
    >>> traces = await synkro.synthesize_traces(policy, scenarios)
    >>> verified = await synkro.verify_traces(policy, traces)

    >>> # Streaming usage
    >>> async for event in synkro.extract_rules_stream(policy):
    ...     if event.type == "rule_found":
    ...         print(f"Found: {event.rule.rule_id}")
    ...     elif event.type == "complete":
    ...         logic_map = event.result.logic_map
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

from synkro.core.policy import Policy
from synkro.llm.client import LLM
from synkro.types.events import (
    CompleteEvent,
    ErrorEvent,
    Event,
    ProgressEvent,
    RefinementStartedEvent,
    RuleFoundEvent,
    ScenarioGeneratedEvent,
    TraceGeneratedEvent,
    TraceRefinedEvent,
    TraceVerifiedEvent,
)
from synkro.types.metrics import PhaseMetrics
from synkro.types.results import (
    ExtractionResult,
    ScenariosResult,
    TracesResult,
    VerificationResult,
)
from synkro.utils.model_detection import (
    get_default_grading_model,
    get_default_model,
    get_default_models,
)

if TYPE_CHECKING:
    from synkro.types.core import Trace
    from synkro.types.logic_map import GoldenScenario, LogicMap
    from synkro.types.tool import ToolDefinition


async def extract_rules_async(
    policy: str | Policy,
    model: str | None = None,
    base_url: str | None = None,
) -> ExtractionResult:
    """
    Extract rules from a policy document as a Logic Map (DAG).

    This is Stage 1 of the pipeline - "The Cartographer" extracts rules
    and their dependencies from the policy text.

    Args:
        policy: Policy text or Policy object
        model: Model to use for extraction (auto-detected if not specified)
        base_url: Optional API base URL for local providers

    Returns:
        ExtractionResult containing the Logic Map and metrics

    Examples:
        >>> result = await synkro.extract_rules("Expenses over $50 require approval")
        >>> print(f"Found {len(result.logic_map.rules)} rules")
        >>> for rule in result.logic_map.rules:
        ...     print(f"{rule.rule_id}: {rule.text}")
    """
    from synkro.generation.logic_extractor import LogicExtractor

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Auto-detect model if not specified
    if model is None:
        model = get_default_grading_model()

    # Create metrics tracker
    metrics = PhaseMetrics(phase="extraction", model=model)
    metrics.start()

    # Create extractor with specified model
    llm = LLM(model=model, base_url=base_url, temperature=0.1)
    extractor = LogicExtractor(llm=llm)

    # Extract rules
    logic_map = await extractor.extract(policy.text)

    # Record metrics
    metrics.cost = llm.total_cost
    metrics.calls = llm.call_count
    metrics.complete()

    return ExtractionResult(logic_map=logic_map, metrics=metrics)


def extract_rules(
    policy: str | Policy,
    model: str | None = None,
    base_url: str | None = None,
) -> ExtractionResult:
    """
    Extract rules from a policy document (sync wrapper).

    See extract_rules_async for full documentation.
    """
    return asyncio.run(extract_rules_async(policy, model, base_url))


async def generate_scenarios_async(
    policy: str | Policy,
    logic_map: "LogicMap | None" = None,
    count: int = 20,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.8,
) -> ScenariosResult:
    """
    Generate test scenarios from a policy and Logic Map.

    This is Stage 2 of the pipeline - "The Adversary" generates diverse
    scenarios (positive, negative, edge_case, irrelevant) that target
    specific rules.

    Args:
        policy: Policy text or Policy object
        logic_map: Pre-extracted Logic Map (will extract if not provided)
        count: Number of scenarios to generate (default: 20)
        model: Model to use for generation (default: gpt-4o-mini)
        base_url: Optional API base URL for local providers
        temperature: Sampling temperature (default: 0.8 for diversity)

    Returns:
        ScenariosResult containing scenarios, distribution, and metrics

    Examples:
        >>> # With pre-extracted Logic Map
        >>> extraction = await synkro.extract_rules(policy)
        >>> scenarios = await synkro.generate_scenarios(policy, extraction.logic_map, count=50)
        >>> print(f"Generated {len(scenarios)} scenarios")
        >>> print(f"Distribution: {scenarios.distribution}")

        >>> # Without Logic Map (will extract automatically)
        >>> scenarios = await synkro.generate_scenarios(policy, count=50)
    """
    from synkro.generation.golden_scenarios import GoldenScenarioGenerator
    from synkro.generation.planner import Planner

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Auto-detect models if not specified
    gen_model, grading_model = get_default_models()
    if model is None:
        model = gen_model

    # Create metrics tracker
    metrics = PhaseMetrics(phase="scenarios", model=model)
    metrics.start()

    # Extract logic map if not provided
    if logic_map is None:
        extraction = await extract_rules_async(policy, model=grading_model, base_url=base_url)
        logic_map = extraction.logic_map

    # Create planner for category distribution
    planner_llm = LLM(model=grading_model, base_url=base_url, temperature=0.3)
    planner = Planner(llm=planner_llm)
    plan = await planner.plan(policy, count, analyze_turns=False)

    # Create scenario generator
    llm = LLM(model=model, base_url=base_url, temperature=temperature)
    generator = GoldenScenarioGenerator(llm=llm)

    # Generate scenarios for each category
    semaphore = asyncio.Semaphore(10)

    async def limited_generate(category):
        async with semaphore:
            return await generator.generate(policy.text, logic_map, category, category.count)

    tasks = [limited_generate(cat) for cat in plan.categories]
    results = await asyncio.gather(*tasks)

    # Flatten scenarios
    scenarios = [s for batch in results for s in batch]

    # Calculate distribution
    distribution = {"positive": 0, "negative": 0, "edge_case": 0, "irrelevant": 0}
    for s in scenarios:
        distribution[s.scenario_type.value] += 1

    # Record metrics
    metrics.cost = llm.total_cost + planner_llm.total_cost
    metrics.calls = llm.call_count + planner_llm.call_count
    metrics.complete()

    return ScenariosResult(
        scenarios=scenarios,
        logic_map=logic_map,
        distribution=distribution,
        coverage_report=None,
        metrics=metrics,
    )


async def synthesize_traces_async(
    policy: str | Policy,
    scenarios: ScenariosResult | list["GoldenScenario"],
    logic_map: "LogicMap | None" = None,
    turns: int = 1,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    dataset_type: str = "conversation",
    tools: list["ToolDefinition"] | None = None,
    concurrency: int = 50,
) -> TracesResult:
    """
    Synthesize conversation traces from scenarios.

    This is Stage 3 of the pipeline - "The Thinker" generates responses
    with grounded Chain-of-Thought reasoning and rule citations.

    Args:
        policy: Policy text or Policy object
        scenarios: ScenariosResult or list of GoldenScenario objects
        logic_map: Logic Map (will extract from scenarios if ScenariosResult provided)
        turns: Number of conversation turns (default: 1, forced to 1 for instruction/evaluation)
        model: Model to use for generation (default: gpt-4o-mini)
        base_url: Optional API base URL for local providers
        temperature: Sampling temperature (default: 0.7)
        dataset_type: Type of dataset (conversation, instruction, evaluation, tool_call)
        tools: List of ToolDefinition for TOOL_CALL dataset type
        concurrency: Maximum concurrent LLM calls (default: 50)

    Returns:
        TracesResult containing traces and metrics

    Examples:
        >>> scenarios = await synkro.generate_scenarios(policy, count=20)
        >>> traces = await synkro.synthesize_traces(policy, scenarios, turns=2)
        >>> print(f"Generated {len(traces)} traces")
        >>> for trace in traces:
        ...     print(trace.assistant_message[:50])
    """
    # Validate TOOL_CALL requires tools (matching Generator validation)
    if dataset_type == "tool_call" and not tools:
        raise ValueError("TOOL_CALL dataset type requires tools parameter")

    # Force turns=1 for instruction and evaluation (matching Generator behavior)
    if dataset_type in ("instruction", "evaluation"):
        turns = 1

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Extract scenarios list and logic map
    if isinstance(scenarios, ScenariosResult):
        scenario_list = scenarios.scenarios
        if logic_map is None:
            logic_map = scenarios.logic_map
    else:
        scenario_list = scenarios
        if logic_map is None:
            raise ValueError("logic_map is required when scenarios is a list")

    # Auto-detect model if not specified
    if model is None:
        model = get_default_model()

    # Create metrics tracker
    metrics = PhaseMetrics(phase="traces", model=model)
    metrics.start()

    # Create LLM client
    llm = LLM(model=model, base_url=base_url, temperature=temperature)

    # Route to correct generator based on dataset_type (matching factory.py pattern)
    if dataset_type == "tool_call":
        from synkro.generation.golden_tool_responses import GoldenToolCallResponseGenerator
        from synkro.generation.tool_simulator import ToolSimulator

        simulator = ToolSimulator(tools=tools, llm=llm)
        generator = GoldenToolCallResponseGenerator(
            tools=tools,
            llm=llm,
            simulator=simulator,
        )
    else:
        from synkro.generation.golden_responses import GoldenResponseGenerator

        generator = GoldenResponseGenerator(llm=llm)

    # Generate traces with concurrency control
    traces = await generator.generate(policy.text, logic_map, scenario_list, turns, concurrency)

    # Record metrics
    metrics.cost = llm.total_cost
    metrics.calls = llm.call_count
    metrics.complete()

    return TracesResult(
        traces=list(traces),
        logic_map=logic_map,
        scenarios=scenario_list,
        metrics=metrics,
    )


def synthesize_traces(
    policy: str | Policy,
    scenarios: ScenariosResult | list["GoldenScenario"],
    logic_map: "LogicMap | None" = None,
    turns: int = 1,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    dataset_type: str = "conversation",
    tools: list["ToolDefinition"] | None = None,
    concurrency: int = 50,
) -> TracesResult:
    """
    Synthesize conversation traces (sync wrapper).

    See synthesize_traces_async for full documentation.
    """
    return asyncio.run(
        synthesize_traces_async(
            policy,
            scenarios,
            logic_map,
            turns,
            model,
            base_url,
            temperature,
            dataset_type,
            tools,
            concurrency,
        )
    )


async def verify_traces_async(
    policy: str | Policy,
    traces: TracesResult | list["Trace"],
    logic_map: "LogicMap | None" = None,
    scenarios: list["GoldenScenario"] | None = None,
    max_iterations: int = 3,
    model: str | None = None,
    refine_model: str | None = None,
    base_url: str | None = None,
    dataset_type: str = "conversation",
    tools: list["ToolDefinition"] | None = None,
    concurrency: int = 50,
) -> VerificationResult:
    """
    Verify and refine traces against the Logic Map.

    This is Stage 4 of the pipeline - "The Auditor" checks traces for:
    - Skipped rules (rules that should have been applied)
    - Hallucinated rules (rules cited that don't apply)
    - Contradictions in reasoning

    Failed traces are refined up to max_iterations times.

    Args:
        policy: Policy text or Policy object
        traces: TracesResult or list of Trace objects
        logic_map: Logic Map for verification (extracted from TracesResult if provided)
        scenarios: Scenarios for verification context (extracted from TracesResult if provided)
        max_iterations: Maximum refinement attempts per trace (default: 3)
        model: Model for verification (default: gpt-4o, stronger is better)
        refine_model: Model for refinement (default: gpt-4o-mini)
        base_url: Optional API base URL for local providers
        dataset_type: Type of dataset (conversation, instruction, evaluation, tool_call)
        tools: List of ToolDefinition for TOOL_CALL dataset type
        concurrency: Maximum concurrent LLM calls (default: 50)

    Returns:
        VerificationResult with verified traces, pass rate, and refinement history

    Examples:
        >>> traces = await synkro.synthesize_traces(policy, scenarios)
        >>> verified = await synkro.verify_traces(policy, traces, max_iterations=2)
        >>> print(f"Pass rate: {verified.pass_rate:.1%}")
        >>> print(f"Refined {verified.refinement_count} traces")
    """
    from synkro.quality.golden_refiner import GoldenRefiner
    from synkro.quality.verifier import TraceVerifier

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Extract trace list and context
    if isinstance(traces, TracesResult):
        trace_list = list(traces.traces)
        if logic_map is None:
            logic_map = traces.logic_map
        if scenarios is None:
            scenarios = traces.scenarios
    else:
        trace_list = list(traces)
        if logic_map is None:
            raise ValueError("logic_map is required when traces is a list")
        if scenarios is None:
            raise ValueError("scenarios is required when traces is a list")

    # Auto-detect models if not specified
    gen_model, grading_model = get_default_models()
    if model is None:
        model = grading_model
    if refine_model is None:
        refine_model = gen_model

    # Create metrics tracker
    metrics = PhaseMetrics(phase="verification", model=model)
    metrics.start()

    # Create verifier and refiner - route based on dataset_type
    verify_llm = LLM(model=model, base_url=base_url, temperature=0.1)
    refine_llm = LLM(model=refine_model, base_url=base_url, temperature=0.5)

    if dataset_type == "tool_call" and tools:
        from synkro.quality.tool_grader import ToolCallGrader
        from synkro.quality.tool_refiner import ToolCallRefiner

        verifier = ToolCallGrader(llm=verify_llm, tools=tools)
        refiner = ToolCallRefiner(llm=refine_llm, tools=tools)
    else:
        verifier = TraceVerifier(llm=verify_llm)
        refiner = GoldenRefiner(llm=refine_llm)

    # Concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def verify_one(trace, scenario):
        """Verify a single trace with concurrency control."""
        async with semaphore:
            return await verifier.verify(trace, logic_map, scenario, policy.text)

    async def refine_one(trace, scenario, result):
        """Refine a single trace with concurrency control."""
        async with semaphore:
            return await refiner.refine(trace, logic_map, scenario, result)

    # ==========================================================================
    # BATCH REFINEMENT WAVES - Maximum I/O parallelism
    # ==========================================================================

    # Wave 0: Verify ALL traces in parallel
    verify_tasks = [verify_one(t, s) for t, s in zip(trace_list, scenarios)]
    verify_results = await asyncio.gather(*verify_tasks)

    # Separate passed and failed
    verified_traces = [None] * len(trace_list)  # Preserve order
    failed_indices = []  # (index, trace, scenario, result)
    refinement_history = []

    for i, (trace, scenario, result) in enumerate(zip(trace_list, scenarios, verify_results)):
        if result.passed:
            trace.grade = type("GradeResult", (), {"passed": True, "issues": [], "feedback": ""})()
            verified_traces[i] = trace
        else:
            failed_indices.append((i, trace, scenario, result))

    # Refinement waves: refine ALL failed in parallel, then verify ALL in parallel
    for iteration in range(max_iterations):
        if not failed_indices:
            break  # All passed

        # Refine ALL failed traces in parallel
        refine_tasks = [
            refine_one(trace, scenario, result) for (_, trace, scenario, result) in failed_indices
        ]
        refined_traces = await asyncio.gather(*refine_tasks)

        # Verify ALL refined traces in parallel
        verify_tasks = [
            verify_one(refined, scenario)
            for refined, (_, _, scenario, _) in zip(refined_traces, failed_indices)
        ]
        new_results = await asyncio.gather(*verify_tasks)

        # Update state: separate newly passed from still-failed
        still_failed = []
        for (i, _, scenario, _), refined, result in zip(
            failed_indices, refined_traces, new_results
        ):
            if result.passed:
                refined.grade = type(
                    "GradeResult", (), {"passed": True, "issues": [], "feedback": ""}
                )()
                verified_traces[i] = refined
                refinement_history.append(
                    {"trace_index": i, "iteration": iteration + 1, "success": True}
                )
            else:
                still_failed.append((i, refined, scenario, result))

        failed_indices = still_failed

    # Mark remaining failures
    for i, trace, _, result in failed_indices:
        trace.grade = type(
            "GradeResult",
            (),
            {"passed": False, "issues": result.issues, "feedback": "; ".join(result.issues)},
        )()
        verified_traces[i] = trace
        refinement_history.append({"trace_index": i, "iteration": max_iterations, "success": False})

    # Count successful refinements
    refinement_count = sum(1 for h in refinement_history if h["success"])

    # Calculate pass rate
    passed = sum(1 for t in verified_traces if t.grade and t.grade.passed)
    pass_rate = (passed / len(verified_traces) * 100) if verified_traces else 0

    # Record metrics
    metrics.cost = verify_llm.total_cost + refine_llm.total_cost
    metrics.calls = verify_llm.call_count + refine_llm.call_count
    metrics.complete()

    return VerificationResult(
        verified_traces=verified_traces,
        pass_rate=pass_rate,
        refinement_count=refinement_count,
        refinement_history=refinement_history,
        metrics=metrics,
    )


def verify_traces(
    policy: str | Policy,
    traces: TracesResult | list["Trace"],
    logic_map: "LogicMap | None" = None,
    scenarios: list["GoldenScenario"] | None = None,
    max_iterations: int = 3,
    model: str | None = None,
    refine_model: str | None = None,
    base_url: str | None = None,
    dataset_type: str = "conversation",
    tools: list["ToolDefinition"] | None = None,
    concurrency: int = 50,
) -> VerificationResult:
    """
    Verify and refine traces (sync wrapper).

    See verify_traces_async for full documentation.
    """
    return asyncio.run(
        verify_traces_async(
            policy,
            traces,
            logic_map,
            scenarios,
            max_iterations,
            model,
            refine_model,
            base_url,
            dataset_type,
            tools,
            concurrency,
        )
    )


# =============================================================================
# STREAMING API FUNCTIONS
# =============================================================================


async def extract_rules_stream(
    policy: str | Policy,
    model: str | None = None,
    base_url: str | None = None,
) -> AsyncIterator[Event]:
    """
    Extract rules with streaming progress events.

    Yields events as rules are extracted, providing real-time visibility
    into the extraction process.

    Args:
        policy: Policy text or Policy object
        model: Model to use for extraction (auto-detected if not specified)
        base_url: Optional API base URL

    Yields:
        Event objects:
        - ProgressEvent: Progress updates
        - RuleFoundEvent: When each rule is extracted
        - CompleteEvent: Final result with ExtractionResult

    Examples:
        >>> async for event in synkro.extract_rules_stream(policy):
        ...     match event.type:
        ...         case "progress":
        ...             print(f"Progress: {event.message}")
        ...         case "rule_found":
        ...             print(f"Found: {event.rule.rule_id}: {event.rule.text[:50]}...")
        ...         case "complete":
        ...             logic_map = event.result.logic_map
    """
    from synkro.generation.logic_extractor import LogicExtractor

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Auto-detect model if not specified
    if model is None:
        model = get_default_grading_model()

    # Create metrics tracker
    metrics = PhaseMetrics(phase="extraction", model=model)
    metrics.start()

    yield ProgressEvent(phase="extraction", message="Starting extraction...", progress=0.0)

    try:
        # Create extractor
        llm = LLM(model=model, base_url=base_url, temperature=0.1)
        extractor = LogicExtractor(llm=llm)

        yield ProgressEvent(phase="extraction", message="Analyzing policy...", progress=0.2)

        # Extract rules
        logic_map = await extractor.extract(policy.text)

        # Emit events for each rule found
        for i, rule in enumerate(logic_map.rules):
            yield RuleFoundEvent(rule=rule, index=i)
            progress = 0.3 + (0.6 * (i + 1) / len(logic_map.rules))
            yield ProgressEvent(
                phase="extraction",
                message=f"Extracted rule {rule.rule_id}",
                progress=progress,
                completed=i + 1,
                total=len(logic_map.rules),
            )

        # Record metrics
        metrics.cost = llm.total_cost
        metrics.calls = llm.call_count
        metrics.complete()

        result = ExtractionResult(logic_map=logic_map, metrics=metrics)
        yield CompleteEvent(result=result, metrics=metrics)

    except Exception as e:
        yield ErrorEvent(error=e, message=str(e), phase="extraction")
        raise


async def generate_scenarios_stream(
    policy: str | Policy,
    logic_map: "LogicMap | None" = None,
    count: int = 20,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.8,
) -> AsyncIterator[Event]:
    """
    Generate scenarios with streaming progress events.

    Yields events as scenarios are generated, enabling real-time progress
    tracking and incremental display.

    Args:
        policy: Policy text or Policy object
        logic_map: Pre-extracted Logic Map (will extract if not provided)
        count: Number of scenarios to generate
        model: Model for generation
        base_url: Optional API base URL
        temperature: Sampling temperature

    Yields:
        Event objects:
        - ProgressEvent: Progress updates
        - ScenarioGeneratedEvent: When each scenario is created
        - CompleteEvent: Final result with ScenariosResult

    Examples:
        >>> async for event in synkro.generate_scenarios_stream(policy, logic_map):
        ...     match event.type:
        ...         case "scenario_generated":
        ...             print(f"[{event.scenario.scenario_type}] {event.scenario.description[:40]}...")
        ...         case "complete":
        ...             scenarios = event.result.scenarios
    """
    from synkro.generation.golden_scenarios import GoldenScenarioGenerator
    from synkro.generation.planner import Planner

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Auto-detect models if not specified
    gen_model, grading_model = get_default_models()
    if model is None:
        model = gen_model

    metrics = PhaseMetrics(phase="scenarios", model=model)
    metrics.start()

    yield ProgressEvent(phase="scenarios", message="Starting scenario generation...", progress=0.0)

    try:
        # Extract logic map if needed
        if logic_map is None:
            yield ProgressEvent(
                phase="scenarios", message="Extracting rules first...", progress=0.1
            )
            extraction = await extract_rules_async(policy, model=grading_model, base_url=base_url)
            logic_map = extraction.logic_map

        # Create planner
        yield ProgressEvent(phase="scenarios", message="Planning categories...", progress=0.2)
        planner_llm = LLM(model=grading_model, base_url=base_url, temperature=0.3)
        planner = Planner(llm=planner_llm)
        plan = await planner.plan(policy, count, analyze_turns=False)

        # Create generator
        llm = LLM(model=model, base_url=base_url, temperature=temperature)
        generator = GoldenScenarioGenerator(llm=llm)

        yield ProgressEvent(phase="scenarios", message="Generating scenarios...", progress=0.3)

        # Generate scenarios for each category
        semaphore = asyncio.Semaphore(10)

        async def limited_generate(category):
            async with semaphore:
                return await generator.generate(policy.text, logic_map, category, category.count)

        tasks = [limited_generate(cat) for cat in plan.categories]
        results = await asyncio.gather(*tasks)

        # Flatten scenarios
        scenarios = [s for batch in results for s in batch]

        # Calculate distribution
        distribution = {"positive": 0, "negative": 0, "edge_case": 0, "irrelevant": 0}
        for s in scenarios:
            distribution[s.scenario_type.value] += 1

        # Emit events for each scenario
        for i, scenario in enumerate(scenarios):
            yield ScenarioGeneratedEvent(scenario=scenario, index=i)
            progress = 0.4 + (0.5 * (i + 1) / len(scenarios))
            yield ProgressEvent(
                phase="scenarios",
                message=f"Generated scenario {i+1}/{len(scenarios)}",
                progress=progress,
                completed=i + 1,
                total=len(scenarios),
            )

        # Record metrics
        metrics.cost = llm.total_cost + planner_llm.total_cost
        metrics.calls = llm.call_count + planner_llm.call_count
        metrics.complete()

        result = ScenariosResult(
            scenarios=scenarios,
            logic_map=logic_map,
            distribution=distribution,
            coverage_report=None,
            metrics=metrics,
        )
        yield CompleteEvent(result=result, metrics=metrics)

    except Exception as e:
        yield ErrorEvent(error=e, message=str(e), phase="scenarios")
        raise


async def synthesize_traces_stream(
    policy: str | Policy,
    scenarios: ScenariosResult | list["GoldenScenario"],
    logic_map: "LogicMap | None" = None,
    turns: int = 1,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
) -> AsyncIterator[Event]:
    """
    Synthesize traces with streaming progress events.

    Yields events as traces are generated, providing real-time progress
    visibility.

    Args:
        policy: Policy text or Policy object
        scenarios: ScenariosResult or list of scenarios
        logic_map: Logic Map (extracted from scenarios if ScenariosResult)
        turns: Conversation turns per trace
        model: Model for generation
        base_url: Optional API base URL
        temperature: Sampling temperature

    Yields:
        Event objects:
        - ProgressEvent: Progress updates
        - TraceGeneratedEvent: When each trace is created
        - CompleteEvent: Final result with TracesResult

    Examples:
        >>> async for event in synkro.synthesize_traces_stream(policy, scenarios):
        ...     match event.type:
        ...         case "trace_generated":
        ...             print(f"Trace {event.index}: {event.trace.user_message[:40]}...")
        ...         case "complete":
        ...             traces = event.result.traces
    """
    from synkro.generation.golden_responses import GoldenResponseGenerator

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Extract scenarios and logic map
    if isinstance(scenarios, ScenariosResult):
        scenario_list = scenarios.scenarios
        if logic_map is None:
            logic_map = scenarios.logic_map
    else:
        scenario_list = scenarios
        if logic_map is None:
            raise ValueError("logic_map is required when scenarios is a list")

    # Auto-detect model if not specified
    if model is None:
        model = get_default_model()

    metrics = PhaseMetrics(phase="traces", model=model)
    metrics.start()

    yield ProgressEvent(phase="traces", message="Starting trace synthesis...", progress=0.0)

    try:
        # Create generator
        llm = LLM(model=model, base_url=base_url, temperature=temperature)
        generator = GoldenResponseGenerator(llm=llm)

        yield ProgressEvent(phase="traces", message="Generating responses...", progress=0.1)

        # Generate traces
        traces = await generator.generate(policy.text, logic_map, scenario_list, turns)
        trace_list = list(traces)

        # Emit events for each trace
        for i, trace in enumerate(trace_list):
            yield TraceGeneratedEvent(trace=trace, index=i)
            progress = 0.2 + (0.7 * (i + 1) / len(trace_list))
            yield ProgressEvent(
                phase="traces",
                message=f"Generated trace {i+1}/{len(trace_list)}",
                progress=progress,
                completed=i + 1,
                total=len(trace_list),
            )

        # Record metrics
        metrics.cost = llm.total_cost
        metrics.calls = llm.call_count
        metrics.complete()

        result = TracesResult(
            traces=trace_list,
            logic_map=logic_map,
            scenarios=scenario_list,
            metrics=metrics,
        )
        yield CompleteEvent(result=result, metrics=metrics)

    except Exception as e:
        yield ErrorEvent(error=e, message=str(e), phase="traces")
        raise


async def verify_traces_stream(
    policy: str | Policy,
    traces: TracesResult | list["Trace"],
    logic_map: "LogicMap | None" = None,
    scenarios: list["GoldenScenario"] | None = None,
    max_iterations: int = 3,
    model: str | None = None,
    refine_model: str | None = None,
    base_url: str | None = None,
) -> AsyncIterator[Event]:
    """
    Verify traces with streaming progress events.

    Yields events as traces are verified and refined, providing real-time
    visibility into the verification process.

    Args:
        policy: Policy text or Policy object
        traces: TracesResult or list of traces
        logic_map: Logic Map for verification
        scenarios: Scenarios for verification context
        max_iterations: Maximum refinement attempts
        model: Model for verification
        refine_model: Model for refinement
        base_url: Optional API base URL

    Yields:
        Event objects:
        - ProgressEvent: Progress updates
        - TraceVerifiedEvent: When each trace is verified
        - RefinementStartedEvent: When refinement begins
        - TraceRefinedEvent: When a trace is refined
        - CompleteEvent: Final result with VerificationResult

    Examples:
        >>> async for event in synkro.verify_traces_stream(policy, traces):
        ...     match event.type:
        ...         case "trace_verified":
        ...             status = "PASS" if event.passed else "FAIL"
        ...             print(f"Trace {event.index}: {status}")
        ...         case "refinement_started":
        ...             print(f"Refining {event.count} failed traces...")
        ...         case "complete":
        ...             print(f"Pass rate: {event.result.pass_rate:.1%}")
    """
    from synkro.quality.golden_refiner import GoldenRefiner
    from synkro.quality.verifier import TraceVerifier

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Extract trace list and context
    if isinstance(traces, TracesResult):
        trace_list = list(traces.traces)
        if logic_map is None:
            logic_map = traces.logic_map
        if scenarios is None:
            scenarios = traces.scenarios
    else:
        trace_list = list(traces)
        if logic_map is None:
            raise ValueError("logic_map is required when traces is a list")
        if scenarios is None:
            raise ValueError("scenarios is required when traces is a list")

    # Auto-detect models if not specified
    gen_model, grading_model = get_default_models()
    if model is None:
        model = grading_model
    if refine_model is None:
        refine_model = gen_model

    metrics = PhaseMetrics(phase="verification", model=model)
    metrics.start()

    yield ProgressEvent(phase="verification", message="Starting verification...", progress=0.0)

    try:
        # Create verifier and refiner
        verify_llm = LLM(model=model, base_url=base_url, temperature=0.1)
        refine_llm = LLM(model=refine_model, base_url=base_url, temperature=0.5)
        verifier = TraceVerifier(llm=verify_llm)
        refiner = GoldenRefiner(llm=refine_llm)

        verified_traces = []
        failed_indices = []
        refinement_count = 0
        refinement_history = []

        # First pass: verify all traces
        for i, (trace, scenario) in enumerate(zip(trace_list, scenarios)):
            result = await verifier.verify(trace, logic_map, scenario, policy.text)

            yield TraceVerifiedEvent(
                trace=trace,
                index=i,
                passed=result.passed,
                issues=result.issues if hasattr(result, "issues") else [],
            )

            if result.passed:
                trace.grade = type(
                    "GradeResult", (), {"passed": True, "issues": [], "feedback": ""}
                )()
                verified_traces.append(trace)
            else:
                failed_indices.append((i, trace, scenario, result))

            progress = 0.1 + (0.4 * (i + 1) / len(trace_list))
            yield ProgressEvent(
                phase="verification",
                message=f"Verified {i+1}/{len(trace_list)}",
                progress=progress,
                completed=i + 1,
                total=len(trace_list),
            )

        # Second pass: refine failed traces
        if failed_indices:
            yield RefinementStartedEvent(
                count=len(failed_indices),
                iteration=1,
                max_iterations=max_iterations,
            )

            for idx, (i, trace, scenario, result) in enumerate(failed_indices):
                current_trace = trace
                refined = False

                for iteration in range(max_iterations):
                    refined_trace = await refiner.refine(current_trace, logic_map, scenario, result)
                    result = await verifier.verify(refined_trace, logic_map, scenario, policy.text)

                    if result.passed:
                        refined_trace.grade = type(
                            "GradeResult", (), {"passed": True, "issues": [], "feedback": ""}
                        )()
                        verified_traces.append(refined_trace)
                        refinement_count += 1
                        refinement_history.append(
                            {
                                "trace_index": i,
                                "iteration": iteration + 1,
                                "success": True,
                            }
                        )

                        yield TraceRefinedEvent(
                            trace=refined_trace,
                            index=i,
                            original_issues=result.issues if hasattr(result, "issues") else [],
                            passed=True,
                        )
                        refined = True
                        break
                    else:
                        current_trace = refined_trace

                if not refined:
                    current_trace.grade = type(
                        "GradeResult",
                        (),
                        {
                            "passed": False,
                            "issues": result.issues if hasattr(result, "issues") else [],
                            "feedback": "",
                        },
                    )()
                    verified_traces.append(current_trace)
                    refinement_history.append(
                        {
                            "trace_index": i,
                            "iteration": max_iterations,
                            "success": False,
                        }
                    )

                progress = 0.6 + (0.3 * (idx + 1) / len(failed_indices))
                yield ProgressEvent(
                    phase="verification",
                    message=f"Refined {idx+1}/{len(failed_indices)}",
                    progress=progress,
                )

        # Calculate pass rate
        passed = sum(1 for t in verified_traces if t.grade and t.grade.passed)
        pass_rate = (passed / len(verified_traces) * 100) if verified_traces else 0

        # Record metrics
        metrics.cost = verify_llm.total_cost + refine_llm.total_cost
        metrics.calls = verify_llm.call_count + refine_llm.call_count
        metrics.complete()

        result = VerificationResult(
            verified_traces=verified_traces,
            pass_rate=pass_rate,
            refinement_count=refinement_count,
            refinement_history=refinement_history,
            metrics=metrics,
        )
        yield CompleteEvent(result=result, metrics=metrics)

    except Exception as e:
        yield ErrorEvent(error=e, message=str(e), phase="verification")
        raise


__all__ = [
    # Non-streaming API
    "extract_rules",
    "extract_rules_async",
    "generate_scenarios_async",
    "synthesize_traces",
    "synthesize_traces_async",
    "verify_traces",
    "verify_traces_async",
    # Streaming API
    "extract_rules_stream",
    "generate_scenarios_stream",
    "synthesize_traces_stream",
    "verify_traces_stream",
]
