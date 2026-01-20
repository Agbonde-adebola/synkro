"""
Synkro - Generate high-quality training datasets from any document.

Quick Start:
    >>> import synkro
    >>> dataset = synkro.generate("Your policy text...")
    >>> dataset.save("training.jsonl")

Fast Generation (recommended for production):
    >>> # Step 1: Ingest policy once (extracts rules, allows review)
    >>> config = synkro.ingest(policy_text, output="./policy.json", review=True)

    >>> # Step 2: Use pre-computed config for fast generation
    >>> pipeline = synkro.create_pipeline(logic_map="./policy.json")
    >>> dataset = pipeline.generate(traces=20)  # Skips extraction!

Pipeline Usage (more control):
    >>> from synkro import create_pipeline, DatasetType
    >>> pipeline = create_pipeline(dataset_type=DatasetType.CONVERSATION)
    >>> dataset = pipeline.generate("policy text", traces=50)

Access Logic Map (for inspection):
    >>> result = pipeline.generate("policy text", return_logic_map=True)
    >>> print(result.logic_map.rules)  # See extracted rules
    >>> dataset = result.dataset

Silent Mode:
    >>> from synkro import SilentReporter, create_pipeline
    >>> pipeline = create_pipeline(reporter=SilentReporter())

Progress Callbacks:
    >>> from synkro import CallbackReporter, create_pipeline
    >>> reporter = CallbackReporter(
    ...     on_progress=lambda event, data: print(f"{event}: {data}")
    ... )
    >>> pipeline = create_pipeline(reporter=reporter)

Tool Call Dataset:
    >>> from synkro import create_pipeline, ToolDefinition, DatasetType
    >>> tools = [ToolDefinition(name="search", description="...", parameters={})]
    >>> pipeline = create_pipeline(dataset_type=DatasetType.TOOL_CALL, tools=tools)

Eval Dataset Generation:
    >>> import synkro
    >>> result = synkro.generate_scenarios("Your policy...", count=100)
    >>> for scenario in result.scenarios:
    ...     response = my_model(scenario.user_message)
    ...     grade = synkro.grade(response, scenario, policy)

Advanced Usage (power users):
    >>> from synkro.advanced import LogicExtractor, TraceVerifier, LogicMap
    >>> # Full access to Golden Trace internals
"""

# Dynamic version from package metadata
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("synkro")
except Exception:
    __version__ = "0.4.58"  # Fallback

# =============================================================================
# PRIMARY API - What most developers need
# =============================================================================

# Step-by-step API functions
from synkro.api import (
    extract_rules,
    extract_rules_async,
    # Streaming variants
    extract_rules_stream,
    generate_scenarios_async,
    generate_scenarios_stream,
    synthesize_traces,
    synthesize_traces_async,
    synthesize_traces_stream,
    verify_traces,
    verify_traces_async,
    verify_traces_stream,
)
from synkro.core.dataset import Dataset
from synkro.core.policy import Policy
from synkro.ingestion import PolicyConfig, ingest, load_config

# HITL editing functions
from synkro.interactive.standalone import (
    edit_rules,
    edit_rules_async,
    edit_scenarios,
    edit_scenarios_async,
)
from synkro.llm import LLM
from synkro.models import Anthropic, Google, Local, LocalModel, OpenAI

# GenerationResult for return_logic_map=True (backward compatibility)
from synkro.pipeline.runner import GenerationResult, ScenariosResult
from synkro.pipelines import create_pipeline
from synkro.reporting import (
    CallbackReporter,
    FileLoggingReporter,
    ProgressReporter,
    RichReporter,
    SilentReporter,
)

# Session class
from synkro.session import Session

# Tool definitions for LLM agents
from synkro.tools import TOOL_DEFINITIONS

# Tool types (needed for TOOL_CALL dataset type)
# =============================================================================
# SECONDARY API - Less commonly needed
# =============================================================================
from synkro.types import (
    Category,
    DatasetType,
    EvalScenario,
    GradeResult,
    Message,
    Plan,
    Scenario,
    ToolCall,
    ToolDefinition,
    ToolFunction,
    ToolResult,
    Trace,
)

# Coverage tracking types
from synkro.types.coverage import CoverageReport, SubCategoryTaxonomy
from synkro.types.events import CompleteEvent, Event, ProgressEvent
from synkro.types.metrics import Metrics, PhaseMetrics

# New result types with metrics
from synkro.types.results import ExtractionResult, PipelineResult, TracesResult, VerificationResult
from synkro.types.results import ScenariosResult as ScenariosResultNew
from synkro.types.state import PipelinePhase, PipelineState

# Model detection utilities
from synkro.utils.model_detection import (
    detect_available_provider,
    get_default_grading_model,
    get_default_model,
    get_default_models,
    get_provider_info,
)

__all__ = [
    # Primary API
    "create_pipeline",
    "generate",
    "generate_scenarios",
    "grade",
    "ingest",
    "load_config",
    "PolicyConfig",
    "DatasetType",
    "Policy",
    "Dataset",
    "ToolDefinition",
    # Step-by-step API
    "extract_rules",
    "extract_rules_async",
    "generate_scenarios_async",
    "synthesize_traces",
    "synthesize_traces_async",
    "verify_traces",
    "verify_traces_async",
    # HITL editing functions
    "edit_rules",
    "edit_rules_async",
    "edit_scenarios",
    "edit_scenarios_async",
    # Streaming API
    "extract_rules_stream",
    "generate_scenarios_stream",
    "synthesize_traces_stream",
    "verify_traces_stream",
    # Session class
    "Session",
    # Tool definitions
    "TOOL_DEFINITIONS",
    # Model detection
    "detect_available_provider",
    "get_default_models",
    "get_default_model",
    "get_default_grading_model",
    "get_provider_info",
    # Reporters
    "SilentReporter",
    "RichReporter",
    "CallbackReporter",
    "FileLoggingReporter",
    "ProgressReporter",
    # Models
    "OpenAI",
    "Anthropic",
    "Google",
    "Local",
    "LocalModel",
    "LLM",
    # Result types (new)
    "PipelineResult",
    "ExtractionResult",
    "TracesResult",
    "VerificationResult",
    "ScenariosResultNew",
    # Result types (backward compatibility)
    "GenerationResult",
    "ScenariosResult",
    "CoverageReport",
    "SubCategoryTaxonomy",
    # Metrics and state types
    "Metrics",
    "PhaseMetrics",
    "PipelinePhase",
    "PipelineState",
    # Event types
    "Event",
    "ProgressEvent",
    "CompleteEvent",
    # Data types (less common)
    "Trace",
    "Scenario",
    "EvalScenario",
    "Message",
    "GradeResult",
    "Plan",
    "Category",
    "ToolCall",
    "ToolFunction",
    "ToolResult",
]


# Note: For advanced usage (LogicMap, TraceVerifier, etc.), use:
# from synkro.advanced import ...


def generate(
    policy: str | Policy,
    traces: int = 20,
    turns: int | str = "auto",
    dataset_type: DatasetType = DatasetType.CONVERSATION,
    generation_model: OpenAI | Anthropic | Google | LocalModel | str | None = None,
    grading_model: OpenAI | Anthropic | Google | LocalModel | str | None = None,
    max_iterations: int = 3,
    skip_grading: bool = False,
    reporter: ProgressReporter | None = None,
    return_logic_map: bool = False,
    return_result: bool = False,
    enable_hitl: bool = True,
    base_url: str | None = None,
    temperature: float = 0.7,
) -> Dataset | GenerationResult | PipelineResult:
    """
    Generate training traces from a policy document.

    This is a convenience function. For more control, use create_pipeline().

    Args:
        policy: Policy text or Policy object
        traces: Number of traces to generate (default: 20)
        turns: Conversation turns per trace. Use int for fixed turns, or "auto"
            for policy complexity-driven turns (Simple=1-2, Conditional=3, Complex=5+)
        dataset_type: Type of dataset - CONVERSATION (default), INSTRUCTION, or TOOL_CALL
        generation_model: Model for generating (auto-detected if not specified)
        grading_model: Model for grading (auto-detected if not specified)
        max_iterations: Max refinement iterations per trace (default: 3)
        skip_grading: Skip grading phase for faster generation (default: False)
        reporter: Progress reporter (default: RichReporter for console output)
        return_logic_map: If True, return GenerationResult with Logic Map access (deprecated, use return_result)
        return_result: If True, return PipelineResult with full stage results and metrics
        enable_hitl: Enable Human-in-the-Loop Logic Map editing (default: False)
        base_url: Optional API base URL for local LLM providers (Ollama, vLLM, etc.)
        temperature: Sampling temperature for generation (0.0-2.0, default: 0.7).
            Lower values (0.1-0.3) produce more deterministic outputs for eval datasets.
            Higher values (0.7-1.0) produce more diverse outputs for training data.

    Returns:
        Dataset (default), GenerationResult if return_logic_map=True,
        or PipelineResult if return_result=True

    Example:
        >>> import synkro
        >>> dataset = synkro.generate("All expenses over $50 require approval")
        >>> dataset.save("training.jsonl")

        >>> # Access full results with metrics (recommended)
        >>> result = synkro.generate(policy, return_result=True)
        >>> print(f"Cost: ${result.metrics.total_cost:.4f}")
        >>> print(f"Pass rate: {result.pass_rate:.1%}")
        >>> dataset = result.dataset

        >>> # Access Logic Map (deprecated, use return_result)
        >>> result = synkro.generate(policy, return_logic_map=True)
        >>> print(result.logic_map.rules)
        >>> dataset = result.dataset

        >>> # Multi-turn with fixed 3 turns
        >>> dataset = synkro.generate(policy, turns=3)

        >>> # Interactive Logic Map editing
        >>> dataset = synkro.generate(policy, enable_hitl=True)

        >>> # Silent mode
        >>> from synkro import SilentReporter
        >>> dataset = synkro.generate(policy, reporter=SilentReporter())
    """
    import warnings

    from synkro.generation.generator import Generator

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Auto-detect models if not specified
    if generation_model is None or grading_model is None:
        gen_model, grade_model = get_default_models()
        if generation_model is None:
            generation_model = gen_model
        if grading_model is None:
            grading_model = grade_model

    # Handle deprecation: return_logic_map -> return_result
    if return_logic_map and not return_result:
        warnings.warn(
            "return_logic_map is deprecated, use return_result=True instead. "
            "return_result provides PipelineResult with full metrics.",
            DeprecationWarning,
            stacklevel=2,
        )
        # For backward compat, still return GenerationResult
        should_return_result = True
    elif return_result:
        should_return_result = True
    else:
        should_return_result = False

    generator = Generator(
        dataset_type=dataset_type,
        generation_model=generation_model,
        grading_model=grading_model,
        max_iterations=max_iterations,
        skip_grading=skip_grading,
        reporter=reporter,
        turns=turns,
        enable_hitl=enable_hitl,
        base_url=base_url,
        temperature=temperature,
    )

    result = generator.generate(policy, traces=traces, return_logic_map=should_return_result)

    # Convert to GenerationResult for backward compat if return_logic_map was used
    if return_logic_map and not return_result and isinstance(result, PipelineResult):
        return GenerationResult(
            dataset=result.dataset,
            logic_map=result.logic_map,
            scenarios=result.scenarios.scenarios if result.scenarios else [],
            distribution=result.scenarios.distribution if result.scenarios else {},
            coverage_report=result.coverage_report,
        )

    return result


def generate_scenarios(
    policy: str | Policy,
    count: int = 100,
    generation_model: OpenAI | Anthropic | Google | LocalModel | str | None = None,
    temperature: float = 0.8,
    reporter: ProgressReporter | None = None,
    enable_hitl: bool = False,
    base_url: str | None = None,
) -> ScenariosResult:
    """
    Generate eval scenarios from a policy without synthetic responses.

    This is the eval-focused API. It generates diverse test scenarios with
    ground truth labels (expected outcomes, target rules) but does NOT generate
    synthetic responses. Use synkro.grade() to evaluate your own model's outputs.

    Args:
        policy: Policy text or Policy object
        count: Number of scenarios to generate (default: 100)
        generation_model: Model for generation (auto-detected if not specified)
        temperature: Sampling temperature (default: 0.8 for scenario diversity)
        reporter: Progress reporter (default: RichReporter for console output)
        enable_hitl: Enable Human-in-the-Loop editing (default: False)
        base_url: Optional API base URL for local LLM providers

    Returns:
        ScenariosResult with scenarios, logic_map, and distribution

    Example:
        >>> import synkro
        >>> result = synkro.generate_scenarios("Your policy...", count=100)
        >>>
        >>> for scenario in result.scenarios:
        ...     # Run YOUR model
        ...     response = my_model(scenario.user_message)
        ...
        ...     # Grade the response
        ...     grade = synkro.grade(response, scenario, policy)
        ...     print(f"Passed: {grade.passed}")
    """
    from synkro.generation.generator import Generator

    if isinstance(policy, str):
        policy = Policy(text=policy)

    # Auto-detect model if not specified
    if generation_model is None:
        generation_model = get_default_model()

    generator = Generator(
        dataset_type=DatasetType.CONVERSATION,  # Type doesn't matter for scenarios-only
        generation_model=generation_model,
        grading_model=generation_model,  # Not used but required
        skip_grading=True,
        reporter=reporter,
        enable_hitl=enable_hitl,
        base_url=base_url,
        temperature=temperature,
    )

    return generator.generate_scenarios(policy, count=count)


def grade(
    response: str,
    scenario: EvalScenario,
    policy: str | Policy,
    model: OpenAI | Anthropic | Google | LocalModel | str | None = None,
    base_url: str | None = None,
) -> GradeResult:
    """
    Grade an external model's response against a scenario and policy.

    Use this to evaluate your own model's outputs against scenarios
    generated by synkro.generate_scenarios().

    Args:
        response: The response from the model being evaluated
        scenario: The eval scenario with expected_outcome and target_rules
        policy: The policy document for grading context
        model: LLM to use for grading (auto-detected if not specified, stronger = better)
        base_url: Optional API base URL for local LLM providers

    Returns:
        GradeResult with passed, feedback, and issues

    Example:
        >>> scenarios = synkro.generate_scenarios(policy, count=100)
        >>> for scenario in scenarios:
        ...     response = my_model(scenario.user_message)
        ...     grade = synkro.grade(response, scenario, policy)
        ...     if not grade.passed:
        ...         print(f"Failed: {grade.feedback}")
    """
    import asyncio

    from synkro.llm.client import LLM
    from synkro.quality.grader import Grader
    from synkro.types.core import Message, Trace
    from synkro.types.core import Scenario as BaseScenario

    if isinstance(policy, str):
        policy_text = policy
    else:
        policy_text = policy.text

    # Auto-detect model if not specified
    if model is None:
        model = get_default_grading_model()

    # Create grader with specified model
    grading_llm = LLM(model=model, base_url=base_url, temperature=0.1)
    grader = Grader(llm=grading_llm)

    # Build a Trace object from the scenario and response
    base_scenario = BaseScenario(
        description=scenario.user_message,
        context=scenario.context,
        category=scenario.category,
        scenario_type=scenario.scenario_type,
        target_rule_ids=scenario.target_rule_ids,
        expected_outcome=scenario.expected_outcome,
    )

    trace = Trace(
        messages=[
            Message(role="user", content=scenario.user_message),
            Message(role="assistant", content=response),
        ],
        scenario=base_scenario,
    )

    # Run grading
    async def _grade():
        return await grader.grade(trace, policy_text)

    return asyncio.run(_grade())
