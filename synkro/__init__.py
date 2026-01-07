"""
Synkro - Generate high-quality training datasets from any document.

Quick Start:
    >>> import synkro
    >>> dataset = synkro.generate("Your policy text...")
    >>> dataset.save("training.jsonl")

Pipeline Usage (more control):
    >>> from synkro import create_pipeline, DatasetType
    >>> pipeline = create_pipeline(dataset_type=DatasetType.SFT)
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

Advanced Usage (power users):
    >>> from synkro.advanced import LogicExtractor, TraceVerifier, LogicMap
    >>> # Full access to Golden Trace internals
"""

# Dynamic version from package metadata
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("synkro")
except Exception:
    __version__ = "0.4.15"  # Fallback

# =============================================================================
# PRIMARY API - What most developers need
# =============================================================================

from synkro.pipelines import create_pipeline
from synkro.models import OpenAI, Anthropic, Google, Local, LocalModel
from synkro.llm import LLM
from synkro.types import DatasetType
from synkro.core.policy import Policy
from synkro.core.dataset import Dataset
from synkro.reporting import SilentReporter, RichReporter, CallbackReporter

# Tool types (needed for TOOL_CALL dataset type)
from synkro.types import ToolDefinition

# =============================================================================
# SECONDARY API - Less commonly needed
# =============================================================================

from synkro.types import Message, Scenario, Trace, GradeResult, Plan, Category
from synkro.types import ToolCall, ToolFunction, ToolResult
from synkro.reporting import ProgressReporter

# GenerationResult for return_logic_map=True
from synkro.pipeline.runner import GenerationResult

__all__ = [
    # Primary API
    "create_pipeline",
    "generate",
    "DatasetType",
    "Policy",
    "Dataset",
    "ToolDefinition",
    # Reporters
    "SilentReporter",
    "RichReporter",
    "CallbackReporter",
    "ProgressReporter",
    # Models
    "OpenAI",
    "Anthropic",
    "Google",
    "Local",
    "LocalModel",
    "LLM",
    # Result types
    "GenerationResult",
    # Data types (less common)
    "Trace",
    "Scenario",
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
    dataset_type: DatasetType = DatasetType.SFT,
    generation_model: OpenAI | Anthropic | Google | LocalModel | str = OpenAI.GPT_5_MINI,
    grading_model: OpenAI | Anthropic | Google | LocalModel | str = OpenAI.GPT_52,
    max_iterations: int = 3,
    skip_grading: bool = False,
    reporter: ProgressReporter | None = None,
    return_logic_map: bool = False,
    enable_hitl: bool = True,
    base_url: str | None = None,
) -> Dataset | GenerationResult:
    """
    Generate training traces from a policy document.

    This is a convenience function. For more control, use create_pipeline().

    Args:
        policy: Policy text or Policy object
        traces: Number of traces to generate (default: 20)
        turns: Conversation turns per trace. Use int for fixed turns, or "auto"
            for policy complexity-driven turns (Simple=1-2, Conditional=3, Complex=5+)
        dataset_type: Type of dataset - SFT (default) or QA
        generation_model: Model for generating (default: gpt-5-mini)
        grading_model: Model for grading (default: gpt-5.2)
        max_iterations: Max refinement iterations per trace (default: 3)
        skip_grading: Skip grading phase for faster generation (default: False)
        reporter: Progress reporter (default: RichReporter for console output)
        return_logic_map: If True, return GenerationResult with Logic Map access
        enable_hitl: Enable Human-in-the-Loop Logic Map editing (default: False)
        base_url: Optional API base URL for local LLM providers (Ollama, vLLM, etc.)

    Returns:
        Dataset (default) or GenerationResult if return_logic_map=True

    Example:
        >>> import synkro
        >>> dataset = synkro.generate("All expenses over $50 require approval")
        >>> dataset.save("training.jsonl")

        >>> # Access Logic Map
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
    from synkro.generation.generator import Generator

    if isinstance(policy, str):
        policy = Policy(text=policy)

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
    )

    return generator.generate(policy, traces=traces, return_logic_map=return_logic_map)
