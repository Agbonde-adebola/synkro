"""Pipeline creation utilities.

Usage:
    from synkro.pipelines import create_pipeline
    from synkro.models.openai import OpenAI
    from synkro.types import DatasetType

    pipeline = create_pipeline(
        model=OpenAI.GPT_5_MINI,
        dataset_type=DatasetType.CONVERSATION,
    )
    dataset = pipeline.generate("policy text", traces=50)

    # Tool calling pipeline
    from synkro import ToolDefinition

    web_search = ToolDefinition(
        name="web_search",
        description="Search the web",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}}
    )

    pipeline = create_pipeline(
        dataset_type=DatasetType.TOOL_CALL,
        tools=[web_search],
    )
    dataset = pipeline.generate("Search guidelines", traces=50)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from synkro.generation.generator import Generator
from synkro.models import Model, OpenAI
from synkro.reporting import ProgressReporter
from synkro.types import DatasetType

if TYPE_CHECKING:
    from synkro.ingestion import PolicyConfig
    from synkro.types.tool import ToolDefinition


def create_pipeline(
    model: Model = OpenAI.GPT_5_MINI,
    dataset_type: DatasetType = DatasetType.CONVERSATION,
    grading_model: Model = OpenAI.GPT_52,
    max_iterations: int = 3,
    skip_grading: bool = False,
    reporter: ProgressReporter | None = None,
    tools: list["ToolDefinition"] | None = None,
    turns: int | str = "auto",
    checkpoint_dir: str | None = None,
    enable_hitl: bool = True,
    base_url: str | None = None,
    thinking: bool = False,
    temperature: float = 0.7,
    logic_map: str | Path | "PolicyConfig" | None = None,
    skip_coverage: bool = False,
) -> Generator:
    """
    Create a pipeline for generating training datasets.

    Args:
        model: Model enum for generation (default: OpenAI.GPT_5_MINI)
        dataset_type: Type of dataset - CONVERSATION, INSTRUCTION, EVALUATION, or TOOL_CALL (default: CONVERSATION)
        grading_model: Model enum for grading (default: OpenAI.GPT_52)
        max_iterations: Max refinement iterations per trace (default: 3)
        skip_grading: Skip grading phase for faster generation (default: False)
        reporter: Progress reporter (default: RichReporter for console output)
        tools: List of ToolDefinition for TOOL_CALL dataset type
        turns: Conversation turns per trace. Use int for fixed turns, or "auto"
            for policy complexity-driven turns (Simple=1-2, Conditional=3, Complex=5+)
        checkpoint_dir: Directory for checkpoints. Enables resumable generation.
        enable_hitl: Enable Human-in-the-Loop Logic Map editing (default: False)
        base_url: Optional API base URL for local LLM providers (Ollama, vLLM, etc.)
        thinking: Enable thinking mode with <think> tags in responses (default: False).
            When enabled, assistant responses will include reasoning wrapped in
            <think>...</think> tags, compatible with Qwen3 and DeepSeek-R1 formats.
        temperature: Sampling temperature for generation (0.0-2.0, default: 0.7).
            Lower values (0.1-0.3) produce more deterministic outputs for eval datasets.
            Higher values (0.7-1.0) produce more diverse outputs for training data.
        logic_map: Pre-computed Logic Map from synkro.ingest(). Can be a file path
            (str or Path) or a PolicyConfig object. When provided, skips the
            planning and logic extraction phases for faster generation.
        skip_coverage: Skip coverage tracking for faster generation (default: False).
            Coverage tracking adds taxonomy extraction and scenario tagging which
            requires additional LLM calls.

    Returns:
        Generator instance ready to use

    Example:
        >>> from synkro.pipelines import create_pipeline
        >>> from synkro.models.openai import OpenAI
        >>> from synkro.types import DatasetType
        >>>
        >>> pipeline = create_pipeline(
        ...     model=OpenAI.GPT_5_MINI,
        ...     dataset_type=DatasetType.CONVERSATION,
        ... )
        >>> dataset = pipeline.generate("policy text", traces=50)
        >>> dataset.save("training.jsonl")

        >>> # Multi-turn with fixed 3 turns
        >>> pipeline = create_pipeline(turns=3)
        >>> dataset = pipeline.generate("policy text", traces=50)

        >>> # Silent mode for embedding
        >>> from synkro.reporting import SilentReporter
        >>> pipeline = create_pipeline(reporter=SilentReporter())

        >>> # Interactive Logic Map editing
        >>> pipeline = create_pipeline(enable_hitl=True)
        >>> dataset = pipeline.generate("policy text", traces=50)

        >>> # Tool calling dataset
        >>> from synkro import ToolDefinition
        >>> search_tool = ToolDefinition(
        ...     name="web_search",
        ...     description="Search the web for information",
        ...     parameters={"type": "object", "properties": {"query": {"type": "string"}}}
        ... )
        >>> pipeline = create_pipeline(
        ...     dataset_type=DatasetType.TOOL_CALL,
        ...     tools=[search_tool],
        ... )
        >>> dataset = pipeline.generate("Search guidelines", traces=50)
    """
    return Generator(
        dataset_type=dataset_type,
        generation_model=model,
        grading_model=grading_model,
        max_iterations=max_iterations,
        skip_grading=skip_grading,
        reporter=reporter,
        tools=tools,
        turns=turns,
        checkpoint_dir=checkpoint_dir,
        enable_hitl=enable_hitl,
        base_url=base_url,
        thinking=thinking,
        temperature=temperature,
        logic_map=logic_map,
        skip_coverage=skip_coverage,
    )


__all__ = ["create_pipeline"]
