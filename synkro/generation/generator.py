"""Main Generator class orchestrating the full trace generation pipeline."""

import asyncio
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from synkro.llm.client import LLM
from synkro.llm.rate_limits import auto_workers
from synkro.models import Model, OpenAI
from synkro.types.dataset_type import DatasetType
from synkro.core.policy import Policy
from synkro.core.dataset import Dataset
from synkro.core.checkpoint import CheckpointManager, hash_policy
from synkro.modes.config import get_mode_config
from synkro.errors import handle_error
from synkro.factory import ComponentFactory
from synkro.reporting import ProgressReporter, RichReporter
from synkro.pipeline.runner import GenerationPipeline, GenerationResult

if TYPE_CHECKING:
    from synkro.types.tool import ToolDefinition


class Generator:
    """
    Main orchestrator for generating training datasets.

    The Generator handles the full pipeline:
    1. Plan: Analyze policy and create category distribution
    2. Generate: Create scenarios and responses
    3. Grade: Evaluate response quality
    4. Refine: Fix failed responses
    5. Return: Dataset of passing traces

    Examples:
        >>> generator = Generator()
        >>> dataset = generator.generate(policy, traces=20)

        >>> # QA dataset
        >>> generator = Generator(dataset_type=DatasetType.QA)
        >>> dataset = generator.generate(policy)

        >>> # Silent mode (no console output)
        >>> from synkro.reporting import SilentReporter
        >>> generator = Generator(reporter=SilentReporter())
        >>> dataset = generator.generate(policy)
        
        >>> # Tool call dataset
        >>> from synkro import ToolDefinition
        >>> tools = [ToolDefinition(name="search", description="...", parameters={})]
        >>> generator = Generator(dataset_type=DatasetType.TOOL_CALL, tools=tools)
        >>> dataset = generator.generate("Usage guidelines", traces=20)
    """

    def __init__(
        self,
        dataset_type: DatasetType = DatasetType.SFT,
        generation_model: Model = OpenAI.GPT_4O_MINI,
        grading_model: Model = OpenAI.GPT_4O,
        max_iterations: int = 1,
        skip_grading: bool = False,
        reporter: ProgressReporter | None = None,
        tools: list["ToolDefinition"] | None = None,
        turns: int | str = "auto",
        checkpoint_dir: str | Path | None = None,
        enable_hitl: bool = True,
        base_url: str | None = None,
    ):
        """
        Initialize the Generator.

        Args:
            dataset_type: Type of dataset to generate (QA, SFT, or TOOL_CALL)
            generation_model: Model for scenarios/responses (default: gpt-4o-mini)
            grading_model: Model for grading (default: gpt-4o, recommend stronger)
            max_iterations: Max refinement iterations per trace (default: 1, no retries)
            skip_grading: Skip grading phase for faster generation (default: False)
            reporter: Progress reporter (default: RichReporter for console output)
            tools: List of ToolDefinition for TOOL_CALL dataset type
            turns: Conversation turns per trace. Use int for fixed turns, or "auto"
                for policy complexity-driven turns (Simple=1-2, Conditional=3, Complex=5+)
            checkpoint_dir: Directory for checkpoints. If provided, enables resumable
                generation. Progress is saved after each stage.
            enable_hitl: Enable Human-in-the-Loop Logic Map editing. When enabled,
                pauses after Logic Map extraction to allow interactive refinement.
            base_url: Optional API base URL for local LLM providers (Ollama, vLLM, etc.)
        """
        self.dataset_type = dataset_type
        self.mode_config = get_mode_config(dataset_type)
        self.max_iterations = max_iterations
        self.skip_grading = skip_grading
        self.tools = tools
        self.turns = turns
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Create checkpoint manager if checkpointing enabled
        self.checkpoint_manager = (
            CheckpointManager(self.checkpoint_dir) if self.checkpoint_dir else None
        )

        # HITL configuration
        self.enable_hitl = enable_hitl

        # Validate tools for TOOL_CALL dataset type
        if dataset_type == DatasetType.TOOL_CALL and not tools:
            raise ValueError("TOOL_CALL dataset type requires tools parameter")
        
        # Store model info for reporting
        self.generation_model = generation_model
        self.grading_model = grading_model

        # Create LLM clients
        self.generation_llm = LLM(model=generation_model, base_url=base_url)
        self.grading_llm = LLM(model=grading_model, base_url=base_url)
        
        # Create factory for component creation
        self.factory = ComponentFactory(
            generation_llm=self.generation_llm,
            grading_llm=self.grading_llm,
            mode_config=self.mode_config,
            tools=tools,
        )

        # Reporter for progress output
        self.reporter = reporter or RichReporter()

        # Auto-scale workers based on provider
        model_str = generation_model.value if isinstance(generation_model, Enum) else str(generation_model)
        self.workers = auto_workers(model_str)

        # Create HITL editor if enabled
        hitl_editor = self.factory.create_logic_map_editor() if enable_hitl else None

        # Create pipeline
        self.pipeline = GenerationPipeline(
            factory=self.factory,
            reporter=self.reporter,
            workers=self.workers,
            max_iterations=max_iterations,
            skip_grading=skip_grading,
            checkpoint_manager=self.checkpoint_manager,
            enable_hitl=enable_hitl,
            hitl_editor=hitl_editor,
        )

    @handle_error
    def generate(
        self,
        policy: Policy | str,
        traces: int = 20,
        return_logic_map: bool = False,
    ) -> Dataset | GenerationResult:
        """
        Generate a training dataset from a policy.

        Args:
            policy: Policy object or text string
            traces: Target number of traces to generate (default: 20)
            return_logic_map: If True, return GenerationResult with access to
                the Logic Map, scenarios, and distribution (default: False)

        Returns:
            Dataset (default) or GenerationResult if return_logic_map=True

        Examples:
            >>> # Standard usage
            >>> dataset = generator.generate(policy, traces=50)

            >>> # Access Logic Map for inspection
            >>> result = generator.generate(policy, return_logic_map=True)
            >>> print(result.logic_map.rules)  # See extracted rules
            >>> print(result.distribution)     # See scenario type counts
            >>> dataset = result.dataset       # Get the dataset
        """
        if isinstance(policy, str):
            policy = Policy(text=policy)

        # Validate policy has enough content
        policy.validate_length()

        return asyncio.run(self._generate_async(policy, traces, return_logic_map))

    async def _generate_async(
        self,
        policy: Policy,
        traces: int,
        return_logic_map: bool = False,
    ) -> Dataset | GenerationResult:
        """Async implementation of generation pipeline."""
        model_str = self.generation_model.value if isinstance(self.generation_model, Enum) else str(self.generation_model)

        return await self.pipeline.run(
            policy=policy,
            traces=traces,
            model=model_str,
            dataset_type=self.dataset_type.value,
            turns=self.turns,
            return_result=return_logic_map,
        )

    async def generate_async(
        self,
        policy: Policy | str,
        traces: int = 20,
        return_logic_map: bool = False,
    ) -> Dataset | GenerationResult:
        """
        Async version of generate for use in async contexts.

        Args:
            policy: Policy object or text string
            traces: Target number of traces to generate (default: 20)
            return_logic_map: If True, return GenerationResult with Logic Map access

        Returns:
            Dataset (default) or GenerationResult if return_logic_map=True
        """
        if isinstance(policy, str):
            policy = Policy(text=policy)

        return await self._generate_async(policy, traces, return_logic_map)
