"""Main Generator class orchestrating the full trace generation pipeline."""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from synkro.core.checkpoint import CheckpointManager
from synkro.core.dataset import Dataset
from synkro.core.policy import Policy
from synkro.errors import handle_error
from synkro.factory import ComponentFactory
from synkro.llm.client import LLM
from synkro.llm.rate_limits import auto_workers
from synkro.models import Model, OpenAI
from synkro.modes.config import get_mode_config
from synkro.pipeline.runner import GenerationPipeline, GenerationResult, ScenariosResult
from synkro.reporting import ProgressReporter, RichReporter
from synkro.types.dataset_type import DatasetType

if TYPE_CHECKING:
    from synkro.ingestion import PolicyConfig
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

        >>> # Conversation dataset (default, multi-turn)
        >>> generator = Generator(dataset_type=DatasetType.CONVERSATION)
        >>> dataset = generator.generate(policy)

        >>> # Instruction dataset (single-turn)
        >>> generator = Generator(dataset_type=DatasetType.INSTRUCTION)
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

        >>> # Eval dataset with low temperature for deterministic outputs
        >>> generator = Generator(dataset_type=DatasetType.EVALUATION, temperature=0.2)
        >>> dataset = generator.generate(policy, traces=50)
    """

    def __init__(
        self,
        dataset_type: DatasetType = DatasetType.CONVERSATION,
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
        thinking: bool = False,
        temperature: float = 0.7,
        logic_map: str | Path | "PolicyConfig" | None = None,
        skip_coverage: bool = False,
    ):
        """
        Initialize the Generator.

        Args:
            dataset_type: Type of dataset to generate (CONVERSATION, INSTRUCTION, or TOOL_CALL)
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
            thinking: Enable thinking mode with <think> tags in responses (default: False).
                When enabled, assistant responses will include reasoning wrapped in
                <think>...</think> tags, compatible with Qwen3 and DeepSeek-R1 formats.
            temperature: Sampling temperature for generation (0.0-2.0, default: 0.7).
                Lower values (0.1-0.3) produce more deterministic outputs for eval datasets.
                Higher values (0.7-1.0) produce more diverse outputs for training data.
            logic_map: Pre-computed Logic Map from synkro.ingest(). Can be a file path
                or PolicyConfig object. Skips planning and extraction phases when provided.
            skip_coverage: Skip coverage tracking for faster generation (default: False).
        """
        self.dataset_type = dataset_type
        self.mode_config = get_mode_config(dataset_type)
        self.max_iterations = max_iterations
        self.skip_grading = skip_grading
        self.skip_coverage = skip_coverage
        self.tools = tools
        self.turns = turns
        self.thinking = thinking
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Load policy config if logic_map provided
        self.policy_config = None
        if logic_map is not None:
            self.policy_config = self._load_policy_config(logic_map)
            # Use turns from config if not explicitly set
            if turns == "auto" and self.policy_config:
                self.turns = self.policy_config.complexity.recommended_turns

        # HITL configuration
        self.enable_hitl = enable_hitl

        # Validate tools for TOOL_CALL dataset type
        if dataset_type == DatasetType.TOOL_CALL and not tools:
            raise ValueError("TOOL_CALL dataset type requires tools parameter")

        # Force turns=1 for INSTRUCTION and EVALUATION types
        if dataset_type in (DatasetType.INSTRUCTION, DatasetType.EVALUATION):
            self.turns = 1

        # Store model info for reporting
        self.generation_model = generation_model
        self.grading_model = grading_model

        # Create LLM clients
        self.generation_llm = LLM(
            model=generation_model, base_url=base_url, temperature=temperature
        )
        self.grading_llm = LLM(model=grading_model, base_url=base_url)

        # Create factory for component creation
        self.factory = ComponentFactory(
            generation_llm=self.generation_llm,
            grading_llm=self.grading_llm,
            mode_config=self.mode_config,
            tools=tools,
            thinking=thinking,
        )

        # Reporter for progress output
        self.reporter = reporter or RichReporter()

        # Create checkpoint manager if checkpointing enabled
        # Create it after reporter so we can route messages through the event log
        checkpoint_logger = None
        if hasattr(self.reporter, "_display") and self.reporter._display:

            def checkpoint_logger(msg):
                return self.reporter._display.add_event(msg)

        self.checkpoint_manager = (
            CheckpointManager(self.checkpoint_dir, logger=checkpoint_logger)
            if self.checkpoint_dir
            else None
        )

        # Auto-scale workers based on provider
        model_str = (
            generation_model.value if isinstance(generation_model, Enum) else str(generation_model)
        )
        self.workers = auto_workers(model_str)

        # Create HITL editors if enabled
        hitl_editor = self.factory.create_logic_map_editor() if enable_hitl else None
        scenario_editor = self.factory.create_scenario_editor() if enable_hitl else None

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
            scenario_editor=scenario_editor,
            skip_coverage=skip_coverage,
        )

    def _load_policy_config(
        self,
        logic_map: str | Path | "PolicyConfig",
    ) -> "PolicyConfig":
        """Load PolicyConfig from file path or return as-is."""
        from synkro.ingestion import PolicyConfig

        if isinstance(logic_map, (str, Path)):
            return PolicyConfig.load(logic_map)
        return logic_map

    @handle_error
    def generate(
        self,
        policy: Policy | str | None = None,
        traces: int = 20,
        return_logic_map: bool = False,
    ) -> Dataset | GenerationResult:
        """
        Generate a training dataset from a policy.

        Args:
            policy: Policy object or text string. Optional if a logic_map
                was provided to create_pipeline() (policy text is stored in config).
            traces: Target number of traces to generate (default: 20)
            return_logic_map: If True, return GenerationResult with access to
                the Logic Map, scenarios, and distribution (default: False)

        Returns:
            Dataset (default) or GenerationResult if return_logic_map=True

        Examples:
            >>> # Standard usage
            >>> dataset = generator.generate(policy, traces=50)

            >>> # With pre-computed logic_map, policy is optional
            >>> pipeline = create_pipeline(logic_map="./policy_config.json")
            >>> dataset = pipeline.generate(traces=50)

            >>> # Access Logic Map for inspection
            >>> result = generator.generate(policy, return_logic_map=True)
            >>> print(result.logic_map.rules)  # See extracted rules
            >>> print(result.distribution)     # See scenario type counts
            >>> dataset = result.dataset       # Get the dataset
        """
        # If no policy provided, try to use policy from config
        if policy is None:
            if self.policy_config is not None:
                policy = Policy(text=self.policy_config.policy_text)
            else:
                raise ValueError(
                    "policy is required when no logic_map was provided to create_pipeline()"
                )
        elif isinstance(policy, str):
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
        model_str = (
            self.generation_model.value
            if isinstance(self.generation_model, Enum)
            else str(self.generation_model)
        )

        return await self.pipeline.run(
            policy=policy,
            traces=traces,
            model=model_str,
            dataset_type=self.dataset_type.value,
            turns=self.turns,
            return_result=return_logic_map,
            policy_config=self.policy_config,
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

    @handle_error
    def generate_scenarios(
        self,
        policy: Policy | str,
        count: int = 20,
    ) -> ScenariosResult:
        """
        Generate eval scenarios without synthetic responses.

        This runs stages 0-2 of the pipeline (planning, logic extraction,
        scenario synthesis) but skips response generation. Use this for
        creating eval datasets where you want to test your own model.

        Args:
            policy: Policy object or text string
            count: Target number of scenarios to generate (default: 20)

        Returns:
            ScenariosResult with scenarios, logic_map, and distribution

        Examples:
            >>> result = generator.generate_scenarios(policy, count=100)
            >>> for scenario in result.scenarios:
            ...     response = my_model(scenario.user_message)
            ...     grade = synkro.grade(response, scenario, policy)
        """
        if isinstance(policy, str):
            policy = Policy(text=policy)

        # Validate policy has enough content
        policy.validate_length()

        return asyncio.run(self._generate_scenarios_async(policy, count))

    async def _generate_scenarios_async(
        self,
        policy: Policy,
        count: int,
    ) -> ScenariosResult:
        """Async implementation of scenario-only generation."""
        model_str = (
            self.generation_model.value
            if isinstance(self.generation_model, Enum)
            else str(self.generation_model)
        )

        return await self.pipeline.run_scenarios_only(
            policy=policy,
            count=count,
            model=model_str,
        )

    async def generate_scenarios_async(
        self,
        policy: Policy | str,
        count: int = 20,
    ) -> ScenariosResult:
        """
        Async version of generate_scenarios for use in async contexts.

        Args:
            policy: Policy object or text string
            count: Target number of scenarios to generate (default: 20)

        Returns:
            ScenariosResult with scenarios, logic_map, and distribution
        """
        if isinstance(policy, str):
            policy = Policy(text=policy)

        return await self._generate_scenarios_async(policy, count)
