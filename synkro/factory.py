"""Component factory for dependency injection.

This module provides a factory for creating pipeline components,
enabling testability and flexible configuration.

Supports both legacy components and Golden Trace components:
- Logic Extractor (The Cartographer)
- Golden Scenario Generator (The Adversary)
- Golden Response Generator (The Thinker)
- Trace Verifier (The Auditor)
- Golden Refiner
"""

from typing import TYPE_CHECKING

from synkro.llm.client import LLM
from synkro.modes.config import ModeConfig
from synkro.generation.planner import Planner
from synkro.generation.scenarios import ScenarioGenerator
from synkro.generation.responses import ResponseGenerator
from synkro.generation.follow_ups import FollowUpGenerator
from synkro.generation.multiturn_responses import MultiTurnResponseGenerator
from synkro.quality.grader import Grader
from synkro.quality.refiner import Refiner
from synkro.quality.multiturn_grader import MultiTurnGrader

if TYPE_CHECKING:
    from synkro.types.tool import ToolDefinition
    from synkro.generation.tool_simulator import ToolSimulator
    from synkro.generation.tool_responses import ToolCallResponseGenerator
    from synkro.quality.tool_grader import ToolCallGrader
    from synkro.quality.tool_refiner import ToolCallRefiner
    from synkro.generation.logic_extractor import LogicExtractor
    from synkro.generation.golden_scenarios import GoldenScenarioGenerator
    from synkro.generation.golden_responses import GoldenResponseGenerator
    from synkro.generation.golden_tool_responses import GoldenToolCallResponseGenerator
    from synkro.quality.verifier import TraceVerifier
    from synkro.quality.golden_refiner import GoldenRefiner
    from synkro.interactive.logic_map_editor import LogicMapEditor
    from synkro.interactive.scenario_editor import ScenarioEditor
    from synkro.coverage.taxonomy_extractor import TaxonomyExtractor
    from synkro.coverage.scenario_tagger import ScenarioTagger
    from synkro.coverage.calculator import CoverageCalculator
    from synkro.coverage.improver import CoverageImprover


class ComponentFactory:
    """
    Factory for creating pipeline components with shared LLM clients.
    
    This centralizes component creation and ensures consistent configuration
    across the pipeline.
    
    Examples:
        >>> factory = ComponentFactory(gen_llm, grade_llm, mode_config)
        >>> planner = factory.create_planner()
        >>> grader = factory.create_grader()
        
        >>> # With tools for tool_call dataset type
        >>> factory = ComponentFactory(gen_llm, grade_llm, mode_config, tools=[...])
        >>> simulator = factory.create_tool_simulator()
    """
    
    def __init__(
        self,
        generation_llm: LLM,
        grading_llm: LLM,
        mode_config: ModeConfig,
        tools: list["ToolDefinition"] | None = None,
        thinking: bool = False,
    ):
        """
        Initialize the factory.

        Args:
            generation_llm: LLM client for generation tasks (scenarios, responses, refinement)
            grading_llm: LLM client for grading and planning (typically stronger model)
            mode_config: Configuration for the dataset type (prompts, etc.)
            tools: Optional list of tool definitions for tool_call dataset type
            thinking: Enable thinking mode with <think> tags in responses
        """
        self.generation_llm = generation_llm
        self.grading_llm = grading_llm
        self.mode_config = mode_config
        self.tools = tools or []
        self.thinking = thinking
    
    def create_planner(self) -> Planner:
        """Create a Planner instance."""
        return Planner(llm=self.grading_llm)
    
    def create_scenario_generator(self) -> ScenarioGenerator:
        """Create a ScenarioGenerator with mode-specific prompts."""
        gen = ScenarioGenerator(llm=self.generation_llm)
        gen.prompt_template = self.mode_config.scenario_prompt
        return gen
    
    def create_response_generator(self) -> ResponseGenerator:
        """Create a ResponseGenerator with mode-specific prompts."""
        gen = ResponseGenerator(llm=self.generation_llm)
        gen.prompt_template = self.mode_config.response_prompt
        return gen
    
    def create_grader(self) -> "Grader | ToolCallGrader":
        """
        Create a Grader with mode-specific prompts.
        
        Auto-selects ToolCallGrader when tools are configured.
        """
        if self.has_tools:
            from synkro.quality.tool_grader import ToolCallGrader
            return ToolCallGrader(llm=self.grading_llm, tools=self.tools)
        
        grader = Grader(llm=self.grading_llm)
        grader.prompt_template = self.mode_config.grade_prompt
        return grader
    
    def create_refiner(self) -> "Refiner | ToolCallRefiner":
        """
        Create a Refiner with mode-specific prompts.
        
        Auto-selects ToolCallRefiner when tools are configured.
        This ensures tool_calls format is preserved during refinement.
        """
        if self.has_tools:
            from synkro.quality.tool_refiner import ToolCallRefiner
            simulator = self.create_tool_simulator()
            return ToolCallRefiner(
                llm=self.generation_llm,
                tools=self.tools,
                simulator=simulator,
            )
        
        refiner = Refiner(llm=self.generation_llm)
        refiner.prompt_template = self.mode_config.refine_prompt
        return refiner
    
    def create_tool_simulator(self) -> "ToolSimulator":
        """Create a ToolSimulator instance for tool_call dataset type."""
        from synkro.generation.tool_simulator import ToolSimulator
        
        if not self.tools:
            raise ValueError("Cannot create ToolSimulator without tools")
        
        return ToolSimulator(tools=self.tools, llm=self.generation_llm)
    
    def create_tool_call_response_generator(self) -> "ToolCallResponseGenerator":
        """
        Create a ToolCallResponseGenerator for generating proper tool call traces.
        
        This generator uses JSON mode to produce structured tool calls in
        OpenAI function calling format.
        """
        from synkro.generation.tool_responses import ToolCallResponseGenerator
        
        if not self.tools:
            raise ValueError("Cannot create ToolCallResponseGenerator without tools")
        
        # Create simulator for generating tool responses
        simulator = self.create_tool_simulator()
        
        return ToolCallResponseGenerator(
            tools=self.tools,
            llm=self.generation_llm,
            simulator=simulator,
        )
    
    def get_tools_description(self) -> str:
        """Get formatted description of all available tools."""
        if not self.tools:
            return "No tools available"
        
        descriptions = []
        for tool in self.tools:
            descriptions.append(tool.to_system_prompt())
        return "\n\n".join(descriptions)
    
    @property
    def has_tools(self) -> bool:
        """Check if tools are configured."""
        return bool(self.tools)

    def create_follow_up_generator(self) -> FollowUpGenerator:
        """Create a FollowUpGenerator for multi-turn conversations."""
        return FollowUpGenerator(llm=self.generation_llm)

    def create_multi_turn_response_generator(self) -> MultiTurnResponseGenerator:
        """Create a MultiTurnResponseGenerator for multi-turn trace generation."""
        return MultiTurnResponseGenerator(llm=self.generation_llm)

    def create_multi_turn_grader(self) -> MultiTurnGrader:
        """Create a MultiTurnGrader for per-turn and overall conversation grading."""
        return MultiTurnGrader(llm=self.grading_llm)

    # =========================================================================
    # GOLDEN TRACE COMPONENTS
    # =========================================================================

    def create_logic_extractor(self) -> "LogicExtractor":
        """
        Create a LogicExtractor (The Cartographer).

        Uses the grading LLM (stronger model) for accurate rule extraction.
        """
        from synkro.generation.logic_extractor import LogicExtractor
        return LogicExtractor(llm=self.grading_llm)

    def create_golden_scenario_generator(self) -> "GoldenScenarioGenerator":
        """
        Create a GoldenScenarioGenerator (The Adversary).

        Generates typed scenarios (positive, negative, edge_case, irrelevant)
        with rule targeting.
        """
        from synkro.generation.golden_scenarios import GoldenScenarioGenerator
        return GoldenScenarioGenerator(llm=self.generation_llm)

    def create_golden_response_generator(self) -> "GoldenResponseGenerator":
        """
        Create a GoldenResponseGenerator (The Thinker).

        Generates traces with grounded Chain-of-Thought reasoning
        and rule citations.
        """
        from synkro.generation.golden_responses import GoldenResponseGenerator
        return GoldenResponseGenerator(llm=self.generation_llm, thinking=self.thinking)

    def create_golden_tool_call_generator(self) -> "GoldenToolCallResponseGenerator":
        """
        Create a GoldenToolCallResponseGenerator (The Thinker for Tools).

        Generates tool call traces with rule citations for tool selection
        decisions. Requires tools to be configured.
        """
        from synkro.generation.golden_tool_responses import GoldenToolCallResponseGenerator

        if not self.tools:
            raise ValueError("Cannot create GoldenToolCallResponseGenerator without tools")

        simulator = self.create_tool_simulator()
        return GoldenToolCallResponseGenerator(
            tools=self.tools,
            llm=self.generation_llm,
            simulator=simulator,
            thinking=self.thinking,
        )

    def create_verifier(self) -> "TraceVerifier":
        """
        Create a TraceVerifier (The Auditor).

        Verifies traces against the Logic Map to ensure:
        - No skipped rules
        - No hallucinated rules
        - No contradictions
        - DAG compliance

        Uses the grading LLM (stronger model) for accurate verification.
        """
        from synkro.quality.verifier import TraceVerifier
        return TraceVerifier(llm=self.grading_llm)

    def create_golden_refiner(self) -> "GoldenRefiner":
        """
        Create a GoldenRefiner.

        Refines traces that fail verification, using Logic Map context
        to fix skipped rules, hallucinations, and contradictions.
        """
        from synkro.quality.golden_refiner import GoldenRefiner
        return GoldenRefiner(llm=self.generation_llm)

    def create_logic_map_editor(self) -> "LogicMapEditor":
        """
        Create a LogicMapEditor for Human-in-the-Loop sessions.

        The editor uses the grading LLM (stronger model) to interpret
        natural language feedback and refine Logic Maps.
        """
        from synkro.interactive.logic_map_editor import LogicMapEditor
        return LogicMapEditor(llm=self.grading_llm)

    def create_scenario_editor(self) -> "ScenarioEditor":
        """
        Create a ScenarioEditor for Human-in-the-Loop scenario editing.

        The editor uses the grading LLM (stronger model) to interpret
        natural language feedback and refine scenarios.
        """
        from synkro.interactive.scenario_editor import ScenarioEditor
        return ScenarioEditor(llm=self.grading_llm)

    # =========================================================================
    # COVERAGE TRACKING COMPONENTS
    # =========================================================================

    def create_taxonomy_extractor(self) -> "TaxonomyExtractor":
        """
        Create a TaxonomyExtractor for sub-category extraction.

        Uses the grading LLM (stronger model) for accurate taxonomy extraction.
        """
        from synkro.coverage.taxonomy_extractor import TaxonomyExtractor
        return TaxonomyExtractor(llm=self.grading_llm)

    def create_scenario_tagger(self) -> "ScenarioTagger":
        """
        Create a ScenarioTagger for tagging scenarios with sub-categories.

        Uses the generation LLM (faster) for batch tagging operations.
        """
        from synkro.coverage.scenario_tagger import ScenarioTagger
        return ScenarioTagger(llm=self.generation_llm)

    def create_coverage_calculator(self) -> "CoverageCalculator":
        """
        Create a CoverageCalculator for computing coverage metrics.

        Uses the generation LLM for generating improvement suggestions.
        """
        from synkro.coverage.calculator import CoverageCalculator
        return CoverageCalculator(llm=self.generation_llm)

    def create_coverage_improver(self) -> "CoverageImprover":
        """
        Create a CoverageImprover for generating gap-filling scenarios.

        Uses the generation LLM for creating new scenarios.
        """
        from synkro.coverage.improver import CoverageImprover
        return CoverageImprover(llm=self.generation_llm)


__all__ = ["ComponentFactory"]

