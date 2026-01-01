"""Component factory for dependency injection.

This module provides a factory for creating pipeline components,
enabling testability and flexible configuration.
"""

from synkro.llm.client import LLM
from synkro.modes.config import ModeConfig
from synkro.generation.planner import Planner
from synkro.generation.scenarios import ScenarioGenerator
from synkro.generation.responses import ResponseGenerator
from synkro.quality.grader import Grader
from synkro.quality.refiner import Refiner


class ComponentFactory:
    """
    Factory for creating pipeline components with shared LLM clients.
    
    This centralizes component creation and ensures consistent configuration
    across the pipeline.
    
    Examples:
        >>> factory = ComponentFactory(gen_llm, grade_llm, mode_config)
        >>> planner = factory.create_planner()
        >>> grader = factory.create_grader()
    """
    
    def __init__(
        self,
        generation_llm: LLM,
        grading_llm: LLM,
        mode_config: ModeConfig,
    ):
        """
        Initialize the factory.
        
        Args:
            generation_llm: LLM client for generation tasks (scenarios, responses, refinement)
            grading_llm: LLM client for grading and planning (typically stronger model)
            mode_config: Configuration for the dataset type (prompts, etc.)
        """
        self.generation_llm = generation_llm
        self.grading_llm = grading_llm
        self.mode_config = mode_config
    
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
    
    def create_grader(self) -> Grader:
        """Create a Grader with mode-specific prompts."""
        grader = Grader(llm=self.grading_llm)
        grader.prompt_template = self.mode_config.grade_prompt
        return grader
    
    def create_refiner(self) -> Refiner:
        """Create a Refiner with mode-specific prompts."""
        refiner = Refiner(llm=self.generation_llm)
        refiner.prompt_template = self.mode_config.refine_prompt
        return refiner


__all__ = ["ComponentFactory"]

