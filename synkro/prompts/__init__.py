"""Prompt templates and customizable prompt classes for Synkro."""

from synkro.prompts.base import (
    GradePrompt,
    PlanPrompt,
    RefinePrompt,
    ResponsePrompt,
    ScenarioPrompt,
    SystemPrompt,
)
from synkro.prompts.multiturn_templates import (
    FOLLOW_UP_GENERATION_PROMPT,
    MULTI_TURN_GRADE_PROMPT,
    MULTI_TURN_INITIAL_PROMPT,
    MULTI_TURN_REFINE_PROMPT,
    MULTI_TURN_RESPONSE_PROMPT,
)
from synkro.prompts.templates import (
    BATCHED_GRADER_PROMPT,
    BATCHED_REFINER_PROMPT,
    BATCHED_RESPONSE_PROMPT,
    CATEGORY_SCENARIO_PROMPT,
    POLICY_COMPLEXITY_PROMPT,
    POLICY_PLANNING_PROMPT,
    SCENARIO_GENERATOR_PROMPT,
    SINGLE_GRADE_PROMPT,
    SINGLE_RESPONSE_PROMPT,
    SYSTEM_PROMPT,
)

__all__ = [
    # Prompt classes
    "SystemPrompt",
    "ScenarioPrompt",
    "ResponsePrompt",
    "GradePrompt",
    "RefinePrompt",
    "PlanPrompt",
    # Raw templates
    "SYSTEM_PROMPT",
    "SCENARIO_GENERATOR_PROMPT",
    "CATEGORY_SCENARIO_PROMPT",
    "POLICY_PLANNING_PROMPT",
    "POLICY_COMPLEXITY_PROMPT",
    "BATCHED_RESPONSE_PROMPT",
    "BATCHED_GRADER_PROMPT",
    "BATCHED_REFINER_PROMPT",
    "SINGLE_RESPONSE_PROMPT",
    "SINGLE_GRADE_PROMPT",
    # Multi-turn templates
    "FOLLOW_UP_GENERATION_PROMPT",
    "MULTI_TURN_RESPONSE_PROMPT",
    "MULTI_TURN_INITIAL_PROMPT",
    "MULTI_TURN_GRADE_PROMPT",
    "MULTI_TURN_REFINE_PROMPT",
]
