"""Tool Call mode configuration."""

from synkro.modes.config import ModeConfig
from synkro.prompts.tool_templates import (
    TOOL_GRADE_PROMPT,
    TOOL_REFINE_PROMPT,
    TOOL_RESPONSE_PROMPT,
    TOOL_SCENARIO_PROMPT,
)

TOOL_CALL_CONFIG = ModeConfig(
    scenario_prompt=TOOL_SCENARIO_PROMPT,
    response_prompt=TOOL_RESPONSE_PROMPT,
    grade_prompt=TOOL_GRADE_PROMPT,
    refine_prompt=TOOL_REFINE_PROMPT,
    output_description="Tool calling: {messages: [system, user, {tool_calls}, {tool}, assistant]}",
)
