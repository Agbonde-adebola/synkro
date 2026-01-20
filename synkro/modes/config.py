"""Mode configuration that bundles prompts, schema, and formatter per dataset type."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.dataset_type import DatasetType


@dataclass
class ModeConfig:
    """
    Configuration bundle for a dataset type.

    Defines all the prompts, schemas, and formatters needed
    for generating a specific type of dataset.
    """

    # Prompts
    scenario_prompt: str
    """Prompt for generating scenarios/questions"""

    response_prompt: str
    """Prompt for generating responses/answers"""

    grade_prompt: str
    """Prompt for grading quality"""

    refine_prompt: str
    """Prompt for refining failed responses"""

    # Output configuration
    output_description: str
    """Human-readable description of output format"""


def get_mode_config(dataset_type: "DatasetType") -> ModeConfig:
    """
    Get the mode configuration for a dataset type.

    Args:
        dataset_type: The type of dataset to generate

    Returns:
        ModeConfig with appropriate prompts and settings

    Example:
        >>> from synkro import DatasetType
        >>> config = get_mode_config(DatasetType.CONVERSATION)
    """
    from synkro.modes.conversation import CONVERSATION_CONFIG, EVALUATION_CONFIG, INSTRUCTION_CONFIG
    from synkro.modes.tool_call import TOOL_CALL_CONFIG

    configs = {
        "conversation": CONVERSATION_CONFIG,
        "instruction": INSTRUCTION_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "tool_call": TOOL_CALL_CONFIG,
    }

    type_value = dataset_type.value if hasattr(dataset_type, "value") else str(dataset_type)

    if type_value not in configs:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return configs[type_value]
