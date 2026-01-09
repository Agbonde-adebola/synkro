"""Type definitions for Synkro.

Usage:
    from synkro.types import DatasetType, Message, Trace
    from synkro.types import ToolDefinition, ToolCall, ToolFunction
"""

from synkro.types.core import (
    Role,
    Message,
    Scenario,
    EvalScenario,
    Trace,
    GradeResult,
    Plan,
    Category,
)
from synkro.types.dataset_type import DatasetType
from synkro.types.tool import (
    ToolDefinition,
    ToolCall,
    ToolFunction,
    ToolResult,
)

__all__ = [
    # Dataset type
    "DatasetType",
    # Core types
    "Role",
    "Message",
    "Scenario",
    "EvalScenario",
    "Trace",
    "GradeResult",
    "Plan",
    "Category",
    # Tool types
    "ToolDefinition",
    "ToolCall",
    "ToolFunction",
    "ToolResult",
]
