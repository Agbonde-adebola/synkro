"""Type definitions for Synkro.

Usage:
    from synkro.types import DatasetType, Message, Trace
    from synkro.types import ToolDefinition, ToolCall, ToolFunction
    from synkro.types import Metrics, PipelineResult, Event
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

# New metrics types
from synkro.types.metrics import (
    PhaseMetrics,
    Metrics,
    TrackedLLM,
)

# New result types
from synkro.types.results import (
    ExtractionResult,
    ScenariosResult,
    TracesResult,
    VerificationResult,
    PipelineResult,
)

# New state types
from synkro.types.state import (
    PipelinePhase,
    PipelineState,
)

# New event types for streaming
from synkro.types.events import (
    EventType,
    Event,
    ProgressEvent,
    RuleFoundEvent,
    ScenarioGeneratedEvent,
    TraceGeneratedEvent,
    TraceVerifiedEvent,
    RefinementStartedEvent,
    TraceRefinedEvent,
    CoverageCalculatedEvent,
    CompleteEvent,
    ErrorEvent,
    StreamEvent,
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
    # Metrics types
    "PhaseMetrics",
    "Metrics",
    "TrackedLLM",
    # Result types
    "ExtractionResult",
    "ScenariosResult",
    "TracesResult",
    "VerificationResult",
    "PipelineResult",
    # State types
    "PipelinePhase",
    "PipelineState",
    # Event types
    "EventType",
    "Event",
    "ProgressEvent",
    "RuleFoundEvent",
    "ScenarioGeneratedEvent",
    "TraceGeneratedEvent",
    "TraceVerifiedEvent",
    "RefinementStartedEvent",
    "TraceRefinedEvent",
    "CoverageCalculatedEvent",
    "CompleteEvent",
    "ErrorEvent",
    "StreamEvent",
]
