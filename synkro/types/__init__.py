"""Type definitions for Synkro.

Usage:
    from synkro.types import DatasetType, Message, Trace
    from synkro.types import ToolDefinition, ToolCall, ToolFunction
    from synkro.types import Metrics, PipelineResult, Event
"""

from synkro.types.core import (
    Category,
    EvalScenario,
    GradeResult,
    Message,
    Plan,
    Role,
    Scenario,
    Trace,
)
from synkro.types.dataset_type import DatasetType

# New event types for streaming
from synkro.types.events import (
    CompleteEvent,
    CoverageCalculatedEvent,
    ErrorEvent,
    Event,
    EventType,
    ProgressEvent,
    RefinementStartedEvent,
    RuleFoundEvent,
    ScenarioGeneratedEvent,
    StreamEvent,
    TraceGeneratedEvent,
    TraceRefinedEvent,
    TraceVerifiedEvent,
)

# New metrics types
from synkro.types.metrics import (
    Metrics,
    PhaseMetrics,
    TrackedLLM,
)

# New result types
from synkro.types.results import (
    ExtractionResult,
    PipelineResult,
    ScenariosResult,
    TracesResult,
    VerificationResult,
)

# New state types
from synkro.types.state import (
    PipelinePhase,
    PipelineState,
)
from synkro.types.tool import (
    ToolCall,
    ToolDefinition,
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
