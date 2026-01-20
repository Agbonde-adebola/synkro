"""Advanced components for power users.

This module exposes internal components for developers who need fine-grained
control over the generation pipeline.

Usage:
    from synkro.advanced import (
        # Golden Trace components
        LogicExtractor,
        GoldenScenarioGenerator,
        GoldenResponseGenerator,
        TraceVerifier,
        GoldenRefiner,

        # Types
        LogicMap,
        Rule,
        GoldenScenario,
        VerificationResult,
        GenerationResult,

        # Pipeline internals
        GenerationPipeline,
        ComponentFactory,
    )

Examples:
    >>> # Extract Logic Map manually
    >>> from synkro.advanced import LogicExtractor, LLM
    >>> extractor = LogicExtractor(llm=LLM(model="gpt-4o"))
    >>> logic_map = await extractor.extract(policy_text)
    >>> print(logic_map.rules)

    >>> # Verify a trace against Logic Map
    >>> from synkro.advanced import TraceVerifier
    >>> verifier = TraceVerifier()
    >>> result = await verifier.verify(trace, logic_map, scenario)
    >>> if not result.passed:
    ...     print(f"Failed: {result.issues}")
"""

# Golden Trace components (The 4 Stages)
from synkro.factory import ComponentFactory

# Formatters
from synkro.formatters.messages import MessagesFormatter
from synkro.formatters.tool_call import ToolCallFormatter
from synkro.generation.follow_ups import FollowUpGenerator

# Low-level generators
from synkro.generation.generator import Generator
from synkro.generation.golden_responses import GoldenResponseGenerator
from synkro.generation.golden_scenarios import GoldenScenarioGenerator
from synkro.generation.golden_tool_responses import GoldenToolCallResponseGenerator
from synkro.generation.logic_extractor import LogicExtractor
from synkro.generation.multiturn_responses import MultiTurnResponseGenerator
from synkro.generation.planner import Planner
from synkro.generation.responses import ResponseGenerator
from synkro.generation.scenarios import ScenarioGenerator

# LLM client
from synkro.llm.client import LLM

# Pipeline phases
from synkro.pipeline.phases import (
    GoldenScenarioPhase,
    GoldenToolCallPhase,
    GoldenTracePhase,
    LogicExtractionPhase,
    PlanPhase,
    VerificationPhase,
)

# Pipeline internals
from synkro.pipeline.runner import GenerationPipeline, GenerationResult

# Prompts (for customization)
from synkro.prompts import GradePrompt, ResponsePrompt, ScenarioPrompt, SystemPrompt
from synkro.prompts.golden_templates import (
    GOLDEN_REFINE_PROMPT,
    GOLDEN_SCENARIO_PROMPT,
    GOLDEN_TOOL_TRACE_PROMPT,
    GOLDEN_TRACE_PROMPT,
    LOGIC_EXTRACTION_PROMPT,
    VERIFICATION_PROMPT,
)
from synkro.quality.golden_refiner import GoldenRefiner

# Quality components
from synkro.quality.grader import Grader
from synkro.quality.multiturn_grader import MultiTurnGrader
from synkro.quality.refiner import Refiner
from synkro.quality.tool_grader import ToolCallGrader
from synkro.quality.tool_refiner import ToolCallRefiner
from synkro.quality.verifier import TraceVerifier

# Schemas (for structured output)
from synkro.schemas import (
    GoldenScenarioOutput,
    GoldenScenariosArray,
    GoldenTraceOutput,
    LogicMapOutput,
    ReasoningStepOutput,
    RuleExtraction,
    VerificationOutput,
)

# Logic Map types
from synkro.types.logic_map import (
    GoldenScenario,
    LogicMap,
    ReasoningStep,
    Rule,
    RuleCategory,
    ScenarioType,
    VerificationResult,
)

__all__ = [
    # Golden Trace components
    "LogicExtractor",
    "GoldenScenarioGenerator",
    "GoldenResponseGenerator",
    "GoldenToolCallResponseGenerator",
    "TraceVerifier",
    "GoldenRefiner",
    # Logic Map types
    "LogicMap",
    "Rule",
    "RuleCategory",
    "GoldenScenario",
    "ScenarioType",
    "ReasoningStep",
    "VerificationResult",
    # Pipeline
    "GenerationPipeline",
    "GenerationResult",
    "ComponentFactory",
    # Phases
    "PlanPhase",
    "LogicExtractionPhase",
    "GoldenScenarioPhase",
    "GoldenTracePhase",
    "GoldenToolCallPhase",
    "VerificationPhase",
    # Generators
    "Generator",
    "ScenarioGenerator",
    "ResponseGenerator",
    "Planner",
    "FollowUpGenerator",
    "MultiTurnResponseGenerator",
    # Quality
    "Grader",
    "Refiner",
    "ToolCallGrader",
    "ToolCallRefiner",
    "MultiTurnGrader",
    # LLM
    "LLM",
    # Prompts
    "SystemPrompt",
    "ScenarioPrompt",
    "ResponsePrompt",
    "GradePrompt",
    "LOGIC_EXTRACTION_PROMPT",
    "GOLDEN_SCENARIO_PROMPT",
    "GOLDEN_TRACE_PROMPT",
    "VERIFICATION_PROMPT",
    "GOLDEN_REFINE_PROMPT",
    "GOLDEN_TOOL_TRACE_PROMPT",
    # Formatters
    "MessagesFormatter",
    "ToolCallFormatter",
    # Schemas
    "RuleExtraction",
    "LogicMapOutput",
    "GoldenScenarioOutput",
    "GoldenScenariosArray",
    "ReasoningStepOutput",
    "GoldenTraceOutput",
    "VerificationOutput",
]
