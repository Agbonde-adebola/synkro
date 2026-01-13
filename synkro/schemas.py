"""Pydantic schemas for structured LLM outputs and validation."""

from typing import Literal
from pydantic import BaseModel, Field


# =============================================================================
# SCENARIO SCHEMAS
# =============================================================================


class ScenarioOutput(BaseModel):
    """Output schema for scenario generation."""

    scenario: str = Field(description="Detailed scenario description")
    context: str = Field(description="Relevant background information")


class ScenariosArray(BaseModel):
    """Array of generated scenarios."""

    scenarios: list[ScenarioOutput]


# =============================================================================
# POLICY ANALYSIS SCHEMAS
# =============================================================================


class PolicyComplexity(BaseModel):
    """Policy complexity analysis for auto-detecting optimal turns."""

    variable_count: int = Field(
        description="Number of variables/conditions in the policy (rules, exceptions, conditions)"
    )
    complexity_level: Literal["simple", "conditional", "complex"] = Field(
        description="Overall complexity: simple (1 var), conditional (2-3 vars), complex (4+ vars)"
    )
    recommended_turns: int = Field(
        ge=1, le=6, description="Recommended conversation turns based on complexity"
    )
    reasoning: str = Field(description="Brief explanation of the complexity assessment")


class PlanCategory(BaseModel):
    """A category in the generation plan."""

    name: str = Field(description='Short category name (e.g., "Consent Violations", "Edge Cases")')
    description: str = Field(description="What this category tests")
    traces: int = Field(ge=1, description="Number of traces to generate for this category")


class PolicyPlan(BaseModel):
    """LLM-generated plan for dataset generation."""

    categories: list[PlanCategory] = Field(
        min_length=2, max_length=10, description="Scenario categories with trace allocations"
    )
    reasoning: str = Field(
        description="Explanation of why these categories were chosen based on policy content"
    )


# =============================================================================
# CHAT MESSAGE SCHEMAS
# =============================================================================


class ChatMessage(BaseModel):
    """A single chat message in OpenAI format."""

    role: Literal["system", "user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")


class ConversationOutput(BaseModel):
    """Output from response generation - a complete conversation."""

    index: int = Field(description="Scenario index (0-based)")
    messages: list[ChatMessage] = Field(
        description="Full conversation with system, user, and assistant messages"
    )


class BatchedConversations(BaseModel):
    """Batch of generated conversations."""

    conversations: list[ConversationOutput]


# =============================================================================
# GRADING SCHEMAS
# =============================================================================


class GradeOutput(BaseModel):
    """Grading result for a single response."""

    index: int = Field(description="Scenario index (0-based)")
    passed: bool = Field(
        alias="pass", description="Is the response FULLY correct, policy-compliant, and format-valid?"
    )
    policy_violations: list[str] = Field(
        default_factory=list,
        description="Specific policy rules that were violated or misinterpreted",
    )
    missing_citations: list[str] = Field(
        default_factory=list,
        description="Policy sections that should have been cited but were not",
    )
    incomplete_reasoning: list[str] = Field(
        default_factory=list, description="Logical gaps or missing steps in the chain of thought"
    )
    vague_recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations that need to be more specific or actionable",
    )
    feedback: str = Field(description="Summary of how to fix the issues")

    class Config:
        populate_by_name = True


class BatchedGrades(BaseModel):
    """Batch of grading results."""

    grades: list[GradeOutput]


# =============================================================================
# SINGLE-ITEM SCHEMAS (for parallel generation)
# =============================================================================


class SingleResponse(BaseModel):
    """Single response output for parallel generation."""

    messages: list[ChatMessage] = Field(
        min_length=3, max_length=3, description="Exactly 3 messages: system, user, assistant"
    )


class MultiTurnResponse(BaseModel):
    """Multi-turn response output for complexity-driven generation."""

    messages: list[ChatMessage] = Field(
        min_length=3,
        description="Conversation messages (variable length based on turn count)"
    )
    turn_count: int = Field(
        ge=1, le=10,
        description="Number of user-assistant exchanges in this conversation"
    )


class SingleGrade(BaseModel):
    """Single grade output for parallel generation."""

    passed: bool = Field(
        alias="pass", description="Is the response FULLY correct, policy-compliant, and format-valid?"
    )
    policy_violations: list[str] = Field(
        default_factory=list, description="Specific policy rules that were violated"
    )
    missing_citations: list[str] = Field(
        default_factory=list, description="Policy sections that should have been cited"
    )
    incomplete_reasoning: list[str] = Field(
        default_factory=list, description="Logical gaps or missing reasoning steps"
    )
    vague_recommendations: list[str] = Field(
        default_factory=list, description="Recommendations that need to be more specific"
    )
    feedback: str = Field(description='Summary of issues or "Correct" if passing')

    class Config:
        populate_by_name = True


# =============================================================================
# MULTI-TURN SCHEMAS
# =============================================================================


class FollowUpQuestion(BaseModel):
    """A follow-up question for multi-turn conversations."""

    index: int = Field(description="Scenario index")
    question: str = Field(description="Follow-up question from the user")
    question_type: Literal["clarification", "edge_case", "what_if", "specificity", "challenge"] = (
        Field(description="Type of follow-up")
    )


class TurnGrade(BaseModel):
    """Grade for a single turn in a multi-turn conversation."""

    turn_index: int = Field(description="Which turn (0-based, only assistant turns)")
    passed: bool = Field(alias="pass", description="Does this turn pass all criteria?")
    policy_violations: list[str] = Field(
        default_factory=list, description="Policy violations in this turn"
    )
    missing_citations: list[str] = Field(
        default_factory=list, description="Missing citations in this turn"
    )
    incomplete_reasoning: list[str] = Field(
        default_factory=list, description="Reasoning gaps in this turn"
    )
    vague_recommendations: list[str] = Field(
        default_factory=list, description="Vague recommendations in this turn"
    )
    feedback: str = Field(description="Specific feedback for this turn")

    class Config:
        populate_by_name = True


class ConversationGrade(BaseModel):
    """Full grading for a multi-turn conversation."""

    index: int = Field(description="Scenario index")
    overall_pass: bool = Field(description="Does the ENTIRE conversation pass?")
    turn_grades: list[TurnGrade] = Field(description="Grade for each assistant turn")
    coherence_pass: bool = Field(
        description="Is the conversation coherent with no contradictions?"
    )
    coherence_issues: list[str] = Field(
        default_factory=list, description="Any contradictions or incoherence across turns"
    )
    progressive_depth: bool = Field(
        description="Does each turn build on previous context appropriately?"
    )
    overall_feedback: str = Field(
        description="Summary of what needs to be fixed across the conversation"
    )


# =============================================================================
# AGENTIC SCHEMAS
# =============================================================================


class ToolCall(BaseModel):
    """A tool call in an agentic trace."""

    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict[str, str] = Field(description="Arguments to pass to the tool")


class AgenticStep(BaseModel):
    """A single step in an agentic trace."""

    reasoning: str = Field(description="Reasoning before tool call")
    tool_name: str = Field(description="Tool to call")
    tool_args: dict = Field(description="Tool arguments")


class AgenticTrace(BaseModel):
    """Complete agentic trace with tool usage."""

    index: int = Field(description="Scenario index")
    steps: list[AgenticStep] = Field(description="Steps of tool usage")
    final_answer: str = Field(description="Final comprehensive answer")


# =============================================================================
# TOOL CALL GRADING SCHEMAS
# =============================================================================


class ToolCallGrade(BaseModel):
    """Grading result for a tool call trace.
    
    Evaluates tool usage on four criteria:
    - Tool Selection: Did they use the right tool?
    - Parameter Accuracy: Were the parameters correct?
    - Response Synthesis: Did they use tool results correctly?
    - Timing: Did they call tools at the right time?
    """
    
    passed: bool = Field(
        alias="pass",
        description="Does the trace pass ALL criteria?"
    )
    
    # Criterion 1: Tool Selection
    tool_selection_correct: bool = Field(
        description="Did the assistant choose the appropriate tool for the task?"
    )
    tool_selection_issues: list[str] = Field(
        default_factory=list,
        description="Specific issues with tool selection (wrong tool, missing tool, unnecessary tool)"
    )
    
    # Criterion 2: Parameter Accuracy
    parameters_valid: bool = Field(
        description="Were the tool parameters correct (types, values, required fields)?"
    )
    parameter_issues: list[str] = Field(
        default_factory=list,
        description="Specific issues with parameters (wrong type, invalid value, missing required)"
    )
    
    # Criterion 3: Response Synthesis
    synthesis_accurate: bool = Field(
        description="Did the assistant correctly use tool results without hallucination?"
    )
    synthesis_issues: list[str] = Field(
        default_factory=list,
        description="Specific issues with synthesis (hallucinated data, ignored results, misinterpreted)"
    )
    
    # Criterion 4: Timing
    timing_appropriate: bool = Field(
        description="Did the assistant call tools at the right moment?"
    )
    timing_issues: list[str] = Field(
        default_factory=list,
        description="Specific issues with timing (premature call, delayed call, should have called earlier)"
    )
    
    # Overall feedback
    feedback: str = Field(
        description="Summary of issues or 'Correct' if passing"
    )
    
    class Config:
        populate_by_name = True
    
    def get_all_issues(self) -> list[str]:
        """Get all issues combined."""
        return (
            self.tool_selection_issues
            + self.parameter_issues
            + self.synthesis_issues
            + self.timing_issues
        )


# =============================================================================
# GOLDEN TRACE SCHEMAS
# =============================================================================


class RuleExtraction(BaseModel):
    """A single rule extracted from the policy."""

    rule_id: str = Field(description="Unique identifier (e.g., 'R001')")
    text: str = Field(description="Exact rule text from the policy")
    condition: str = Field(description="The 'if' part - when this rule applies")
    action: str = Field(description="The 'then' part - what happens")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Rule IDs that must be evaluated before this rule"
    )
    category: Literal["constraint", "permission", "procedure", "exception"] = Field(
        description="Type of rule"
    )


class LogicMapOutput(BaseModel):
    """Output schema for logic extraction - the complete DAG of rules."""

    rules: list[RuleExtraction] = Field(
        description="All rules extracted from the policy"
    )
    root_rules: list[str] = Field(
        description="Rule IDs with no dependencies (entry points)"
    )
    reasoning: str = Field(
        description="Explanation of rule extraction and relationship identification"
    )


class RefinedLogicMapOutput(BaseModel):
    """Output schema for Logic Map refinement based on user feedback."""

    rules: list[RuleExtraction] = Field(
        description="All rules in the refined Logic Map (modified and unmodified)"
    )
    root_rules: list[str] = Field(
        description="Rule IDs with no dependencies (entry points)"
    )
    changes_summary: str = Field(
        description="Brief summary of changes made (e.g., 'Added R009, removed R005')"
    )
    reasoning: str = Field(
        description="Explanation of how user feedback was interpreted and applied"
    )


class HITLIntent(BaseModel):
    """Classified user intent in unified HITL session."""

    intent_type: Literal["turns", "rules", "scenarios", "compound", "coverage", "command", "unclear"] = Field(
        description="Type of user intent: turns adjustment, rule modification, scenario editing, compound (rules + scenarios together), coverage operations, command, or unclear"
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the classification"
    )

    # For turns changes
    target_turns: int | None = Field(
        default=None, ge=1, le=6, description="Target conversation turns (1-6) if intent is turns"
    )
    turns_reasoning: str | None = Field(
        default=None, description="Explanation of why this turn count was chosen"
    )

    # For rule changes (passthrough to existing LogicMapEditor)
    rule_feedback: str | None = Field(
        default=None, description="Original user feedback for rule modification"
    )

    # For scenario changes
    scenario_operation: Literal["add", "delete", "modify", "distribution"] | None = Field(
        default=None, description="Type of scenario operation"
    )
    scenario_target: str | None = Field(
        default=None, description="Target scenario (S3, 'the refund scenario', 'all edge cases')"
    )
    scenario_feedback: str | None = Field(
        default=None, description="Original user feedback for scenario modification"
    )

    # For coverage operations
    coverage_operation: Literal["view", "increase", "target"] | None = Field(
        default=None, description="Type of coverage operation"
    )
    coverage_target_sub_category: str | None = Field(
        default=None, description="Target sub-category name or ID"
    )
    coverage_target_percent: int | None = Field(
        default=None, ge=0, le=100, description="Target coverage percentage"
    )
    coverage_increase_amount: int | None = Field(
        default=None, ge=1, le=100, description="Percentage points to increase coverage"
    )
    coverage_scenario_type: str | None = Field(
        default=None, description="Specific scenario type to add (positive/negative/edge_case)"
    )
    coverage_view_mode: Literal["summary", "gaps", "heatmap", "detail"] | None = Field(
        default=None, description="What to display for coverage view commands"
    )


class GoldenScenarioOutput(BaseModel):
    """Output schema for a single golden scenario."""

    description: str = Field(description="The user's request or question")
    context: str = Field(default="", description="Additional context")
    scenario_type: Literal["positive", "negative", "edge_case", "irrelevant"] = Field(
        description="Type of scenario"
    )
    target_rule_ids: list[str] = Field(
        description="Rule IDs this scenario tests"
    )
    expected_outcome: str = Field(
        description="Expected behavior based on rules"
    )


class GoldenScenariosArray(BaseModel):
    """Array of generated golden scenarios."""

    scenarios: list[GoldenScenarioOutput]


class RefinedScenariosOutput(BaseModel):
    """Result of scenario refinement in HITL session."""

    scenarios: list[GoldenScenarioOutput] = Field(
        description="All scenarios after applying changes"
    )
    changes_summary: str = Field(
        description="Brief summary of what changed (e.g., 'Added S21, deleted S3')"
    )
    reasoning: str = Field(
        description="Explanation of how user feedback was interpreted"
    )


class ReasoningStepOutput(BaseModel):
    """A single step in the Chain-of-Thought reasoning."""

    rule_id: str = Field(description="The rule being evaluated")
    rule_text: str = Field(description="The text of the rule")
    applies: bool = Field(description="Whether this rule applies")
    reasoning: str = Field(description="Why the rule does/doesn't apply")
    exclusions: list[str] = Field(
        default_factory=list,
        description="Rule IDs excluded because this rule applies"
    )


class GoldenTraceOutput(BaseModel):
    """Output schema for a golden trace with grounded reasoning."""

    messages: list[ChatMessage] = Field(
        description="The conversation messages"
    )
    reasoning_chain: list[ReasoningStepOutput] = Field(
        description="Step-by-step reasoning with rule citations"
    )
    rules_applied: list[str] = Field(
        description="Rule IDs that were applied in the response"
    )
    rules_excluded: list[str] = Field(
        default_factory=list,
        description="Rule IDs that were explicitly excluded and why"
    )


class VerificationOutput(BaseModel):
    """Output schema for trace verification against Logic Map."""

    passed: bool = Field(description="Whether the trace passed verification")
    issues: list[str] = Field(
        default_factory=list,
        description="List of issues found"
    )
    skipped_rules: list[str] = Field(
        default_factory=list,
        description="Rule IDs that should have been applied but weren't"
    )
    hallucinated_rules: list[str] = Field(
        default_factory=list,
        description="Rule IDs cited that don't exist or don't apply"
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Logical contradictions found"
    )
    rules_verified: list[str] = Field(
        default_factory=list,
        description="Rule IDs correctly applied"
    )
    feedback: str = Field(
        description="Summary of verification result"
    )


# =============================================================================
# COVERAGE SCHEMAS
# =============================================================================


class SubCategoryOutput(BaseModel):
    """Output schema for a single sub-category extraction."""

    id: str = Field(description="Unique identifier (e.g., 'SC001')")
    name: str = Field(description="Short, descriptive name")
    description: str = Field(description="What this sub-category covers")
    parent_category: str = Field(description="Name of the parent category")
    related_rule_ids: list[str] = Field(
        default_factory=list,
        description="Rule IDs from LogicMap that relate to this sub-category"
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Coverage priority based on policy importance"
    )


class TaxonomyOutput(BaseModel):
    """Output schema for sub-category taxonomy extraction."""

    sub_categories: list[SubCategoryOutput] = Field(
        description="All extracted sub-categories"
    )
    reasoning: str = Field(
        description="Explanation of how the taxonomy was organized"
    )


class ScenarioTaggingOutput(BaseModel):
    """Output schema for tagging a scenario with sub-categories."""

    scenario_index: int = Field(description="Index of the scenario being tagged")
    sub_category_ids: list[str] = Field(
        description="Sub-category IDs this scenario covers"
    )


class BatchedScenarioTagging(BaseModel):
    """Batch output for scenario tagging."""

    taggings: list[ScenarioTaggingOutput] = Field(
        description="Tagging results for each scenario"
    )


class CoverageSuggestionsOutput(BaseModel):
    """Output schema for coverage improvement suggestions."""

    suggestions: list[str] = Field(
        description="Actionable suggestions to improve coverage"
    )
    reasoning: str = Field(
        description="Explanation of how suggestions were prioritized"
    )

