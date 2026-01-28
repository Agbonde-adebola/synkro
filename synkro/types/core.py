"""Core Pydantic models for Synkro."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """
    A single message in a conversation.

    Supports both regular chat messages and tool-calling messages.

    Examples:
        >>> # Regular message
        >>> Message(role="user", content="Hello")

        >>> # Assistant with tool call (tool_calls is list of dicts or ToolCall objects)
        >>> Message(role="assistant", content=None, tool_calls=[...])

        >>> # Tool response
        >>> Message(role="tool", content="Result", tool_call_id="call_123")
    """

    role: Role
    content: str | None = None
    tool_calls: list[Any] | None = Field(
        default=None, description="Tool calls made by the assistant (list of ToolCall or dicts)"
    )
    tool_call_id: str | None = Field(
        default=None, description="ID of the tool call this message responds to (for tool role)"
    )

    def model_post_init(self, __context) -> None:
        """Validate message structure based on role."""
        # For backwards compatibility, ensure content is string for non-tool roles
        if self.role in ("system", "user") and self.content is None:
            self.content = ""

    def model_dump(self, **kwargs) -> dict:
        """Exclude None tool fields for clean output."""
        data = super().model_dump(**kwargs)
        # Remove null tool fields for non-tool messages
        if data.get("tool_calls") is None:
            del data["tool_calls"]
        if data.get("tool_call_id") is None:
            del data["tool_call_id"]
        return data


class Scenario(BaseModel):
    """A test scenario for trace generation."""

    description: str = Field(description="The scenario description")
    context: str = Field(description="Additional context and background")
    category: str | None = Field(default=None, description="Category this scenario belongs to")

    # Evaluation fields (populated from GoldenScenario)
    scenario_type: str | None = Field(
        default=None, description="Type: positive, negative, edge_case, irrelevant"
    )
    target_rule_ids: list[str] | None = Field(
        default=None, description="Rule IDs this scenario tests"
    )
    expected_outcome: str | None = Field(
        default=None, description="Expected behavior based on rules"
    )


class GradeResult(BaseModel):
    """Result of grading a trace."""

    passed: bool = Field(description="Whether the trace passes quality checks")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    feedback: str = Field(default="", description="Summary feedback for improvement")


class Trace(BaseModel):
    """A complete training trace with messages and metadata."""

    messages: list[Message] = Field(description="The conversation messages")
    scenario: Scenario | None = Field(
        default=None, description="The scenario this trace was generated from"
    )
    grade: GradeResult | None = Field(default=None, description="Grading result if graded")

    # Golden Trace metadata (for verification)
    reasoning_chain: list[Any] | None = Field(
        default=None, description="Chain-of-thought reasoning steps with rule citations"
    )
    rules_applied: list[str] | None = Field(
        default=None, description="Rule IDs that were applied in the response"
    )
    rules_excluded: list[str] | None = Field(
        default=None, description="Rule IDs that were explicitly excluded"
    )

    @property
    def system_message(self) -> str | None:
        """Get the system message content."""
        for m in self.messages:
            if m.role == "system":
                return m.content
        return None

    @property
    def user_message(self) -> str:
        """Get the first user message content."""
        for m in self.messages:
            if m.role == "user":
                return m.content or ""
        return ""

    @property
    def assistant_message(self) -> str:
        """Get the last assistant message content."""
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m.content or ""
        return ""

    @property
    def has_tool_calls(self) -> bool:
        """Check if this trace contains any tool calls."""
        for m in self.messages:
            if m.tool_calls:
                return True
        return False


class EvalScenario(BaseModel):
    """
    A scenario for evaluation with ground truth labels.

    Used by generate_scenarios() for eval dataset generation.
    Contains the test input and expected behavior, but no synthetic response.

    Examples:
        >>> scenarios = synkro.generate_scenarios(policy, count=100)
        >>> for s in scenarios:
        ...     response = my_model(s.user_message)
        ...     grade = synkro.grade(response, s, policy)
    """

    user_message: str = Field(description="The user's request or question (test input)")
    expected_outcome: str = Field(description="Expected behavior based on policy rules")
    target_rule_ids: list[str] = Field(
        default_factory=list, description="Rule IDs this scenario tests"
    )
    scenario_type: str = Field(description="Type: positive, negative, edge_case, irrelevant")
    category: str = Field(default="", description="Policy category this scenario belongs to")
    context: str = Field(default="", description="Additional context for the scenario")


class Category(BaseModel):
    """A category for organizing scenarios."""

    name: str = Field(description="Category name")
    description: str = Field(description="What this category tests")
    count: int = Field(description="Number of traces to generate for this category")


class Plan(BaseModel):
    """A generation plan with categories and complexity analysis."""

    categories: list[Category] = Field(description="Categories with trace allocations")
    reasoning: str = Field(description="Explanation of why these categories were chosen")
    recommended_turns: int = Field(
        default=1, description="Recommended conversation turns based on policy complexity"
    )
    complexity_level: Literal["simple", "conditional", "complex"] = Field(
        default="simple",
        description="Policy complexity level: simple (1-2 turns), conditional (3 turns), complex (5+ turns)",
    )
    taxonomy: Any = Field(
        default=None,
        description="Sub-category taxonomy for coverage tracking (SubCategoryTaxonomy)",
    )
