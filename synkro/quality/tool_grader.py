"""Specialized grading for tool call traces."""

from typing import TYPE_CHECKING

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.tool_templates import TOOL_GRADE_PROMPT
from synkro.quality.grader import Grader
from synkro.schemas import ToolCallGrade
from synkro.types.core import GradeResult, Trace

if TYPE_CHECKING:
    from synkro.types.tool import ToolDefinition


class ToolCallGrader(Grader):
    """
    Specialized grader for tool call traces.

    Evaluates tool usage on four criteria:
    - Tool Selection: Did they use the right tool?
    - Parameter Accuracy: Were the parameters correct?
    - Response Synthesis: Did they use tool results correctly?
    - Timing: Did they call tools at the right time?

    Examples:
        >>> grader = ToolCallGrader(tools=[web_search, db_lookup])
        >>> result = await grader.grade(trace, policy_text)
        >>> if not result.passed:
        ...     print(f"Issues: {result.issues}")
    """

    def __init__(
        self,
        tools: list["ToolDefinition"],
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_52,
    ):
        """
        Initialize the tool call grader.

        Args:
            tools: List of available tool definitions (for context)
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (recommend stronger model)
        """
        super().__init__(llm=llm, model=model)
        self.tools = tools

    def _get_tools_description(self) -> str:
        """Get formatted description of all tools for grading context."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(tool.to_system_prompt())
        return "\n\n".join(descriptions)

    def _format_conversation(self, trace: Trace) -> str:
        """Format the trace messages for the grading prompt, including tool_calls."""
        lines = []
        for msg in trace.messages:
            if msg.role == "system":
                lines.append(f"[SYSTEM]\n{msg.content}")
            elif msg.role == "user":
                lines.append(f"[USER]\n{msg.content}")
            elif msg.role == "assistant":
                if msg.tool_calls:
                    # Format assistant message with tool calls
                    tool_calls_str = []
                    for tc in msg.tool_calls:
                        tool_calls_str.append(f"  - {tc.function.name}({tc.function.arguments})")
                    lines.append("[ASSISTANT - TOOL CALLS]\n" + "\n".join(tool_calls_str))
                else:
                    lines.append(f"[ASSISTANT]\n{msg.content}")
            elif msg.role == "tool":
                lines.append(f"[TOOL RESULT - {msg.tool_call_id}]\n{msg.content}")
        return "\n\n".join(lines)

    async def grade(self, trace: Trace, policy_text: str) -> GradeResult:
        """
        Grade a tool call trace using tool-specific criteria.

        Args:
            trace: The trace to grade
            policy_text: The policy/guidelines text

        Returns:
            GradeResult with pass/fail and detailed feedback
        """
        tools_desc = self._get_tools_description()
        conversation = self._format_conversation(trace)

        prompt = TOOL_GRADE_PROMPT.format(
            TOOLS_DESCRIPTION=tools_desc,
            GUIDELINES=policy_text,
            SCENARIO=trace.scenario.description,
            CONVERSATION=conversation,
        )

        try:
            # Use structured output for consistent grading
            parsed = await self.llm.generate_structured(prompt, ToolCallGrade)

            # Convert to standard GradeResult format
            return GradeResult(
                passed=parsed.passed,
                issues=parsed.get_all_issues(),
                feedback=parsed.feedback,
            )
        except Exception:
            # Fallback: assume fail if we can't parse
            return GradeResult(
                passed=False,
                issues=["Unable to parse grade response"],
                feedback="Grading failed - unable to parse response",
            )


__all__ = ["ToolCallGrader"]
