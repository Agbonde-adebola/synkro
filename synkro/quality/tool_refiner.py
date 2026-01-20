"""Specialized refinement for tool call traces that preserves format."""

from typing import TYPE_CHECKING

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.quality.refiner import Refiner
from synkro.types.core import GradeResult, Scenario, Trace

if TYPE_CHECKING:
    from synkro.generation.tool_simulator import ToolSimulator
    from synkro.types.tool import ToolDefinition


class ToolCallRefiner(Refiner):
    """
    Specialized refiner for tool call traces.

    Unlike the base Refiner which generates plain text responses, this refiner
    uses the ToolCallResponseGenerator to regenerate traces, ensuring the
    tool_calls format is preserved in the output.

    The grading feedback is incorporated into the scenario context so the
    LLM knows what to fix during regeneration.

    Examples:
        >>> refiner = ToolCallRefiner(
        ...     tools=[web_search, db_lookup],
        ...     simulator=tool_simulator,
        ... )
        >>> improved = await refiner.refine(failed_trace, grade, policy_text)
        >>> # improved trace has proper tool_calls format
    """

    def __init__(
        self,
        tools: list["ToolDefinition"],
        simulator: "ToolSimulator",
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O_MINI,
    ):
        """
        Initialize the tool call refiner.

        Args:
            tools: List of available tool definitions
            simulator: Tool simulator for generating tool responses
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
        """
        super().__init__(llm=llm, model=model)
        self.tools = tools
        self.simulator = simulator
        self._response_generator = None

    def _get_response_generator(self):
        """Lazily create the ToolCallResponseGenerator."""
        if self._response_generator is None:
            from synkro.generation.tool_responses import ToolCallResponseGenerator

            self._response_generator = ToolCallResponseGenerator(
                tools=self.tools,
                llm=self.llm,
                simulator=self.simulator,
            )
        return self._response_generator

    def _build_enhanced_scenario(self, trace: Trace, grade: GradeResult) -> Scenario:
        """
        Build an enhanced scenario that includes grading feedback.

        The feedback helps the LLM understand what went wrong and how to fix it.
        """
        # Build feedback context
        feedback_parts = []
        if grade.issues:
            feedback_parts.append("PREVIOUS ISSUES TO FIX:")
            for issue in grade.issues:
                feedback_parts.append(f"  - {issue}")
        if grade.feedback:
            feedback_parts.append(f"\nGRADER FEEDBACK: {grade.feedback}")

        feedback_context = "\n".join(feedback_parts) if feedback_parts else ""

        # Enhance the context with feedback
        enhanced_context = trace.scenario.context
        if feedback_context:
            enhanced_context = (
                f"{trace.scenario.context}\n\n--- REFINEMENT GUIDANCE ---\n{feedback_context}"
            )

        return Scenario(
            description=trace.scenario.description,
            context=enhanced_context,
            category=trace.scenario.category,
        )

    async def refine(self, trace: Trace, grade: GradeResult, policy_text: str) -> Trace:
        """
        Refine a failed tool call trace by regenerating with feedback.

        Uses the ToolCallResponseGenerator to ensure the regenerated trace
        maintains proper tool_calls format.

        Args:
            trace: The trace that failed grading
            grade: The grade result with feedback
            policy_text: The policy/guidelines text

        Returns:
            New trace with improved response and preserved tool_calls format
        """
        # Create enhanced scenario with grading feedback
        enhanced_scenario = self._build_enhanced_scenario(trace, grade)

        # Regenerate using ToolCallResponseGenerator (preserves format)
        generator = self._get_response_generator()
        refined_trace = await generator.generate_single(policy_text, enhanced_scenario)

        # Preserve the original scenario reference (without the feedback context)
        refined_trace.scenario = trace.scenario

        return refined_trace


__all__ = ["ToolCallRefiner"]
