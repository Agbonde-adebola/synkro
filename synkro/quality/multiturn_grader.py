"""Multi-turn conversation grading with per-turn and overall evaluation."""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.multiturn_templates import MULTI_TURN_GRADE_PROMPT
from synkro.schemas import ConversationGrade, TurnGrade
from synkro.types.core import GradeResult, Message, Trace


class MultiTurnGrader:
    """
    Grades multi-turn conversations using per-turn and overall criteria.

    Uses existing schemas:
    - TurnGrade: Per-turn policy violations, citations, reasoning
    - ConversationGrade: Overall pass, coherence, progressive depth

    Examples:
        >>> grader = MultiTurnGrader()
        >>> result = await grader.grade(trace, policy_text)
        >>> print(result.passed, result.feedback)
    """

    def __init__(self, llm: LLM | None = None, model: Model = OpenAI.GPT_4O):
        """
        Initialize the multi-turn grader.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (recommend stronger model)
        """
        self.llm = llm or LLM(model=model)

    def _count_assistant_turns(self, trace: Trace) -> int:
        """Count the number of assistant messages (turns) in a trace."""
        return sum(1 for m in trace.messages if m.role == "assistant")

    def _format_conversation(self, messages: list[Message]) -> str:
        """Format conversation messages for prompt inclusion."""
        formatted = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content or "[No content]"
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)

    def _extract_all_issues(self, conversation_grade: ConversationGrade) -> list[str]:
        """Extract all issues from conversation grade into flat list."""
        issues = []

        # Add coherence issues
        issues.extend(conversation_grade.coherence_issues)

        # Add per-turn issues
        for turn_grade in conversation_grade.turn_grades:
            issues.extend(turn_grade.policy_violations)
            issues.extend(turn_grade.missing_citations)
            issues.extend(turn_grade.incomplete_reasoning)
            issues.extend(turn_grade.vague_recommendations)

        return issues

    async def _grade_conversation(
        self,
        trace: Trace,
        policy_text: str,
    ) -> ConversationGrade:
        """
        Grade the full conversation using ConversationGrade schema.

        Args:
            trace: The trace to grade
            policy_text: The policy for evaluation

        Returns:
            ConversationGrade with per-turn and overall assessment
        """
        conversation = self._format_conversation(trace.messages)

        prompt = f"""{MULTI_TURN_GRADE_PROMPT.format(
            conversation=conversation,
            policy=policy_text,
        )}"""

        try:
            return await self.llm.generate_structured(prompt, ConversationGrade)
        except Exception:
            # Fallback - create a failing grade
            num_turns = self._count_assistant_turns(trace)
            turn_grades = [
                TurnGrade(
                    turn_index=i,
                    passed=False,
                    policy_violations=[],
                    missing_citations=[],
                    incomplete_reasoning=[],
                    vague_recommendations=[],
                    feedback="Unable to grade - parsing error",
                )
                for i in range(num_turns)
            ]
            return ConversationGrade(
                index=0,
                overall_pass=False,
                turn_grades=turn_grades,
                coherence_pass=False,
                coherence_issues=["Unable to evaluate - grading error"],
                progressive_depth=False,
                overall_feedback="Grading failed - please retry",
            )

    async def grade(self, trace: Trace, policy_text: str) -> GradeResult:
        """
        Grade a multi-turn conversation.

        Args:
            trace: The trace to grade
            policy_text: The policy for evaluation

        Returns:
            GradeResult with pass/fail, issues, and feedback
        """
        # Get full conversation grade
        conversation_grade = await self._grade_conversation(trace, policy_text)

        # Convert to standard GradeResult
        return GradeResult(
            passed=conversation_grade.overall_pass,
            issues=self._extract_all_issues(conversation_grade),
            feedback=conversation_grade.overall_feedback,
        )

    async def grade_detailed(
        self,
        trace: Trace,
        policy_text: str,
    ) -> ConversationGrade:
        """
        Get detailed per-turn grading for a conversation.

        Use this when you need access to individual turn grades.

        Args:
            trace: The trace to grade
            policy_text: The policy for evaluation

        Returns:
            ConversationGrade with full per-turn breakdown
        """
        return await self._grade_conversation(trace, policy_text)
