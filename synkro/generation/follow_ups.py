"""Follow-up question generation for multi-turn conversations."""

from typing import Literal

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.multiturn_templates import FOLLOW_UP_GENERATION_PROMPT
from synkro.schemas import FollowUpQuestion
from synkro.types.core import Message

QuestionType = Literal["clarification", "edge_case", "what_if", "specificity", "challenge"]

# Question type progression for multi-turn conversations
# Earlier turns focus on clarification, later turns probe deeper
QUESTION_TYPE_BY_TURN = {
    1: "clarification",
    2: "specificity",
    3: "edge_case",
    4: "what_if",
    5: "challenge",
}


class FollowUpGenerator:
    """
    Generates follow-up questions for multi-turn conversations.

    Uses different question types based on turn index:
    - Turn 1: clarification - Ask for more details
    - Turn 2: specificity - Drill into specifics
    - Turn 3: edge_case - Probe boundary conditions
    - Turn 4: what_if - Explore hypotheticals
    - Turn 5+: challenge - Question reasoning

    Examples:
        >>> gen = FollowUpGenerator()
        >>> follow_up = await gen.generate(policy_text, messages, turn_index=2)
        >>> print(follow_up.question)
    """

    def __init__(self, llm: LLM | None = None, model: Model = OpenAI.GPT_4O_MINI):
        """
        Initialize the follow-up generator.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
        """
        self.llm = llm or LLM(model=model)

    def _select_question_type(self, turn_index: int) -> QuestionType:
        """
        Select question type based on turn index.

        Args:
            turn_index: Which turn this is (1-based, counting user-assistant exchanges)

        Returns:
            Appropriate question type for this turn
        """
        if turn_index in QUESTION_TYPE_BY_TURN:
            return QUESTION_TYPE_BY_TURN[turn_index]
        # For turns beyond 5, cycle through challenging questions
        return "challenge"

    def _format_conversation(self, messages: list[Message]) -> str:
        """Format conversation messages for prompt inclusion."""
        formatted = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content or "[No content]"
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)

    async def generate(
        self,
        policy_text: str,
        messages: list[Message],
        turn_index: int,
        question_type: QuestionType | None = None,
        scenario_index: int = 0,
    ) -> FollowUpQuestion:
        """
        Generate a follow-up question for the conversation.

        Args:
            policy_text: The policy text for context
            messages: Conversation messages so far
            turn_index: Which turn this is (1-based)
            question_type: Override auto-selected question type
            scenario_index: Index for the scenario (default 0)

        Returns:
            FollowUpQuestion with the generated question
        """
        # Select question type if not specified
        if question_type is None:
            question_type = self._select_question_type(turn_index)

        # Format conversation for prompt
        conversation = self._format_conversation(messages)

        # Build prompt
        prompt = FOLLOW_UP_GENERATION_PROMPT.format(
            question_type=question_type,
            conversation=conversation,
            policy=policy_text,
        )

        try:
            # Generate the follow-up question
            response = await self.llm.generate(prompt)
            question_text = response.strip()

            return FollowUpQuestion(
                index=scenario_index,
                question=question_text,
                question_type=question_type,
            )
        except Exception:
            # Fallback generic follow-up
            fallback_questions = {
                "clarification": "Can you clarify that further?",
                "edge_case": "What about edge cases?",
                "what_if": "What if the situation changes?",
                "specificity": "Can you be more specific?",
                "challenge": "Why is that the best approach?",
            }
            return FollowUpQuestion(
                index=scenario_index,
                question=fallback_questions.get(question_type, "Can you elaborate?"),
                question_type=question_type,
            )
