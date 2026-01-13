"""HITL Intent Classifier - LLM-powered classification of user feedback."""

from __future__ import annotations

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import HITLIntent
from synkro.prompts.interactive_templates import HITL_INTENT_CLASSIFIER_PROMPT


class HITLIntentClassifier:
    """
    LLM-powered classifier for user feedback in unified HITL sessions.

    Classifies user input into one of five intent types:
    - "turns": User wants to adjust conversation turns
    - "rules": User wants to modify the Logic Map
    - "scenarios": User wants to add/delete/modify scenarios or adjust distribution
    - "command": User typed a built-in command
    - "unclear": Cannot determine intent

    Examples:
        >>> classifier = HITLIntentClassifier(llm=LLM(model=OpenAI.GPT_4O))
        >>> intent = await classifier.classify(
        ...     user_input="I want shorter conversations",
        ...     current_turns=3,
        ...     complexity_level="conditional",
        ...     rule_count=10,
        ...     scenario_count=20
        ... )
        >>> intent.intent_type
        'turns'
        >>> intent.target_turns
        2
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O,
    ):
        """
        Initialize the HITL Intent Classifier.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (default: GPT-4O for accuracy)
        """
        self.llm = llm or LLM(model=model, temperature=0.2)

    async def classify(
        self,
        user_input: str,
        current_turns: int,
        complexity_level: str,
        rule_count: int,
        scenario_count: int = 0,
        conversation_history: str = "No previous feedback in this session.",
        coverage_summary: str = "Not available",
    ) -> HITLIntent:
        """
        Classify user input and extract structured intent.

        Args:
            user_input: The user's natural language feedback
            current_turns: Current conversation turns setting
            complexity_level: Policy complexity level (simple/conditional/complex)
            rule_count: Number of rules in the Logic Map
            scenario_count: Number of scenarios generated
            conversation_history: Formatted history of previous feedback in this session
            coverage_summary: Summary of coverage metrics (e.g., "75% overall, 3 gaps")

        Returns:
            HITLIntent with classified intent_type and relevant fields populated
        """
        prompt = HITL_INTENT_CLASSIFIER_PROMPT.format(
            user_input=user_input,
            current_turns=current_turns,
            complexity_level=complexity_level,
            rule_count=rule_count,
            scenario_count=scenario_count,
            conversation_history=conversation_history,
            coverage_summary=coverage_summary,
        )

        try:
            return await self.llm.generate_structured(prompt, HITLIntent)
        except Exception:
            # Default to treating as rule feedback (preserves existing behavior)
            return HITLIntent(
                intent_type="rules",
                confidence=0.5,
                rule_feedback=user_input,
            )


__all__ = ["HITLIntentClassifier"]
