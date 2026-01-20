"""Multi-turn response generation for complex conversations."""

from synkro.generation.follow_ups import FollowUpGenerator
from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.multiturn_templates import (
    MULTI_TURN_INITIAL_PROMPT,
    MULTI_TURN_RESPONSE_PROMPT,
)
from synkro.prompts.templates import SYSTEM_PROMPT
from synkro.types.core import Message, Scenario, Trace


class MultiTurnResponseGenerator:
    """
    Generates multi-turn conversations based on policy complexity.

    Turn allocation:
    - Simple (1-2 turns): Single query -> Straight answer
    - Conditional (3 turns): Query -> Clarification -> Verdict
    - Complex (5+ turns): Multiple rounds of exploration

    Examples:
        >>> gen = MultiTurnResponseGenerator()
        >>> trace = await gen.generate_single(policy_text, scenario, target_turns=3)
        >>> print(len([m for m in trace.messages if m.role == "assistant"]))
        3
    """

    def __init__(self, llm: LLM | None = None, model: Model = OpenAI.GPT_4O_MINI):
        """
        Initialize the multi-turn response generator.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
        """
        self.llm = llm or LLM(model=model)
        self.follow_up_gen = FollowUpGenerator(llm=self.llm)

    def _format_conversation(self, messages: list[Message]) -> str:
        """Format conversation messages for prompt inclusion."""
        formatted = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content or "[No content]"
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)

    async def _generate_initial_response(
        self,
        policy_text: str,
        scenario: Scenario,
        target_turns: int,
    ) -> list[Message]:
        """
        Generate the initial exchange (system + user + assistant).

        Args:
            policy_text: The policy text
            scenario: The scenario to respond to
            target_turns: Total target turns for context

        Returns:
            List of initial messages [system, user, assistant]
        """
        prompt = MULTI_TURN_INITIAL_PROMPT.format(
            target_turns=target_turns,
            scenario=scenario.description,
            context=scenario.context,
            policy=policy_text,
        )

        response = await self.llm.generate(prompt)

        return [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"{scenario.description}\n\nContext: {scenario.context}"),
            Message(role="assistant", content=response.strip()),
        ]

    async def _generate_response(
        self,
        policy_text: str,
        messages: list[Message],
        question: str,
    ) -> str:
        """
        Generate a response to a follow-up question.

        Args:
            policy_text: The policy text
            messages: Conversation history
            question: The follow-up question to answer

        Returns:
            Assistant response text
        """
        conversation = self._format_conversation(messages)

        prompt = MULTI_TURN_RESPONSE_PROMPT.format(
            conversation=conversation,
            question=question,
            policy=policy_text,
        )

        response = await self.llm.generate(prompt)
        return response.strip()

    async def generate_single(
        self,
        policy_text: str,
        scenario: Scenario,
        target_turns: int,
    ) -> Trace:
        """
        Generate a multi-turn trace for one scenario.

        Args:
            policy_text: The policy text
            scenario: The scenario to generate for
            target_turns: Number of user-assistant exchanges

        Returns:
            Trace with multi-turn conversation
        """
        # Generate initial exchange
        messages = await self._generate_initial_response(policy_text, scenario, target_turns)

        # Generate follow-up turns
        for turn in range(1, target_turns):
            # Generate follow-up question
            follow_up = await self.follow_up_gen.generate(
                policy_text=policy_text,
                messages=messages,
                turn_index=turn,
            )

            # Add user message with follow-up question
            messages.append(Message(role="user", content=follow_up.question))

            # Generate assistant response
            response = await self._generate_response(
                policy_text=policy_text,
                messages=messages,
                question=follow_up.question,
            )

            # Add assistant response
            messages.append(Message(role="assistant", content=response))

        return Trace(messages=messages, scenario=scenario)

    async def generate(
        self,
        policy_text: str,
        scenarios: list[Scenario],
        target_turns: int,
    ) -> list[Trace]:
        """
        Generate multi-turn traces for multiple scenarios.

        Args:
            policy_text: The policy text
            scenarios: List of scenarios to generate for
            target_turns: Number of turns per trace

        Returns:
            List of traces with multi-turn conversations
        """
        traces = []
        for scenario in scenarios:
            trace = await self.generate_single(policy_text, scenario, target_turns)
            traces.append(trace)
        return traces
