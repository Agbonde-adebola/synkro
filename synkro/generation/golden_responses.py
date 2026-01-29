"""Golden Response Generator - The Thinker.

Generates traces with grounded Chain-of-Thought reasoning and rule citations.
This is Stage 3 of the Golden Trace pipeline for CONVERSATION/INSTRUCTION datasets.
"""

import asyncio

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.golden_templates import (
    GOLDEN_TRACE_MULTI_TURN_PROMPT,
    GOLDEN_TRACE_PROMPT,
)
from synkro.schemas import GoldenTraceOutput
from synkro.types.core import Message, Trace
from synkro.types.logic_map import (
    GoldenScenario,
    LogicMap,
)


class GoldenResponseGenerator:
    """
    The Thinker - Generates traces with grounded reasoning.

    Produces traces with:
    - Explicit Chain-of-Thought reasoning
    - Rule citations (Rule IDs) for each reasoning step
    - Exclusionary reasoning (why rules DON'T apply)
    - DAG-compliant dependency order

    Examples:
        >>> generator = GoldenResponseGenerator(llm=LLM(model=OpenAI.GPT_4O_MINI))
        >>> trace = await generator.generate_single(
        ...     policy_text="...",
        ...     logic_map=logic_map,
        ...     scenario=scenario,
        ...     target_turns=1,
        ... )
    """

    # Instruction to inject when thinking mode is enabled
    THINKING_INSTRUCTION = """
THINKING MODE:
Your assistant response MUST include reasoning wrapped in <think> and </think> tags.
Place your step-by-step reasoning inside the think tags BEFORE your actual response.

Format:
<think>
[Your reasoning about which rules apply, why they apply/don't apply, etc.]
</think>

[Your actual response to the user]
"""

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O_MINI,
        thinking: bool = False,
    ):
        """
        Initialize the Golden Response Generator.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
            thinking: Enable thinking mode with <think> tags in responses
        """
        self.llm = llm or LLM(model=model, temperature=0.7)
        self.thinking = thinking

    async def generate_single(
        self,
        policy_text: str,
        logic_map: LogicMap,
        scenario: GoldenScenario,
        target_turns: int = 1,
    ) -> Trace:
        """
        Generate a single trace with grounded reasoning.

        Args:
            policy_text: The policy document text
            logic_map: The extracted Logic Map (DAG of rules)
            scenario: The golden scenario to respond to
            target_turns: Number of conversation turns

        Returns:
            Trace with messages and reasoning metadata
        """
        if target_turns > 1:
            return await self._generate_multi_turn(policy_text, logic_map, scenario, target_turns)

        return await self._generate_single_turn(policy_text, logic_map, scenario)

    async def _generate_single_turn(
        self,
        policy_text: str,
        logic_map: LogicMap,
        scenario: GoldenScenario,
    ) -> Trace:
        """Generate a single-turn trace."""
        # Format Logic Map for prompt
        logic_map_str = self._format_logic_map(logic_map)

        # Build prompt
        prompt = GOLDEN_TRACE_PROMPT.format(
            policy_text=policy_text,
            logic_map=logic_map_str,
            scenario_description=scenario.description,
            scenario_context=scenario.context,
            target_rule_ids=", ".join(scenario.target_rule_ids),
            scenario_type=scenario.scenario_type.value.upper(),
            expected_outcome=scenario.expected_outcome,
        )

        # Inject thinking instruction if enabled
        if self.thinking:
            prompt = prompt + self.THINKING_INSTRUCTION

        # Generate structured output
        result = await self.llm.generate_structured(prompt, GoldenTraceOutput)

        # Convert to Trace
        messages = [Message(role=m.role, content=m.content) for m in result.messages]

        # Convert GoldenScenario to base Scenario for Trace
        base_scenario = scenario.to_base_scenario()

        # Convert reasoning chain to serializable format
        reasoning_chain = None
        if result.reasoning_chain:
            reasoning_chain = [
                {
                    "rule_id": step.rule_id,
                    "rule_text": step.rule_text,
                    "applies": step.applies,
                    "reasoning": step.reasoning,
                    "exclusions": step.exclusions,
                }
                for step in result.reasoning_chain
            ]

        return Trace(
            messages=messages,
            scenario=base_scenario,
            reasoning_chain=reasoning_chain,
            rules_applied=result.rules_applied,
            rules_excluded=result.rules_excluded,
        )

    async def _generate_multi_turn(
        self,
        policy_text: str,
        logic_map: LogicMap,
        scenario: GoldenScenario,
        target_turns: int,
    ) -> Trace:
        """Generate a multi-turn trace."""
        # Format Logic Map for prompt
        logic_map_str = self._format_logic_map(logic_map)

        # Build prompt
        prompt = GOLDEN_TRACE_MULTI_TURN_PROMPT.format(
            policy_text=policy_text,
            logic_map=logic_map_str,
            scenario_description=scenario.description,
            scenario_context=scenario.context,
            target_rule_ids=", ".join(scenario.target_rule_ids),
            scenario_type=scenario.scenario_type.value.upper(),
            target_turns=target_turns,
        )

        # Inject thinking instruction if enabled
        if self.thinking:
            prompt = prompt + self.THINKING_INSTRUCTION

        # Generate structured output
        result = await self.llm.generate_structured(prompt, GoldenTraceOutput)

        # Convert to Trace
        messages = [Message(role=m.role, content=m.content) for m in result.messages]

        # Convert GoldenScenario to base Scenario for Trace
        base_scenario = scenario.to_base_scenario()

        # Convert reasoning chain to serializable format
        reasoning_chain = None
        if result.reasoning_chain:
            reasoning_chain = [
                {
                    "rule_id": step.rule_id,
                    "rule_text": step.rule_text,
                    "applies": step.applies,
                    "reasoning": step.reasoning,
                    "exclusions": step.exclusions,
                }
                for step in result.reasoning_chain
            ]

        return Trace(
            messages=messages,
            scenario=base_scenario,
            reasoning_chain=reasoning_chain,
            rules_applied=result.rules_applied,
            rules_excluded=result.rules_excluded,
        )

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map for prompt inclusion."""
        lines = []
        lines.append("RULES:")
        for rule in logic_map.rules:
            deps = f" [depends on: {', '.join(rule.dependencies)}]" if rule.dependencies else ""
            lines.append(f"  {rule.rule_id} ({rule.category.value}): {rule.text}{deps}")
            lines.append(f"    IF: {rule.condition}")
            lines.append(f"    THEN: {rule.action}")

        lines.append("\nDEPENDENCY ORDER (evaluate in this order):")
        # Show topological order for root rules and their chains
        for root_id in logic_map.root_rules:
            chain = logic_map.get_chain(root_id)
            if chain:
                chain_str = " -> ".join(r.rule_id for r in chain)
                lines.append(f"  {chain_str}")

        return "\n".join(lines)

    async def generate(
        self,
        policy_text: str,
        logic_map: LogicMap,
        scenarios: list[GoldenScenario],
        target_turns: int = 1,
        concurrency: int = 50,
    ) -> list[Trace]:
        """
        Generate traces for multiple scenarios.

        Args:
            policy_text: The policy document text
            logic_map: The extracted Logic Map
            scenarios: List of golden scenarios
            target_turns: Number of conversation turns
            concurrency: Maximum concurrent LLM calls (default: 50)

        Returns:
            List of traces with grounded reasoning
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_limit(scenario):
            async with semaphore:
                return await self.generate_single(policy_text, logic_map, scenario, target_turns)

        tasks = [generate_with_limit(s) for s in scenarios]
        return await asyncio.gather(*tasks)


__all__ = ["GoldenResponseGenerator"]
