"""Scenario Editor - LLM-powered interactive refinement of golden scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import RefinedScenariosOutput
from synkro.types.logic_map import LogicMap, GoldenScenario, ScenarioType
from synkro.prompts.interactive_templates import SCENARIO_REFINEMENT_PROMPT

if TYPE_CHECKING:
    pass


class ScenarioEditor:
    """
    LLM-powered scenario editor that interprets natural language feedback.

    The editor takes user feedback in natural language (e.g., "add a scenario for...",
    "delete S3", "more edge cases") and uses an LLM to interpret and apply
    the changes to the scenario list.

    Examples:
        >>> editor = ScenarioEditor(llm=LLM(model=OpenAI.GPT_4O))
        >>> new_scenarios, distribution, summary = await editor.refine(
        ...     scenarios=current_scenarios,
        ...     distribution=current_distribution,
        ...     user_feedback="Add a scenario for expenses at exactly $50",
        ...     policy_text=policy.text,
        ...     logic_map=logic_map,
        ... )
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O,
    ):
        """
        Initialize the Scenario Editor.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (default: GPT-4O for accuracy)
        """
        self.llm = llm or LLM(model=model, temperature=0.3)

    async def refine(
        self,
        scenarios: list[GoldenScenario],
        distribution: dict[str, int],
        user_feedback: str,
        policy_text: str,
        logic_map: LogicMap,
        conversation_history: str = "No previous feedback in this session.",
    ) -> tuple[list[GoldenScenario], dict[str, int], str]:
        """
        Refine scenarios based on natural language feedback.

        Args:
            scenarios: Current list of golden scenarios
            distribution: Current type distribution (e.g., {"positive": 5, "negative": 3})
            user_feedback: Natural language instruction from user
            policy_text: Original policy text for context
            logic_map: Logic Map for rule references
            conversation_history: Formatted history of previous feedback in this session

        Returns:
            Tuple of (refined scenarios, updated distribution, changes summary)
        """
        # Format current scenarios for prompt
        scenarios_str = self._format_scenarios_for_prompt(scenarios)
        distribution_str = self._format_distribution(distribution)
        logic_map_str = self._format_logic_map_for_prompt(logic_map)

        # Format the prompt
        prompt = SCENARIO_REFINEMENT_PROMPT.format(
            logic_map=logic_map_str,
            scenarios_formatted=scenarios_str,
            distribution=distribution_str,
            policy_text=policy_text,
            user_feedback=user_feedback,
            conversation_history=conversation_history,
        )

        # Generate structured output
        result = await self.llm.generate_structured(prompt, RefinedScenariosOutput)

        # Convert to domain model
        refined_scenarios = self._convert_to_scenarios(result.scenarios)

        # Calculate new distribution
        new_distribution = self._calculate_distribution(refined_scenarios)

        return refined_scenarios, new_distribution, result.changes_summary

    def _format_scenarios_for_prompt(self, scenarios: list[GoldenScenario]) -> str:
        """Format scenarios with S1, S2... IDs for the LLM prompt."""
        lines = []
        for i, scenario in enumerate(scenarios, start=1):
            lines.append(f"S{i}:")
            lines.append(f"  Description: {scenario.description}")
            lines.append(f"  Type: {scenario.scenario_type.value}")
            lines.append(f"  Target Rules: {', '.join(scenario.target_rule_ids) or 'None'}")
            lines.append(f"  Expected Outcome: {scenario.expected_outcome}")
            if scenario.context:
                lines.append(f"  Context: {scenario.context}")
            lines.append("")
        return "\n".join(lines)

    def _format_distribution(self, distribution: dict[str, int]) -> str:
        """Format distribution as a string."""
        lines = []
        for type_name, count in sorted(distribution.items()):
            lines.append(f"  {type_name}: {count}")
        return "\n".join(lines)

    def _format_logic_map_for_prompt(self, logic_map: LogicMap) -> str:
        """Format Logic Map rules for context."""
        lines = []
        lines.append(f"Total Rules: {len(logic_map.rules)}")
        lines.append("Rules:")
        for rule in logic_map.rules:
            lines.append(f"  {rule.rule_id}: {rule.text}")
        return "\n".join(lines)

    def _convert_to_scenarios(self, schema_scenarios: list) -> list[GoldenScenario]:
        """Convert schema output to domain model."""
        scenarios = []
        for s in schema_scenarios:
            # Handle scenario_type as string or enum
            if isinstance(s.scenario_type, str):
                scenario_type = ScenarioType(s.scenario_type)
            else:
                scenario_type = s.scenario_type

            scenario = GoldenScenario(
                description=s.description,
                context=getattr(s, 'context', ''),
                category=getattr(s, 'category', ''),
                scenario_type=scenario_type,
                target_rule_ids=s.target_rule_ids,
                expected_outcome=s.expected_outcome,
            )
            scenarios.append(scenario)
        return scenarios

    def _calculate_distribution(self, scenarios: list[GoldenScenario]) -> dict[str, int]:
        """Calculate type distribution from scenarios."""
        distribution: dict[str, int] = {
            "positive": 0,
            "negative": 0,
            "edge_case": 0,
            "irrelevant": 0,
        }
        for scenario in scenarios:
            type_key = scenario.scenario_type.value
            distribution[type_key] = distribution.get(type_key, 0) + 1
        return distribution

    def validate_scenarios(
        self,
        scenarios: list[GoldenScenario],
        logic_map: LogicMap,
    ) -> tuple[bool, list[str]]:
        """
        Validate scenarios against the Logic Map.

        Args:
            scenarios: List of scenarios to validate
            logic_map: Logic Map for rule reference validation

        Returns:
            Tuple of (is_valid, list of issue descriptions)
        """
        issues = []
        rule_ids = {r.rule_id for r in logic_map.rules}

        for i, scenario in enumerate(scenarios, start=1):
            # Check target_rule_ids reference existing rules
            for rule_id in scenario.target_rule_ids:
                if rule_id not in rule_ids:
                    issues.append(f"S{i} references non-existent rule {rule_id}")

            # Check scenario_type is valid
            if scenario.scenario_type.value not in ["positive", "negative", "edge_case", "irrelevant"]:
                issues.append(f"S{i} has invalid scenario_type: {scenario.scenario_type}")

            # Check expected_outcome is not empty
            if not scenario.expected_outcome.strip():
                issues.append(f"S{i} has empty expected_outcome")

        return len(issues) == 0, issues


__all__ = ["ScenarioEditor"]
