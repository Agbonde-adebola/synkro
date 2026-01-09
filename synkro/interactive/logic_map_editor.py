"""Logic Map Editor - LLM-powered interactive refinement of Logic Maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import RefinedLogicMapOutput
from synkro.types.logic_map import LogicMap, Rule, RuleCategory
from synkro.prompts.interactive_templates import LOGIC_MAP_REFINEMENT_PROMPT

if TYPE_CHECKING:
    pass


class LogicMapEditor:
    """
    LLM-powered Logic Map editor that interprets natural language feedback.

    The editor takes user feedback in natural language (e.g., "add a rule for...",
    "remove R005", "merge R002 and R003") and uses an LLM to interpret and apply
    the changes to the Logic Map.

    Examples:
        >>> editor = LogicMapEditor(llm=LLM(model=OpenAI.GPT_4O))
        >>> new_logic_map = await editor.refine(
        ...     logic_map=current_map,
        ...     user_feedback="Add a rule for overtime approval",
        ...     policy_text=policy.text
        ... )
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O,
    ):
        """
        Initialize the Logic Map Editor.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (default: GPT-4O for accuracy)
        """
        self.llm = llm or LLM(model=model, temperature=0.3)

    async def refine(
        self,
        logic_map: LogicMap,
        user_feedback: str,
        policy_text: str,
        conversation_history: str = "No previous feedback in this session.",
    ) -> tuple[LogicMap, str]:
        """
        Refine the Logic Map based on natural language feedback.

        Args:
            logic_map: Current Logic Map to refine
            user_feedback: Natural language instruction from user
            policy_text: Original policy text for context
            conversation_history: Formatted history of previous feedback in this session

        Returns:
            Tuple of (refined LogicMap, changes summary string)

        Raises:
            ValueError: If refinement produces invalid DAG
        """
        # Format current Logic Map as string for prompt
        current_map_str = self._format_logic_map_for_prompt(logic_map)

        # Format the prompt
        prompt = LOGIC_MAP_REFINEMENT_PROMPT.format(
            current_logic_map=current_map_str,
            policy_text=policy_text,
            user_feedback=user_feedback,
            conversation_history=conversation_history,
        )

        # Generate structured output
        result = await self.llm.generate_structured(prompt, RefinedLogicMapOutput)

        # Convert to domain model
        refined_map = self._convert_to_logic_map(result)

        # Validate DAG properties
        if not refined_map.validate_dag():
            raise ValueError(
                "Refined Logic Map contains circular dependencies. "
                "Please try a different modification."
            )

        return refined_map, result.changes_summary

    def _format_logic_map_for_prompt(self, logic_map: LogicMap) -> str:
        """Format a Logic Map as a string for the LLM prompt."""
        lines = []
        lines.append(f"Total Rules: {len(logic_map.rules)}")
        lines.append(f"Root Rules: {', '.join(logic_map.root_rules)}")
        lines.append("")
        lines.append("Rules:")

        for rule in logic_map.rules:
            deps = f" -> {', '.join(rule.dependencies)}" if rule.dependencies else ""
            lines.append(f"  {rule.rule_id}: {rule.text}")
            lines.append(f"    Category: {rule.category.value}")
            lines.append(f"    Condition: {rule.condition}")
            lines.append(f"    Action: {rule.action}")
            if deps:
                lines.append(f"    Dependencies: {', '.join(rule.dependencies)}")
            lines.append("")

        return "\n".join(lines)

    def _convert_to_logic_map(self, output: RefinedLogicMapOutput) -> LogicMap:
        """Convert schema output to domain model."""
        rules = []
        for rule_out in output.rules:
            # Convert category string to enum
            category = RuleCategory(rule_out.category)

            rule = Rule(
                rule_id=rule_out.rule_id,
                text=rule_out.text,
                condition=rule_out.condition,
                action=rule_out.action,
                dependencies=rule_out.dependencies,
                category=category,
            )
            rules.append(rule)

        return LogicMap(
            rules=rules,
            root_rules=output.root_rules,
        )

    def validate_refinement(
        self,
        original: LogicMap,
        refined: LogicMap,
    ) -> tuple[bool, list[str]]:
        """
        Validate that refinement maintains DAG properties and is sensible.

        Args:
            original: Original Logic Map
            refined: Refined Logic Map

        Returns:
            Tuple of (is_valid, list of issue descriptions)
        """
        issues = []

        # Check DAG validity
        if not refined.validate_dag():
            issues.append("Refined Logic Map has circular dependencies")

        # Check that all dependencies reference existing rules
        rule_ids = {r.rule_id for r in refined.rules}
        for rule in refined.rules:
            for dep in rule.dependencies:
                if dep not in rule_ids:
                    issues.append(f"Rule {rule.rule_id} depends on non-existent rule {dep}")

        # Check root_rules consistency
        for root_id in refined.root_rules:
            if root_id not in rule_ids:
                issues.append(f"Root rule {root_id} does not exist")

        # Check that rules with no dependencies are in root_rules
        for rule in refined.rules:
            if not rule.dependencies and rule.rule_id not in refined.root_rules:
                issues.append(f"Rule {rule.rule_id} has no dependencies but is not in root_rules")

        return len(issues) == 0, issues
