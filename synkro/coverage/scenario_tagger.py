"""Scenario Tagger - Tag scenarios with sub-category coverage.

Tags each scenario with the sub-categories it covers, enabling
coverage calculation and gap analysis.
"""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import BatchedScenarioTagging
from synkro.types.coverage import SubCategoryTaxonomy
from synkro.types.logic_map import LogicMap, GoldenScenario
from synkro.prompts.coverage_templates import SCENARIO_SUBCATEGORY_TAGGING_PROMPT


class ScenarioTagger:
    """
    Tag scenarios with sub-category coverage.

    Analyzes each scenario to determine which sub-categories it covers
    based on target rules, description, and expected outcome.

    Examples:
        >>> tagger = ScenarioTagger(llm=LLM(model=OpenAI.GPT_4O_MINI))
        >>> tagged_scenarios = await tagger.tag(
        ...     scenarios=scenarios,
        ...     taxonomy=taxonomy,
        ...     logic_map=logic_map,
        ... )
        >>> for s in tagged_scenarios:
        ...     print(f"{s.description[:50]}... covers {s.sub_category_ids}")
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O_MINI,
        batch_size: int = 20,
    ):
        """
        Initialize the Scenario Tagger.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (default: GPT-4O-mini for speed)
            batch_size: Number of scenarios to tag per LLM call
        """
        self.llm = llm or LLM(model=model, temperature=0.1)
        self.batch_size = batch_size

    async def tag(
        self,
        scenarios: list[GoldenScenario],
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap,
    ) -> list[GoldenScenario]:
        """
        Tag scenarios with sub-category IDs.

        Args:
            scenarios: Scenarios to tag
            taxonomy: Sub-category taxonomy
            logic_map: Logic Map for rule reference

        Returns:
            Scenarios with sub_category_ids populated
        """
        if not scenarios:
            return scenarios

        # First, do rule-based tagging (fast, no LLM needed)
        rule_to_subcats = self._build_rule_mapping(taxonomy)
        scenarios = self._tag_by_rules(scenarios, rule_to_subcats)

        # Then, use LLM to tag scenarios that still have no sub-categories
        # or to refine existing tags
        untagged = [s for s in scenarios if not s.sub_category_ids]
        if untagged:
            scenarios = await self._tag_with_llm(
                scenarios, taxonomy, logic_map
            )

        return scenarios

    def _build_rule_mapping(
        self,
        taxonomy: SubCategoryTaxonomy,
    ) -> dict[str, list[str]]:
        """Build mapping from rule IDs to sub-category IDs."""
        mapping: dict[str, list[str]] = {}
        for sc in taxonomy.sub_categories:
            for rule_id in sc.related_rule_ids:
                if rule_id not in mapping:
                    mapping[rule_id] = []
                mapping[rule_id].append(sc.id)
        return mapping

    def _tag_by_rules(
        self,
        scenarios: list[GoldenScenario],
        rule_to_subcats: dict[str, list[str]],
    ) -> list[GoldenScenario]:
        """Tag scenarios based on their target_rule_ids."""
        for scenario in scenarios:
            sub_cat_ids = set()
            for rule_id in scenario.target_rule_ids:
                if rule_id in rule_to_subcats:
                    sub_cat_ids.update(rule_to_subcats[rule_id])
            scenario.sub_category_ids = list(sub_cat_ids)
        return scenarios

    async def _tag_with_llm(
        self,
        scenarios: list[GoldenScenario],
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap,
    ) -> list[GoldenScenario]:
        """Use LLM to tag scenarios with sub-categories."""
        # Process in batches
        for i in range(0, len(scenarios), self.batch_size):
            batch = scenarios[i:i + self.batch_size]
            batch_indices = list(range(i, min(i + self.batch_size, len(scenarios))))

            # Format inputs
            sub_categories_str = self._format_sub_categories(taxonomy)
            scenarios_str = self._format_scenarios(batch, batch_indices)
            logic_map_str = self._format_logic_map(logic_map)

            prompt = SCENARIO_SUBCATEGORY_TAGGING_PROMPT.format(
                sub_categories_formatted=sub_categories_str,
                scenarios_formatted=scenarios_str,
                logic_map=logic_map_str,
            )

            # Generate structured output
            result = await self.llm.generate_structured(prompt, BatchedScenarioTagging)

            # Apply tags to scenarios
            for tagging in result.taggings:
                idx = tagging.scenario_index
                if 0 <= idx < len(scenarios):
                    # Merge with existing tags
                    existing = set(scenarios[idx].sub_category_ids)
                    existing.update(tagging.sub_category_ids)
                    scenarios[idx].sub_category_ids = list(existing)

        return scenarios

    def _format_sub_categories(self, taxonomy: SubCategoryTaxonomy) -> str:
        """Format sub-categories for the prompt."""
        lines = []
        for sc in taxonomy.sub_categories:
            rules = ", ".join(sc.related_rule_ids) if sc.related_rule_ids else "none"
            lines.append(
                f"- {sc.id}: {sc.name}\n"
                f"  Description: {sc.description}\n"
                f"  Parent: {sc.parent_category}\n"
                f"  Related Rules: {rules}"
            )
        return "\n".join(lines)

    def _format_scenarios(
        self,
        scenarios: list[GoldenScenario],
        indices: list[int],
    ) -> str:
        """Format scenarios for the prompt."""
        lines = []
        for idx, scenario in zip(indices, scenarios):
            rules = ", ".join(scenario.target_rule_ids) if scenario.target_rule_ids else "none"
            lines.append(
                f"[{idx}] {scenario.description[:100]}...\n"
                f"  Type: {scenario.scenario_type.value}\n"
                f"  Category: {scenario.category}\n"
                f"  Target Rules: {rules}\n"
                f"  Expected: {scenario.expected_outcome[:100]}..."
            )
        return "\n\n".join(lines)

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map summary for the prompt."""
        lines = []
        for rule in logic_map.rules[:20]:  # Limit to first 20 for context
            lines.append(f"- {rule.rule_id}: {rule.text[:80]}...")
        if len(logic_map.rules) > 20:
            lines.append(f"... and {len(logic_map.rules) - 20} more rules")
        return "\n".join(lines)
