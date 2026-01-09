"""Golden Scenario Generator - The Adversary.

Generates typed scenarios (positive, negative, edge_case, irrelevant)
with explicit rule targeting. This is Stage 2 of the Golden Trace pipeline.
"""

import asyncio
from typing import Literal

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import GoldenScenariosArray
from synkro.types.core import Category
from synkro.types.logic_map import LogicMap, GoldenScenario, ScenarioType
from synkro.prompts.golden_templates import (
    GOLDEN_SCENARIO_PROMPT,
    GOLDEN_SCENARIO_BATCHED_PROMPT,
    POSITIVE_SCENARIO_INSTRUCTIONS,
    NEGATIVE_SCENARIO_INSTRUCTIONS,
    EDGE_CASE_SCENARIO_INSTRUCTIONS,
    IRRELEVANT_SCENARIO_INSTRUCTIONS,
)


# Default scenario type distribution
DEFAULT_DISTRIBUTION = {
    ScenarioType.POSITIVE: 0.35,    # 35% happy path
    ScenarioType.NEGATIVE: 0.30,    # 30% violations
    ScenarioType.EDGE_CASE: 0.25,   # 25% edge cases
    ScenarioType.IRRELEVANT: 0.10,  # 10% out of scope
}


TYPE_INSTRUCTIONS = {
    ScenarioType.POSITIVE: POSITIVE_SCENARIO_INSTRUCTIONS,
    ScenarioType.NEGATIVE: NEGATIVE_SCENARIO_INSTRUCTIONS,
    ScenarioType.EDGE_CASE: EDGE_CASE_SCENARIO_INSTRUCTIONS,
    ScenarioType.IRRELEVANT: IRRELEVANT_SCENARIO_INSTRUCTIONS,
}


class GoldenScenarioGenerator:
    """
    The Adversary - Generates typed scenarios with rule targeting.

    Produces scenarios across four types:
    - POSITIVE (35%): Happy path, all criteria met
    - NEGATIVE (30%): Violation, exactly one criterion fails
    - EDGE_CASE (25%): Boundary conditions, exact limits
    - IRRELEVANT (10%): Outside policy scope

    Each scenario includes:
    - Target rule IDs it's designed to test
    - Expected outcome based on the rules
    - Scenario type for classification

    Examples:
        >>> generator = GoldenScenarioGenerator(llm=LLM(model=OpenAI.GPT_4O_MINI))
        >>> scenarios = await generator.generate(
        ...     policy_text="...",
        ...     logic_map=logic_map,
        ...     category=category,
        ...     count=10,
        ... )
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O_MINI,
        distribution: dict[ScenarioType, float] | None = None,
    ):
        """
        Initialize the Golden Scenario Generator.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
            distribution: Custom scenario type distribution (defaults to 35/30/25/10)
        """
        self.llm = llm or LLM(model=model, temperature=0.8)
        self.distribution = distribution or DEFAULT_DISTRIBUTION

    async def generate(
        self,
        policy_text: str,
        logic_map: LogicMap,
        category: Category,
        count: int,
    ) -> list[GoldenScenario]:
        """
        Generate scenarios for a category with balanced type distribution.

        Uses batched generation (single LLM call per category) for efficiency.

        Args:
            policy_text: The policy document text
            logic_map: The extracted Logic Map (DAG of rules)
            category: The category to generate scenarios for
            count: Total number of scenarios to generate

        Returns:
            List of GoldenScenarios with type distribution
        """
        # Calculate counts per type based on distribution
        type_counts = self._calculate_type_distribution(count)

        # Use batched generation (single call for all types)
        return await self._generate_batched(
            policy_text=policy_text,
            logic_map=logic_map,
            category=category,
            type_counts=type_counts,
        )

    def _calculate_type_distribution(self, total: int) -> dict[ScenarioType, int]:
        """Calculate how many scenarios of each type to generate."""
        counts = {
            ScenarioType.POSITIVE: 0,
            ScenarioType.NEGATIVE: 0,
            ScenarioType.EDGE_CASE: 0,
            ScenarioType.IRRELEVANT: 0,
        }

        if total == 0:
            return counts

        if total == 1:
            # Single scenario - use the configured distribution ratio to pick type
            # This ensures variety across categories
            import random
            types = list(self.distribution.keys())
            weights = list(self.distribution.values())
            chosen = random.choices(types, weights=weights, k=1)[0]
            counts[chosen] = 1
        elif total == 2:
            # Two scenarios - always include variety (positive + one other)
            counts[ScenarioType.POSITIVE] = 1
            # Randomly pick between negative and edge_case for the second
            import random
            other = random.choice([ScenarioType.NEGATIVE, ScenarioType.EDGE_CASE])
            counts[other] = 1
        elif total == 3:
            # Three scenarios - positive, negative, edge_case
            counts[ScenarioType.POSITIVE] = 1
            counts[ScenarioType.NEGATIVE] = 1
            counts[ScenarioType.EDGE_CASE] = 1
        else:
            # Normal distribution for larger counts (4+)
            remaining = total
            types_list = list(self.distribution.items())
            for i, (stype, ratio) in enumerate(types_list):
                if i == len(types_list) - 1:
                    # Last type gets remaining to ensure total is exact
                    counts[stype] = max(0, remaining)
                else:
                    count = round(total * ratio)
                    counts[stype] = count
                    remaining -= count

        return counts

    async def _generate_batched(
        self,
        policy_text: str,
        logic_map: LogicMap,
        category: Category,
        type_counts: dict[ScenarioType, int],
    ) -> list[GoldenScenario]:
        """
        Generate all scenario types in a single LLM call (batched).

        This is more efficient than making separate calls per type.
        Includes retry logic if the LLM doesn't return the exact count.
        """
        # Format Logic Map for prompt
        logic_map_str = self._format_logic_map(logic_map)

        # Calculate total
        total_count = sum(type_counts.values())

        # Build batched prompt
        prompt = GOLDEN_SCENARIO_BATCHED_PROMPT.format(
            policy_text=policy_text,
            logic_map=logic_map_str,
            category=category.name,
            positive_count=type_counts.get(ScenarioType.POSITIVE, 0),
            negative_count=type_counts.get(ScenarioType.NEGATIVE, 0),
            edge_case_count=type_counts.get(ScenarioType.EDGE_CASE, 0),
            irrelevant_count=type_counts.get(ScenarioType.IRRELEVANT, 0),
            total_count=total_count,
        )

        # Generate with retry (max 1 retry if count is wrong)
        scenarios = await self._generate_and_parse(prompt, category.name, total_count)

        # Retry once if count doesn't match
        if len(scenarios) != total_count:
            retry_prompt = prompt + f"\n\nIMPORTANT: You must generate EXACTLY {total_count} scenarios. You previously generated {len(scenarios)}."
            retry_scenarios = await self._generate_and_parse(retry_prompt, category.name, total_count)

            # Use retry result if it's closer to target, otherwise keep original
            if abs(len(retry_scenarios) - total_count) < abs(len(scenarios) - total_count):
                scenarios = retry_scenarios

        # Truncate if over, accept if under (after retry)
        return scenarios[:total_count]

    async def _generate_and_parse(
        self,
        prompt: str,
        category_name: str,
        expected_count: int,
    ) -> list[GoldenScenario]:
        """Generate scenarios and parse to domain models."""
        result = await self.llm.generate_structured(prompt, GoldenScenariosArray)

        scenarios = []
        for s in result.scenarios:
            scenario = GoldenScenario(
                description=s.description,
                context=s.context,
                category=category_name,
                scenario_type=ScenarioType(s.scenario_type),
                target_rule_ids=s.target_rule_ids,
                expected_outcome=s.expected_outcome,
            )
            scenarios.append(scenario)

        return scenarios

    async def _generate_type(
        self,
        policy_text: str,
        logic_map: LogicMap,
        category: Category,
        scenario_type: ScenarioType,
        count: int,
    ) -> list[GoldenScenario]:
        """Generate scenarios of a specific type."""
        # Get type-specific instructions
        type_instructions = TYPE_INSTRUCTIONS[scenario_type]

        # Format Logic Map for prompt
        logic_map_str = self._format_logic_map(logic_map)

        # Build prompt
        prompt = GOLDEN_SCENARIO_PROMPT.format(
            scenario_type=scenario_type.value.upper(),
            policy_text=policy_text,
            logic_map=logic_map_str,
            category=category.name,
            count=count,
            type_specific_instructions=type_instructions,
        )

        # Generate structured output
        result = await self.llm.generate_structured(prompt, GoldenScenariosArray)

        # Convert to domain models
        scenarios = []
        for s in result.scenarios:
            scenario = GoldenScenario(
                description=s.description,
                context=s.context,
                category=category.name,
                scenario_type=ScenarioType(s.scenario_type),
                target_rule_ids=s.target_rule_ids,
                expected_outcome=s.expected_outcome,
            )
            scenarios.append(scenario)

        # Enforce requested count (LLM may return more or fewer)
        return scenarios[:count]

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map for prompt inclusion."""
        lines = []
        lines.append("RULES:")
        for rule in logic_map.rules:
            deps = f" (depends on: {', '.join(rule.dependencies)})" if rule.dependencies else ""
            lines.append(
                f"  {rule.rule_id} [{rule.category.value}]: {rule.text}{deps}"
            )

        lines.append("\nROOT RULES (Entry Points):")
        lines.append(f"  {', '.join(logic_map.root_rules)}")

        return "\n".join(lines)

    async def generate_for_categories(
        self,
        policy_text: str,
        logic_map: LogicMap,
        categories: list[Category],
    ) -> tuple[list[GoldenScenario], dict[str, int]]:
        """
        Generate scenarios for multiple categories with distribution tracking.

        Uses GLOBAL distribution across all categories to ensure proper type variety.

        Args:
            policy_text: The policy document text
            logic_map: The extracted Logic Map
            categories: List of categories with counts

        Returns:
            Tuple of (all scenarios, type distribution counts)
        """
        # Calculate GLOBAL distribution across all scenarios
        total_scenarios = sum(cat.count for cat in categories)
        global_distribution = self._calculate_type_distribution(total_scenarios)

        # Create a pool of types to assign to categories
        type_pool = []
        for stype, count in global_distribution.items():
            type_pool.extend([stype] * count)

        # Shuffle for randomness, but maintain overall distribution
        import random
        random.shuffle(type_pool)

        # Assign type distributions to each category from the pool
        category_type_counts = []
        pool_idx = 0
        for cat in categories:
            cat_types = {
                ScenarioType.POSITIVE: 0,
                ScenarioType.NEGATIVE: 0,
                ScenarioType.EDGE_CASE: 0,
                ScenarioType.IRRELEVANT: 0,
            }
            for _ in range(cat.count):
                if pool_idx < len(type_pool):
                    cat_types[type_pool[pool_idx]] += 1
                    pool_idx += 1
            category_type_counts.append(cat_types)

        # Generate for each category with assigned type distributions
        tasks = [
            self._generate_with_distribution(policy_text, logic_map, cat, type_counts)
            for cat, type_counts in zip(categories, category_type_counts)
        ]
        results = await asyncio.gather(*tasks)

        # Flatten scenarios
        all_scenarios = []
        for batch in results:
            all_scenarios.extend(batch)

        # Calculate actual distribution from results
        distribution = {
            ScenarioType.POSITIVE.value: 0,
            ScenarioType.NEGATIVE.value: 0,
            ScenarioType.EDGE_CASE.value: 0,
            ScenarioType.IRRELEVANT.value: 0,
        }
        for s in all_scenarios:
            distribution[s.scenario_type.value] += 1

        return all_scenarios, distribution

    async def _generate_with_distribution(
        self,
        policy_text: str,
        logic_map: LogicMap,
        category: Category,
        type_counts: dict[ScenarioType, int],
    ) -> list[GoldenScenario]:
        """Generate scenarios for a category with a specific type distribution."""
        total_count = sum(type_counts.values())
        if total_count == 0:
            return []

        return await self._generate_batched(
            policy_text=policy_text,
            logic_map=logic_map,
            category=category,
            type_counts=type_counts,
        )

    def get_distribution_summary(self, scenarios: list[GoldenScenario]) -> dict[str, int]:
        """Get a summary of scenario type distribution."""
        distribution = {
            "positive": 0,
            "negative": 0,
            "edge_case": 0,
            "irrelevant": 0,
        }
        for s in scenarios:
            distribution[s.scenario_type.value] += 1
        return distribution


__all__ = ["GoldenScenarioGenerator", "DEFAULT_DISTRIBUTION"]
