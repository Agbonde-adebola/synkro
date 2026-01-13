"""Coverage Improver - Generate scenarios to fill coverage gaps.

Generates targeted scenarios to improve coverage for specific
sub-categories based on natural language commands.
"""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import GoldenScenariosArray, HITLIntent
from synkro.types.coverage import (
    SubCategoryTaxonomy,
    CoverageReport,
    CoverageIntent,
)
from synkro.types.logic_map import GoldenScenario, ScenarioType, LogicMap
from synkro.prompts.coverage_templates import (
    COVERAGE_COMMAND_PROMPT,
    TARGETED_SCENARIO_GENERATION_PROMPT,
)


class CoverageImprover:
    """
    Generate scenarios to improve coverage for specific sub-categories.

    Parses natural language coverage commands and generates targeted
    scenarios to fill coverage gaps.

    Examples:
        >>> improver = CoverageImprover(llm=generation_llm)
        >>> # From natural language command
        >>> new_scenarios = await improver.improve_from_command(
        ...     command="increase coverage for refunds by 20%",
        ...     coverage_report=report,
        ...     taxonomy=taxonomy,
        ...     logic_map=logic_map,
        ...     policy_text=policy_text,
        ... )
        >>> # Or directly target a sub-category
        >>> new_scenarios = await improver.improve_coverage(
        ...     target_sub_category_id="SC001",
        ...     count=5,
        ...     scenario_type="edge_case",
        ... )
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O,
    ):
        """
        Initialize the Coverage Improver.

        Args:
            llm: LLM client to use for generation
            model: Model to use if creating LLM
        """
        self.llm = llm or LLM(model=model, temperature=0.7)

    async def parse_command(
        self,
        command: str,
        coverage_report: CoverageReport,
        taxonomy: SubCategoryTaxonomy,
    ) -> CoverageIntent:
        """
        Parse a natural language coverage command.

        Args:
            command: User's natural language command
            coverage_report: Current coverage report
            taxonomy: Sub-category taxonomy

        Returns:
            Parsed CoverageIntent
        """
        # Format coverage summary
        coverage_summary = coverage_report.to_summary_string()

        # Format sub-categories list
        sub_cats_list = "\n".join(
            f"- {sc.id}: {sc.name} ({sc.parent_category})"
            for sc in taxonomy.sub_categories
        )

        prompt = COVERAGE_COMMAND_PROMPT.format(
            coverage_summary=coverage_summary,
            sub_categories_list=sub_cats_list,
            user_input=command,
        )

        result = await self.llm.generate_structured(prompt, CoverageIntent)
        return result

    async def improve_from_command(
        self,
        command: str,
        coverage_report: CoverageReport,
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap,
        policy_text: str,
        existing_scenarios: list[GoldenScenario] | None = None,
    ) -> list[GoldenScenario]:
        """
        Improve coverage based on natural language command.

        Args:
            command: User's natural language command (e.g., "increase coverage for refunds by 20%")
            coverage_report: Current coverage report
            taxonomy: Sub-category taxonomy
            logic_map: Logic Map for rule context
            policy_text: Policy text for context
            existing_scenarios: Existing scenarios to avoid duplicating

        Returns:
            New scenarios to add
        """
        # Parse the command
        intent = await self.parse_command(command, coverage_report, taxonomy)

        if intent.operation == "view":
            # View commands don't generate scenarios
            return []

        # Find the target sub-category
        target_sc = self._find_sub_category(
            intent.target_sub_category or "",
            taxonomy,
        )

        if not target_sc:
            # If no specific target, pick the lowest coverage sub-category
            sorted_coverage = sorted(
                coverage_report.sub_category_coverage,
                key=lambda c: c.coverage_percent,
            )
            if sorted_coverage:
                target_sc = taxonomy.get_by_id(sorted_coverage[0].sub_category_id)

        if not target_sc:
            return []

        # Determine how many scenarios to generate
        if intent.operation == "target":
            # Target a specific percentage
            current_cov = coverage_report.get_coverage_for(target_sc.id)
            current_percent = current_cov.coverage_percent if current_cov else 0
            target_percent = intent.target_percent or 80
            # Estimate scenarios needed (rough calculation)
            count = max(1, int((target_percent - current_percent) / 20))
        elif intent.increase_amount:
            # Increase by percentage points
            count = max(1, int(intent.increase_amount / 20))
        else:
            # Default: add 3 scenarios
            count = 3

        # Determine preferred scenario types
        preferred_types = [intent.scenario_type] if intent.scenario_type else None

        return await self.improve_coverage(
            target_sub_category_id=target_sc.id,
            taxonomy=taxonomy,
            logic_map=logic_map,
            policy_text=policy_text,
            count=count,
            preferred_types=preferred_types,
            existing_scenarios=existing_scenarios,
        )

    async def improve_coverage(
        self,
        target_sub_category_id: str,
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap,
        policy_text: str,
        count: int = 3,
        preferred_types: list[str] | None = None,
        existing_scenarios: list[GoldenScenario] | None = None,
    ) -> list[GoldenScenario]:
        """
        Generate scenarios to improve coverage for a specific sub-category.

        Args:
            target_sub_category_id: ID of sub-category to target
            taxonomy: Sub-category taxonomy
            logic_map: Logic Map for rule context
            policy_text: Policy text for generation
            count: Number of scenarios to generate
            preferred_types: Preferred scenario types (e.g., ["negative", "edge_case"])
            existing_scenarios: Existing scenarios to avoid duplicating

        Returns:
            New scenarios for the target sub-category
        """
        target_sc = taxonomy.get_by_id(target_sub_category_id)
        if not target_sc:
            return []

        # Get existing scenarios for this sub-category
        existing_for_sc = []
        existing_types: dict[str, int] = {}
        if existing_scenarios:
            for s in existing_scenarios:
                if target_sub_category_id in s.sub_category_ids:
                    existing_for_sc.append(s)
                    t = s.scenario_type.value
                    existing_types[t] = existing_types.get(t, 0) + 1

        # Determine types to generate
        if preferred_types:
            types_str = ", ".join(preferred_types)
        else:
            # Balance types - prefer types that are underrepresented
            all_types = ["positive", "negative", "edge_case"]
            missing_types = [t for t in all_types if t not in existing_types]
            if missing_types:
                types_str = ", ".join(missing_types)
            else:
                types_str = "balanced mix of positive, negative, and edge_case"

        # Format existing descriptions to avoid
        existing_descriptions = "\n".join(
            f"- {s.description[:80]}..." for s in existing_for_sc[:10]
        ) or "None"

        # Format logic map
        logic_map_str = self._format_logic_map(logic_map)

        prompt = TARGETED_SCENARIO_GENERATION_PROMPT.format(
            policy_text=policy_text[:3000] + "..." if len(policy_text) > 3000 else policy_text,
            logic_map=logic_map_str,
            sub_category_id=target_sc.id,
            sub_category_name=target_sc.name,
            sub_category_description=target_sc.description,
            related_rule_ids=", ".join(target_sc.related_rule_ids) or "none",
            priority=target_sc.priority,
            current_count=len(existing_for_sc),
            current_percent=len(existing_for_sc) * 20,  # Rough estimate
            existing_types=", ".join(f"{t}:{c}" for t, c in existing_types.items()) or "none",
            count=count,
            preferred_types=types_str,
            existing_descriptions=existing_descriptions,
        )

        result = await self.llm.generate_structured(prompt, GoldenScenariosArray)

        # Convert to domain models
        scenarios = []
        for s_out in result.scenarios:
            scenario = GoldenScenario(
                description=s_out.description,
                context=s_out.context,
                category=target_sc.parent_category,
                scenario_type=ScenarioType(s_out.scenario_type),
                target_rule_ids=s_out.target_rule_ids,
                expected_outcome=s_out.expected_outcome,
                sub_category_ids=[target_sub_category_id],
            )
            scenarios.append(scenario)

        return scenarios

    def _find_sub_category(
        self,
        query: str,
        taxonomy: SubCategoryTaxonomy,
    ):
        """Find a sub-category by name or ID (fuzzy matching)."""
        query_lower = query.lower()

        # Exact ID match
        for sc in taxonomy.sub_categories:
            if sc.id.lower() == query_lower:
                return sc

        # Exact name match
        for sc in taxonomy.sub_categories:
            if sc.name.lower() == query_lower:
                return sc

        # Partial name match
        for sc in taxonomy.sub_categories:
            if query_lower in sc.name.lower():
                return sc

        # Partial description match
        for sc in taxonomy.sub_categories:
            if query_lower in sc.description.lower():
                return sc

        return None

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map for the prompt."""
        lines = []
        for rule in logic_map.rules:
            deps = f" (depends on: {', '.join(rule.dependencies)})" if rule.dependencies else ""
            lines.append(f"- {rule.rule_id}: {rule.text}{deps}")
        return "\n".join(lines)

    def parse_hitl_intent_to_coverage(
        self,
        intent: HITLIntent,
    ) -> CoverageIntent | None:
        """Convert HITLIntent with coverage fields to CoverageIntent."""
        if intent.intent_type != "coverage":
            return None

        if not intent.coverage_operation:
            return None

        return CoverageIntent(
            operation=intent.coverage_operation,
            view_mode=intent.coverage_view_mode,
            target_sub_category=intent.coverage_target_sub_category,
            target_percent=intent.coverage_target_percent,
            increase_amount=intent.coverage_increase_amount,
            scenario_type=intent.coverage_scenario_type,
        )
