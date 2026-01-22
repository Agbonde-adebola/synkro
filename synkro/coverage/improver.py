"""Coverage Improver - Generate scenarios to fill coverage gaps.

Generates targeted scenarios to improve coverage for specific
sub-categories based on natural language commands.
"""

from __future__ import annotations

from collections.abc import Callable

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.coverage_templates import (
    COVERAGE_COMMAND_PROMPT,
    COVERAGE_EXECUTION_PROMPT,
    COVERAGE_PLANNING_PROMPT,
    SCENARIO_DEDUPLICATION_PROMPT,
    TARGETED_SCENARIO_GENERATION_PROMPT,
)
from synkro.schemas import (
    CoveragePlan,
    DeduplicatedScenarios,
    GoldenScenariosArray,
    HITLIntent,
)
from synkro.types.coverage import (
    CoverageIntent,
    CoverageReport,
    SubCategoryTaxonomy,
)
from synkro.types.logic_map import GoldenScenario, LogicMap, ScenarioType


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
            f"- {sc.id}: {sc.name} ({sc.parent_category})" for sc in taxonomy.sub_categories
        )

        prompt = COVERAGE_COMMAND_PROMPT.format(
            coverage_summary=coverage_summary,
            sub_categories_list=sub_cats_list,
            user_input=command,
        )

        result = await self.llm.generate_structured(prompt, CoverageIntent)
        return result

    async def improve_from_intent(
        self,
        operation: str | None,
        target_percent: int | None,
        target_sub_category: str | None,
        coverage_report: CoverageReport,
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap,
        policy_text: str,
        existing_scenarios: list[GoldenScenario] | None = None,
        on_scenario_generated: Callable[[GoldenScenario], None] | None = None,
    ) -> list[GoldenScenario]:
        """
        Improve coverage using pre-parsed intent (skips redundant LLM call).

        Use this when intent was already classified by HITLIntentClassifier.
        """
        # Default to "target" operation if not specified
        operation = operation or "target"

        if operation == "target":
            return await self.improve_to_target(
                target_percent=target_percent or 80,
                target_sub_category=target_sub_category,
                coverage_report=coverage_report,
                taxonomy=taxonomy,
                logic_map=logic_map,
                policy_text=policy_text,
                existing_scenarios=existing_scenarios or [],
                on_scenario_generated=on_scenario_generated,
            )

        # For "increase" operation, target a specific sub-category
        target_sc = self._find_sub_category(
            target_sub_category or "",
            taxonomy,
        )

        if not target_sc:
            # Fallback to lowest coverage sub-category
            sorted_coverage = sorted(
                coverage_report.sub_category_coverage,
                key=lambda c: c.coverage_percent,
            )
            if sorted_coverage:
                target_sc = taxonomy.get_by_id(sorted_coverage[0].sub_category_id)

        if not target_sc:
            # Last resort: pick first sub-category
            if taxonomy.sub_categories:
                target_sc = taxonomy.sub_categories[0]

        if not target_sc:
            return []

        return await self.improve_coverage(
            target_sub_category_id=target_sc.id,
            taxonomy=taxonomy,
            logic_map=logic_map,
            policy_text=policy_text,
            count=3,
            existing_scenarios=existing_scenarios,
            on_scenario_generated=on_scenario_generated,
        )

    async def improve_from_command(
        self,
        command: str,
        coverage_report: CoverageReport,
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap,
        policy_text: str,
        existing_scenarios: list[GoldenScenario] | None = None,
        on_scenario_generated: Callable[[GoldenScenario], None] | None = None,
        on_step_change: Callable[[str], None] | None = None,
    ) -> list[GoldenScenario]:
        """
        Improve coverage based on natural language command.

        Note: If intent was already classified, use improve_from_intent() instead
        to skip the redundant parse_command() LLM call.
        """
        # Parse the command (LLM call - skip this if intent already classified)
        intent = await self.parse_command(command, coverage_report, taxonomy)

        if intent.operation == "view":
            return []

        return await self.improve_from_intent(
            operation=intent.operation,
            target_percent=intent.target_percent,
            target_sub_category=intent.target_sub_category,
            coverage_report=coverage_report,
            taxonomy=taxonomy,
            logic_map=logic_map,
            policy_text=policy_text,
            existing_scenarios=existing_scenarios,
            on_scenario_generated=on_scenario_generated,
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
        on_scenario_generated: Callable[[GoldenScenario], None] | None = None,
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
            on_scenario_generated: Callback called after each scenario is generated (for live updates)

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
        existing_descriptions = (
            "\n".join(f"- {s.description[:80]}..." for s in existing_for_sc[:10]) or "None"
        )

        # Format logic map
        logic_map_str = self._format_logic_map(logic_map)

        # Generate scenarios one at a time for streaming updates
        scenarios: list[GoldenScenario] = []
        generated_descriptions: list[str] = []

        for i in range(count):
            # Update existing descriptions to avoid duplicates
            all_existing = existing_descriptions
            if generated_descriptions:
                all_existing = (
                    existing_descriptions
                    + "\n"
                    + "\n".join(f"- {d[:80]}..." for d in generated_descriptions)
                )

            prompt = TARGETED_SCENARIO_GENERATION_PROMPT.format(
                policy_text=(
                    policy_text[:3000] + "..." if len(policy_text) > 3000 else policy_text
                ),
                logic_map=logic_map_str,
                sub_category_id=target_sc.id,
                sub_category_name=target_sc.name,
                sub_category_description=target_sc.description,
                related_rule_ids=", ".join(target_sc.related_rule_ids) or "none",
                priority=target_sc.priority,
                current_count=len(existing_for_sc) + len(scenarios),
                current_percent=(len(existing_for_sc) + len(scenarios)) * 20,
                existing_types=", ".join(f"{t}:{c}" for t, c in existing_types.items()) or "none",
                count=1,  # Generate one at a time
                preferred_types=types_str,
                existing_descriptions=all_existing,
            )

            result = await self.llm.generate_structured(prompt, GoldenScenariosArray)

            # Convert to domain model and add
            for s_out in result.scenarios[:1]:  # Take only the first one
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
                generated_descriptions.append(s_out.description)

                # Update existing types count for next iteration
                t = scenario.scenario_type.value
                existing_types[t] = existing_types.get(t, 0) + 1

                # Call the callback for live updates
                if on_scenario_generated:
                    on_scenario_generated(scenario)

        return scenarios

    async def improve_to_target(
        self,
        target_percent: float,
        target_sub_category: str | None,
        coverage_report: CoverageReport,
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap,
        policy_text: str,
        existing_scenarios: list[GoldenScenario],
        on_scenario_generated: Callable[[GoldenScenario], None] | None = None,
        on_step_change: Callable[[str], None] | None = None,
    ) -> list[GoldenScenario]:
        """
        Generate scenarios to reach target coverage using 3-call workflow.

        Call 1: Planning - Analyze gaps and create generation plan
        Call 2: Generation - Generate scenarios according to plan
        Call 3: Deduplication - Remove duplicates/near-duplicates

        Args:
            target_percent: Target overall coverage percentage
            target_sub_category: Specific sub-category to focus on (or None for overall)
            coverage_report: Current coverage report
            taxonomy: Sub-category taxonomy
            logic_map: Logic Map for rule context
            policy_text: Policy text for context
            existing_scenarios: Existing scenarios to avoid duplicating
            on_scenario_generated: Callback for live updates
            on_step_change: Callback for step progress updates (e.g., "Planning", "Generating")

        Returns:
            New scenarios to improve coverage (deduplicated)
        """
        # Build coverage context
        context = self._build_coverage_context(coverage_report, taxonomy, existing_scenarios)

        # Determine current coverage based on target
        current_coverage = coverage_report.overall_coverage_percent
        focus_instruction = ""

        if target_sub_category:
            target_sc = self._find_sub_category(target_sub_category, taxonomy)
            if target_sc:
                sc_coverage = coverage_report.get_coverage_for(target_sc.id)
                current_coverage = sc_coverage.coverage_percent if sc_coverage else 0.0
                focus_instruction = (
                    f"\n\nUSER FOCUS: Improve ONLY '{target_sc.name}' "
                    f"(currently {current_coverage:.0f}%) to {target_percent}%."
                )

        gap = target_percent - current_coverage

        # If target is already met, nothing to do
        if gap <= 0:
            return []

        # =====================================================================
        # CALL 1: Planning - Analyze gaps and create plan
        # =====================================================================
        if on_step_change:
            on_step_change("Planning coverage improvement...")

        planning_prompt = (
            COVERAGE_PLANNING_PROMPT.format(
                current_overall=current_coverage,
                target_percent=target_percent,
                gap=gap,
                sub_category_coverage_table=context["coverage_table"],
                existing_count=len(existing_scenarios),
            )
            + focus_instruction
        )

        plan = await self.llm.generate_structured(planning_prompt, CoveragePlan)

        # If no plan items, generate at least 2 scenarios for the target sub-category
        if not plan.plan_items:
            # Fallback: generate 2 scenarios if we have a specific target
            if target_sub_category:
                target_sc = self._find_sub_category(target_sub_category, taxonomy)
                if target_sc:
                    return await self.improve_coverage(
                        target_sub_category_id=target_sc.id,
                        taxonomy=taxonomy,
                        logic_map=logic_map,
                        policy_text=policy_text,
                        count=2,
                        existing_scenarios=existing_scenarios,
                        on_scenario_generated=on_scenario_generated,
                    )
            return []

        # =====================================================================
        # CALL 2: Generation - Generate scenarios based on plan
        # =====================================================================
        if on_step_change:
            on_step_change(f"Generating {plan.total_scenarios} scenarios...")

        # Format plan for generation prompt
        plan_summary = plan.strategy_summary
        plan_details = self._format_plan_for_generation(plan)

        generation_prompt = COVERAGE_EXECUTION_PROMPT.format(
            plan_summary=plan_summary,
            plan_details=plan_details,
            policy_text=policy_text[:4000] + "..." if len(policy_text) > 4000 else policy_text,
            logic_map=self._format_logic_map(logic_map),
        )

        generation_result = await self.llm.generate_structured(
            generation_prompt, GoldenScenariosArray
        )

        # If no scenarios generated, return empty
        if not generation_result.scenarios:
            return []

        # =====================================================================
        # CALL 3: Deduplication - Skip if few existing scenarios (saves LLM call)
        # =====================================================================
        kept_indices: set[int] = set(range(len(generation_result.scenarios)))  # Keep all by default

        # Only run dedup if there are enough existing scenarios to warrant it
        if len(existing_scenarios) >= 5:
            if on_step_change:
                on_step_change(f"Deduplicating {len(generation_result.scenarios)} scenarios...")

            existing_formatted = self._format_scenarios_for_dedup(existing_scenarios)
            generated_formatted = self._format_generated_for_dedup(generation_result.scenarios)

            dedup_prompt = SCENARIO_DEDUPLICATION_PROMPT.format(
                existing_scenarios=existing_formatted,
                generated_scenarios=generated_formatted,
            )

            dedup_result = await self.llm.generate_structured(dedup_prompt, DeduplicatedScenarios)
            kept_indices = set(dedup_result.kept_indices)

        # Convert to domain models and stream updates (only kept ones)
        scenarios: list[GoldenScenario] = []
        for i, s_out in enumerate(generation_result.scenarios):
            if i not in kept_indices:
                continue  # Skip removed duplicates

            # Determine category from sub_category_ids
            category = "General"
            if s_out.sub_category_ids:
                sc = taxonomy.get_by_id(s_out.sub_category_ids[0])
                if sc:
                    category = sc.parent_category

            scenario = GoldenScenario(
                description=s_out.description,
                context=s_out.context,
                category=category,
                scenario_type=ScenarioType(s_out.scenario_type),
                target_rule_ids=s_out.target_rule_ids,
                expected_outcome=s_out.expected_outcome,
                sub_category_ids=s_out.sub_category_ids or [],
            )
            scenarios.append(scenario)

            # Call callback for live updates
            if on_scenario_generated:
                on_scenario_generated(scenario)

        return scenarios

    def _format_plan_for_generation(self, plan: CoveragePlan) -> str:
        """Format the coverage plan for the generation prompt."""
        lines = []
        for item in plan.plan_items:
            types_str = ", ".join(item.scenario_types)
            focus_str = "; ".join(item.focus_areas) if item.focus_areas else "general coverage"
            lines.append(
                f"- {item.sub_category_name} ({item.sub_category_id}): "
                f"Generate {item.scenario_count} scenarios ({types_str}). "
                f"Focus: {focus_str}"
            )
        return "\n".join(lines)

    def _format_scenarios_for_dedup(self, scenarios: list[GoldenScenario]) -> str:
        """Format existing scenarios for deduplication prompt."""
        if not scenarios:
            return "None"
        lines = []
        for i, s in enumerate(scenarios[:20]):  # Limit to 20 for context
            sc_ids = ", ".join(s.sub_category_ids) if s.sub_category_ids else "none"
            lines.append(
                f"{i + 1}. [{s.scenario_type.value}] {s.description[:80]}... "
                f"(covers: {sc_ids}, rules: {', '.join(s.target_rule_ids[:3])})"
            )
        if len(scenarios) > 20:
            lines.append(f"... and {len(scenarios) - 20} more existing scenarios")
        return "\n".join(lines)

    def _format_generated_for_dedup(self, scenarios: list) -> str:
        """Format generated scenarios for deduplication prompt."""
        lines = []
        for i, s in enumerate(scenarios):
            sc_ids = ", ".join(s.sub_category_ids) if s.sub_category_ids else "none"
            rules = ", ".join(s.target_rule_ids[:3]) if s.target_rule_ids else "none"
            lines.append(
                f"[{i}] [{s.scenario_type}] {s.description[:100]}... "
                f"(covers: {sc_ids}, rules: {rules})"
            )
        return "\n".join(lines)

    def _build_coverage_context(
        self,
        coverage_report: CoverageReport,
        taxonomy: SubCategoryTaxonomy,
        existing_scenarios: list[GoldenScenario],
    ) -> dict:
        """
        Build comprehensive coverage context for LLM prompts.

        Returns:
            Dict with "coverage_table" and "scenarios_summary" strings
        """
        # Build coverage table
        table_lines = []
        for cov in coverage_report.sub_category_coverage:
            sc = taxonomy.get_by_id(cov.sub_category_id)
            priority = sc.priority if sc else "medium"
            table_lines.append(
                f"- {cov.sub_category_name}: {cov.coverage_percent:.0f}% "
                f"({cov.scenario_count} scenarios) [{cov.coverage_status}] "
                f"[{priority.upper()} priority]"
            )
        coverage_table = "\n".join(table_lines) or "No sub-categories defined"

        # Build existing scenarios summary (avoid duplicates)
        if existing_scenarios:
            scenario_lines = []
            for i, s in enumerate(existing_scenarios[:15]):  # Limit to 15
                sc_ids = ", ".join(s.sub_category_ids) if s.sub_category_ids else "none"
                scenario_lines.append(
                    f"{i + 1}. [{s.scenario_type.value}] {s.description[:60]}... "
                    f"(covers: {sc_ids})"
                )
            if len(existing_scenarios) > 15:
                scenario_lines.append(f"... and {len(existing_scenarios) - 15} more")
            scenarios_summary = "\n".join(scenario_lines)
        else:
            scenarios_summary = "None"

        return {
            "coverage_table": coverage_table,
            "scenarios_summary": scenarios_summary,
        }

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
