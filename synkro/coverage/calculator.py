"""Coverage Calculator - Calculate coverage metrics.

Calculates coverage metrics from scenarios and taxonomy,
providing a detailed coverage report with gaps and suggestions.
"""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import CoverageSuggestionsOutput
from synkro.types.coverage import (
    SubCategoryTaxonomy,
    SubCategoryCoverage,
    CoverageReport,
    CoverageThresholds,
)
from synkro.types.logic_map import GoldenScenario, LogicMap
from synkro.prompts.coverage_templates import COVERAGE_SUGGESTIONS_PROMPT


class CoverageCalculator:
    """
    Calculate coverage metrics from scenarios and taxonomy.

    Analyzes how well scenarios cover each sub-category, identifying
    gaps and generating suggestions for improvement.

    Examples:
        >>> calculator = CoverageCalculator()
        >>> report = await calculator.calculate(
        ...     scenarios=scenarios,
        ...     taxonomy=taxonomy,
        ... )
        >>> print(f"Overall coverage: {report.overall_coverage_percent}%")
        >>> print(f"Gaps: {report.gaps}")
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O_MINI,
        thresholds: CoverageThresholds | None = None,
    ):
        """
        Initialize the Coverage Calculator.

        Args:
            llm: LLM client for generating suggestions (optional)
            model: Model to use if creating LLM
            thresholds: Coverage thresholds for status determination
        """
        self.llm = llm
        self.model = model
        self.thresholds = thresholds or CoverageThresholds()

    async def calculate(
        self,
        scenarios: list[GoldenScenario],
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap | None = None,
        policy_text: str = "",
        generate_suggestions: bool = True,
    ) -> CoverageReport:
        """
        Calculate coverage metrics.

        Args:
            scenarios: Tagged scenarios with sub_category_ids
            taxonomy: Sub-category taxonomy
            logic_map: Logic Map for context (optional)
            policy_text: Policy text for context (optional)
            generate_suggestions: Whether to generate improvement suggestions

        Returns:
            CoverageReport with detailed metrics
        """
        # Calculate coverage for each sub-category
        sub_category_coverage = self._calculate_sub_category_coverage(
            scenarios, taxonomy
        )

        # Calculate overall metrics
        total_sub_categories = len(taxonomy.sub_categories)
        covered_count = sum(
            1 for c in sub_category_coverage if c.coverage_status == "covered"
        )
        partial_count = sum(
            1 for c in sub_category_coverage if c.coverage_status == "partial"
        )
        uncovered_count = sum(
            1 for c in sub_category_coverage if c.coverage_status == "uncovered"
        )

        # Calculate overall coverage percentage
        if total_sub_categories > 0:
            total_coverage = sum(c.coverage_percent for c in sub_category_coverage)
            overall_coverage_percent = total_coverage / total_sub_categories
        else:
            overall_coverage_percent = 0.0

        # Identify gaps
        gaps = self._identify_gaps(sub_category_coverage, taxonomy)

        # Build heatmap data
        heatmap_data = self._build_heatmap(sub_category_coverage, taxonomy)

        # Generate suggestions if requested
        suggestions: list[str] = []
        if generate_suggestions and gaps and self.llm:
            suggestions = await self._generate_suggestions(
                sub_category_coverage,
                taxonomy,
                logic_map,
                policy_text,
            )
        elif generate_suggestions and gaps:
            # Generate basic suggestions without LLM
            suggestions = self._basic_suggestions(sub_category_coverage, taxonomy)

        return CoverageReport(
            total_scenarios=len(scenarios),
            total_sub_categories=total_sub_categories,
            covered_count=covered_count,
            partial_count=partial_count,
            uncovered_count=uncovered_count,
            overall_coverage_percent=overall_coverage_percent,
            sub_category_coverage=sub_category_coverage,
            gaps=gaps,
            suggestions=suggestions,
            heatmap_data=heatmap_data,
        )

    def _calculate_sub_category_coverage(
        self,
        scenarios: list[GoldenScenario],
        taxonomy: SubCategoryTaxonomy,
    ) -> list[SubCategoryCoverage]:
        """Calculate coverage for each sub-category."""
        # Initialize coverage tracking
        coverage_map: dict[str, SubCategoryCoverage] = {}
        for sc in taxonomy.sub_categories:
            coverage_map[sc.id] = SubCategoryCoverage(
                sub_category_id=sc.id,
                sub_category_name=sc.name,
                parent_category=sc.parent_category,
                scenario_count=0,
                scenario_ids=[],
                type_distribution={},
            )

        # Count scenarios per sub-category
        for i, scenario in enumerate(scenarios):
            scenario_id = f"S{i}"
            for sc_id in scenario.sub_category_ids:
                if sc_id in coverage_map:
                    cov = coverage_map[sc_id]
                    cov.scenario_count += 1
                    cov.scenario_ids.append(scenario_id)

                    # Track type distribution
                    sc_type = scenario.scenario_type.value
                    if sc_type not in cov.type_distribution:
                        cov.type_distribution[sc_type] = 0
                    cov.type_distribution[sc_type] += 1

        # Calculate percentages and status
        for sc_id, cov in coverage_map.items():
            sc = taxonomy.get_by_id(sc_id)
            if sc:
                expected = self._expected_scenarios(sc)
                cov.coverage_percent = min(
                    100.0, (cov.scenario_count / expected) * 100 if expected > 0 else 0.0
                )
                cov.coverage_status = self._determine_status(cov.coverage_percent)

        return list(coverage_map.values())

    def _expected_scenarios(self, sub_category) -> int:
        """Calculate expected scenarios for a sub-category."""
        base = self.thresholds.min_scenarios_per_sub_category
        multiplier = self.thresholds.priority_multipliers.get(
            sub_category.priority, 1.0
        )
        # Add bonus for more related rules
        rule_bonus = len(sub_category.related_rule_ids) * 0.5
        return int(base * multiplier + rule_bonus)

    def _determine_status(self, coverage_percent: float) -> str:
        """Determine coverage status based on percentage."""
        percent = coverage_percent / 100.0  # Convert to 0-1 range
        if percent >= self.thresholds.covered_threshold:
            return "covered"
        elif percent >= self.thresholds.partial_threshold:
            return "partial"
        else:
            return "uncovered"

    def _identify_gaps(
        self,
        coverage: list[SubCategoryCoverage],
        taxonomy: SubCategoryTaxonomy,
    ) -> list[str]:
        """Identify coverage gaps."""
        gaps = []

        # Sort by coverage status and priority
        for cov in coverage:
            if cov.coverage_status == "uncovered":
                sc = taxonomy.get_by_id(cov.sub_category_id)
                priority = f"[{sc.priority.upper()}]" if sc else ""
                gaps.append(
                    f"{cov.sub_category_name} {priority} "
                    f"({cov.coverage_percent:.0f}% coverage, "
                    f"{cov.scenario_count} scenarios)"
                )
            elif cov.coverage_status == "partial":
                sc = taxonomy.get_by_id(cov.sub_category_id)
                priority = f"[{sc.priority.upper()}]" if sc else ""
                gaps.append(
                    f"{cov.sub_category_name} {priority} "
                    f"(partial: {cov.coverage_percent:.0f}% coverage)"
                )

        return gaps

    def _build_heatmap(
        self,
        coverage: list[SubCategoryCoverage],
        taxonomy: SubCategoryTaxonomy,
    ) -> dict[str, dict[str, float]]:
        """Build heatmap data structure."""
        heatmap: dict[str, dict[str, float]] = {}

        for cov in coverage:
            category = cov.parent_category
            if category not in heatmap:
                heatmap[category] = {}
            heatmap[category][cov.sub_category_name] = cov.coverage_percent

        return heatmap

    def _basic_suggestions(
        self,
        coverage: list[SubCategoryCoverage],
        taxonomy: SubCategoryTaxonomy,
    ) -> list[str]:
        """Generate basic suggestions without LLM."""
        suggestions = []

        # Sort by priority (uncovered high-priority first)
        uncovered_high = []
        uncovered_medium = []
        uncovered_low = []
        partial = []

        for cov in coverage:
            sc = taxonomy.get_by_id(cov.sub_category_id)
            if not sc:
                continue

            if cov.coverage_status == "uncovered":
                if sc.priority == "high":
                    uncovered_high.append((cov, sc))
                elif sc.priority == "medium":
                    uncovered_medium.append((cov, sc))
                else:
                    uncovered_low.append((cov, sc))
            elif cov.coverage_status == "partial":
                partial.append((cov, sc))

        # Generate suggestions in priority order
        for cov, sc in uncovered_high[:3]:
            rules_str = ", ".join(sc.related_rule_ids[:3]) if sc.related_rule_ids else "related rules"
            suggestions.append(
                f"Add 3+ scenarios for '{sc.name}' (HIGH priority) "
                f"testing {rules_str}"
            )

        for cov, sc in uncovered_medium[:3]:
            rules_str = ", ".join(sc.related_rule_ids[:3]) if sc.related_rule_ids else "related rules"
            suggestions.append(
                f"Add 2+ scenarios for '{sc.name}' testing {rules_str}"
            )

        for cov, sc in partial[:3]:
            # Check what types are missing
            existing_types = set(cov.type_distribution.keys())
            missing_types = {"positive", "negative", "edge_case"} - existing_types
            if missing_types:
                type_str = ", ".join(missing_types)
                suggestions.append(
                    f"Add {type_str} scenarios for '{sc.name}' "
                    f"to improve coverage from {cov.coverage_percent:.0f}%"
                )

        return suggestions[:10]

    async def _generate_suggestions(
        self,
        coverage: list[SubCategoryCoverage],
        taxonomy: SubCategoryTaxonomy,
        logic_map: LogicMap | None,
        policy_text: str,
    ) -> list[str]:
        """Generate suggestions using LLM."""
        if not self.llm:
            self.llm = LLM(model=self.model, temperature=0.3)

        # Format coverage details
        coverage_details = self._format_coverage_details(coverage, taxonomy)
        gaps = self._identify_gaps(coverage, taxonomy)
        gaps_str = "\n".join(f"- {gap}" for gap in gaps)

        # Format logic map summary
        logic_map_summary = ""
        if logic_map:
            logic_map_summary = f"{len(logic_map.rules)} rules extracted"

        # Format policy summary (truncate if needed)
        policy_summary = policy_text[:1000] + "..." if len(policy_text) > 1000 else policy_text

        prompt = COVERAGE_SUGGESTIONS_PROMPT.format(
            total_sub_categories=len(taxonomy.sub_categories),
            covered_count=sum(1 for c in coverage if c.coverage_status == "covered"),
            partial_count=sum(1 for c in coverage if c.coverage_status == "partial"),
            uncovered_count=sum(1 for c in coverage if c.coverage_status == "uncovered"),
            overall_coverage_percent=sum(c.coverage_percent for c in coverage) / max(len(coverage), 1),
            covered_threshold=int(self.thresholds.covered_threshold * 100),
            partial_threshold=int(self.thresholds.partial_threshold * 100),
            coverage_details=coverage_details,
            gaps=gaps_str,
            policy_summary=policy_summary,
            logic_map_summary=logic_map_summary,
        )

        result = await self.llm.generate_structured(prompt, CoverageSuggestionsOutput)
        return result.suggestions

    def _format_coverage_details(
        self,
        coverage: list[SubCategoryCoverage],
        taxonomy: SubCategoryTaxonomy,
    ) -> str:
        """Format coverage details for the prompt."""
        lines = []
        for cov in coverage:
            sc = taxonomy.get_by_id(cov.sub_category_id)
            priority = f"[{sc.priority}]" if sc else ""
            status_icon = {
                "covered": "OK",
                "partial": "~",
                "uncovered": "X",
            }[cov.coverage_status]

            types_str = ", ".join(
                f"{t}:{c}" for t, c in cov.type_distribution.items()
            ) or "none"

            lines.append(
                f"[{status_icon}] {cov.sub_category_name} {priority}: "
                f"{cov.coverage_percent:.0f}% ({cov.scenario_count} scenarios) "
                f"Types: {types_str}"
            )
        return "\n".join(lines)
