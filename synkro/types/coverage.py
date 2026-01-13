"""Coverage tracking types for scenario diversity analysis.

The coverage system tracks how well generated scenarios cover the policy's
sub-categories, similar to code coverage for tests.
"""

from typing import Literal
from pydantic import BaseModel, Field


class SubCategory(BaseModel):
    """
    A sub-category within a policy category, LLM-extracted during planning.

    Sub-categories represent specific testable aspects of a policy category.
    For example, a "Refund Policy" category might have sub-categories like:
    - "Time-based eligibility" (30-day window rules)
    - "Amount thresholds" (refund limits)
    - "Method restrictions" (cash vs. credit rules)

    Examples:
        >>> sc = SubCategory(
        ...     id="SC001",
        ...     name="Time-based eligibility",
        ...     description="Rules about refund timing windows",
        ...     parent_category="Refund Policy",
        ...     related_rule_ids=["R001", "R002"],
        ...     priority="high",
        ... )
    """

    id: str = Field(
        description="Unique identifier (e.g., 'SC001', 'SC002')"
    )
    name: str = Field(
        description="Short, descriptive name for the sub-category"
    )
    description: str = Field(
        description="What this sub-category covers"
    )
    parent_category: str = Field(
        description="Name of the parent category this belongs to"
    )
    related_rule_ids: list[str] = Field(
        default_factory=list,
        description="Rule IDs from LogicMap that relate to this sub-category"
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Coverage priority based on policy importance"
    )


class SubCategoryTaxonomy(BaseModel):
    """
    Complete taxonomy of sub-categories extracted from a policy.

    The taxonomy organizes all sub-categories across all policy categories,
    enabling coverage tracking and gap analysis.

    Examples:
        >>> taxonomy = SubCategoryTaxonomy(
        ...     sub_categories=[sc1, sc2, sc3],
        ...     reasoning="Organized by rule type and policy section",
        ... )
        >>> refund_scs = taxonomy.get_by_category("Refund Policy")
    """

    sub_categories: list[SubCategory] = Field(
        description="All extracted sub-categories"
    )
    reasoning: str = Field(
        description="Explanation of how the taxonomy was organized"
    )

    def get_by_id(self, sc_id: str) -> SubCategory | None:
        """Get a sub-category by its ID."""
        for sc in self.sub_categories:
            if sc.id == sc_id:
                return sc
        return None

    def get_by_category(self, category: str) -> list[SubCategory]:
        """Get all sub-categories for a parent category."""
        return [sc for sc in self.sub_categories if sc.parent_category == category]

    def get_by_rule(self, rule_id: str) -> list[SubCategory]:
        """Get all sub-categories that relate to a specific rule."""
        return [sc for sc in self.sub_categories if rule_id in sc.related_rule_ids]

    def get_categories(self) -> list[str]:
        """Get list of unique parent categories."""
        return list(set(sc.parent_category for sc in self.sub_categories))


class SubCategoryCoverage(BaseModel):
    """
    Coverage metrics for a single sub-category.

    Tracks how well a sub-category is covered by generated scenarios,
    including count, percentage, and distribution by scenario type.

    Examples:
        >>> cov = SubCategoryCoverage(
        ...     sub_category_id="SC001",
        ...     sub_category_name="Time eligibility",
        ...     scenario_count=5,
        ...     coverage_percent=100.0,
        ...     coverage_status="covered",
        ...     type_distribution={"positive": 2, "negative": 2, "edge_case": 1},
        ... )
    """

    sub_category_id: str = Field(
        description="ID of the sub-category"
    )
    sub_category_name: str = Field(
        description="Name of the sub-category"
    )
    parent_category: str = Field(
        default="",
        description="Parent category name"
    )
    scenario_count: int = Field(
        default=0,
        description="Number of scenarios covering this sub-category"
    )
    scenario_ids: list[str] = Field(
        default_factory=list,
        description="IDs/indices of scenarios that cover this sub-category"
    )
    coverage_percent: float = Field(
        default=0.0,
        description="Percentage of expected coverage achieved (0-100)"
    )
    coverage_status: Literal["covered", "partial", "uncovered"] = Field(
        default="uncovered",
        description="Coverage status based on thresholds"
    )
    type_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count by scenario type (positive/negative/edge_case/irrelevant)"
    )

    @property
    def is_covered(self) -> bool:
        """Check if this sub-category is fully covered."""
        return self.coverage_status == "covered"

    @property
    def needs_attention(self) -> bool:
        """Check if this sub-category needs more coverage."""
        return self.coverage_status in ("partial", "uncovered")


class CoverageThresholds(BaseModel):
    """
    Configurable thresholds for coverage status determination.

    Examples:
        >>> thresholds = CoverageThresholds(
        ...     covered_threshold=0.8,   # 80%+ = covered
        ...     partial_threshold=0.3,   # 30-80% = partial
        ...     min_scenarios_per_sub_category=2,
        ... )
    """

    covered_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum percentage to be considered 'covered' (default: 80%)"
    )
    partial_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum percentage to be considered 'partial' (default: 30%)"
    )
    min_scenarios_per_sub_category: int = Field(
        default=2,
        ge=1,
        description="Minimum scenarios expected per sub-category"
    )
    priority_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"high": 2.0, "medium": 1.0, "low": 0.5},
        description="Multipliers for expected scenarios based on priority"
    )


class CoverageReport(BaseModel):
    """
    Complete coverage report for a generation run.

    Provides a comprehensive view of scenario coverage across all sub-categories,
    including metrics, gaps, and actionable suggestions.

    Examples:
        >>> report = CoverageReport(
        ...     total_scenarios=20,
        ...     total_sub_categories=10,
        ...     covered_count=7,
        ...     partial_count=2,
        ...     uncovered_count=1,
        ...     overall_coverage_percent=85.0,
        ...     sub_category_coverage=[...],
        ...     gaps=["Method restrictions (0% coverage)"],
        ...     suggestions=["Add 3 negative scenarios for Method restrictions"],
        ... )
        >>> print(f"Coverage: {report.overall_coverage_percent}%")
    """

    total_scenarios: int = Field(
        description="Total number of scenarios in the dataset"
    )
    total_sub_categories: int = Field(
        description="Total number of sub-categories in the taxonomy"
    )
    covered_count: int = Field(
        description="Number of sub-categories with full coverage"
    )
    partial_count: int = Field(
        description="Number of sub-categories with partial coverage"
    )
    uncovered_count: int = Field(
        description="Number of sub-categories with no coverage"
    )
    overall_coverage_percent: float = Field(
        description="Overall coverage percentage (0-100)"
    )

    sub_category_coverage: list[SubCategoryCoverage] = Field(
        default_factory=list,
        description="Detailed coverage for each sub-category"
    )

    gaps: list[str] = Field(
        default_factory=list,
        description="Identified coverage gaps (uncovered or underrepresented)"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable suggestions to improve coverage"
    )

    heatmap_data: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Nested dict: category -> sub_category -> coverage_percent"
    )

    def get_coverage_for(self, sub_category_id: str) -> SubCategoryCoverage | None:
        """Get coverage details for a specific sub-category."""
        for cov in self.sub_category_coverage:
            if cov.sub_category_id == sub_category_id:
                return cov
        return None

    def get_uncovered(self) -> list[SubCategoryCoverage]:
        """Get all sub-categories with no coverage."""
        return [c for c in self.sub_category_coverage if c.coverage_status == "uncovered"]

    def get_partial(self) -> list[SubCategoryCoverage]:
        """Get all sub-categories with partial coverage."""
        return [c for c in self.sub_category_coverage if c.coverage_status == "partial"]

    def get_by_category(self, category: str) -> list[SubCategoryCoverage]:
        """Get coverage for all sub-categories in a parent category."""
        return [c for c in self.sub_category_coverage if c.parent_category == category]

    def to_summary_string(self) -> str:
        """Generate a human-readable summary of the coverage report."""
        lines = [
            f"Coverage Report",
            f"=" * 40,
            f"Overall: {self.overall_coverage_percent:.1f}%",
            f"Sub-categories: {self.covered_count} covered, {self.partial_count} partial, {self.uncovered_count} uncovered",
            f"Total scenarios: {self.total_scenarios}",
        ]

        if self.gaps:
            lines.append(f"\nGaps ({len(self.gaps)}):")
            for gap in self.gaps[:5]:
                lines.append(f"  - {gap}")
            if len(self.gaps) > 5:
                lines.append(f"  ... and {len(self.gaps) - 5} more")

        if self.suggestions:
            lines.append(f"\nSuggestions:")
            for i, sugg in enumerate(self.suggestions[:3], 1):
                lines.append(f"  {i}. {sugg}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Return coverage report as a dictionary.

        Returns:
            Dictionary representation of the coverage report

        Example:
            >>> report = result.coverage_report
            >>> d = report.to_dict()
            >>> print(f"Coverage: {d['overall_coverage_percent']}%")
        """
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """
        Return coverage report as a JSON string.

        Args:
            indent: Number of spaces for indentation (default: 2)

        Returns:
            JSON string representation of the coverage report

        Example:
            >>> report = result.coverage_report
            >>> json_str = report.to_json()
            >>> print(json_str)
        """
        import json
        return json.dumps(self.model_dump(), indent=indent)

    def print(self) -> None:
        """
        Print formatted coverage report to console.

        Uses the RichReporter to display a nicely formatted coverage report
        with colors and tables.

        Example:
            >>> result = synkro.generate(policy, return_logic_map=True)
            >>> result.coverage_report.print()
        """
        from synkro.reporting import RichReporter
        reporter = RichReporter()
        reporter.on_coverage_calculated(self)


class CoverageIntent(BaseModel):
    """
    Parsed intent from a natural language coverage command.

    Used in HITL sessions to interpret commands like:
    - "show coverage"
    - "increase coverage for refunds by 20%"
    - "get amount thresholds to 80%"

    Examples:
        >>> intent = CoverageIntent(
        ...     operation="increase",
        ...     target_sub_category="refunds",
        ...     increase_amount=20,
        ... )
    """

    operation: Literal["view", "increase", "target"] = Field(
        description="Type of coverage operation"
    )

    # For view operations
    view_mode: Literal["summary", "gaps", "heatmap", "detail"] | None = Field(
        default=None,
        description="What to display (for view operations)"
    )

    # For increase/target operations
    target_sub_category: str | None = Field(
        default=None,
        description="Sub-category name or ID to target"
    )
    target_percent: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Target coverage percentage (for target operations)"
    )
    increase_amount: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Percentage points to increase (for increase operations)"
    )
    scenario_type: str | None = Field(
        default=None,
        description="Specific scenario type to add (positive/negative/edge_case)"
    )


__all__ = [
    "SubCategory",
    "SubCategoryTaxonomy",
    "SubCategoryCoverage",
    "CoverageThresholds",
    "CoverageReport",
    "CoverageIntent",
]
