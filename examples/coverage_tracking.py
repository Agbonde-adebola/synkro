"""
Example: Coverage Tracking for Scenario Diversity

This example demonstrates how to use the coverage tracking system to:
1. View coverage metrics for generated scenarios
2. Identify coverage gaps
3. Understand sub-category distribution

Coverage tracking helps ensure your generated dataset covers all important
aspects of your policy, similar to code coverage for tests.
"""

import asyncio
from synkro import (
    Policy,
    CoverageReport,
    SubCategoryTaxonomy,
    coverage_report,
)
from synkro.types.coverage import (
    SubCategory,
    SubCategoryCoverage,
    CoverageThresholds,
)
from synkro.types.logic_map import GoldenScenario, ScenarioType, LogicMap, Rule, RuleCategory
from synkro.types.core import Category
from synkro.coverage import (
    TaxonomyExtractor,
    ScenarioTagger,
    CoverageCalculator,
    CoverageImprover,
)


# Sample policy for demonstration
SAMPLE_POLICY = """
EXPENSE REIMBURSEMENT POLICY

1. APPROVAL THRESHOLDS
- Expenses under $50: No approval required
- Expenses $50-$500: Manager approval required
- Expenses over $500: VP approval required

2. MEAL EXPENSES
- Daily meal limit: $75 per day
- Client meals: Must include client name and business purpose
- Team meals: Maximum $25 per person for team events

3. TRAVEL EXPENSES
- Flights must be booked 14 days in advance for reimbursement
- Economy class only for flights under 6 hours
- Business class allowed for flights over 6 hours with VP approval

4. RECEIPT REQUIREMENTS
- All expenses over $25 require itemized receipts
- Digital receipts are accepted
- Missing receipts require manager exception approval
"""


def demo_coverage_types():
    """Demonstrate the coverage data types."""
    print("\n" + "=" * 60)
    print("DEMO: Coverage Data Types")
    print("=" * 60)

    # Create a sample sub-category
    sub_category = SubCategory(
        id="SC001",
        name="Approval thresholds",
        description="Rules about expense approval limits based on amount",
        parent_category="Expense Limits",
        related_rule_ids=["R001", "R002", "R003"],
        priority="high",
    )
    print(f"\nSubCategory: {sub_category.name}")
    print(f"  ID: {sub_category.id}")
    print(f"  Parent: {sub_category.parent_category}")
    print(f"  Priority: {sub_category.priority}")
    print(f"  Related Rules: {sub_category.related_rule_ids}")

    # Create a sample taxonomy
    taxonomy = SubCategoryTaxonomy(
        sub_categories=[
            sub_category,
            SubCategory(
                id="SC002",
                name="Meal limits",
                description="Daily and per-person meal expense limits",
                parent_category="Meal Expenses",
                related_rule_ids=["R004", "R005"],
                priority="medium",
            ),
            SubCategory(
                id="SC003",
                name="Travel booking rules",
                description="Advance booking and class requirements",
                parent_category="Travel",
                related_rule_ids=["R006", "R007"],
                priority="medium",
            ),
            SubCategory(
                id="SC004",
                name="Receipt requirements",
                description="Documentation needed for reimbursement",
                parent_category="Documentation",
                related_rule_ids=["R008", "R009"],
                priority="high",
            ),
        ],
        reasoning="Organized by expense type and compliance importance",
    )
    print(f"\nTaxonomy: {len(taxonomy.sub_categories)} sub-categories")
    for sc in taxonomy.sub_categories:
        print(f"  - {sc.id}: {sc.name} [{sc.priority}]")

    # Create sample coverage metrics
    coverage = SubCategoryCoverage(
        sub_category_id="SC001",
        sub_category_name="Approval thresholds",
        parent_category="Expense Limits",
        scenario_count=5,
        scenario_ids=["S1", "S2", "S3", "S4", "S5"],
        coverage_percent=100.0,
        coverage_status="covered",
        type_distribution={"positive": 2, "negative": 2, "edge_case": 1},
    )
    print(f"\nCoverage for '{coverage.sub_category_name}':")
    print(f"  Scenarios: {coverage.scenario_count}")
    print(f"  Coverage: {coverage.coverage_percent}%")
    print(f"  Status: {coverage.coverage_status}")
    print(f"  Type distribution: {coverage.type_distribution}")


def demo_coverage_report():
    """Demonstrate the coverage report structure."""
    print("\n" + "=" * 60)
    print("DEMO: Coverage Report")
    print("=" * 60)

    # Create a sample coverage report
    report = CoverageReport(
        total_scenarios=20,
        total_sub_categories=4,
        covered_count=2,
        partial_count=1,
        uncovered_count=1,
        overall_coverage_percent=68.75,
        sub_category_coverage=[
            SubCategoryCoverage(
                sub_category_id="SC001",
                sub_category_name="Approval thresholds",
                parent_category="Expense Limits",
                scenario_count=5,
                coverage_percent=100.0,
                coverage_status="covered",
                type_distribution={"positive": 2, "negative": 2, "edge_case": 1},
            ),
            SubCategoryCoverage(
                sub_category_id="SC002",
                sub_category_name="Meal limits",
                parent_category="Meal Expenses",
                scenario_count=4,
                coverage_percent=80.0,
                coverage_status="covered",
                type_distribution={"positive": 2, "negative": 1, "edge_case": 1},
            ),
            SubCategoryCoverage(
                sub_category_id="SC003",
                sub_category_name="Travel booking rules",
                parent_category="Travel",
                scenario_count=2,
                coverage_percent=40.0,
                coverage_status="partial",
                type_distribution={"positive": 1, "negative": 1},
            ),
            SubCategoryCoverage(
                sub_category_id="SC004",
                sub_category_name="Receipt requirements",
                parent_category="Documentation",
                scenario_count=0,
                coverage_percent=0.0,
                coverage_status="uncovered",
                type_distribution={},
            ),
        ],
        gaps=[
            "Receipt requirements [HIGH] (0% coverage, 0 scenarios)",
            "Travel booking rules [MEDIUM] (partial: 40% coverage)",
        ],
        suggestions=[
            "Add 3+ scenarios for 'Receipt requirements' (HIGH priority) testing R008, R009",
            "Add edge_case scenarios for 'Travel booking rules' to improve from 40%",
        ],
        heatmap_data={
            "Expense Limits": {"Approval thresholds": 100.0},
            "Meal Expenses": {"Meal limits": 80.0},
            "Travel": {"Travel booking rules": 40.0},
            "Documentation": {"Receipt requirements": 0.0},
        },
    )

    # Print the summary
    print(f"\n{report.to_summary_string()}")

    # Show heatmap data
    print("\nHeatmap Data:")
    for category, sub_cats in report.heatmap_data.items():
        print(f"\n  {category}:")
        for name, pct in sub_cats.items():
            bar = "#" * int(pct / 10) + "-" * (10 - int(pct / 10))
            print(f"    {name}: [{bar}] {pct:.0f}%")


def demo_coverage_thresholds():
    """Demonstrate configurable coverage thresholds."""
    print("\n" + "=" * 60)
    print("DEMO: Coverage Thresholds")
    print("=" * 60)

    # Default thresholds
    default = CoverageThresholds()
    print(f"\nDefault Thresholds:")
    print(f"  Covered: >= {default.covered_threshold * 100}%")
    print(f"  Partial: >= {default.partial_threshold * 100}%")
    print(f"  Min scenarios/sub-category: {default.min_scenarios_per_sub_category}")

    # Custom thresholds for stricter coverage
    strict = CoverageThresholds(
        covered_threshold=0.9,  # 90%+
        partial_threshold=0.5,  # 50-90%
        min_scenarios_per_sub_category=3,
        priority_multipliers={"high": 3.0, "medium": 1.5, "low": 0.5},
    )
    print(f"\nStrict Thresholds:")
    print(f"  Covered: >= {strict.covered_threshold * 100}%")
    print(f"  Partial: >= {strict.partial_threshold * 100}%")
    print(f"  Min scenarios/sub-category: {strict.min_scenarios_per_sub_category}")
    print(f"  Priority multipliers: {strict.priority_multipliers}")


async def demo_coverage_calculator():
    """Demonstrate the coverage calculator (offline, no LLM needed)."""
    print("\n" + "=" * 60)
    print("DEMO: Coverage Calculator (Offline)")
    print("=" * 60)

    # Create a simple taxonomy
    taxonomy = SubCategoryTaxonomy(
        sub_categories=[
            SubCategory(
                id="SC001",
                name="Amount thresholds",
                description="Approval limits by amount",
                parent_category="Approvals",
                related_rule_ids=["R001"],
                priority="high",
            ),
            SubCategory(
                id="SC002",
                name="Time constraints",
                description="Deadline and timing rules",
                parent_category="Timing",
                related_rule_ids=["R002"],
                priority="medium",
            ),
        ],
        reasoning="Test taxonomy",
    )

    # Create some scenarios with sub_category_ids
    scenarios = [
        GoldenScenario(
            description="I need to expense a $45 lunch",
            context="Employee lunch",
            category="Approvals",
            scenario_type=ScenarioType.POSITIVE,
            target_rule_ids=["R001"],
            expected_outcome="Approved without manager approval",
            sub_category_ids=["SC001"],
        ),
        GoldenScenario(
            description="Can I expense $600 for a conference?",
            context="Professional development",
            category="Approvals",
            scenario_type=ScenarioType.NEGATIVE,
            target_rule_ids=["R001"],
            expected_outcome="Needs VP approval",
            sub_category_ids=["SC001"],
        ),
        GoldenScenario(
            description="I spent exactly $50 on supplies",
            context="Office supplies",
            category="Approvals",
            scenario_type=ScenarioType.EDGE_CASE,
            target_rule_ids=["R001"],
            expected_outcome="Boundary case - requires manager approval",
            sub_category_ids=["SC001"],
        ),
    ]

    # Calculate coverage (no LLM needed for basic calculation)
    calculator = CoverageCalculator(llm=None)  # No LLM = no AI suggestions
    report = await calculator.calculate(
        scenarios=scenarios,
        taxonomy=taxonomy,
        generate_suggestions=True,  # Will use basic suggestions without LLM
    )

    print(f"\nCalculated Coverage Report:")
    print(f"  Total scenarios: {report.total_scenarios}")
    print(f"  Total sub-categories: {report.total_sub_categories}")
    print(f"  Overall coverage: {report.overall_coverage_percent:.1f}%")
    print(f"  Covered: {report.covered_count}")
    print(f"  Partial: {report.partial_count}")
    print(f"  Uncovered: {report.uncovered_count}")

    print("\nPer Sub-Category:")
    for cov in report.sub_category_coverage:
        status_icon = {"covered": "V", "partial": "~", "uncovered": "X"}[cov.coverage_status]
        print(f"  [{status_icon}] {cov.sub_category_name}: {cov.coverage_percent:.0f}% ({cov.scenario_count} scenarios)")

    if report.gaps:
        print("\nGaps:")
        for gap in report.gaps:
            print(f"  - {gap}")

    if report.suggestions:
        print("\nSuggestions:")
        for i, sugg in enumerate(report.suggestions[:3], 1):
            print(f"  {i}. {sugg}")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# SYNKRO COVERAGE TRACKING EXAMPLES")
    print("#" * 60)

    # Run synchronous demos
    demo_coverage_types()
    demo_coverage_report()
    demo_coverage_thresholds()

    # Run async demo
    asyncio.run(demo_coverage_calculator())

    print("\n" + "=" * 60)
    print("USAGE WITH SYNKRO.GENERATE()")
    print("=" * 60)
    print("""
To use coverage tracking with synkro.generate():

    import synkro

    # Generate with return_logic_map=True to access coverage
    result = synkro.generate(
        policy,
        traces=20,
        return_logic_map=True,
    )

    # View coverage report (if available)
    synkro.coverage_report(result)

    # Or get as dict for programmatic use
    report = synkro.coverage_report(result, format="dict")
    if report:
        print(f"Coverage: {report['overall_coverage_percent']}%")
        print(f"Gaps: {len(report['gaps'])}")

Note: Full coverage tracking requires pipeline integration
which enables taxonomy extraction and scenario tagging during
generation. The coverage components can also be used standalone
for custom workflows.
""")


if __name__ == "__main__":
    main()
