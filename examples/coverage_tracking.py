"""
Example: Coverage Tracking for Scenario Diversity

Coverage tracking measures how well your generated scenarios cover
different aspects of your policy - like code coverage, but for policies.

This helps you:
- Identify gaps in scenario coverage
- Ensure all policy areas have test cases
- Improve coverage with natural language commands
"""

import synkro
from synkro.models.google import Google

# Sample policy
POLICY = """
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


def main():
    print("=" * 60)
    print("COVERAGE TRACKING EXAMPLE")
    print("=" * 60)

    # Generate scenarios with logic map access
    print("\nGenerating scenarios...")
    result = synkro.generate(
        POLICY,
        traces=5,
        generation_model=Google.GEMINI_25_FLASH,
        grading_model=Google.GEMINI_25_FLASH,
        return_logic_map=True,
        enable_hitl=False,  # Disable interactive mode for scripted example
    )

    # View coverage report (prints to console)
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    synkro.coverage_report(result)

    # Get coverage as dictionary for programmatic use
    print("\n" + "=" * 60)
    print("PROGRAMMATIC ACCESS")
    print("=" * 60)
    report = synkro.coverage_report(result, format="dict")
    if report:
        print(f"Overall coverage: {report['overall_coverage_percent']:.1f}%")
        print(f"Covered: {report['covered_count']}")
        print(f"Partial: {report['partial_count']}")
        print(f"Uncovered: {report['uncovered_count']}")
        print(f"Gaps: {len(report['gaps'])}")

        if report['gaps']:
            print("\nGaps to address:")
            for gap in report['gaps'][:3]:
                print(f"  - {gap}")

    # Get the dataset
    dataset = result.dataset
    print(f"\nDataset: {len(dataset)} traces")
    dataset.save("coverage_example_output.jsonl")
    print("Saved to coverage_example_output.jsonl")


if __name__ == "__main__":
    main()
