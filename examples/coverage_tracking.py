"""
Example: Coverage Tracking for Scenario Diversity

Coverage tracking measures how well your generated scenarios cover
different aspects of your policy - like code coverage, but for policies.

This helps you:
- Identify gaps in scenario coverage
- Ensure all policy areas have test cases
- Improve coverage with natural language commands (HITL)
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
    # Generate scenarios with logic map access
    result = synkro.generate(
        POLICY,
        traces=5,
        generation_model=Google.GEMINI_25_FLASH,
        grading_model=Google.GEMINI_25_FLASH,
        return_logic_map=True,
        enable_hitl=True,  # Enable for interactive coverage improvement
    )

    # Get the dataset
    dataset = result.dataset
    dataset.save("coverage_example_output.jsonl")


if __name__ == "__main__":
    main()
