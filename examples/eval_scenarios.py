"""
Eval Scenario Generation Example
================================

Generate eval scenarios from a policy and grade external model responses.

This demonstrates the eval-focused API:
1. generate_scenarios() - creates test scenarios with ground truth (no synthetic responses)
2. grade() - evaluates your own model's responses against scenarios

This is the proper workflow for:
- Drift detection (compare model versions)
- Compliance testing (verify policy adherence)
- Regression testing (catch behavior changes)
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import synkro
from synkro.models import Google
from synkro.examples import EXPENSE_POLICY
from synkro.reporting import FileLoggingReporter

# =============================================================================
# STEP 1: Generate eval scenarios (no synthetic responses)
# =============================================================================

# Use FileLoggingReporter for both CLI output and file logging
reporter = FileLoggingReporter(log_dir="./logs")

print("Generating eval scenarios...")
result = synkro.generate_scenarios(
    policy=EXPENSE_POLICY,
    count=5,
    generation_model=Google.GEMINI_25_FLASH,
    temperature=0.8,  # High temp for diverse scenarios
    enable_hitl=False,
    reporter=reporter,                  # Log to both CLI and file
)

print(f"\nGenerated {len(result.scenarios)} scenarios")
print(f"Distribution: {result.distribution}")

# =============================================================================
# STEP 2: Inspect the scenarios
# =============================================================================

print("\n" + "=" * 60)
print("GENERATED SCENARIOS")
print("=" * 60)

for i, scenario in enumerate(result.scenarios):
    print(f"\n--- Scenario {i+1} ---")
    print(f"Type: {scenario.scenario_type}")
    print(f"Category: {scenario.category}")
    print(f"Target rules: {scenario.target_rule_ids}")
    print(f"User message: {scenario.user_message[:100]}...")
    print(f"Expected outcome: {scenario.expected_outcome}")

# =============================================================================
# STEP 3: Simulate grading an external model's response
# =============================================================================

print("\n" + "=" * 60)
print("GRADING EXAMPLE")
print("=" * 60)

# Pick the first scenario
scenario = result.scenarios[0]

# Simulate what YOUR model might respond
# (In practice, you'd call your actual model here)
fake_model_response = """
I can help you with that expense report. Based on our policy:

1. For expenses over $50, you'll need manager approval
2. Please ensure you have your receipt ready
3. Submit within 30 days of the purchase date

Would you like me to help you prepare the submission?
"""

print(f"\nScenario: {scenario.user_message[:80]}...")
print(f"Expected: {scenario.expected_outcome}")
print(f"\nModel response: {fake_model_response[:100]}...")

# Grade the response
grade_result = synkro.grade(
    response=fake_model_response,
    scenario=scenario,
    policy=EXPENSE_POLICY,
    model=Google.GEMINI_25_FLASH,
)

print(f"\n--- Grade Result ---")
print(f"Passed: {grade_result.passed}")
print(f"Feedback: {grade_result.feedback}")
if grade_result.issues:
    print(f"Issues: {grade_result.issues}")

# =============================================================================
# STEP 4: Batch evaluation example
# =============================================================================

print("\n" + "=" * 60)
print("BATCH EVALUATION")
print("=" * 60)

# In a real scenario, you'd evaluate all scenarios against your model
passed = 0
failed = 0

for scenario in result.scenarios:
    # Your model would generate a response here
    # response = my_model(scenario.user_message)

    # For demo, we'll use a placeholder
    response = "I'll help you with that request according to our policies."

    grade = synkro.grade(
        response=response,
        scenario=scenario,
        policy=EXPENSE_POLICY,
        model=Google.GEMINI_25_FLASH,
    )

    if grade.passed:
        passed += 1
    else:
        failed += 1
        print(f"Failed scenario: {scenario.scenario_type} - {grade.feedback[:50]}...")

print(f"\nResults: {passed}/{len(result.scenarios)} passed ({passed/len(result.scenarios)*100:.0f}%)")
