"""
Evaluation Dataset Generation Test
===================================

Generate eval scenarios and grade model responses using the eval API.

This demonstrates:
- generate_scenarios() - creates test scenarios with ground truth (no synthetic responses)
- grade() - evaluates your model's responses against scenarios
- High temperature (0.8) for diverse scenario coverage
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import synkro
from synkro.models import Google
from synkro.examples import EXPENSE_POLICY

# Generate 5 eval scenarios (no synthetic responses)
# High temperature = diverse scenario coverage
result = synkro.generate_scenarios(
    policy=EXPENSE_POLICY,
    count=5,
    generation_model=Google.GEMINI_25_FLASH,
    temperature=0.8,
    enable_hitl=False,
)

# Show what we generated
print(f"\nGenerated {len(result.scenarios)} eval scenarios")
print(f"Distribution: {result.distribution}")

# Preview scenarios
for i, scenario in enumerate(result.scenarios):
    print(f"\n--- Scenario {i+1} ---")
    print(f"Type: {scenario.scenario_type}")
    print(f"Category: {scenario.category}")
    print(f"Target rules: {scenario.target_rule_ids}")
    print(f"User message: {scenario.user_message[:80]}...")
    print(f"Expected outcome: {scenario.expected_outcome[:80]}...")

# Example: Grade a fake model response
print("\n" + "=" * 50)
print("GRADING EXAMPLE")
print("=" * 50)

scenario = result.scenarios[0]
fake_response = "I'll help you submit that expense. Please provide your receipt."

grade = synkro.grade(
    response=fake_response,
    scenario=scenario,
    policy=EXPENSE_POLICY,
    model=Google.GEMINI_25_FLASH,
)

print(f"\nScenario: {scenario.user_message[:60]}...")
print(f"Response: {fake_response}")
print(f"Passed: {grade.passed}")
print(f"Feedback: {grade.feedback[:100]}...")
