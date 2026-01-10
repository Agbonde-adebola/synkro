"""
Evaluation Dataset Generation Test
===================================

Generate eval scenarios using the eval API.

This demonstrates:
- generate_scenarios() - creates test scenarios with ground truth (no synthetic responses)
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

# Save to LangSmith format (auto-names: synkro_eval_YYYY-MM-DD_HHMM.jsonl)
result.save(format="langsmith")