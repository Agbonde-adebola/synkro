"""
Evaluation Dataset Generation Test
===================================

Generate an eval dataset with low temperature for deterministic outputs.
Uses DatasetType.EVALUATION with QA format including ground truth.

This demonstrates:
- Low temperature (0.2) for reproducible eval scenarios
- EVALUATION dataset type with expected outcomes
- QA format export with ground truth fields
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from synkro import create_pipeline, DatasetType
from synkro.models import Google
from synkro.examples import EXPENSE_POLICY

# Create pipeline with low temperature for eval datasets
# Low temperature = more deterministic outputs for consistent evals
pipeline = create_pipeline(
    dataset_type=DatasetType.EVALUATION,
    model=Google.GEMINI_25_FLASH,
    grading_model=Google.GEMINI_25_FLASH,
    temperature=0.2,  # Low temp for eval reproducibility
    skip_grading=True,
    enable_hitl=False,
)

# Generate 5 eval traces
dataset = pipeline.generate(EXPENSE_POLICY, traces=5)

# Save as LangSmith format (includes ground truth)
dataset.save("eval_test.jsonl", format="langsmith")

# Show what we generated
print(f"\nGenerated {len(dataset)} eval traces")
print(f"Categories: {dataset.categories}")

# Preview first trace
if dataset.traces:
    trace = dataset.traces[0]
    print(f"\n--- Sample Trace ---")
    print(f"Scenario type: {trace.scenario.scenario_type}")
    print(f"Target rules: {trace.scenario.target_rule_ids}")
    print(f"Expected outcome: {trace.scenario.expected_outcome}")
    print(f"User: {trace.user_message[:100]}...")
    print(f"Assistant: {trace.assistant_message[:100]}...")
