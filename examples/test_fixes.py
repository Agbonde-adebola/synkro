"""Test script to verify prompt fixes."""

from pathlib import Path

from dotenv import load_dotenv

from synkro.examples import EXPENSE_POLICY
from synkro.models.openai import OpenAI
from synkro.pipelines import create_pipeline
from synkro.types import DatasetType

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Create pipeline
pipeline = create_pipeline(
    model=OpenAI.GPT_4O_MINI,
    grading_model=OpenAI.GPT_4O,
    dataset_type=DatasetType.CONVERSATION,
    max_iterations=2,
)

# Generate small dataset to test
print("Generating 5 traces to test fixes...")
dataset = pipeline.generate(EXPENSE_POLICY, traces=5)

# Save results
output_file = "test_fixes_output.jsonl"
dataset.save(output_file)

print(f"\nSaved to {output_file}")
print(f"Total traces: {len(dataset)}")
print(f"Passing: {len(dataset.filter(passed=True))}")

# Print first few traces for inspection
print("\n" + "=" * 60)
print("SAMPLE TRACES FOR INSPECTION:")
print("=" * 60)

for i, trace in enumerate(dataset.traces[:3]):
    print(f"\n--- Trace {i+1} ---")
    messages = trace.formatted_output.get("messages", [])
    for msg in messages[:4]:  # First 4 messages
        role = msg.get("role", "?")
        content = msg.get("content", "")[:200]  # Truncate
        print(f"{role.upper()}: {content}...")
    print()
