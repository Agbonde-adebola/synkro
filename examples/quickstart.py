"""
Synkro Quickstart - RLAIF Data Generation
==========================================

Generate high-quality chat fine-tuning datasets:
- LLM generates diverse scenarios from your policy
- Strong model creates expert responses
- AI grader filters by quality against policy rules
- Output: Ready-to-train JSONL with only passing traces

Dataset types: CONVERSATION (multi-turn), INSTRUCTION (single-turn), TOOL_CALL (with function calling)

For production use, consider the two-step workflow:
  1. synkro ingest policy.pdf  # Extract rules once, review if needed
  2. synkro generate --config policy_config.json  # Fast generation, reusable

See other examples for advanced features.
"""

from pathlib import Path

from dotenv import load_dotenv

from synkro.examples import EXPENSE_POLICY
from synkro.models.google import Google
from synkro.pipelines import create_pipeline
from synkro.reporting import FileLoggingReporter
from synkro.types import DatasetType

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Use FileLoggingReporter for both CLI output and file logging
# Logs are saved to a timestamped file in the current directory
reporter = FileLoggingReporter(log_dir="./logs")

# Create a pipeline: generates scenarios, responses, and grades them
# - model: Used for scenario and response generation (fast model recommended)
# - grading_model: Used for quality grading (stronger model recommended for better filtering)
# - dataset_type: CONVERSATION = multi-turn, INSTRUCTION = single-turn, TOOL_CALL = with function calling
pipeline = create_pipeline(
    model=Google.GEMINI_25_FLASH,  # Fast generation
    grading_model=Google.GEMINI_25_FLASH,  # Quality grading
    # grading_model=Google.GEMINI_25_PRO, # Quality grading (stronger = better filtering)
    dataset_type=DatasetType.CONVERSATION,  # Chat format for fine-tuning
    # max_iterations=3,                   # Max refinement iterations per trace
    # skip_grading=True,  # Skip grading phase for faster generation
    # enable_hitl=False,                  # Disable HITL for non-interactive testing
    reporter=reporter,  # Log to both CLI and file
)

# Output: Only traces that passed quality checks (when grading enabled)
dataset = pipeline.generate(EXPENSE_POLICY, traces=5)

# Save to JSONL file (ready for training)
# Note: When skip_grading=True, don't filter by passed since traces have no grades
dataset.save(pretty_print=True)
