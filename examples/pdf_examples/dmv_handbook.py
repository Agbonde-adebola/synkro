"""
DMV California Handbook Example
===============================

Generate training data from the California DMV Driver Handbook PDF.
This example demonstrates:
- Loading policy from a PDF file
- Automatic chunked extraction for large documents
- File logging for observability
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from synkro import create_pipeline, Policy, DatasetType
from synkro.models.google import Google
from synkro.reporting import FileLoggingReporter

# =============================================================================
# Load the DMV Handbook PDF
# =============================================================================

print("=" * 80)
print("Loading California DMV Driver Handbook")
print("=" * 80)
print()

pdf_path = Path(__file__).parent / "dmv_ca_handbook.pdf"
policy = Policy.from_file(pdf_path)

print(f"✓ Loaded PDF: {pdf_path.name}")
print(f"  Word count: {policy.word_count:,} words")

# Large documents (>5k words) automatically use chunked extraction:
# - Splits into ~4k word chunks with overlap
# - Extracts rules from each chunk in parallel
# - Merges and deduplicates rules
# This avoids hallucination and handles documents of any size!

print()

# =============================================================================
# Create Pipeline with File Logging
# =============================================================================

print("=" * 80)
print(f"Generating 20 Traces from DMV Handbook")
print("=" * 80)
print()

# Use FileLoggingReporter for both CLI output and file logging
reporter = FileLoggingReporter(log_dir="./logs")

pipeline = create_pipeline(
    model=Google.GEMINI_25_FLASH,           # Fast generation
    grading_model=Google.GEMINI_25_FLASH,   # Quality grading
    dataset_type=DatasetType.CONVERSATION,  # Chat format for fine-tuning
    skip_grading=True,                      # Skip grading for faster generation
    enable_hitl=False,                      # Skip interactive session for speed
    reporter=reporter,                      # Log to both CLI and file
)

# Generate 20 traces - this will:
# 1. Extract logic map (rules) using chunked extraction
# 2. Generate diverse scenarios covering different rules
# 3. Create multi-turn conversations for each scenario
dataset = pipeline.generate(policy, traces=20)

# =============================================================================
# Save Results
# =============================================================================

output_file = "dmv_handbook_training.jsonl"
dataset.save(output_file, pretty_print=True)

print()
print("=" * 80)
print("Results")
print("=" * 80)
print()
print(f"✓ Generated {len(dataset)} traces")
print(f"✓ Saved to: {output_file}")
print(f"✓ Log file: {reporter.log_path}")
