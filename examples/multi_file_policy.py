"""
Synkro Multi-File Policy Example
=================================

Generate training data from multiple policy documents.

This example demonstrates:
1. Loading policy from a folder (all supported files)
2. Loading multiple specific files
3. Combining documents into a single policy
4. Generating training data from combined policies

Use cases:
- Company handbooks split across multiple documents
- Policies organized by department (HR, Finance, Security)
- Combining related documents for comprehensive training
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from synkro import create_pipeline, Policy, DatasetType
from synkro.models.google import Google
from synkro.reporting import FileLoggingReporter

# =============================================================================
# Method 1: Load from Folder
# =============================================================================

print("=" * 80)
print("Method 1: Loading Policy from Folder")
print("=" * 80)
print()

# Load all supported files from the policies folder
policies_folder = Path(__file__).parent / "policies"
policy_from_folder = Policy.from_file(policies_folder)

print(f"✓ Loaded {policy_from_folder.word_count} words from folder")
print(f"  Source: {policy_from_folder.source}")
print()

# =============================================================================
# Method 2: Load Specific Files
# =============================================================================

print("=" * 80)
print("Method 2: Loading Specific Files")
print("=" * 80)
print()

# Load specific files explicitly
policy_files = [
    policies_folder / "Expense-Reimbursement-Policy.docx",
    policies_folder / "hr_policy.md",
    policies_folder / "security_policy.txt",
]

policy_from_files = Policy.from_files(policy_files)

print(f"✓ Loaded {policy_from_files.word_count} words from {len(policy_files)} files")
print(f"  Source: {policy_from_files.source}")
print()

# =============================================================================
# Method 3: Custom Separator
# =============================================================================

print("=" * 80)
print("Method 3: Custom Document Separator")
print("=" * 80)
print()

# Use a custom separator between documents
policy_custom = Policy.from_files(
    policy_files,
    separator="\n\n" + "=" * 60 + "\n\n",
    source_prefix="company_policies"
)

print(f"✓ Loaded with custom separator: {policy_custom.word_count} words")
print()

# =============================================================================
# Generate Training Data
# =============================================================================

print("=" * 80)
print("Generating Training Data from Combined Policy")
print("=" * 80)
print()

# Use FileLoggingReporter for both CLI output and file logging
reporter = FileLoggingReporter(log_dir="./logs")

# Create pipeline
pipeline = create_pipeline(
    model=Google.GEMINI_25_FLASH,
    grading_model=Google.GEMINI_25_FLASH,
    dataset_type=DatasetType.CONVERSATION,
    max_iterations=2,
    reporter=reporter,                  # Log to both CLI and file
)

# Generate from the combined policy (using folder method)
print(f"Generating 5 traces from combined policy...")
print(f"Policy covers: Expense (txt + docx), HR, and Security policies")
print()

dataset = pipeline.generate(policy_from_folder, traces=5)

# =============================================================================
# Save and Display Results
# =============================================================================

# Save dataset
output_file = "multi_file_training.jsonl"
dataset.save(output_file)

print()
print("=" * 80)
print("Results")
print("=" * 80)
print()

# Display summary
print(dataset.summary())

# Show sample trace
if len(dataset) > 0:
    print("\n--- Sample Trace ---")
    trace = dataset[0]
    print(f"Scenario: {trace.scenario.description[:80]}...")
    print(f"Response: {trace.assistant_message[:80]}...")
    if trace.grade:
        print(f"Grade: {'✓ PASS' if trace.grade.passed else '✗ FAIL'}")

print(f"\n✓ Saved to: {output_file}")
print(f"✓ Generated from combined policy documents (folder contains {len(policy_files)} files)")

