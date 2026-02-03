"""
Test Session API with Detection and Remediation.

Demonstrates the extended Session API for:
1. Extract rules from policy
2. Detect violations in bad traces
3. Generate golden traces for finetuning

Run: python examples/test_session_remediation.py
Requires: API key for LLM provider (OpenAI, Anthropic, etc.)
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Sample policy
POLICY = """
Customer Support Refund Policy:

1. REFUND WINDOW: Refunds are only allowed within 30 days of purchase.
   - Purchases older than 30 days are NOT eligible for refunds.
   - The purchase date is the date shown on the receipt.

2. ITEM CONDITION: Items must be unused and in original packaging.
   - Used items cannot be refunded.
   - Items without original packaging may only receive store credit.

3. DIGITAL PRODUCTS: Digital products (ebooks, software, downloads) are non-refundable.
   - Once downloaded or activated, no refunds are possible.
   - Exceptions only for technical issues preventing use.

4. DEFECTIVE ITEMS: Defective items can be refunded at any time.
   - Customer must provide proof of defect (photo or video).
   - Defective refunds bypass the 30-day window.
"""

# Bad traces that violate the policy
BAD_TRACES = [
    # Violation: Approved refund outside 30-day window
    [
        {"role": "user", "content": "I want to return my laptop I bought 2 months ago"},
        {
            "role": "assistant",
            "content": "Sure, I can process that refund for you right away! "
            "I'll initiate the refund to your original payment method.",
        },
    ],
    # Violation: Approved refund for digital product
    [
        {"role": "user", "content": "Can I get a refund on this ebook I purchased yesterday?"},
        {
            "role": "assistant",
            "content": "Of course! I'll refund the ebook to your account immediately. "
            "The refund should appear within 3-5 business days.",
        },
    ],
    # Violation: Approved refund for used item
    [
        {
            "role": "user",
            "content": "I'd like to return these headphones. I used them a few times.",
        },
        {
            "role": "assistant",
            "content": "No problem! I'll process a full refund for the headphones. "
            "Just ship them back and we'll credit your account.",
        },
    ],
]


async def main():
    from synkro import Session
    from synkro.models import Cerebras

    print("=" * 60)
    print("SESSION REMEDIATION TEST")
    print("=" * 60)

    # Use Cerebras model
    MODEL = Cerebras.GPT_OSS_120B

    # 1. Create session with policy
    print("\n1. Creating session with policy...")
    session = await Session.create(policy=POLICY)
    session.model = MODEL
    session.grading_model = MODEL
    print(f"   Session ID: {session.session_id}")

    # 2. Extract rules
    print("\n2. Extracting rules...")
    await session.extract_rules(POLICY)
    print(session.show_rules())

    # 3. Detect violations in external bad traces
    print("\n3. Detecting violations...")
    violations = await session.detect_violations(traces=BAD_TRACES)
    print(f"   Found {len(violations)} violations")
    print(session.show_violations())

    # 4. Generate golden traces (remediation)
    print("\n4. Generating golden traces (remediation)...")
    golden = await session.remediate(traces_per_violation=2)
    print(f"   Generated {len(golden)} golden traces")
    print(session.show_remediation())

    # 5. Export results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    violations_file = output_dir / "session_violations.jsonl"
    golden_file = output_dir / "session_golden_traces.jsonl"

    session.save_violations(violations_file)
    session.save_remediation(golden_file)

    print("\n5. Saved outputs:")
    print(f"   Violations: {violations_file}")
    print(f"   Golden traces: {golden_file}")

    # 6. Show session status
    print("\n6. Session status:")
    print(f"   {session.status()}")

    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Bad traces analyzed: {len(BAD_TRACES)}")
    print(f"Violations detected: {len(violations)}")
    print(f"Golden traces generated: {len(golden)}")
    print("Traces per violation: 2")
    print("\nSession API methods used:")
    print("  - await session.extract_rules(policy)")
    print("  - await session.detect_violations(traces=bad_traces)")
    print("  - await session.remediate(traces_per_violation=2)")
    print("  - session.save_violations(path)")
    print("  - session.save_remediation(path)")
    print("=" * 60)


if __name__ == "__main__":
    # Trigger auto-loading of .env files before checking
    from synkro.utils.model_detection import detect_available_provider

    if not detect_available_provider():
        print("WARNING: No LLM API key found in environment.")
        print("Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, CEREBRAS_API_KEY")
        print()

    asyncio.run(main())
