"""
Test Policy Violation Detection and Remediation.

Demonstrates:
1. Detect policy violations in traces
2. Store violations (LangSmith/Langfuse/Datadog compatible format)
3. Generate golden traces for finetuning

Run: python examples/test_policy_remediation.py
Requires: API key for LLM provider (OpenAI, Anthropic, etc.)
"""

import asyncio
import os
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
        {"role": "user", "content": "I'd like to return these headphones. I used them a few times."},
        {
            "role": "assistant",
            "content": "No problem! I'll process a full refund for the headphones. "
            "Just ship them back and we'll credit your account.",
        },
    ],
]


async def run_detection_and_remediation(logic_map, model: str):
    """Run the async detection and remediation steps."""
    from synkro import PolicyDetector, PolicyRefiner, ViolationStore

    # 2. Detect violations
    print("\n2. Detecting violations...")
    detector = PolicyDetector(logic_map, model=model)
    violations = await detector.detect(BAD_TRACES)
    print(f"   Found {len(violations)} violations")

    # 3. Store violations
    print("\n3. Storing violations...")
    store = ViolationStore()
    store.add_batch(violations)
    print(f"   Stored {store.count()} violations")

    # 4. Inspect violations
    print("\n4. Violation details:")
    for v in violations:
        print(f"\n   --- Violation {v.id[:12]}... ---")
        print(f"   Rules violated: {v.rules_violated}")
        print(f"   Severity: {v.severity}")
        print(f"   User intent: {v.user_intent}")
        print(f"   Bad response: {v.trace[-1]['content'][:60]}...")
        print(f"   Expected: {v.expected_outcome}")

        # Show platform-compatible format
        print("\n   LangSmith format:")
        ls = v.to_langsmith()
        print(f"     key: {ls['key']}, score: {ls['score']}")

        print("   Langfuse format:")
        lf = v.to_langfuse()
        print(f"     name: {lf['name']}, value: {lf['value']}, dataType: {lf['dataType']}")

        print("   Datadog format:")
        dd = v.to_datadog()
        print(f"     label: {dd['label']}, metric_type: {dd['metric_type']}")

    # 5. Generate golden traces
    print("\n5. Generating golden traces...")
    refiner = PolicyRefiner(logic_map, model=model)
    golden = await refiner.refine(store, traces_per_violation=2)
    print(f"   Generated {len(golden)} golden traces")

    # 6. Inspect golden traces
    print("\n6. Golden trace samples:")
    for i, trace in enumerate(golden[:4]):  # Show first 4
        print(f"\n   --- Golden Trace {i + 1} ---")
        print(f"   User: {trace.user_message[:60]}...")
        print(f"   Assistant: {trace.assistant_message[:80]}...")
        if trace.rules_applied:
            print(f"   Rules applied: {trace.rules_applied}")

    # 7. Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    violations_file = output_dir / "violations.jsonl"
    golden_file = output_dir / "golden_traces.jsonl"

    store.save(violations_file)
    golden.save(golden_file)

    print(f"\n7. Saved outputs:")
    print(f"   Violations: {violations_file}")
    print(f"   Golden traces: {golden_file}")

    return violations, golden


def main():
    from synkro import ingest
    from synkro.models import Cerebras

    print("=" * 60)
    print("POLICY REMEDIATION TEST")
    print("=" * 60)

    # Use Cerebras model
    MODEL = Cerebras.GPT_OSS_120B

    # 1. Process policy into LogicMap (sync function)
    print("\n1. Ingesting policy...")
    config = ingest(POLICY, model=MODEL)
    logic_map = config.logic_map
    print(f"   Extracted {len(logic_map.rules)} rules:")
    for rule in logic_map.rules:
        print(f"   - {rule.rule_id}: {rule.text[:50]}...")

    # Run async detection and remediation
    violations, golden = asyncio.run(run_detection_and_remediation(logic_map, MODEL))

    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Bad traces analyzed: {len(BAD_TRACES)}")
    print(f"Violations detected: {len(violations)}")
    print(f"Golden traces generated: {len(golden)}")
    print(f"Traces per violation: 2")
    print("\nViolation storage is compatible with:")
    print("  - LangSmith (use violation.to_langsmith())")
    print("  - Langfuse (use violation.to_langfuse())")
    print("  - Datadog (use violation.to_datadog())")
    print("=" * 60)


if __name__ == "__main__":
    # Check for API key
    has_key = any(
        [
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("ANTHROPIC_API_KEY"),
            os.environ.get("GOOGLE_API_KEY"),
            os.environ.get("CEREBRAS_API_KEY"),
        ]
    )
    if not has_key:
        print("WARNING: No LLM API key found in environment.")
        print("Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, CEREBRAS_API_KEY")
        print()

    main()
