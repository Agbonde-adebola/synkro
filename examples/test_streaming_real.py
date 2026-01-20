#!/usr/bin/env python3
"""Test streaming with a real policy document."""

import asyncio
import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synkro


async def main():
    # Check for API key
    provider = synkro.detect_available_provider()
    if not provider:
        print("No API key found. Please set one of:")
        print("  export ANTHROPIC_API_KEY=your-key")
        print("  export OPENAI_API_KEY=your-key")
        print("  export GOOGLE_API_KEY=your-key")
        return

    print(f"Using {provider} provider\n")

    # Sample policy - replace with your own or load from file
    policy = """
    Expense Reimbursement Policy

    1. All expense reports must be submitted within 30 days of the expense date.
    2. Expenses under $25 can be self-approved.
    3. Expenses between $25 and $100 require manager approval.
    4. Expenses over $100 require director approval.
    5. Travel expenses always require pre-approval before the trip.
    6. Meals with clients can be expensed up to $75 per person.
    7. Emergency expenses can be approved retroactively with justification.
    8. Receipts are required for all expenses over $10.
    9. Personal expenses are never reimbursable.
    10. Alcohol is not reimbursable except for client entertainment.
    """

    # Or load from a file:
    # with open("path/to/your/policy.txt") as f:
    #     policy = f.read()

    print("=" * 60)
    print("STEP 1: Extracting Rules")
    print("=" * 60)

    logic_map = None
    async for event in synkro.extract_rules_stream(policy):
        match event.type:
            case "progress":
                print(f"  {event.progress:.0%} - {event.message}")
            case "rule_found":
                print(f"  ✅ {event.rule.rule_id}: {event.rule.text[:70]}...")
            case "complete":
                logic_map = event.result.logic_map
                print(f"\n✨ Extracted {len(logic_map.rules)} rules")
                print(f"   Cost: ${event.metrics.cost:.4f}")

    if not logic_map or len(logic_map.rules) == 0:
        print("\nNo rules extracted. Check your API key and policy text.")
        return

    print("\n" + "=" * 60)
    print("STEP 2: Generating Scenarios")
    print("=" * 60)

    scenarios_result = None
    async for event in synkro.generate_scenarios_stream(policy, logic_map=logic_map, count=10):
        match event.type:
            case "progress":
                print(f"  {event.progress:.0%} - {event.message}")
            case "scenario_generated":
                stype = event.scenario.scenario_type.value
                print(f"  🎯 [{stype:10}] {event.scenario.description[:50]}...")
            case "complete":
                scenarios_result = event.result
                print(f"\n✨ Generated {len(scenarios_result.scenarios)} scenarios")
                print(f"   Distribution: {scenarios_result.distribution}")
                print(f"   Cost: ${event.metrics.cost:.4f}")

    if not scenarios_result:
        print("\nNo scenarios generated.")
        return

    print("\n" + "=" * 60)
    print("STEP 3: Synthesizing Traces")
    print("=" * 60)

    traces_result = None
    async for event in synkro.synthesize_traces_stream(policy, scenarios=scenarios_result):
        match event.type:
            case "progress":
                print(f"  {event.progress:.0%} - {event.message}")
            case "trace_generated":
                print(f"  ✍️  Trace {event.index + 1}: {event.trace.user_message[:50]}...")
            case "complete":
                traces_result = event.result
                print(f"\n✨ Synthesized {len(traces_result.traces)} traces")
                print(f"   Cost: ${event.metrics.cost:.4f}")

    if not traces_result:
        print("\nNo traces generated.")
        return

    # Save the dataset
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    from synkro.core.dataset import Dataset

    dataset = Dataset(traces=traces_result.traces)
    dataset.save("test_output.jsonl")
    print(f"✅ Saved {len(dataset)} traces to test_output.jsonl")

    # Show a sample trace
    print("\n" + "=" * 60)
    print("SAMPLE TRACE")
    print("=" * 60)
    if traces_result.traces:
        trace = traces_result.traces[0]
        print(f"User: {trace.user_message[:200]}...")
        print(f"\nAssistant: {trace.assistant_message[:300]}...")


if __name__ == "__main__":
    asyncio.run(main())
