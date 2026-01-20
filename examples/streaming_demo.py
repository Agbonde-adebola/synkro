#!/usr/bin/env python3
"""
Streaming API Demonstration

This script demonstrates the new streaming APIs in synkro.
Run with: python examples/streaming_demo.py

Requires one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY/GEMINI_API_KEY.
The API will auto-detect which provider is available and use appropriate models.
"""

import asyncio
import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def demo_streaming_extraction():
    """Demonstrate streaming rule extraction."""
    import synkro

    print("\n" + "=" * 60)
    print("DEMO 1: Streaming Rule Extraction")
    print("=" * 60 + "\n")

    policy = """
    Expense Policy:
    1. All expenses under $25 can be self-approved
    2. Expenses between $25-$100 require manager approval
    3. Expenses over $100 require director approval
    4. Travel expenses always require pre-approval
    5. Emergency expenses can be approved retroactively
    """

    print(f"Policy:\n{policy}\n")
    print("Extracting rules (streaming)...\n")

    async for event in synkro.extract_rules_stream(policy):
        match event.type:
            case "progress":
                print(f"  📊 Progress: {event.progress:.0%} - {event.message}")
            case "rule_found":
                rule = event.rule
                print(f"  ✅ Found {rule.rule_id}: {rule.text[:60]}...")
            case "complete":
                print("\n✨ Extraction complete!")
                print(f"   Total rules: {len(event.result.logic_map.rules)}")
                print(f"   Cost: ${event.metrics.cost:.4f}")
                return event.result


async def demo_streaming_scenarios(extraction_result):
    """Demonstrate streaming scenario generation."""
    import synkro

    print("\n" + "=" * 60)
    print("DEMO 2: Streaming Scenario Generation")
    print("=" * 60 + "\n")

    print("Generating scenarios (streaming)...\n")

    async for event in synkro.generate_scenarios_stream(
        "Expense policy with approval levels",
        logic_map=extraction_result.logic_map,
        count=5,
    ):
        match event.type:
            case "progress":
                print(f"  📊 Progress: {event.progress:.0%} - {event.message}")
            case "scenario_generated":
                s = event.scenario
                stype = (
                    s.scenario_type.value if hasattr(s.scenario_type, "value") else s.scenario_type
                )
                print(f"  🎯 [{stype}] {s.description[:50]}...")
            case "complete":
                print("\n✨ Scenario generation complete!")
                print(f"   Total scenarios: {len(event.result.scenarios)}")
                print(f"   Distribution: {event.result.distribution}")
                print(f"   Cost: ${event.metrics.cost:.4f}")
                return event.result


async def demo_session_workflow():
    """Demonstrate Session-based workflow."""
    import synkro

    print("\n" + "=" * 60)
    print("DEMO 3: Session-Based Workflow")
    print("=" * 60 + "\n")

    session = synkro.Session()

    policy = (
        "All purchases over $500 require two approvals. Urgent purchases can skip one approval."
    )

    print("Step 1: Extract rules...")
    await session.extract_rules(policy)
    print(f"   ✅ Extracted {len(session.logic_map.rules)} rules")

    print("\nStep 2: Generate scenarios...")
    await session.generate_scenarios(count=3)
    print(f"   ✅ Generated {len(session.scenarios)} scenarios")

    print("\nStep 3: Edit rules with natural language...")
    try:
        new_map, summary = await session.edit_rules(
            "Add a rule: Refunds require finance team approval"
        )
        print(f"   ✅ {summary}")
    except Exception as e:
        print(f"   ⚠️ Edit skipped: {e}")

    print("\nSession Status:")
    print("-" * 40)
    print(session.format_status())

    return session


async def demo_step_by_step_api():
    """Demonstrate step-by-step API."""
    import synkro

    print("\n" + "=" * 60)
    print("DEMO 4: Step-by-Step API")
    print("=" * 60 + "\n")

    policy = "Documents must be reviewed within 5 business days. Critical documents require same-day review."

    print("Step 1: Extract rules...")
    extraction = await synkro.extract_rules_async(policy)
    print(f"   Rules: {[r.rule_id for r in extraction.logic_map.rules]}")
    print(f"   Cost: ${extraction.metrics.cost:.4f}")

    print("\nStep 2: Generate scenarios...")
    scenarios = await synkro.generate_scenarios_async(
        policy,
        logic_map=extraction.logic_map,
        count=3,
    )
    print(f"   Scenarios: {len(scenarios.scenarios)}")
    print(f"   Distribution: {scenarios.distribution}")

    print("\nStep 3: Synthesize traces...")
    traces = await synkro.synthesize_traces_async(
        policy,
        scenarios=scenarios,
    )
    print(f"   Traces: {len(traces.traces)}")
    for i, trace in enumerate(traces.traces):
        print(f"   T{i+1}: {trace.user_message[:40]}...")

    return traces


async def demo_tool_definitions():
    """Demonstrate tool definitions for LLM agents."""
    from synkro import TOOL_DEFINITIONS
    from synkro.tools import get_tool_by_name, get_tool_names

    print("\n" + "=" * 60)
    print("DEMO 5: Tool Definitions for LLM Agents")
    print("=" * 60 + "\n")

    print(f"Available tools ({len(TOOL_DEFINITIONS)}):")
    for name in get_tool_names():
        tool = get_tool_by_name(name)
        desc = tool["function"]["description"][:60]
        print(f"  • {name}: {desc}...")

    print("\nExample tool schema (extract_rules):")
    import json

    extract_tool = get_tool_by_name("extract_rules")
    print(json.dumps(extract_tool, indent=2)[:500] + "...")


async def main():
    """Run all demos."""
    print("\n" + "🚀" * 20)
    print("\n  SYNKRO STREAMING API DEMO")
    print("\n" + "🚀" * 20)

    # Check for API key (import directly to avoid loading full synkro package yet)
    import os

    def _detect_provider():
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return "google"
        return None

    provider = _detect_provider()
    if not provider:
        print("\n⚠️  Warning: No API key found. Please set one of:")
        print("   - ANTHROPIC_API_KEY (for Claude)")
        print("   - OPENAI_API_KEY (for GPT)")
        print("   - GOOGLE_API_KEY or GEMINI_API_KEY (for Gemini)\n")
    else:
        print(f"\n✅ Using {provider} provider (models will be auto-detected)\n")

    try:
        # Demo 1: Streaming extraction
        extraction = await demo_streaming_extraction()

        # Demo 2: Streaming scenarios (if extraction succeeded)
        if extraction:
            await demo_streaming_scenarios(extraction)

        # Demo 3: Session workflow
        await demo_session_workflow()

        # Demo 4: Step-by-step API
        await demo_step_by_step_api()

        # Demo 5: Tool definitions (no API calls needed)
        await demo_tool_definitions()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
