"""
Simulated HITL with Database Persistence
=========================================

Demonstrates the clean Session API with show commands and done().
Simulates what an agent would do - step by step with inspection.

Tests all Session API features:
- dataset_type (conversation, instruction, evaluation, tool_call)
- skip_grading (skip verification phase)
- turns (conversation turns)

No interactive prompts - everything is programmatic.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from synkro import Session
from synkro.examples import EXPENSE_POLICY

# from synkro.models.google import Google
from synkro.models import Cerebras

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


async def main():
    print("=" * 70)
    print("Simulated HITL - Clean Abstractions Demo")
    print("=" * 70)
    print()

    # =========================================================================
    # Create session with dataset_type (default: conversation)
    # =========================================================================
    print("Creating session (dataset_type=conversation)...")
    session = await Session.create(
        policy=EXPENSE_POLICY,
        session_id="clean-api-demo",
        dataset_type="conversation",  # Explicit dataset type
    )
    session.model = Cerebras.GPT_OSS_120B
    session.grading_model = Cerebras.GPT_OSS_120B
    print(f"Session: {session.session_id}")
    print(f"Dataset type: {session.dataset_type}")
    print()

    # =========================================================================
    # Extract rules and show them
    # =========================================================================
    print("Extracting rules...")
    await session.extract_rules(session.policy)
    print()
    print(session.show_rules(limit=5))
    print()

    # =========================================================================
    # Refine rules (simulated HITL)
    # =========================================================================
    print("Refining rules...")
    summary = await session.refine_rules("Add rule: Conference attendance requires pre-approval")
    print(f"  → {summary}")
    print()

    # =========================================================================
    # Generate scenarios and show distribution
    # =========================================================================
    print("Generating scenarios...")
    await session.generate_scenarios(count=5)
    print()
    print(session.show_distribution())
    print()

    # =========================================================================
    # Show scenarios filtered by type
    # =========================================================================
    print(session.show_scenarios(filter="edge_case", limit=3))
    print()

    # =========================================================================
    # Generate taxonomy and show it
    # =========================================================================
    print("Generating taxonomy...")
    await session.generate_taxonomy()
    print()
    print(session.show_taxonomy(limit=5))
    print()

    # =========================================================================
    # Refine taxonomy (simulated HITL)
    # =========================================================================
    print("Refining taxonomy...")
    summary = await session.refine_taxonomy("Add sub-category for Remote Work Equipment")
    print(f"  → {summary}")
    print()

    # =========================================================================
    # Refine scenarios (simulated HITL)
    # =========================================================================
    print("Refining scenarios...")
    summary = await session.refine_scenarios("Add 2 edge cases for the conference rule")
    print(f"  → {summary}")
    print()

    # =========================================================================
    # Done - synthesize, verify, export in one call
    # =========================================================================
    print("Running done()...")
    await session.done(output="clean_api_output.jsonl")
    print()
    print(session.show_passed())
    print()

    # =========================================================================
    # Show any failed traces
    # =========================================================================
    print(session.show_failed())
    print()

    # =========================================================================
    # Verify persistence
    # =========================================================================
    print("Verifying persistence...")
    reloaded = await Session.load_from_db("clean-api-demo")
    await reloaded.ensure_loaded()  # Load data for inspection
    print(f"  Reloaded: {reloaded.session_id}")
    print(f"  Rules: {len(reloaded.logic_map.rules)}")
    print(f"  Scenarios: {len(reloaded.scenarios)}")
    print(f"  Traces: {len(reloaded.verified_traces)}")
    print()

    # =========================================================================
    # Show all commands work on reloaded session
    # =========================================================================
    print("Show commands on reloaded session:")
    print(reloaded.show_rules(limit=3))
    print()
    print(reloaded.show_passed())
    print()

    # =========================================================================
    # Get dataset from DB (no file needed)
    # =========================================================================
    print("Getting dataset from DB...")
    dataset = reloaded.to_dataset()
    print(f"  Dataset traces: {len(dataset.traces)}")
    print(f"  Pass rate: {dataset.passing_rate:.1%}")
    print(f"  First trace messages: {len(dataset.traces[0].messages)}")
    print()

    # =========================================================================
    # Test new session management methods
    # =========================================================================
    print("Testing session management...")
    print()

    # List all sessions
    print("Listing all sessions...")
    sessions = await Session.list_sessions()
    print(f"  Found {len(sessions)} session(s)")
    for s in sessions[:3]:
        print(f"    - {s['session_id']} (updated: {s['updated_at'][:10]})")
    print()

    # Session status (now includes cost)
    print("Session status...")
    print(session.status())
    print()

    # Cost tracking
    print("Cost summary...")
    print(session.show_cost_summary())
    print()

    print("Cost breakdown...")
    print(session.show_cost())
    print()

    # Show individual trace
    print("Showing trace #0...")
    print(session.show_trace(0))
    print()

    # Undo last change
    print("Testing undo...")
    undo_result = await session.undo()
    print(f"  → {undo_result}")
    print()

    # Delete a test session (create one first)
    print("Testing delete...")
    temp_session = await Session.create(session_id="temp-delete-test")
    deleted = await temp_session.delete()
    print(f"  → Deleted temp session: {deleted}")
    print()

    # =========================================================================
    # Test dataset_type="instruction" with skip_grading=True
    # =========================================================================
    print("=" * 70)
    print("Testing dataset_type='instruction' with skip_grading=True")
    print("=" * 70)
    print()

    instruction_session = await Session.create(
        policy=EXPENSE_POLICY,
        session_id="instruction-test",
        dataset_type="instruction",  # Single-turn forced
        skip_grading=True,  # Skip verification for speed
    )
    instruction_session.model = Cerebras.GPT_OSS_120B
    instruction_session.grading_model = Cerebras.GPT_OSS_120B
    print(f"Session: {instruction_session.session_id}")
    print(f"Dataset type: {instruction_session.dataset_type}")
    print(f"Skip grading: {instruction_session.skip_grading}")
    print()

    # Quick flow - extract, generate, done (no verify)
    await instruction_session.extract_rules(instruction_session.policy)
    print(f"Rules extracted: {len(instruction_session.logic_map.rules)}")

    await instruction_session.generate_scenarios(count=3)
    print(f"Scenarios generated: {len(instruction_session.scenarios)}")

    # Synthesize with explicit turns (will be forced to 1 for instruction)
    await instruction_session.synthesize_traces(turns=5)  # Should be forced to 1
    print(f"Traces synthesized: {len(instruction_session.traces)}")

    # Check message count (should be 2-3 for single-turn: system + user + assistant)
    first_trace = instruction_session.traces[0]
    print(f"Messages in first trace: {len(first_trace.messages)} (should be 2-3 for single-turn)")

    # done() should skip verification
    dataset = await instruction_session.done(output="instruction_output.jsonl")
    print(f"Dataset traces: {len(dataset.traces)}")
    print(f"Verified traces: {instruction_session.verified_traces}")  # Should be None
    print()

    # Cleanup
    await instruction_session.delete()
    print("Instruction session deleted.")
    print()

    # =========================================================================
    # Test custom turns with conversation type
    # =========================================================================
    print("=" * 70)
    print("Testing custom turns (turns=3)")
    print("=" * 70)
    print()

    turns_session = await Session.create(
        policy=EXPENSE_POLICY,
        session_id="turns-test",
        dataset_type="conversation",
        skip_grading=True,
    )
    turns_session.model = Cerebras.GPT_OSS_120B
    turns_session.grading_model = Cerebras.GPT_OSS_120B

    await turns_session.extract_rules(turns_session.policy)
    await turns_session.generate_scenarios(count=2)

    # Synthesize with 3 turns
    await turns_session.synthesize_traces(turns=3)
    print(f"Traces synthesized: {len(turns_session.traces)}")

    # Check message count (should be more than single-turn)
    first_trace = turns_session.traces[0]
    print(f"Messages in first trace: {len(first_trace.messages)} (multi-turn)")

    await turns_session.done(output="turns_output.jsonl")

    # Cleanup
    await turns_session.delete()
    print("Turns session deleted.")
    print()

    print("=" * 70)
    print("Done! All Session API features tested:")
    print("  - dataset_type: conversation, instruction")
    print("  - skip_grading: True/False")
    print("  - turns: custom turn count")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
