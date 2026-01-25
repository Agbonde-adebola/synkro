"""
Simulated HITL with Database Persistence
=========================================

Demonstrates the clean Session API with show commands and done().
Simulates what an agent would do - step by step with inspection.

No interactive prompts - everything is programmatic.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from synkro import Session
from synkro.examples import EXPENSE_POLICY
from synkro.models.google import Google

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


async def main():
    print("=" * 70)
    print("Simulated HITL - Clean Abstractions Demo")
    print("=" * 70)
    print()

    # =========================================================================
    # Create session
    # =========================================================================
    print("Creating session...")
    session = await Session.create(
        policy=EXPENSE_POLICY,
        session_id="clean-api-demo",
    )
    session.model = Google.GEMINI_25_FLASH
    session.grading_model = Google.GEMINI_25_FLASH
    print(f"Session: {session.session_id}")
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

    # Session status
    print("Session status...")
    print(session.status())
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

    print("=" * 70)
    print("Done! All abstractions tested.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
