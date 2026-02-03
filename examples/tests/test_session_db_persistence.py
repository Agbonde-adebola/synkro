"""
Test Session DB Persistence for Violations and Remediation.

This test verifies that:
1. Violations are persisted to the database
2. Remediated traces are persisted to the database
3. Data can be loaded back from the database
4. Session stats include violation/remediation counts

Run: python examples/test_session_db_persistence.py
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
1. Refunds are only allowed within 30 days of purchase.
2. Used items cannot be refunded.
3. Digital products are non-refundable.
"""

# Bad traces
BAD_TRACES = [
    [
        {"role": "user", "content": "I want to return my laptop I bought 2 months ago"},
        {"role": "assistant", "content": "Sure, I'll process that refund right away!"},
    ],
    [
        {"role": "user", "content": "Can I get a refund on this ebook?"},
        {"role": "assistant", "content": "Of course! Refund processed."},
    ],
]


async def main():
    from synkro import Session
    from synkro.models import Cerebras
    from synkro.storage import Storage

    print("=" * 60)
    print("SESSION DB PERSISTENCE TEST")
    print("=" * 60)

    # Use Cerebras model
    MODEL = Cerebras.GPT_OSS_120B

    # Use a unique test session ID
    TEST_SESSION_ID = "db-persist-test"

    # Delete existing test DB to ensure clean schema
    db_path = Path("~/.synkro/sessions.db").expanduser()
    if db_path.exists():
        print(f"\n1. Removing old DB at {db_path}...")
        db_path.unlink()

    # 1. Create session
    print("\n2. Creating session...")
    session = await Session.create(policy=POLICY, session_id=TEST_SESSION_ID)
    session.model = MODEL
    session.grading_model = MODEL
    print(f"   Session ID: {session.session_id}")

    # 2. Extract rules
    print("\n3. Extracting rules...")
    await session.extract_rules(POLICY)
    print(f"   Extracted {len(session.logic_map.rules)} rules")

    # 3. Detect violations
    print("\n4. Detecting violations...")
    violations = await session.detect_violations(traces=BAD_TRACES)
    print(f"   Found {len(violations)} violations")

    # 4. Generate golden traces
    print("\n5. Generating golden traces...")
    golden = await session.remediate(traces_per_violation=2)
    print(f"   Generated {len(golden)} golden traces")

    # 5. Verify DB persistence
    print("\n6. Verifying DB persistence...")

    # Direct DB access to verify data
    store = Storage()

    # Check violations in DB
    db_violations = await store.get_violations(TEST_SESSION_ID)
    print(f"   DB violations: {len(db_violations)}")
    assert len(db_violations) == len(
        violations
    ), f"Expected {len(violations)}, got {len(db_violations)}"

    # Check remediated traces in DB
    db_remediated = await store.get_remediated_traces(TEST_SESSION_ID)
    print(f"   DB remediated traces: {len(db_remediated)}")
    assert len(db_remediated) == len(golden), f"Expected {len(golden)}, got {len(db_remediated)}"

    # Check stats
    stats = await store.get_stats(TEST_SESSION_ID)
    print(
        f"   DB stats: violation_count={stats.get('violation_count', 0)}, remediated_count={stats.get('remediated_count', 0)}"
    )
    assert stats.get("violation_count") == len(violations), "Violation count mismatch"
    assert stats.get("remediated_count") == len(golden), "Remediated count mismatch"

    # 6. Verify violation data integrity
    print("\n7. Verifying data integrity...")
    for i, db_v in enumerate(db_violations):
        orig_v = violations[i]
        assert db_v["id"] == orig_v.id, f"Violation ID mismatch at {i}"
        assert db_v["severity"] == orig_v.severity, f"Severity mismatch at {i}"
        assert db_v["rules_violated"] == orig_v.rules_violated, f"Rules mismatch at {i}"
        print(f"   Violation {i+1}: ID={db_v['id'][:12]}... ✓")

    # 7. Verify remediated trace data integrity
    for i, db_t in enumerate(db_remediated):
        assert "messages" in db_t, f"Missing messages at {i}"
        assert len(db_t["messages"]) > 0, f"Empty messages at {i}"
        print(f"   Remediated trace {i+1}: {len(db_t['messages'])} messages ✓")

    # 8. Load session from DB and verify
    print("\n8. Loading session from DB...")
    loaded = await Session.load_from_db(TEST_SESSION_ID, full=True)
    await loaded.ensure_loaded()

    # Note: Violations and remediated_traces are not loaded back into Session yet
    # They're stored in DB but would need additional load methods
    print(f"   Loaded session: {loaded.session_id}")
    print(f"   Rules: {len(loaded.logic_map.rules) if loaded.logic_map else 0}")

    # 9. Clean up
    print("\n9. Cleaning up test session...")
    await session.delete()
    print("   Session deleted.")

    # Final summary
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print(f"Violations persisted: {len(violations)}")
    print(f"Remediated traces persisted: {len(golden)}")
    print("DB stores exact JSON format (same as hashmap)")
    print("=" * 60)


if __name__ == "__main__":
    # Trigger auto-loading of .env files before checking
    from synkro.utils.model_detection import detect_available_provider

    if not detect_available_provider():
        print("WARNING: No LLM API key found in environment.")
        print()

    asyncio.run(main())
