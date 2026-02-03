"""
Test SQL Verification.

Demonstrates:
1. sql_file: SQL in separate file (proper syntax highlighting)
2. DSN in config with ${ENV_VAR} expansion
3. verify_batch: Efficient batch verification
4. SQLRefiner: Auto-fix violations using database ground truth

Run: python examples/test_sql_verify.py
Requires: pip install synkro[verify]
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Paths
EXAMPLES_DIR = Path(__file__).parent
DB_PATH = EXAMPLES_DIR / "test_verify.sqlite"
CONFIG_FILE = EXAMPLES_DIR / "verifiers" / "funding.yaml"
SQL_FILE = EXAMPLES_DIR / "verifiers" / "funding.sql"

# Fake company data - the "ground truth"
FAKE_COMPANIES = [
    ("Acme Corp", "Series A", 8_000_000, "Sequoia Capital", "2024-03-15"),
    ("Bolt Industries", "Series B", 25_000_000, "Andreessen Horowitz", "2024-06-20"),
    ("CloudNine AI", "Seed", 2_500_000, "Y Combinator", "2024-01-10"),
]


async def setup_database():
    """Create SQLite database and seed with fake data."""
    from synkro.verify.sql import SQL

    # Remove existing DB
    if DB_PATH.exists():
        DB_PATH.unlink()

    db = SQL(str(DB_PATH))

    # Create table
    await db.execute_raw("""
        CREATE TABLE funding_rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            round_type TEXT NOT NULL,
            amount_usd INTEGER NOT NULL,
            lead_investor TEXT NOT NULL,
            announced_date TEXT NOT NULL
        )
    """)

    # Insert fake data
    for company in FAKE_COMPANIES:
        await db.execute_raw(f"""
            INSERT INTO funding_rounds (company_name, round_type, amount_usd, lead_investor, announced_date)
            VALUES ('{company[0]}', '{company[1]}', {company[2]}, '{company[3]}', '{company[4]}')
        """)

    await db.close()
    print(f"Database created: {DB_PATH}")


async def run_tests():
    """Test SQL verification features."""
    from synkro import Verdict, verify_batch, verify_sql
    from synkro.models import Cerebras
    from synkro.verify.llm import VerifierLLM

    # Set DATABASE_URL env var (simulating real usage)
    os.environ["DATABASE_URL"] = str(DB_PATH)

    # Create LLM
    llm = VerifierLLM(
        route_model=Cerebras.GPT_OSS_120B,
        judge_model=Cerebras.GPT_OSS_120B,
    )

    print(f"\nConfig: {CONFIG_FILE}")
    print(f"SQL:    {SQL_FILE}")

    print("\n" + "=" * 60)
    print("TEST 1: YAML config with sql_file and ${DATABASE_URL}")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "Tell me about Acme Corp's funding"},
        {
            "role": "assistant",
            "content": "Acme Corp raised $8 million in their Series A round, led by Sequoia Capital.",
        },
    ]

    report = await verify_sql(str(CONFIG_FILE), messages=messages, llm=llm)

    print(f"AI Response: {report.response}")
    print(f"Value: {report.results[0].value.value}")
    print(f"Score: {report.results[0].score}")
    print(f"Comment: {report.results[0].comment}")

    assert report.results[0].value == Verdict.PASS
    print("✅ PASS\n")

    print("=" * 60)
    print("TEST 2: Batch verification")
    print("=" * 60)

    traces = [
        [
            {"role": "user", "content": "Tell me about Acme Corp"},
            {"role": "assistant", "content": "Acme Corp raised $8 million Series A from Sequoia."},
        ],
        [
            {"role": "user", "content": "What about Bolt?"},
            {
                "role": "assistant",
                "content": "Bolt raised $100 million from Andreessen.",
            },  # Wrong: $25M
        ],
        [
            {"role": "user", "content": "Tell me about FakeCo"},
            {"role": "assistant", "content": "FakeCo raised $50 million."},
        ],
    ]

    reports = await verify_batch(str(CONFIG_FILE), traces=traces, llm=llm)

    print(f"Verified {len(traces)} traces:")
    for i, report in enumerate(reports):
        value = report.results[0].value.value if report.results else "none"
        print(f"  Trace {i + 1}: {value}")

    assert reports[0].results[0].value == Verdict.PASS
    assert reports[1].results[0].value == Verdict.VIOLATION
    assert reports[2].results[0].value == Verdict.SKIPPED
    print("✅ PASS\n")

    print("=" * 60)
    print("TEST 3: SQLRefiner - fix violations")
    print("=" * 60)

    from synkro import SQLRefiner

    # The trace with wrong amount
    wrong_trace = [
        {"role": "user", "content": "What about Bolt?"},
        {"role": "assistant", "content": "Bolt raised $100 million from Andreessen."},
    ]

    # Verify it (should be violation)
    report = await verify_sql(str(CONFIG_FILE), messages=wrong_trace, llm=llm)
    print(f"Before: {report.response}")
    print(f"Verdict: {report.results[0].value.value}")
    assert report.results[0].value == Verdict.VIOLATION

    # Refine it
    refiner = SQLRefiner()
    fixed_trace = await refiner.refine(wrong_trace, report)
    print(f"After:  {fixed_trace[-1]['content']}")

    # Verify the fixed trace
    fixed_report = await verify_sql(str(CONFIG_FILE), messages=fixed_trace, llm=llm)
    print(f"Fixed verdict: {fixed_report.results[0].value.value}")

    assert fixed_report.results[0].value == Verdict.PASS
    print("✅ PASS\n")

    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)


async def main():
    await setup_database()

    try:
        await run_tests()
    finally:
        if DB_PATH.exists():
            DB_PATH.unlink()
            print(f"\nCleaned up: {DB_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
