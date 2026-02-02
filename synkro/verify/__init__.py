"""SQL Verification Module.

Verify LLM responses against database ground truth using YAML-configured verifiers.

Example:
    from synkro import verify_sql, verify_batch

    # Single trace with DSN in config
    report = await verify_sql("./verifiers/funding.yaml", messages=messages)

    # funding.yaml:
    # name: funding_verification
    # sql_file: ./funding.sql
    # params: [company_name]
    # dsn: ${DATABASE_URL}

    print(report.passed)  # True if no violations
    for v in report.violations:
        print(f"{v.verifier}: {v.reasoning}")

    # Batch verification (reuses connection)
    reports = await verify_batch("./verifiers/", traces=[msgs1, msgs2, msgs3])

    # ORM with user-defined query
    from synkro.verify import Verifier

    def my_query(session, params):
        return session.query(MyModel).filter(...).all()

    verifier = Verifier(query=my_query, params=["company_name"], dsn="...")
    report = await verifier.check(messages)
"""

from synkro.verify.config import VerifierConfig, parse_config, parse_configs
from synkro.verify.engine import verify, verify_batch
from synkro.verify.llm import VerifierLLM
from synkro.verify.orm import Verifier
from synkro.verify.refiner import SQLRefiner
from synkro.verify.sql import SQL
from synkro.verify.types import Report, Result, Verdict

__all__ = [
    # Main API
    "verify",
    "verify_batch",
    "Verifier",
    "SQLRefiner",
    # Types
    "Report",
    "Result",
    "Verdict",
    # Config
    "VerifierConfig",
    "parse_config",
    "parse_configs",
    # Internals
    "VerifierLLM",
    "SQL",
]
