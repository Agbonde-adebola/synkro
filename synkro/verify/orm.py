"""SQLAlchemy ORM support for verification.

Create verifiers from user-defined query functions.
Developer writes their own query - no SQL auto-generation.

Example:
    from sqlalchemy.orm import Session
    from myapp.models import FundingRound
    from synkro.verify import Verifier

    def funding_query(session: Session, params: dict) -> list:
        return session.query(FundingRound).filter(
            FundingRound.company_name.ilike(f"%{params['company_name']}%")
        ).all()

    verifier = Verifier(
        query=funding_query,
        params=["company_name"],
        dsn="postgresql://...",
        intent="Verify company funding claims",
    )
    report = await verifier.check(messages)
"""

from typing import Any, Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from synkro.verify.config import expand_env_vars
from synkro.verify.llm import VerifierLLM
from synkro.verify.types import Report, Result, Verdict


class Verifier:
    """SQL Verifier with user-defined query function.

    Developer writes their own ORM query - no magic SQL generation.
    """

    def __init__(
        self,
        query: Callable[[Session, dict], list[Any]],
        params: list[str],
        dsn: str | None = None,
        name: str = "",
        intent: str = "",
        on_no_results: str = "skip",
    ):
        """Create verifier with user-defined query.

        Args:
            query: Function that takes (session, params_dict) and returns rows.
                   User writes this - full control over query logic.
            params: Parameter names to extract from LLM response
            dsn: Database connection string (supports ${ENV_VAR})
            name: Verifier name for reporting
            intent: Description for LLM routing (what claims to look for)
            on_no_results: "skip" or "violation" when no DB results
        """
        self.query_fn = query
        self.params = params
        self.name = name or "orm_verifier"
        self.intent = intent
        self.on_no_results = on_no_results

        self._dsn = expand_env_vars(dsn) if dsn and "${" in dsn else dsn
        self._engine = None
        self._session_maker = None

    def _get_session(self) -> Session:
        """Get a database session."""
        if self._engine is None:
            if not self._dsn:
                raise ValueError("No DSN provided")
            self._engine = create_engine(self._dsn)
            self._session_maker = sessionmaker(bind=self._engine)
        return self._session_maker()

    def _rows_to_dicts(self, rows: list[Any]) -> list[dict]:
        """Convert ORM objects to dicts for judging."""
        result = []
        for row in rows:
            if hasattr(row, "__dict__"):
                d = {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
                result.append(d)
            elif hasattr(row, "_asdict"):
                result.append(row._asdict())
            elif isinstance(row, dict):
                result.append(row)
            else:
                result.append({"value": row})
        return result

    async def check(
        self,
        messages: list[dict],
        trace_id: str = "",
        llm: VerifierLLM | None = None,
    ) -> Report:
        """Verify messages against this verifier."""
        response = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    response = " ".join(
                        block.get("text", "") for block in content if block.get("type") == "text"
                    )
                else:
                    response = content
                break

        if not response:
            return Report(
                trace_id=trace_id,
                response="",
                results=[
                    Result(
                        name=self.name,
                        score=0.0,
                        value=Verdict.SKIPPED,
                        comment="No assistant response found",
                    )
                ],
            )

        if llm is None:
            llm = VerifierLLM()

        result = await self._verify_response(response, llm)
        return Report(trace_id=trace_id, response=response, results=[result])

    async def _verify_response(self, response: str, llm: VerifierLLM) -> Result:
        """Run verification on a single response."""
        verifier_desc = {
            "intent": self.intent,
            "params": self.params,
        }

        try:
            route_result = await llm.route(response, verifier_desc)
        except Exception as e:
            return Result(
                name=self.name,
                score=0.0,
                value=Verdict.ERROR,
                comment=f"Routing error: {e}",
            )

        if not route_result.get("trigger", False):
            return Result(
                name=self.name,
                score=0.0,
                value=Verdict.SKIPPED,
                comment="Not triggered - no relevant claims found",
            )

        params = route_result.get("params", {})

        rows: list[dict] = []
        try:
            session = self._get_session()
            try:
                raw_rows = self.query_fn(session, params)
                rows = self._rows_to_dicts(raw_rows)
            finally:
                session.close()
        except Exception as e:
            return Result(
                name=self.name,
                score=0.0,
                value=Verdict.ERROR,
                comment=f"Query error: {e}",
                metadata={"params": params},
            )

        if not rows:
            if self.on_no_results == "skip":
                return Result(
                    name=self.name,
                    score=0.0,
                    value=Verdict.SKIPPED,
                    comment="No matching rows in database",
                    metadata={"params": params},
                )
            else:
                return Result(
                    name=self.name,
                    score=1.0,
                    value=Verdict.VIOLATION,
                    comment="No matching rows in database (on_no_results=violation)",
                    metadata={"params": params},
                )

        evidence = {
            "verifier": self.name,
            "params": params,
            "sql_results": rows,
        }

        try:
            grade = await llm.judge(response, evidence)
        except Exception as e:
            return Result(
                name=self.name,
                score=0.0,
                value=Verdict.ERROR,
                comment=f"Judging error: {e}",
                metadata={"params": params, "ground_truth": rows},
            )

        return Result(
            name=self.name,
            score=grade.get("confidence", 0.0),
            value=Verdict(grade.get("verdict", "error")),
            comment=grade.get("reasoning", ""),
            metadata={"params": params, "ground_truth": rows},
        )

    async def check_batch(
        self,
        traces: list[list[dict]],
        llm: VerifierLLM | None = None,
    ) -> list[Report]:
        """Verify multiple traces efficiently."""
        if llm is None:
            llm = VerifierLLM()

        reports = []
        for i, messages in enumerate(traces):
            report = await self.check(messages, trace_id=str(i), llm=llm)
            reports.append(report)

        return reports

    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_maker = None
