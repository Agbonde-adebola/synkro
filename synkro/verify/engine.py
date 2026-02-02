"""Main verification engine."""

from pathlib import Path
from typing import Any

from synkro.verify.config import VerifierConfig, parse_configs
from synkro.verify.llm import VerifierLLM
from synkro.verify.sql import SQL
from synkro.verify.types import Report, Result, Verdict


def _extract_response(messages: list[dict]) -> str:
    """Extract the last assistant response from messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "") for block in content if block.get("type") == "text"
                )
            return content
    return ""


async def _verify_single(
    config: VerifierConfig,
    response: str,
    llm: VerifierLLM,
    db: SQL | None,
) -> Result:
    """Run a single verifier against the response."""
    name = config.name or "unnamed"
    on_no_results = config.on_no_results
    sql_template = config.sql

    # Step 1: Route - decide if verifier is relevant
    verifier_desc = {
        "intent": config.intent,
        "params": config.params,
    }

    try:
        route_result = await llm.route(response, verifier_desc)
    except Exception as e:
        return Result(
            name=name,
            score=0.0,
            value=Verdict.ERROR,
            comment=f"Routing error: {e}",
        )

    if not route_result.get("trigger", False):
        return Result(
            name=name,
            score=0.0,
            value=Verdict.SKIPPED,
            comment="Not triggered - no relevant claims found",
        )

    params = route_result.get("params", {})

    # Step 2: Query database
    rows: list[dict] = []
    if db and sql_template:
        try:
            rows = await db.execute(sql_template, params)
        except Exception as e:
            return Result(
                name=name,
                score=0.0,
                value=Verdict.ERROR,
                comment=f"SQL error: {e}",
                metadata={"params": params},
            )

    if not rows:
        if on_no_results == "skip":
            return Result(
                name=name,
                score=0.0,
                value=Verdict.SKIPPED,
                comment="No matching rows in database",
                metadata={"params": params},
            )
        else:
            return Result(
                name=name,
                score=1.0,
                value=Verdict.VIOLATION,
                comment="No matching rows in database (on_no_results=violation)",
                metadata={"params": params},
            )

    # Step 3: Judge - compare response against ground truth
    evidence = {
        "verifier": name,
        "params": params,
        "sql_results": rows,
    }

    try:
        grade = await llm.judge(response, evidence)
    except Exception as e:
        return Result(
            name=name,
            score=0.0,
            value=Verdict.ERROR,
            comment=f"Judging error: {e}",
            metadata={"params": params, "ground_truth": rows},
        )

    return Result(
        name=name,
        score=grade.get("confidence", 0.0),
        value=Verdict(grade.get("verdict", "error")),
        comment=grade.get("reasoning", ""),
        metadata={"params": params, "ground_truth": rows},
    )


async def verify(
    *configs: str | Path | dict | list[Any],
    messages: list[dict],
    trace_id: str = "",
    dsn: str | None = None,
    llm: VerifierLLM | None = None,
    db: SQL | None = None,
) -> Report:
    """Verify an LLM response against SQL ground truth.

    Args:
        *configs: Any number of configs - YAML files, directories, dicts, or lists
        messages: OpenAI-format chat messages
        trace_id: Optional trace identifier
        dsn: Database connection string (overrides config DSN if provided)
        llm: Optional VerifierLLM instance (creates default if not provided)
        db: Optional SQL instance (creates from dsn if not provided)

    Returns:
        Report with results from all verifiers
    """
    config_list = parse_configs(*configs)

    if not config_list:
        return Report(trace_id=trace_id, response="", results=[])

    response = _extract_response(messages)
    if not response:
        return Report(
            trace_id=trace_id,
            response="",
            results=[
                Result(
                    name="__all__",
                    score=0.0,
                    value=Verdict.SKIPPED,
                    comment="No assistant response found in messages",
                )
            ],
        )

    if llm is None:
        llm = VerifierLLM()

    effective_dsn = dsn
    if effective_dsn is None:
        for cfg in config_list:
            if cfg.dsn:
                effective_dsn = cfg.dsn
                break

    own_db = False
    if db is None and effective_dsn:
        db = SQL(effective_dsn)
        own_db = True

    try:
        results = []
        for config in config_list:
            result = await _verify_single(config, response, llm, db)
            results.append(result)

        return Report(trace_id=trace_id, response=response, results=results)

    finally:
        if own_db and db:
            await db.close()


async def verify_batch(
    *configs: str | Path | dict | list[Any],
    traces: list[list[dict]],
    dsn: str | None = None,
    llm: VerifierLLM | None = None,
) -> list[Report]:
    """Verify multiple traces efficiently.

    Reuses database connection and LLM instance across all traces.

    Args:
        *configs: Any number of configs - YAML files, directories, dicts, or lists
        traces: List of message lists (each is a conversation)
        dsn: Database connection string (overrides config DSN if provided)
        llm: Optional VerifierLLM instance (creates default if not provided)

    Returns:
        List of Reports, one per trace
    """
    config_list = parse_configs(*configs)

    if not config_list:
        return [Report(trace_id=str(i), response="", results=[]) for i in range(len(traces))]

    effective_dsn = dsn
    if effective_dsn is None:
        for cfg in config_list:
            if cfg.dsn:
                effective_dsn = cfg.dsn
                break

    if llm is None:
        llm = VerifierLLM()

    db: SQL | None = None
    if effective_dsn:
        db = SQL(effective_dsn)

    try:
        reports = []
        for i, messages in enumerate(traces):
            response = _extract_response(messages)

            if not response:
                reports.append(
                    Report(
                        trace_id=str(i),
                        response="",
                        results=[
                            Result(
                                name="__all__",
                                score=0.0,
                                value=Verdict.SKIPPED,
                                comment="No assistant response found in messages",
                            )
                        ],
                    )
                )
                continue

            results = []
            for config in config_list:
                result = await _verify_single(config, response, llm, db)
                results.append(result)

            reports.append(Report(trace_id=str(i), response=response, results=results))

        return reports

    finally:
        if db:
            await db.close()
