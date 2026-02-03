"""Policy violation detection for traces.

Detects violations by grading traces against a LogicMap (structured policy).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from synkro.llm.client import LLM
from synkro.remediation.types import Violation
from synkro.schemas import BaseModel, Field
from synkro.types.logic_map import LogicMap


class DetectionResult(BaseModel):
    """Structured output for violation detection."""

    is_violation: bool = Field(description="True if the response violates policy")
    rules_violated: list[str] = Field(
        default_factory=list, description="Rule IDs that were violated"
    )
    issues: list[str] = Field(default_factory=list, description="Description of each issue")
    severity: str = Field(default="high", description="Severity: critical, high, medium, low")
    user_intent: str = Field(default="", description="What the user was trying to accomplish")
    context: str = Field(default="", description="Relevant context from the trace")
    expected_outcome: str = Field(default="", description="What a correct response should do")
    explanation: str = Field(default="", description="Summary explanation of the violation")


DETECTION_PROMPT = """You are a policy compliance detector. Analyze this conversation trace against the policy rules.

POLICY RULES:
{rules}

CONVERSATION TRACE:
{trace}

Analyze whether the assistant's response violates any policy rules.

For each violation found:
1. Identify which specific rule(s) were violated (use rule IDs like R001, R002)
2. Explain the issue clearly
3. Determine severity (critical/high/medium/low)

Also extract:
- user_intent: What the user was trying to accomplish
- context: Relevant facts from the conversation (amounts, dates, conditions)
- expected_outcome: What a compliant response should have done

Return your analysis as structured JSON."""


class PolicyDetector:
    """Detect policy violations in traces.

    Uses a LogicMap (structured policy) to identify violations and extract
    context for remediation.

    Example:
        >>> from synkro import ingest, PolicyDetector
        >>>
        >>> config = ingest("./policy.pdf")
        >>> detector = PolicyDetector(config.logic_map)
        >>>
        >>> violations = await detector.detect(bad_traces)
        >>> print(f"Found {len(violations)} violations")
    """

    def __init__(
        self,
        logic_map: LogicMap,
        llm: LLM | None = None,
        model: str | None = None,
    ):
        """Initialize the detector.

        Args:
            logic_map: Structured policy (LogicMap from ingest())
            llm: LLM client (creates one if not provided)
            model: Model to use if creating LLM (auto-detected if not specified)
        """
        self.logic_map = logic_map
        if llm is not None:
            self.llm = llm
        elif model is not None:
            self.llm = LLM(model=model, temperature=0.1)
        else:
            self.llm = LLM(temperature=0.1)  # Use default model

    def _format_rules(self) -> str:
        """Format LogicMap rules for prompt."""
        lines = []
        for rule in self.logic_map.rules:
            deps = f" [depends on: {', '.join(rule.dependencies)}]" if rule.dependencies else ""
            lines.append(f"{rule.rule_id} ({rule.category.value}): {rule.text}")
            if rule.condition:
                lines.append(f"  IF: {rule.condition}")
            if rule.action:
                lines.append(f"  THEN: {rule.action}")
            if deps:
                lines.append(f"  {deps}")
            lines.append("")
        return "\n".join(lines)

    def _format_trace(self, trace: list[dict[str, Any]]) -> str:
        """Format trace for prompt."""
        lines = []
        for msg in trace:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _load_traces(self, traces: str | Path | list) -> list[list[dict[str, Any]]]:
        """Load traces from various sources.

        Args:
            traces: JSONL file path, or list of trace dicts

        Returns:
            List of traces (each trace is a list of message dicts)
        """
        if isinstance(traces, (str, Path)):
            path = Path(traces)
            loaded = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Handle both formats: list of messages or {"messages": [...]}
                        if isinstance(data, list):
                            loaded.append(data)
                        elif isinstance(data, dict) and "messages" in data:
                            loaded.append(data["messages"])
                        else:
                            loaded.append([data])
            return loaded
        elif isinstance(traces, list):
            # Check if it's a single trace (list of dicts with "role")
            if traces and isinstance(traces[0], dict) and "role" in traces[0]:
                return [traces]
            return traces
        else:
            raise ValueError(f"Unsupported traces type: {type(traces)}")

    async def detect_single(
        self,
        trace: list[dict[str, Any]],
        trace_id: str | None = None,
    ) -> Violation | None:
        """Detect violations in a single trace.

        Args:
            trace: Conversation trace (list of message dicts)
            trace_id: Optional trace ID (generated if not provided)

        Returns:
            Violation if detected, None otherwise
        """
        # Generate trace ID if not provided
        if trace_id is None:
            trace_id = Violation.generate_id(trace)

        # Build prompt
        prompt = DETECTION_PROMPT.format(
            rules=self._format_rules(),
            trace=self._format_trace(trace),
        )

        # Detect violations
        try:
            result = await self.llm.generate_structured(prompt, DetectionResult)
        except Exception as e:
            # If structured output fails, return a generic violation
            return Violation(
                id=Violation.generate_id(trace),
                trace_id=trace_id,
                comment=f"Detection failed: {e}",
                issues=["Unable to analyze trace"],
                trace=trace,
            )

        if not result.is_violation:
            return None

        return Violation(
            id=Violation.generate_id(trace),
            trace_id=trace_id,
            name="policy_compliance",
            score=0.0,
            value="violation",
            data_type="categorical",
            comment=result.explanation,
            rules_violated=result.rules_violated,
            issues=result.issues,
            severity=result.severity,
            trace=trace,
            user_intent=result.user_intent,
            context=result.context,
            expected_outcome=result.expected_outcome,
        )

    async def detect(
        self,
        traces: str | Path | list[dict[str, Any]] | list[list[dict[str, Any]]],
        concurrency: int = 20,
    ) -> list[Violation]:
        """Detect policy violations in traces.

        Args:
            traces: Traces to analyze (JSONL path or list)
            concurrency: Max parallel detection calls

        Returns:
            List of Violations (only traces with violations)
        """
        loaded = self._load_traces(traces)

        if not loaded:
            return []

        semaphore = asyncio.Semaphore(concurrency)

        async def detect_with_limit(trace: list[dict], idx: int) -> Violation | None:
            async with semaphore:
                return await self.detect_single(trace, trace_id=f"trace_{idx}")

        tasks = [detect_with_limit(trace, i) for i, trace in enumerate(loaded)]
        results = await asyncio.gather(*tasks)

        # Filter out None (traces without violations)
        return [v for v in results if v is not None]

    async def detect_all(
        self,
        traces: str | Path | list[dict[str, Any]] | list[list[dict[str, Any]]],
        concurrency: int = 20,
    ) -> tuple[list[Violation], list[list[dict[str, Any]]]]:
        """Detect violations and return both violations and passing traces.

        Args:
            traces: Traces to analyze (JSONL path or list)
            concurrency: Max parallel detection calls

        Returns:
            Tuple of (violations, passing_traces)
        """
        loaded = self._load_traces(traces)

        if not loaded:
            return [], []

        semaphore = asyncio.Semaphore(concurrency)

        async def detect_with_limit(
            trace: list[dict], idx: int
        ) -> tuple[Violation | None, list[dict]]:
            async with semaphore:
                violation = await self.detect_single(trace, trace_id=f"trace_{idx}")
                return violation, trace

        tasks = [detect_with_limit(trace, i) for i, trace in enumerate(loaded)]
        results = await asyncio.gather(*tasks)

        violations = []
        passing = []
        for violation, trace in results:
            if violation is not None:
                violations.append(violation)
            else:
                passing.append(trace)

        return violations, passing
