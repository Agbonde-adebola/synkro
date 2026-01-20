"""Result types for synkro pipeline stages.

These types provide structured access to results from each pipeline stage,
with metrics, display formatters, and serialization support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from synkro.types.metrics import Metrics, PhaseMetrics

if TYPE_CHECKING:
    from synkro.core.dataset import Dataset
    from synkro.types.core import EvalScenario, Trace
    from synkro.types.coverage import CoverageReport
    from synkro.types.logic_map import GoldenScenario, LogicMap


@dataclass
class ExtractionResult:
    """Result of rule extraction from a policy document.

    Contains the extracted Logic Map (DAG of rules) along with
    metrics about the extraction process.

    Examples:
        >>> result = await synkro.extract_rules(policy)
        >>> print(f"Extracted {len(result.logic_map.rules)} rules")
        >>> print(result.format_summary())
        >>> print(result.metrics.format_summary())
    """

    logic_map: "LogicMap"
    metrics: PhaseMetrics = field(default_factory=lambda: PhaseMetrics(phase="extraction"))

    def format_summary(self) -> str:
        """One-line summary of extraction result."""
        rule_count = len(self.logic_map.rules)
        root_count = len(self.logic_map.root_rules)
        return (
            f"Extracted {rule_count} rules ({root_count} root, {rule_count - root_count} dependent)"
        )

    def format_table(self) -> str:
        """Formatted table of extracted rules."""
        return self.logic_map.to_display_string()

    def format_diff(self, old: "ExtractionResult") -> str:
        """Show what changed compared to another extraction."""
        old_ids = {r.rule_id for r in old.logic_map.rules}
        new_ids = {r.rule_id for r in self.logic_map.rules}

        added = new_ids - old_ids
        removed = old_ids - new_ids

        lines = []
        if added:
            lines.append(f"+ Added: {', '.join(sorted(added))}")
        if removed:
            lines.append(f"- Removed: {', '.join(sorted(removed))}")
        if not added and not removed:
            lines.append("No changes")

        return "\n".join(lines)

    def to_agent_context(self) -> str:
        """Compact representation for LLM reasoning context."""
        rules_summary = []
        for rule in self.logic_map.rules:
            deps = f" (depends on: {', '.join(rule.dependencies)})" if rule.dependencies else ""
            rules_summary.append(f"- {rule.rule_id}: {rule.text[:80]}...{deps}")
        return f"Logic Map ({len(self.logic_map.rules)} rules):\n" + "\n".join(rules_summary)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "logic_map": self.logic_map.model_dump(),
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class ScenariosResult:
    """Result of scenario generation.

    Contains generated scenarios with their distribution across types
    (positive, negative, edge_case, irrelevant), optional coverage report,
    and generation metrics.

    Examples:
        >>> result = await synkro.generate_scenarios(policy, logic_map, count=50)
        >>> print(f"Generated {len(result.scenarios)} scenarios")
        >>> print(f"Distribution: {result.distribution}")
        >>> print(result.coverage_report.overall_coverage_percent if result.coverage_report else "N/A")
    """

    scenarios: list["GoldenScenario"]
    logic_map: "LogicMap"
    distribution: dict[str, int] = field(default_factory=dict)
    coverage_report: "CoverageReport | None" = None
    metrics: PhaseMetrics = field(default_factory=lambda: PhaseMetrics(phase="scenarios"))

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    def format_summary(self) -> str:
        """One-line summary of scenarios result."""
        dist_str = ", ".join(f"{k}: {v}" for k, v in self.distribution.items())
        return f"Generated {len(self.scenarios)} scenarios ({dist_str})"

    def format_table(self) -> str:
        """Formatted table of scenarios by type."""
        lines = ["Scenarios:"]
        lines.append("-" * 60)

        for stype in ["positive", "negative", "edge_case", "irrelevant"]:
            count = self.distribution.get(stype, 0)
            lines.append(f"\n{stype.upper()} ({count}):")
            for i, s in enumerate(self.scenarios):
                if s.scenario_type.value == stype:
                    rules = ", ".join(s.target_rule_ids) if s.target_rule_ids else "none"
                    lines.append(f"  S{i+1}: {s.description[:50]}... (rules: {rules})")

        return "\n".join(lines)

    def format_coverage(self) -> str:
        """Format coverage report if available."""
        if not self.coverage_report:
            return "Coverage report not available"

        report = self.coverage_report
        return (
            f"Coverage: {report.overall_coverage_percent:.0f}% "
            f"({report.covered_count} covered, {report.partial_count} partial, "
            f"{report.uncovered_count} uncovered)"
        )

    def to_agent_context(self) -> str:
        """Compact representation for LLM reasoning context."""
        lines = [f"Scenarios ({len(self.scenarios)} total):"]
        lines.append(f"Distribution: {self.distribution}")

        for i, s in enumerate(self.scenarios[:10]):  # Limit to 10 for context
            rules = ", ".join(s.target_rule_ids) if s.target_rule_ids else "none"
            lines.append(
                f"- S{i+1} [{s.scenario_type.value}]: {s.description[:60]}... (rules: {rules})"
            )

        if len(self.scenarios) > 10:
            lines.append(f"... and {len(self.scenarios) - 10} more")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "scenarios": [s.model_dump() for s in self.scenarios],
            "logic_map": self.logic_map.model_dump(),
            "distribution": self.distribution,
            "coverage_report": self.coverage_report.model_dump() if self.coverage_report else None,
            "metrics": self.metrics.to_dict(),
        }

    def to_eval_scenarios(self) -> list["EvalScenario"]:
        """Convert to EvalScenario objects for evaluation API."""
        from synkro.types.core import EvalScenario

        return [
            EvalScenario(
                user_message=s.description,
                expected_outcome=s.expected_outcome,
                target_rule_ids=s.target_rule_ids,
                scenario_type=s.scenario_type.value
                if hasattr(s.scenario_type, "value")
                else s.scenario_type,
                category=s.category,
                context=s.context,
            )
            for s in self.scenarios
        ]


@dataclass
class TracesResult:
    """Result of trace synthesis.

    Contains generated conversation traces with their associated
    scenarios and Logic Map context.

    Examples:
        >>> result = await synkro.synthesize_traces(policy, scenarios)
        >>> print(f"Generated {len(result.traces)} traces")
        >>> for trace in result.traces:
        ...     print(trace.assistant_message[:50])
    """

    traces: list["Trace"]
    logic_map: "LogicMap"
    scenarios: list["GoldenScenario"]
    metrics: PhaseMetrics = field(default_factory=lambda: PhaseMetrics(phase="traces"))

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self):
        return iter(self.traces)

    def format_summary(self) -> str:
        """One-line summary of traces result."""
        tool_call_count = sum(1 for t in self.traces if t.has_tool_calls)
        if tool_call_count > 0:
            return f"Generated {len(self.traces)} traces ({tool_call_count} with tool calls)"
        return f"Generated {len(self.traces)} traces"

    def format_table(self) -> str:
        """Formatted table of traces."""
        lines = ["Traces:"]
        lines.append("-" * 60)

        for i, trace in enumerate(self.traces):
            user_msg = trace.user_message[:40] if trace.user_message else "N/A"
            asst_msg = trace.assistant_message[:40] if trace.assistant_message else "N/A"
            rules = ", ".join(trace.rules_applied or []) if trace.rules_applied else "none"
            lines.append(f"T{i+1}: {user_msg}...")
            lines.append(f"     Response: {asst_msg}...")
            lines.append(f"     Rules: {rules}")

        return "\n".join(lines)

    def to_agent_context(self) -> str:
        """Compact representation for LLM reasoning context."""
        lines = [f"Traces ({len(self.traces)} total):"]

        for i, trace in enumerate(self.traces[:5]):  # Limit to 5 for context
            user_msg = trace.user_message[:50] if trace.user_message else "N/A"
            rules = ", ".join(trace.rules_applied or []) if trace.rules_applied else "none"
            lines.append(f"- T{i+1}: {user_msg}... (rules: {rules})")

        if len(self.traces) > 5:
            lines.append(f"... and {len(self.traces) - 5} more")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "traces": [t.model_dump() for t in self.traces],
            "logic_map": self.logic_map.model_dump(),
            "scenarios": [s.model_dump() for s in self.scenarios],
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class VerificationResult:
    """Result of trace verification and refinement.

    Contains verified traces along with pass rate, refinement history,
    and detailed metrics about the verification process.

    Examples:
        >>> result = await synkro.verify_traces(policy, traces)
        >>> print(f"Pass rate: {result.pass_rate:.1%}")
        >>> print(f"Refined {result.refinement_count} traces")
    """

    verified_traces: list["Trace"]
    pass_rate: float
    refinement_count: int = 0
    refinement_history: list[dict] = field(default_factory=list)
    metrics: PhaseMetrics = field(default_factory=lambda: PhaseMetrics(phase="verification"))

    def __len__(self) -> int:
        return len(self.verified_traces)

    def __iter__(self):
        return iter(self.verified_traces)

    def format_summary(self) -> str:
        """One-line summary of verification result."""
        passed = sum(1 for t in self.verified_traces if t.grade and t.grade.passed)
        total = len(self.verified_traces)
        return f"Verified: {passed}/{total} passed ({self.pass_rate:.1%}), {self.refinement_count} refined"

    def format_table(self) -> str:
        """Formatted table of verification results."""
        lines = ["Verification Results:"]
        lines.append("-" * 60)

        for i, trace in enumerate(self.verified_traces):
            status = "PASS" if (trace.grade and trace.grade.passed) else "FAIL"
            user_msg = trace.user_message[:40] if trace.user_message else "N/A"
            issues = trace.grade.issues if trace.grade else []
            lines.append(f"T{i+1} [{status}]: {user_msg}...")
            if issues:
                for issue in issues[:2]:
                    lines.append(f"     Issue: {issue[:50]}...")

        lines.append("-" * 60)
        lines.append(f"Pass Rate: {self.pass_rate:.1%} | Refined: {self.refinement_count}")

        return "\n".join(lines)

    def to_agent_context(self) -> str:
        """Compact representation for LLM reasoning context."""
        passed = sum(1 for t in self.verified_traces if t.grade and t.grade.passed)
        failed = len(self.verified_traces) - passed

        lines = [
            f"Verification: {passed} passed, {failed} failed, {self.refinement_count} refined",
            f"Pass rate: {self.pass_rate:.1%}",
        ]

        # Show failed traces with issues
        failed_traces = [t for t in self.verified_traces if t.grade and not t.grade.passed]
        if failed_traces:
            lines.append("Failed traces:")
            for t in failed_traces[:3]:
                issues = ", ".join(t.grade.issues[:2]) if t.grade and t.grade.issues else "unknown"
                lines.append(f"  - {t.user_message[:30]}... Issues: {issues}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "verified_traces": [t.model_dump() for t in self.verified_traces],
            "pass_rate": self.pass_rate,
            "refinement_count": self.refinement_count,
            "refinement_history": self.refinement_history,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class PipelineResult:
    """Complete result from the full generation pipeline.

    Aggregates results from all stages (extraction, scenarios, traces,
    verification) with unified metrics and convenient accessors.

    Examples:
        >>> result = await synkro.generate(policy, traces=50, return_result=True)
        >>> print(f"Cost: ${result.metrics.total_cost:.2f}")
        >>> print(f"Pass rate: {result.pass_rate:.1%}")
        >>> dataset = result.dataset
        >>> logic_map = result.logic_map
    """

    dataset: "Dataset"
    metrics: Metrics = field(default_factory=Metrics)

    # Stage results (optional for backward compatibility)
    extraction: ExtractionResult | None = None
    scenarios: ScenariosResult | None = None
    traces: TracesResult | None = None
    verification: VerificationResult | None = None

    @property
    def logic_map(self) -> "LogicMap | None":
        """Get the Logic Map from extraction result."""
        return self.extraction.logic_map if self.extraction else None

    @property
    def pass_rate(self) -> float | None:
        """Get pass rate from verification result."""
        return self.verification.pass_rate if self.verification else None

    @property
    def refinement_history(self) -> list[dict]:
        """Get refinement history from verification result."""
        return self.verification.refinement_history if self.verification else []

    @property
    def coverage_report(self) -> "CoverageReport | None":
        """Get coverage report from scenarios result."""
        return self.scenarios.coverage_report if self.scenarios else None

    def __len__(self) -> int:
        """Number of traces in the dataset."""
        return len(self.dataset)

    def __iter__(self):
        """Iterate over traces in the dataset."""
        return iter(self.dataset)

    # Delegate to dataset for backwards compatibility
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to dataset."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def format_summary(self) -> str:
        """One-line summary of pipeline result."""
        parts = [f"{len(self.dataset)} traces"]

        if self.pass_rate is not None:
            parts.append(f"{self.pass_rate:.1%} pass rate")

        parts.append(f"${self.metrics.total_cost:.4f}")

        return " | ".join(parts)

    def format_table(self) -> str:
        """Formatted breakdown of all stages."""
        lines = ["Pipeline Result:"]
        lines.append("=" * 60)

        if self.extraction:
            lines.append(f"\nExtraction: {self.extraction.format_summary()}")

        if self.scenarios:
            lines.append(f"\nScenarios: {self.scenarios.format_summary()}")
            if self.scenarios.coverage_report:
                lines.append(f"Coverage: {self.scenarios.format_coverage()}")

        if self.traces:
            lines.append(f"\nTraces: {self.traces.format_summary()}")

        if self.verification:
            lines.append(f"\nVerification: {self.verification.format_summary()}")

        lines.append("\n" + self.metrics.format_table())

        return "\n".join(lines)

    def to_agent_context(self) -> str:
        """Compact representation for LLM reasoning context."""
        lines = [f"Pipeline Result ({len(self.dataset)} traces):"]
        lines.append(f"- Cost: ${self.metrics.total_cost:.4f}")
        lines.append(f"- Pass rate: {self.pass_rate:.1%}" if self.pass_rate else "- Pass rate: N/A")

        if self.extraction:
            lines.append(f"- Rules: {len(self.extraction.logic_map.rules)}")

        if self.scenarios:
            lines.append(f"- Scenarios: {len(self.scenarios.scenarios)}")

        if self.verification:
            lines.append(f"- Refined: {self.verification.refinement_count}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dataset": [t.model_dump() for t in self.dataset.traces],
            "metrics": self.metrics.to_dict(),
            "extraction": self.extraction.to_dict() if self.extraction else None,
            "scenarios": self.scenarios.to_dict() if self.scenarios else None,
            "traces": self.traces.to_dict() if self.traces else None,
            "verification": self.verification.to_dict() if self.verification else None,
        }


__all__ = [
    "ExtractionResult",
    "ScenariosResult",
    "TracesResult",
    "VerificationResult",
    "PipelineResult",
]
