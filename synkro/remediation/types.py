"""Violation type for policy compliance detection.

Universal format compatible with LangSmith, Langfuse, and Datadog evaluation APIs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Violation:
    """A detected policy violation in universal observability format.

    Designed to map cleanly to LangSmith, Langfuse, and Datadog evaluation APIs:

    | Field     | LangSmith | Langfuse  | Datadog      |
    |-----------|-----------|-----------|--------------|
    | name      | key       | name      | label        |
    | score     | score     | value     | value        |
    | value     | -         | value     | value        |
    | comment   | comment   | comment   | reasoning    |
    | trace_id  | run_id    | traceId   | span_context |
    | data_type | inferred  | dataType  | metric_type  |
    """

    # === Core identifiers (required by all platforms) ===
    id: str
    trace_id: str

    # === Evaluation result (LangSmith/Langfuse/Datadog compatible) ===
    name: str = "policy_compliance"
    score: float = 0.0
    value: str = "violation"
    data_type: str = "categorical"
    comment: str = ""

    # === Violation details ===
    rules_violated: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    severity: str = "high"

    # === Context for remediation ===
    trace: list[dict[str, Any]] = field(default_factory=list)
    user_intent: str = ""
    context: str = ""
    expected_outcome: str = ""

    # === Metadata (extensible for platform-specific fields) ===
    metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "source": "synkro",
            "evaluation_type": "policy_compliance",
        }
    )

    # === Timestamps ===
    timestamp: datetime = field(default_factory=datetime.now)

    def to_langsmith(self) -> dict[str, Any]:
        """Convert to LangSmith feedback format.

        Returns:
            Dict compatible with langsmith_client.create_feedback()
        """
        return {
            "run_id": self.trace_id,
            "key": self.name,
            "score": self.score,
            "comment": self.comment,
            "metadata": {
                **self.metadata,
                "rules_violated": self.rules_violated,
                "severity": self.severity,
                "issues": self.issues,
            },
        }

    def to_langfuse(self) -> dict[str, Any]:
        """Convert to Langfuse score format.

        Returns:
            Dict compatible with langfuse.score()
        """
        return {
            "traceId": self.trace_id,
            "name": self.name,
            "value": self.score if self.data_type == "numeric" else self.value,
            "dataType": self.data_type.upper(),
            "comment": self.comment,
            "metadata": {
                **self.metadata,
                "rules_violated": self.rules_violated,
                "severity": self.severity,
                "issues": self.issues,
            },
        }

    def to_datadog(self) -> dict[str, Any]:
        """Convert to Datadog evaluation format.

        Returns:
            Dict compatible with LLMObs.submit_evaluation()
        """
        return {
            "span_context": self.trace_id,
            "label": self.name,
            "metric_type": "categorical" if self.data_type == "categorical" else "score",
            "value": self.value if self.data_type == "categorical" else self.score,
            "reasoning": self.comment,
            "tags": [f"rule:{r}" for r in self.rules_violated] + [f"severity:{self.severity}"],
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "name": self.name,
            "score": self.score,
            "value": self.value,
            "data_type": self.data_type,
            "comment": self.comment,
            "rules_violated": self.rules_violated,
            "issues": self.issues,
            "severity": self.severity,
            "trace": self.trace,
            "user_intent": self.user_intent,
            "context": self.context,
            "expected_outcome": self.expected_outcome,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Violation:
        """Create Violation from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            id=data["id"],
            trace_id=data["trace_id"],
            name=data.get("name", "policy_compliance"),
            score=data.get("score", 0.0),
            value=data.get("value", "violation"),
            data_type=data.get("data_type", "categorical"),
            comment=data.get("comment", ""),
            rules_violated=data.get("rules_violated", []),
            issues=data.get("issues", []),
            severity=data.get("severity", "high"),
            trace=data.get("trace", []),
            user_intent=data.get("user_intent", ""),
            context=data.get("context", ""),
            expected_outcome=data.get("expected_outcome", ""),
            metadata=data.get("metadata", {}),
            timestamp=timestamp,
        )

    @staticmethod
    def generate_id(trace: list[dict[str, Any]]) -> str:
        """Generate a unique ID from trace content."""
        content = json.dumps(trace, sort_keys=True)
        return f"v_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
