"""Violation storage with hashmap default and protocol for external integrations.

Designed for future LangSmith/Langfuse/Datadog integrations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from synkro.remediation.types import Violation


@runtime_checkable
class ViolationStoreProtocol(Protocol):
    """Protocol for violation storage backends.

    Implement this to create custom storage backends (LangSmith, Langfuse, Datadog, etc.)
    """

    def add(self, violation: Violation) -> None:
        """Add a single violation."""
        ...

    def add_batch(self, violations: list[Violation]) -> None:
        """Add multiple violations."""
        ...

    def get(self, violation_id: str) -> Violation | None:
        """Get a violation by ID."""
        ...

    def get_all(self) -> list[Violation]:
        """Get all violations."""
        ...

    def get_by_rule(self, rule_id: str) -> list[Violation]:
        """Get violations that violated a specific rule."""
        ...

    def count(self) -> int:
        """Count total violations."""
        ...

    def clear(self) -> None:
        """Clear all violations."""
        ...


class ViolationStore:
    """In-memory hashmap store for violations.

    Default implementation using a dictionary. Format is compatible with
    LangSmith, Langfuse, and Datadog - use violation.to_*() methods to export.

    Example:
        >>> store = ViolationStore()
        >>> store.add(violation)
        >>> store.add_batch(violations)
        >>>
        >>> # Get all violations
        >>> for v in store:
        ...     print(v.rules_violated)
        >>>
        >>> # Export to LangSmith
        >>> for v in store.get_all():
        ...     langsmith_client.create_feedback(**v.to_langsmith())
    """

    def __init__(self) -> None:
        """Initialize empty store."""
        self._store: dict[str, Violation] = {}

    def add(self, violation: Violation) -> None:
        """Add a single violation.

        Args:
            violation: The violation to store
        """
        self._store[violation.id] = violation

    def add_batch(self, violations: list[Violation]) -> None:
        """Add multiple violations.

        Args:
            violations: List of violations to store
        """
        for v in violations:
            self._store[v.id] = v

    def get(self, violation_id: str) -> Violation | None:
        """Get a violation by ID.

        Args:
            violation_id: The violation ID

        Returns:
            Violation if found, None otherwise
        """
        return self._store.get(violation_id)

    def get_all(self) -> list[Violation]:
        """Get all violations.

        Returns:
            List of all violations
        """
        return list(self._store.values())

    def get_by_rule(self, rule_id: str) -> list[Violation]:
        """Get violations that violated a specific rule.

        Args:
            rule_id: The rule ID to filter by

        Returns:
            List of violations that violated the rule
        """
        return [v for v in self._store.values() if rule_id in v.rules_violated]

    def get_by_severity(self, severity: str) -> list[Violation]:
        """Get violations by severity level.

        Args:
            severity: Severity level ("critical", "high", "medium", "low")

        Returns:
            List of violations with matching severity
        """
        return [v for v in self._store.values() if v.severity == severity]

    def count(self) -> int:
        """Count total violations.

        Returns:
            Number of violations in store
        """
        return len(self._store)

    def clear(self) -> None:
        """Clear all violations."""
        self._store.clear()

    def __len__(self) -> int:
        """Return number of violations."""
        return len(self._store)

    def __iter__(self) -> Iterator[Violation]:
        """Iterate over violations."""
        return iter(self._store.values())

    def __contains__(self, violation_id: str) -> bool:
        """Check if violation ID exists."""
        return violation_id in self._store

    def save(self, path: str | Path) -> None:
        """Save violations to JSONL file.

        Args:
            path: Output file path
        """
        path = Path(path)
        with open(path, "w") as f:
            for v in self._store.values():
                f.write(json.dumps(v.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ViolationStore:
        """Load violations from JSONL file.

        Args:
            path: Input file path

        Returns:
            ViolationStore with loaded violations
        """
        store = cls()
        path = Path(path)
        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    store.add(Violation.from_dict(data))
        return store
