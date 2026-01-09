"""LangSmith formatter for evaluation datasets."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.core import Trace


class LangSmithFormatter:
    """
    Format traces for LangSmith datasets.

    LangSmith format uses nested inputs/outputs structure:
    - inputs: dict of input fields
    - outputs: dict of expected output fields
    - metadata: optional additional info

    Example output:
        {
            "inputs": {
                "question": "Can I submit a $200 expense without a receipt?",
                "context": "Expense: $200, No receipt"
            },
            "outputs": {
                "answer": "All expenses require receipts...",
                "expected_outcome": "Deny - missing receipt violates R003"
            },
            "metadata": {
                "ground_truth_rules": ["R003"],
                "difficulty": "negative",
                "category": "Receipt Requirements"
            }
        }
    """

    def format(self, traces: list["Trace"]) -> list[dict]:
        """
        Format traces as LangSmith dataset examples.

        Args:
            traces: List of traces to format

        Returns:
            List of LangSmith-compatible examples
        """
        examples = []

        for trace in traces:
            # Use assistant_message if available, otherwise use expected_outcome
            # This handles both full traces and scenario-only generation
            answer = trace.assistant_message or trace.scenario.expected_outcome or ""

            example = {
                "inputs": {
                    "question": trace.user_message,
                    "context": trace.scenario.context or "",
                },
                "outputs": {
                    "answer": answer,
                    "expected_outcome": trace.scenario.expected_outcome or "",
                },
                "metadata": {
                    "ground_truth_rules": trace.scenario.target_rule_ids or [],
                    "difficulty": trace.scenario.scenario_type or "unknown",
                    "category": trace.scenario.category or "",
                    "passed": trace.grade.passed if trace.grade else None,
                },
            }

            examples.append(example)

        return examples

    def save(self, traces: list["Trace"], path: str | Path) -> None:
        """
        Save formatted traces to a JSONL file.

        Args:
            traces: List of traces to save
            path: Output file path
        """
        path = Path(path)
        examples = self.format(traces)

        with open(path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

    def to_jsonl(self, traces: list["Trace"]) -> str:
        """
        Convert traces to JSONL string.

        Args:
            traces: List of traces to convert

        Returns:
            JSONL formatted string
        """
        examples = self.format(traces)
        return "\n".join(json.dumps(e) for e in examples)
