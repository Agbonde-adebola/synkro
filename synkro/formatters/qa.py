"""QA (Question-Answer) formatter for evaluation datasets."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.core import Trace


class QAFormatter:
    """
    Format traces for evaluation datasets (Q&A format with ground truth).

    QA format includes:
    - question: The user's question
    - answer: The assistant's response
    - expected_outcome: Ground truth expected behavior
    - ground_truth_rules: Rule IDs that should be applied
    - difficulty: Scenario type (positive, negative, edge_case, irrelevant)
    - category: Policy category
    - context: Additional context for the scenario
    - passed: Whether the response was graded as correct

    Example output:
        {
            "question": "Can I submit a $200 expense without a receipt?",
            "answer": "No, all expenses require receipts...",
            "expected_outcome": "Deny - missing receipt violates R003",
            "ground_truth_rules": ["R003", "R005"],
            "difficulty": "negative",
            "category": "Receipt Requirements",
            "context": "Expense: $200, No receipt, Within 30 days",
            "passed": true
        }
    """

    def __init__(self, include_reasoning: bool = False):
        """
        Initialize the QA formatter.

        Args:
            include_reasoning: If True, include reasoning chain in output
        """
        self.include_reasoning = include_reasoning

    def format(self, traces: list["Trace"]) -> list[dict]:
        """
        Format traces as QA evaluation examples.

        Args:
            traces: List of traces to format

        Returns:
            List of QA examples with ground truth
        """
        examples = []

        for trace in traces:
            # Use assistant_message if available, otherwise use expected_outcome
            # This handles both full traces and scenario-only generation
            answer = trace.assistant_message or trace.scenario.expected_outcome or ""

            example = {
                "question": trace.user_message,
                "answer": answer,
                "expected_outcome": trace.scenario.expected_outcome or "",
                "ground_truth_rules": trace.scenario.target_rule_ids or [],
                "difficulty": trace.scenario.scenario_type or "unknown",
                "category": trace.scenario.category or "",
                "context": trace.scenario.context or "",
                "passed": trace.grade.passed if trace.grade else None,
            }

            # Optionally include reasoning
            if self.include_reasoning:
                example["reasoning_chain"] = trace.reasoning_chain
                example["rules_applied"] = trace.rules_applied
                example["rules_excluded"] = trace.rules_excluded

            # Include grading feedback if available
            if trace.grade:
                example["grade_feedback"] = trace.grade.feedback
                example["grade_issues"] = trace.grade.issues

            examples.append(example)

        return examples

    def save(self, traces: list["Trace"], path: str | Path) -> None:
        """
        Save formatted traces to a JSONL file.

        Args:
            traces: List of traces to save
            path: Output file path (should end in .jsonl)
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
