"""
BERT formatter for encoder model fine-tuning datasets.

Supports multiple BERT tasks with an extensible registry pattern.
Contributors can add new task types by subclassing BERTTaskFormatter
and registering with the @register_bert_task decorator.

Supported tasks:
    - classification: Single text -> label
    - pair: Text pair -> label (NLI, similarity)
    - ner: Tokens -> per-token labels (NER, POS)
    - qa: Question + context -> answer span (SQuAD-style)
    - choice: Question + choices -> label index

Example usage:
    >>> formatter = BERTFormatter(task="classification")
    >>> formatter.format(traces)
    [{"text": "...", "label": "positive"}, ...]

    >>> formatter = BERTFormatter(task="qa")
    >>> formatter.format(traces)
    [{"question": "...", "context": "...", "answers": {...}}, ...]

Extensibility:
    >>> from synkro.formatters.bert import register_bert_task, BERTTaskFormatter
    >>>
    >>> @register_bert_task("my_task")
    ... class MyTaskFormatter(BERTTaskFormatter):
    ...     def format_trace(self, trace):
    ...         return {"custom": trace.user_message}
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from synkro.types.core import Trace


# =============================================================================
# Task Registry
# =============================================================================

_BERT_TASK_REGISTRY: dict[str, type["BERTTaskFormatter"]] = {}

_TASK_ALIASES: dict[str, str] = {
    "classification": "text_classification",
    "pair": "sequence_pair",
    "ner": "token_classification",
    "qa": "extractive_qa",
    "choice": "multiple_choice",
}


def register_bert_task(
    task_name: str,
) -> Callable[[type["BERTTaskFormatter"]], type["BERTTaskFormatter"]]:
    """
    Decorator to register a new BERT task formatter.

    Use this to add custom BERT task types without modifying core code.

    Example:
        >>> @register_bert_task("sentiment")
        ... class SentimentFormatter(BERTTaskFormatter):
        ...     def format_trace(self, trace):
        ...         return {"text": trace.user_message, "sentiment": ...}

    Args:
        task_name: Unique identifier for this task type

    Returns:
        Decorator function that registers the class
    """

    def decorator(
        cls: type["BERTTaskFormatter"],
    ) -> type["BERTTaskFormatter"]:
        _BERT_TASK_REGISTRY[task_name] = cls
        return cls

    return decorator


def get_available_bert_tasks() -> list[str]:
    """Return list of all registered BERT task types including aliases."""
    tasks = list(_BERT_TASK_REGISTRY.keys())
    aliases = [alias for alias in _TASK_ALIASES.keys() if alias not in tasks]
    return sorted(set(tasks + aliases))


# =============================================================================
# Base Class
# =============================================================================


class BERTTaskFormatter(ABC):
    """
    Abstract base for BERT task-specific formatters.

    Subclass this and implement `format_trace()` to add new task types.
    Register your subclass with @register_bert_task("task_name").

    The format_trace method receives a single Trace and should return
    a dictionary in the format expected by HuggingFace datasets for
    that specific task.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize with task-specific configuration.

        Args:
            **kwargs: Task-specific options passed from BERTFormatter
        """
        self.config = kwargs

    @abstractmethod
    def format_trace(self, trace: "Trace") -> dict | None:
        """
        Format a single trace for this BERT task.

        Args:
            trace: The trace to format

        Returns:
            Dict in HuggingFace dataset format for this task,
            or None to skip this trace
        """
        pass


# =============================================================================
# Main Formatter
# =============================================================================


class BERTFormatter:
    """
    Universal BERT formatter with pluggable task types.

    This is the main entry point for BERT dataset formatting.
    It delegates to task-specific formatters based on the `task` parameter.

    Supported tasks (extensible via @register_bert_task):
        - classification: Classify text into categories
        - pair: Classify relationship between text pairs
        - ner: Label each token (NER, POS tagging)
        - qa: Extract answer spans from context
        - choice: Multiple choice questions

    Example:
        >>> # Text classification from scenario categories
        >>> formatter = BERTFormatter(task="classification")
        >>> dataset = formatter.format(traces)

        >>> # Custom label mapping for toxicity detection
        >>> formatter = BERTFormatter(
        ...     task="classification",
        ...     text_field="user",
        ...     label_field="scenario_type",
        ...     label_map={"positive": 0, "negative": 1, "edge_case": 1}
        ... )

        >>> # Extractive QA
        >>> formatter = BERTFormatter(task="qa")
        >>> dataset = formatter.format(traces)

    Args:
        task: The BERT task type (see get_available_bert_tasks())
        include_metadata: Include trace metadata in output
        **kwargs: Task-specific configuration options
    """

    def __init__(
        self,
        task: str = "classification",
        include_metadata: bool = False,
        **kwargs: Any,
    ):
        # Resolve alias to full task name
        resolved_task = _TASK_ALIASES.get(task, task)

        if resolved_task not in _BERT_TASK_REGISTRY:
            available = ", ".join(get_available_bert_tasks())
            raise ValueError(
                f"Unknown BERT task: '{task}'. "
                f"Available tasks: {available}. "
                f"Use @register_bert_task to add custom tasks."
            )

        self.task = task
        self.resolved_task = resolved_task
        self.include_metadata = include_metadata
        self.task_formatter = _BERT_TASK_REGISTRY[resolved_task](**kwargs)

    def format(self, traces: list["Trace"]) -> list[dict]:
        """
        Format traces for BERT fine-tuning.

        Args:
            traces: List of traces to format

        Returns:
            List of examples in HuggingFace dataset format
        """
        examples = []

        for trace in traces:
            example = self.task_formatter.format_trace(trace)

            if example is None:
                continue

            if self.include_metadata:
                example["_metadata"] = {
                    "scenario": trace.scenario.description,
                    "category": trace.scenario.category,
                    "scenario_type": trace.scenario.scenario_type,
                    "passed": trace.grade.passed if trace.grade else None,
                }

            examples.append(example)

        return examples

    def save(
        self, traces: list["Trace"], path: str | Path, pretty_print: bool = False
    ) -> None:
        """
        Save formatted traces to a JSONL file.

        Args:
            traces: List of traces to save
            path: Output file path
            pretty_print: If True, format JSON with indentation
        """
        path = Path(path)
        examples = self.format(traces)

        with open(path, "w") as f:
            for example in examples:
                if pretty_print:
                    f.write(json.dumps(example, indent=2) + "\n\n")
                else:
                    f.write(json.dumps(example) + "\n")

    def to_jsonl(self, traces: list["Trace"], pretty_print: bool = False) -> str:
        """
        Convert traces to JSONL string.

        Args:
            traces: List of traces to convert
            pretty_print: If True, format JSON with indentation

        Returns:
            JSONL formatted string
        """
        examples = self.format(traces)
        if pretty_print:
            return "\n\n".join(json.dumps(e, indent=2) for e in examples)
        return "\n".join(json.dumps(e) for e in examples)


# =============================================================================
# Built-in Task Formatters
# =============================================================================


@register_bert_task("text_classification")
class TextClassificationFormatter(BERTTaskFormatter):
    """
    Format traces for text classification (sentiment, intent, topic, toxicity, etc.).

    Output format (HuggingFace compatible):
        {"text": "user message", "label": "category"}
        {"text": "user message", "label": 0}  # with label_map

    Config options:
        text_field: Which text to use - "user", "assistant", "combined" (default: "user")
        label_field: Source of label - "category", "scenario_type", "passed" (default: "category")
        label_map: Optional dict mapping label strings to integers
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.text_field = kwargs.get("text_field", "user")
        self.label_field = kwargs.get("label_field", "category")
        self.label_map: dict[str, int] | None = kwargs.get("label_map")

    def format_trace(self, trace: "Trace") -> dict | None:
        # Extract text based on config
        if self.text_field == "user":
            text = trace.user_message
        elif self.text_field == "assistant":
            text = trace.assistant_message
        elif self.text_field == "combined":
            text = f"{trace.user_message}\n\n{trace.assistant_message}"
        else:
            text = trace.user_message

        if not text:
            return None

        # Extract label based on config
        if self.label_field == "category":
            label: Any = trace.scenario.category or "unknown"
        elif self.label_field == "scenario_type":
            label = trace.scenario.scenario_type or "unknown"
        elif self.label_field == "passed":
            label = "passed" if (trace.grade and trace.grade.passed) else "failed"
        else:
            label = trace.scenario.category or "unknown"

        # Apply label mapping if provided
        if self.label_map is not None:
            label = self.label_map.get(str(label), label)

        return {"text": text, "label": label}


@register_bert_task("sequence_pair")
class SequencePairFormatter(BERTTaskFormatter):
    """
    Format traces for sequence pair classification (NLI, similarity, entailment).

    Output format (HuggingFace compatible):
        {"text1": "...", "text2": "...", "label": 1}

    By default, uses user message as text1 and assistant response as text2.

    Config options:
        text1_field: "user", "context", "system" (default: "user")
        text2_field: "assistant", "expected_outcome" (default: "assistant")
        label_field: Source of label - "passed", "scenario_type" (default: "passed")
        label_map: Optional mapping for custom labels
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.text1_field = kwargs.get("text1_field", "user")
        self.text2_field = kwargs.get("text2_field", "assistant")
        self.label_field = kwargs.get("label_field", "passed")
        self.label_map: dict[str, int] | None = kwargs.get("label_map")

    def format_trace(self, trace: "Trace") -> dict | None:
        # Extract text1
        if self.text1_field == "user":
            text1 = trace.user_message
        elif self.text1_field == "context":
            text1 = trace.scenario.context
        elif self.text1_field == "system":
            text1 = trace.system_message or ""
        else:
            text1 = trace.user_message

        # Extract text2
        if self.text2_field == "assistant":
            text2 = trace.assistant_message
        elif self.text2_field == "expected_outcome":
            text2 = trace.scenario.expected_outcome or ""
        else:
            text2 = trace.assistant_message

        if not text1 or not text2:
            return None

        # Extract label
        label: Any
        if self.label_field == "passed":
            label = 1 if (trace.grade and trace.grade.passed) else 0
        elif self.label_field == "scenario_type":
            label = trace.scenario.scenario_type or "unknown"
            if self.label_map:
                label = self.label_map.get(str(label), 0)
        else:
            label = 1 if (trace.grade and trace.grade.passed) else 0

        return {"text1": text1, "text2": text2, "label": label}


@register_bert_task("token_classification")
class TokenClassificationFormatter(BERTTaskFormatter):
    """
    Format traces for token classification (NER, POS tagging, chunking).

    Output format (HuggingFace compatible):
        {"tokens": ["word1", "word2", ...], "labels": ["O", "B-ENT", ...]}

    Config options:
        text_field: Which text to tokenize - "user", "assistant" (default: "assistant")
        tokenizer: Custom tokenizer function (default: whitespace split)
        default_label: Label for non-entity tokens (default: "O")
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.text_field = kwargs.get("text_field", "assistant")
        self.tokenizer = kwargs.get("tokenizer", lambda x: x.split())
        self.default_label = kwargs.get("default_label", "O")

    def format_trace(self, trace: "Trace") -> dict | None:
        # Extract text
        if self.text_field == "user":
            text = trace.user_message
        elif self.text_field == "assistant":
            text = trace.assistant_message
        else:
            text = trace.assistant_message

        if not text:
            return None

        # Tokenize
        tokens = self.tokenizer(text)
        if not tokens:
            return None

        # Default: all tokens get default label
        labels = [self.default_label] * len(tokens)

        # Mark rule IDs as entities if they appear in tokens
        if trace.rules_applied:
            for i, token in enumerate(tokens):
                for rule_id in trace.rules_applied:
                    if rule_id in token:
                        labels[i] = "B-RULE"
                        break

        return {"tokens": tokens, "labels": labels}


@register_bert_task("extractive_qa")
class ExtractiveQAFormatter(BERTTaskFormatter):
    """
    Format traces for extractive question answering (SQuAD-style).

    Output format (HuggingFace SQuAD compatible):
        {
            "id": "unique_id",
            "question": "user question",
            "context": "passage containing the answer",
            "answers": {
                "text": ["answer text"],
                "answer_start": [position in context]
            }
        }

    Config options:
        context_field: Source of context - "system", "scenario_context" (default: "system")
        answer_field: Source of answer - "assistant", "expected_outcome" (default: "assistant")
        include_impossible: Include unanswerable examples with empty answers (default: False)
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.context_field = kwargs.get("context_field", "system")
        self.answer_field = kwargs.get("answer_field", "assistant")
        self.include_impossible = kwargs.get("include_impossible", False)
        self._id_counter = 0

    def format_trace(self, trace: "Trace") -> dict | None:
        question = trace.user_message

        # Extract context
        if self.context_field == "system":
            context = trace.system_message or ""
        elif self.context_field == "scenario_context":
            context = trace.scenario.context
        else:
            context = trace.system_message or trace.scenario.context

        # Extract answer
        if self.answer_field == "assistant":
            answer_text = trace.assistant_message
        elif self.answer_field == "expected_outcome":
            answer_text = trace.scenario.expected_outcome or ""
        else:
            answer_text = trace.assistant_message

        if not question or not context:
            return None

        # Generate unique ID
        self._id_counter += 1
        example_id = f"synkro_{self._id_counter}"

        # Find answer position in context (for true extractive QA)
        answer_start = context.find(answer_text) if answer_text else -1

        # Build answers dict
        if answer_start >= 0:
            answers = {"text": [answer_text], "answer_start": [answer_start]}
        elif self.include_impossible:
            # SQuAD 2.0 style: unanswerable questions
            answers = {"text": [], "answer_start": []}
        else:
            # Answer not in context - include with -1 position
            answers = {
                "text": [answer_text] if answer_text else [],
                "answer_start": [-1] if answer_text else [],
            }

        return {
            "id": example_id,
            "question": question,
            "context": context,
            "answers": answers,
        }


@register_bert_task("multiple_choice")
class MultipleChoiceFormatter(BERTTaskFormatter):
    """
    Format traces for multiple choice tasks (SWAG, ARC, CommonsenseQA).

    Output format (HuggingFace compatible):
        {
            "question": "...",
            "choices": ["option A", "option B", "option C", "option D"],
            "label": 0  # index of correct answer
        }

    Config options:
        num_choices: Number of choices to include (default: 4)
        correct_field: Source of correct answer - "assistant", "expected_outcome" (default: "assistant")
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_choices = kwargs.get("num_choices", 4)
        self.correct_field = kwargs.get("correct_field", "assistant")

    def format_trace(self, trace: "Trace") -> dict | None:
        question = trace.user_message

        if self.correct_field == "assistant":
            correct_answer = trace.assistant_message
        elif self.correct_field == "expected_outcome":
            correct_answer = trace.scenario.expected_outcome or ""
        else:
            correct_answer = trace.assistant_message

        if not question or not correct_answer:
            return None

        # Correct answer is always first (label=0)
        choices = [correct_answer]

        # Placeholder distractors - override or post-process for real distractors
        for i in range(1, self.num_choices):
            choices.append(f"[Distractor {i}]")

        return {"question": question, "choices": choices, "label": 0}
