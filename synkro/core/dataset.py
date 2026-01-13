"""Dataset class for managing generated traces."""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field
from rich.console import Console

from synkro.types.core import Trace

console = Console()


class Dataset(BaseModel):
    """
    A collection of generated training traces.

    Provides methods for filtering, saving, and exporting traces
    in various formats.

    Examples:
        >>> dataset = generator.generate(policy, traces=100)

        >>> # Filter to only passing traces
        >>> passing = dataset.filter(passed=True)

        >>> # Save to JSONL
        >>> dataset.save("training.jsonl")

        >>> # Push to HuggingFace
        >>> dataset.to_huggingface().push_to_hub("my-org/dataset")
    """

    traces: list[Trace] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self) -> Iterator[Trace]:
        return iter(self.traces)

    def __getitem__(self, idx: int) -> Trace:
        return self.traces[idx]

    def filter(
        self,
        passed: bool | None = None,
        category: str | None = None,
        min_length: int | None = None,
    ) -> "Dataset":
        """
        Filter traces by criteria.

        Args:
            passed: Filter by grade pass/fail status
            category: Filter by scenario category
            min_length: Minimum response length in characters

        Returns:
            New Dataset with filtered traces
        """
        filtered = self.traces

        if passed is not None:
            filtered = [
                t for t in filtered if t.grade and t.grade.passed == passed
            ]

        if category is not None:
            filtered = [
                t for t in filtered if t.scenario.category == category
            ]

        if min_length is not None:
            filtered = [
                t for t in filtered if len(t.assistant_message) >= min_length
            ]

        return Dataset(traces=filtered)

    def dedupe(
        self,
        threshold: float = 0.85,
        method: str = "semantic",
        field: str = "user",
    ) -> "Dataset":
        """
        Remove duplicate or near-duplicate traces.

        Args:
            threshold: Similarity threshold (0-1). Higher = stricter dedup.
                       Only used for semantic method. (default: 0.85)
            method: Deduplication method:
                    - "exact": Remove exact text duplicates (fast)
                    - "semantic": Remove semantically similar traces (requires sentence-transformers)
            field: Which field to dedupe on - "user", "assistant", or "both"

        Returns:
            New Dataset with duplicates removed

        Examples:
            >>> # Remove exact duplicates (fast)
            >>> deduped = dataset.dedupe(method="exact")

            >>> # Remove semantically similar (needs sentence-transformers)
            >>> deduped = dataset.dedupe(threshold=0.9, method="semantic")

            >>> # Dedupe based on assistant responses
            >>> deduped = dataset.dedupe(field="assistant")
        """
        if not self.traces:
            return Dataset(traces=[])

        if method == "exact":
            return self._dedupe_exact(field)
        elif method == "semantic":
            return self._dedupe_semantic(threshold, field)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'exact' or 'semantic'")

    def _dedupe_exact(self, field: str) -> "Dataset":
        """Remove exact text duplicates."""
        seen = set()
        unique_traces = []

        for trace in self.traces:
            if field == "user":
                key = trace.user_message
            elif field == "assistant":
                key = trace.assistant_message
            else:  # both
                key = (trace.user_message, trace.assistant_message)

            if key not in seen:
                seen.add(key)
                unique_traces.append(trace)

        removed = len(self.traces) - len(unique_traces)
        if removed > 0:
            console.print(f"[yellow]🔍 Dedupe:[/yellow] Removed {removed} exact duplicates")

        return Dataset(traces=unique_traces)

    def _dedupe_semantic(self, threshold: float, field: str) -> "Dataset":
        """Remove semantically similar traces using embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic deduplication. "
                "Install with: pip install sentence-transformers"
            )

        # Get texts to embed
        if field == "user":
            texts = [t.user_message for t in self.traces]
        elif field == "assistant":
            texts = [t.assistant_message for t in self.traces]
        else:  # both
            texts = [f"{t.user_message} {t.assistant_message}" for t in self.traces]

        # Compute embeddings
        console.print("[dim]Computing embeddings for deduplication...[/dim]")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Find duplicates using cosine similarity
        unique_indices = []
        duplicate_of = {}  # Maps duplicate index to original index

        for i in range(len(embeddings)):
            is_duplicate = False
            for j in unique_indices:
                similarity = np.dot(embeddings[i], embeddings[j])
                if similarity >= threshold:
                    is_duplicate = True
                    duplicate_of[i] = j
                    break
            if not is_duplicate:
                unique_indices.append(i)

        unique_traces = [self.traces[i] for i in unique_indices]
        removed = len(self.traces) - len(unique_traces)

        if removed > 0:
            console.print(f"[yellow]🔍 Dedupe:[/yellow] Removed {removed} semantic duplicates (threshold={threshold})")

        return Dataset(traces=unique_traces)

    @property
    def passing_rate(self) -> float:
        """Get the percentage of traces that passed grading."""
        if not self.traces:
            return 0.0

        passed = sum(1 for t in self.traces if t.grade and t.grade.passed)
        return passed / len(self.traces)

    @property
    def categories(self) -> list[str]:
        """Get unique categories in the dataset."""
        return list(set(t.scenario.category for t in self.traces if t.scenario.category))

    def save(self, path: str | Path | None = None, format: str = "messages", pretty_print: bool = False) -> "Dataset":
        """
        Save dataset to a JSONL file.

        Args:
            path: Output file path (auto-generated if not provided)
            format: Output format - "messages", "qa", "langsmith", "langfuse", "tool_call", "chatml",
                    or "bert" / "bert:<task>" for BERT models
            pretty_print: If True, format JSON with indentation (multi-line)

        Returns:
            Self for method chaining

        Example:
            >>> dataset.save()  # Auto-names: synkro_messages_2024-01-15.jsonl
            >>> dataset.save("training.jsonl")
            >>> dataset.save("eval.jsonl", format="qa")  # Q&A with ground truth
            >>> dataset.save("eval.jsonl", format="langsmith")  # LangSmith format
            >>> dataset.save("eval.jsonl", format="langfuse")  # Langfuse format
            >>> dataset.save("tools.jsonl", format="tool_call")
            >>> dataset.save("chatml.jsonl", format="chatml")
            >>> dataset.save("bert.jsonl", format="bert")  # BERT classification
            >>> dataset.save("bert_qa.jsonl", format="bert:qa")  # BERT extractive QA
            >>> dataset.save("readable.jsonl", pretty_print=True)  # Human-readable
        """
        from synkro.formatters import (
            MessagesFormatter, ToolCallFormatter, ChatMLFormatter,
            QAFormatter, LangSmithFormatter, LangfuseFormatter,
            BERTFormatter,
        )

        # Auto-generate filename if not provided
        if path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            format_name = format.replace(":", "_")
            path = f"synkro_{format_name}_{timestamp}.jsonl"

        path = Path(path)

        if format == "messages":
            MessagesFormatter().save(self.traces, path, pretty_print=pretty_print)
        elif format == "qa":
            QAFormatter().save(self.traces, path, pretty_print=pretty_print)
        elif format == "langsmith":
            LangSmithFormatter().save(self.traces, path, pretty_print=pretty_print)
        elif format == "langfuse":
            LangfuseFormatter().save(self.traces, path, pretty_print=pretty_print)
        elif format == "tool_call":
            ToolCallFormatter().save(self.traces, path, pretty_print=pretty_print)
        elif format == "chatml":
            ChatMLFormatter().save(self.traces, path, pretty_print=pretty_print)
        elif format == "bert" or format.startswith("bert:"):
            task = format.split(":")[1] if ":" in format else "classification"
            BERTFormatter(task=task).save(self.traces, path, pretty_print=pretty_print)
        else:
            raise ValueError(
                f"Unknown format: {format}. Use 'messages', 'qa', 'langsmith', 'langfuse', "
                f"'tool_call', 'chatml', or 'bert'/'bert:<task>'"
            )
        
        # Print confirmation
        file_size = path.stat().st_size
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / 1024 / 1024:.1f} MB"
        console.print(f"[green]📁 Saved:[/green] {path} ({size_str})")
        
        return self

    def to_jsonl(self, format: str = "messages", pretty_print: bool = False) -> str:
        """
        Convert dataset to JSONL string.

        Args:
            format: Output format - "messages", "qa", "langsmith", "langfuse", "tool_call", "chatml",
                    or "bert" / "bert:<task>" for BERT models
            pretty_print: If True, format JSON with indentation (multi-line)

        Returns:
            JSONL formatted string
        """
        from synkro.formatters import (
            MessagesFormatter, ToolCallFormatter, ChatMLFormatter,
            QAFormatter, LangSmithFormatter, LangfuseFormatter,
            BERTFormatter,
        )

        if format == "messages":
            return MessagesFormatter().to_jsonl(self.traces, pretty_print=pretty_print)
        elif format == "qa":
            return QAFormatter().to_jsonl(self.traces, pretty_print=pretty_print)
        elif format == "langsmith":
            return LangSmithFormatter().to_jsonl(self.traces, pretty_print=pretty_print)
        elif format == "langfuse":
            return LangfuseFormatter().to_jsonl(self.traces, pretty_print=pretty_print)
        elif format == "tool_call":
            return ToolCallFormatter().to_jsonl(self.traces, pretty_print=pretty_print)
        elif format == "chatml":
            return ChatMLFormatter().to_jsonl(self.traces, pretty_print=pretty_print)
        elif format == "bert" or format.startswith("bert:"):
            task = format.split(":")[1] if ":" in format else "classification"
            return BERTFormatter(task=task).to_jsonl(self.traces, pretty_print=pretty_print)
        else:
            raise ValueError(
                f"Unknown format: {format}. Use 'messages', 'qa', 'langsmith', 'langfuse', "
                f"'tool_call', 'chatml', or 'bert'/'bert:<task>'"
            )

    def to_hf_dataset(self, format: str = "messages"):
        """
        Convert to HuggingFace Dataset.

        Args:
            format: Output format - "messages", "qa", "langsmith", "langfuse", "tool_call", "chatml",
                    or "bert" / "bert:<task>" for BERT models

        Returns:
            HuggingFace datasets.Dataset object

        Example:
            >>> hf_dataset = dataset.to_hf_dataset()
            >>> hf_dataset.push_to_hub("my-org/policy-traces")

            >>> # With train/test split
            >>> hf_dataset = dataset.to_hf_dataset()
            >>> split = hf_dataset.train_test_split(test_size=0.1)
            >>> split.push_to_hub("my-org/policy-traces")

            >>> # BERT format for encoder models
            >>> hf_dataset = dataset.to_hf_dataset(format="bert:classification")
        """
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError(
                "datasets is required for HuggingFace export. "
                "Install with: pip install datasets"
            )

        from synkro.formatters import (
            MessagesFormatter, ToolCallFormatter, ChatMLFormatter,
            QAFormatter, LangSmithFormatter, LangfuseFormatter,
            BERTFormatter,
        )

        if format == "messages":
            examples = MessagesFormatter(include_metadata=True).format(self.traces)
        elif format == "qa":
            examples = QAFormatter().format(self.traces)
        elif format == "langsmith":
            examples = LangSmithFormatter().format(self.traces)
        elif format == "langfuse":
            examples = LangfuseFormatter().format(self.traces)
        elif format == "tool_call":
            examples = ToolCallFormatter().format(self.traces)
        elif format == "chatml":
            examples = ChatMLFormatter().format(self.traces)
        elif format == "bert" or format.startswith("bert:"):
            task = format.split(":")[1] if ":" in format else "classification"
            examples = BERTFormatter(task=task, include_metadata=True).format(self.traces)
        else:
            raise ValueError(
                f"Unknown format: {format}. Use 'messages', 'qa', 'langsmith', 'langfuse', "
                f"'tool_call', 'chatml', or 'bert'/'bert:<task>'"
            )

        return HFDataset.from_list(examples)

    # Alias for backwards compatibility
    to_huggingface = to_hf_dataset

    def push_to_hub(
        self,
        repo_id: str,
        format: str = "messages",
        private: bool = False,
        split: str = "train",
        token: str | None = None,
    ) -> str:
        """
        Push dataset directly to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "my-org/policy-data")
            format: Output format - "messages", "qa", or "tool_call"
            private: Whether the repo should be private
            split: Dataset split name (default: "train")
            token: HuggingFace token (uses cached token if not provided)

        Returns:
            URL of the uploaded dataset

        Example:
            >>> dataset.push_to_hub("my-org/policy-data")
            >>> dataset.push_to_hub("my-org/policy-data", private=True)
        """
        hf_dataset = self.to_hf_dataset(format=format)
        hf_dataset.push_to_hub(
            repo_id,
            private=private,
            split=split,
            token=token,
        )
        url = f"https://huggingface.co/datasets/{repo_id}"
        console.print(f"[green]🤗 Pushed to Hub:[/green] {url}")
        return url

    def to_dict(self) -> dict:
        """
        Convert dataset to a dictionary.

        Returns:
            Dictionary with trace data
        """
        return {
            "traces": [t.model_dump() for t in self.traces],
            "stats": {
                "total": len(self.traces),
                "passing_rate": self.passing_rate,
                "categories": self.categories,
            },
        }

    def summary(self) -> str:
        """
        Get a summary of the dataset.

        Returns:
            Human-readable summary string
        """
        lines = [
            f"Dataset Summary",
            f"===============",
            f"Total traces: {len(self.traces)}",
            f"Passing rate: {self.passing_rate:.1%}",
            f"Categories: {len(self.categories)}",
        ]

        if self.categories:
            lines.append("")
            lines.append("By category:")
            for cat in self.categories:
                count = sum(1 for t in self.traces if t.scenario.category == cat)
                lines.append(f"  - {cat}: {count}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return f"Dataset(traces={len(self.traces)}, passing={self.passing_rate:.1%})"

    def __repr__(self) -> str:
        return self.__str__()

