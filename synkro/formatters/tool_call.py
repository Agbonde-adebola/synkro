"""Tool Call formatter for training data."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.core import Trace


class ToolCallFormatter:
    """
    Format traces with tool calls for fine-tuning.

    Outputs OpenAI function calling format compatible with most fine-tuning platforms.

    Example output:
        {
          "messages": [
            {"role": "system", "content": "You have access to: web_search(query)"},
            {"role": "user", "content": "What's the weather in NYC?"},
            {"role": "assistant", "content": null, "tool_calls": [
              {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{\\"query\\": \\"weather NYC\\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "NYC: 72°F, sunny"},
            {"role": "assistant", "content": "The weather in NYC is currently 72°F and sunny."}
          ]
        }
    """

    def __init__(self, include_metadata: bool = False):
        """
        Initialize the ToolCallFormatter.

        Args:
            include_metadata: If True, include trace metadata in output
        """
        self.include_metadata = include_metadata

    def format(self, traces: list["Trace"]) -> list[dict]:
        """
        Format traces as tool-calling training examples.

        Args:
            traces: List of traces to format

        Returns:
            List of formatted examples with tool calls
        """
        examples = []

        for trace in traces:
            messages = []

            for m in trace.messages:
                msg = {"role": m.role}

                # Handle content (can be None for tool-calling assistant messages)
                if m.content is not None:
                    msg["content"] = m.content
                elif m.role == "assistant" and m.tool_calls:
                    msg["content"] = None
                else:
                    msg["content"] = ""

                # Handle tool calls
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in m.tool_calls
                    ]

                # Handle tool response
                if m.tool_call_id:
                    msg["tool_call_id"] = m.tool_call_id

                messages.append(msg)

            example = {"messages": messages}

            if self.include_metadata:
                example["metadata"] = {
                    "scenario": trace.scenario.description,
                    "category": trace.scenario.category,
                    "grade": trace.grade.model_dump() if trace.grade else None,
                    "has_tool_calls": trace.has_tool_calls,
                }

            examples.append(example)

        return examples

    def save(self, traces: list["Trace"], path: str | Path, pretty_print: bool = False) -> None:
        """
        Save formatted traces to a JSONL file.

        Args:
            traces: List of traces to save
            path: Output file path (should end in .jsonl)
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
