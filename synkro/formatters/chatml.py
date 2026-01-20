"""ChatML formatter with XML tags for tool calling."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.core import Trace


class ChatMLFormatter:
    """
    Format traces as ChatML with XML tags for tool calls.

    Uses <tool_call> and <tool_response> XML tags for tool interactions,
    compatible with Hermes/Mistral style fine-tuning.

    Example output:
        {
          "messages": [
            {"role": "system", "content": "You have access to tools."},
            {"role": "user", "content": "What's the weather in NYC?"},
            {"role": "assistant", "content": "<tool_call>\\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}\\n</tool_call>"},
            {"role": "tool", "content": "<tool_response>\\n{\"temp\": \"72F\"}\\n</tool_response>"},
            {"role": "assistant", "content": "The weather in NYC is 72°F."}
          ]
        }
    """

    def __init__(self, include_metadata: bool = False):
        """
        Initialize the ChatMLFormatter.

        Args:
            include_metadata: If True, include trace metadata in output
        """
        self.include_metadata = include_metadata

    def format(self, traces: list["Trace"]) -> list[dict]:
        """
        Format traces as ChatML with XML tags.

        Args:
            traces: List of traces to format

        Returns:
            List of formatted examples
        """
        examples = []

        for trace in traces:
            messages = []

            for m in trace.messages:
                # Handle assistant messages with tool calls
                if m.role == "assistant" and m.tool_calls:
                    # Convert tool calls to XML format
                    tool_call_xmls = []
                    for tc in m.tool_calls:
                        tool_call_json = json.dumps(
                            {
                                "name": tc.function.name,
                                "arguments": json.loads(tc.function.arguments),
                            }
                        )
                        tool_call_xmls.append(f"<tool_call>\n{tool_call_json}\n</tool_call>")

                    content = "\n".join(tool_call_xmls)
                    messages.append({"role": "assistant", "content": content})

                # Handle tool responses
                elif m.role == "tool":
                    content = f"<tool_response>\n{m.content}\n</tool_response>"
                    messages.append({"role": "tool", "content": content})

                # Regular messages (system, user, assistant without tools)
                else:
                    messages.append({"role": m.role, "content": m.content or ""})

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
