"""Tool definitions for LLM agent integration.

Exports OpenAI-compatible tool schemas that allow LLM agents to use
synkro functions via function calling.

Examples:
    >>> from synkro import TOOL_DEFINITIONS
    >>> # Use with OpenAI function calling
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o",
    ...     messages=[...],
    ...     tools=TOOL_DEFINITIONS,
    ... )
"""

from __future__ import annotations

from typing import Any

# Tool definitions in OpenAI function calling format
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "extract_rules",
            "description": "Extract rules from a policy document as a Logic Map (DAG of rules). This is the first step in generating training data from a policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy": {
                        "type": "string",
                        "description": "The policy text to extract rules from",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for extraction (default: gpt-4o)",
                        "default": "gpt-4o",
                    },
                },
                "required": ["policy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_rules",
            "description": "Edit a Logic Map using natural language instructions. Supports adding, removing, merging, or modifying rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural language instruction for editing rules (e.g., 'add rule: overtime needs approval', 'remove R005', 'merge R002 and R003')",
                    },
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_scenarios",
            "description": "Generate test scenarios from a policy and Logic Map. Creates diverse scenarios (positive, negative, edge_case, irrelevant) that target specific rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of scenarios to generate (default: 20)",
                        "default": 20,
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for generation (default: gpt-4o-mini)",
                        "default": "gpt-4o-mini",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_scenarios",
            "description": "Edit scenarios using natural language instructions. Supports adding, removing, or modifying scenarios.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural language instruction for editing scenarios (e.g., 'add 5 edge cases', 'delete S3', 'more negative cases for R002')",
                    },
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_traces",
            "description": "Generate conversation traces from scenarios. Creates responses with Chain-of-Thought reasoning and rule citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "integer",
                        "description": "Number of conversation turns per trace (default: 1)",
                        "default": 1,
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for generation (default: gpt-4o-mini)",
                        "default": "gpt-4o-mini",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_traces",
            "description": "Verify and refine traces against the Logic Map. Checks for skipped rules, hallucinations, and contradictions. Refines failed traces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum refinement attempts per trace (default: 3)",
                        "default": 3,
                    },
                    "model": {
                        "type": "string",
                        "description": "Model for verification (default: gpt-4o)",
                        "default": "gpt-4o",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_coverage",
            "description": "Get the current coverage report showing how well scenarios cover the rules.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_status",
            "description": "Get the current session status including extracted rules, scenarios, traces, and metrics.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_dataset",
            "description": "Export the current traces as a dataset in the specified format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["jsonl", "json", "csv", "parquet", "hf"],
                        "description": "Output format (default: jsonl)",
                        "default": "jsonl",
                    },
                    "path": {
                        "type": "string",
                        "description": "Output file path (optional, returns data if not specified)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_session",
            "description": "Save the current session state to a file for later resumption.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to save the session state",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_session",
            "description": "Load a previously saved session state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to load the session from",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo",
            "description": "Undo the last operation and restore the previous state.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def get_tool_definitions() -> list[dict[str, Any]]:
    """
    Get the list of tool definitions for LLM function calling.

    Returns:
        List of OpenAI-compatible tool definition dictionaries
    """
    return TOOL_DEFINITIONS.copy()


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """
    Get a specific tool definition by name.

    Args:
        name: Name of the tool

    Returns:
        Tool definition dict or None if not found
    """
    for tool in TOOL_DEFINITIONS:
        if tool["function"]["name"] == name:
            return tool.copy()
    return None


def get_tool_names() -> list[str]:
    """
    Get the list of available tool names.

    Returns:
        List of tool names
    """
    return [tool["function"]["name"] for tool in TOOL_DEFINITIONS]


__all__ = [
    "TOOL_DEFINITIONS",
    "get_tool_definitions",
    "get_tool_by_name",
    "get_tool_names",
]
