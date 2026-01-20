"""Tool response simulator for training data generation."""

import json
import uuid
from typing import TYPE_CHECKING

from synkro.prompts.tool_templates import TOOL_SIMULATION_PROMPT

if TYPE_CHECKING:
    from synkro.llm.client import LLM
    from synkro.types.tool import ToolCall, ToolDefinition


class ToolSimulator:
    """
    Simulates tool responses for training data generation.

    Uses an LLM to generate realistic, contextual tool responses
    based on tool definitions and call arguments.

    Example:
        >>> from synkro.types.tool import ToolDefinition, ToolCall, ToolFunction
        >>> simulator = ToolSimulator(tools=[web_search_tool], llm=llm)
        >>> call = ToolCall(
        ...     id="call_1",
        ...     function=ToolFunction(name="web_search", arguments='{"query": "weather NYC"}')
        ... )
        >>> response = await simulator.simulate(call)
        >>> print(response)
        "NYC: 72°F, sunny with a high of 75°F expected"
    """

    def __init__(self, tools: list["ToolDefinition"], llm: "LLM"):
        """
        Initialize the simulator.

        Args:
            tools: List of available tool definitions
            llm: LLM client for generating responses
        """
        self.tools = {t.name: t for t in tools}
        self.llm = llm

    async def simulate(self, tool_call: "ToolCall") -> str:
        """
        Simulate a tool response for the given call.

        Args:
            tool_call: The tool call to simulate

        Returns:
            Simulated tool response content
        """
        tool_name = tool_call.function.name

        if tool_name not in self.tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        tool = self.tools[tool_name]

        # Format mock responses for the prompt
        mock_responses = (
            "\n".join(f"- {r}" for r in tool.mock_responses)
            if tool.mock_responses
            else "No example responses provided"
        )

        prompt = TOOL_SIMULATION_PROMPT.format(
            TOOL_NAME=tool.name,
            TOOL_DESCRIPTION=tool.description,
            TOOL_PARAMETERS=json.dumps(tool.parameters, indent=2),
            ARGUMENTS=tool_call.function.arguments,
            MOCK_RESPONSES=mock_responses,
        )

        response = await self.llm.generate(prompt)
        return response.strip()

    async def simulate_batch(self, tool_calls: list["ToolCall"]) -> list[str]:
        """
        Simulate responses for multiple tool calls.

        Args:
            tool_calls: List of tool calls to simulate

        Returns:
            List of simulated responses in order
        """
        import asyncio

        return await asyncio.gather(*[self.simulate(tc) for tc in tool_calls])

    def generate_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:12]}"

    def get_tools_description(self) -> str:
        """
        Get a formatted description of all available tools.

        Returns:
            Formatted string describing all tools
        """
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.to_system_prompt())
        return "\n\n".join(descriptions)

    def get_tools_json(self) -> list[dict]:
        """
        Get tools in OpenAI function format.

        Returns:
            List of tool definitions in OpenAI format
        """
        return [tool.to_openai_format() for tool in self.tools.values()]
