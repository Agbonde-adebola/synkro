"""Tool-related types for tool call trace generation."""

from pydantic import BaseModel, Field


class ToolFunction(BaseModel):
    """Function details within a tool call."""

    name: str = Field(description="Name of the function to call")
    arguments: str = Field(description="JSON string of function arguments")


class ToolCall(BaseModel):
    """A tool call made by the assistant."""

    id: str = Field(description="Unique identifier for this tool call")
    type: str = Field(default="function", description="Type of tool call")
    function: ToolFunction = Field(description="Function details")


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_call_id: str = Field(description="ID of the tool call this responds to")
    content: str = Field(description="The tool's response content")


class ToolDefinition(BaseModel):
    """
    Definition of a tool that an agent can use.

    Examples:
        >>> web_search = ToolDefinition(
        ...     name="web_search",
        ...     description="Search the web for current information",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "query": {"type": "string", "description": "Search query"}
        ...         },
        ...         "required": ["query"]
        ...     },
        ...     examples=[{"query": "weather in NYC"}],
        ...     mock_responses=["NYC: 72°F, sunny"]
        ... )
    """

    name: str = Field(description="Name of the tool")
    description: str = Field(description="What the tool does")
    parameters: dict = Field(
        description="JSON Schema for the tool's parameters",
        default_factory=lambda: {"type": "object", "properties": {}},
    )
    examples: list[dict] = Field(
        default_factory=list, description="Example tool calls for few-shot learning"
    )
    mock_responses: list[str] = Field(
        default_factory=list, description="Example responses for simulation"
    )

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_system_prompt(self) -> str:
        """Generate a system prompt description of this tool."""
        params_desc = []
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        for param_name, param_info in props.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            req_marker = " (required)" if param_name in required else ""
            params_desc.append(f"    - {param_name}: {param_type}{req_marker} - {param_desc}")

        params_str = "\n".join(params_desc) if params_desc else "    (no parameters)"

        return f"""**{self.name}**: {self.description}
  Parameters:
{params_str}"""


__all__ = ["ToolDefinition", "ToolCall", "ToolFunction", "ToolResult"]
