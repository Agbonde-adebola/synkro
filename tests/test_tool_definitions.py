"""Test 15: Tests for tool definitions structure and validation."""

import json

from synkro.types.tool import ToolCall, ToolDefinition, ToolFunction, ToolResult


def test_tool_function_structure():
    """ToolFunction has name and arguments."""
    func = ToolFunction(name="search", arguments='{"query": "test"}')
    assert func.name == "search"
    assert func.arguments == '{"query": "test"}'


def test_tool_call_structure():
    """ToolCall has id, type, and function."""
    call = ToolCall(
        id="call_123",
        type="function",
        function=ToolFunction(name="search", arguments="{}"),
    )
    assert call.id == "call_123"
    assert call.type == "function"
    assert call.function.name == "search"


def test_tool_call_default_type():
    """ToolCall defaults type to 'function'."""
    call = ToolCall(
        id="call_456",
        function=ToolFunction(name="test", arguments="{}"),
    )
    assert call.type == "function"


def test_tool_result_structure():
    """ToolResult has tool_call_id and content."""
    result = ToolResult(tool_call_id="call_123", content="Search results here")
    assert result.tool_call_id == "call_123"
    assert result.content == "Search results here"


def test_tool_definition_basic():
    """ToolDefinition has name and description."""
    tool = ToolDefinition(
        name="web_search",
        description="Search the web for information",
    )
    assert tool.name == "web_search"
    assert tool.description == "Search the web for information"


def test_tool_definition_with_parameters():
    """ToolDefinition accepts JSON schema parameters."""
    tool = ToolDefinition(
        name="calculator",
        description="Perform calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
            },
            "required": ["expression"],
        },
    )
    assert "properties" in tool.parameters
    assert "expression" in tool.parameters["properties"]


def test_tool_definition_default_parameters():
    """ToolDefinition has default empty parameters."""
    tool = ToolDefinition(name="simple", description="Simple tool")
    assert tool.parameters == {"type": "object", "properties": {}}


def test_tool_definition_with_examples():
    """ToolDefinition accepts example calls."""
    tool = ToolDefinition(
        name="search",
        description="Search",
        examples=[{"query": "weather"}, {"query": "news"}],
    )
    assert len(tool.examples) == 2
    assert tool.examples[0]["query"] == "weather"


def test_tool_definition_with_mock_responses():
    """ToolDefinition accepts mock responses for simulation."""
    tool = ToolDefinition(
        name="api",
        description="API call",
        mock_responses=["Success", "Error: not found"],
    )
    assert len(tool.mock_responses) == 2
    assert "Success" in tool.mock_responses


def test_tool_definition_to_openai_format():
    """to_openai_format produces correct structure."""
    tool = ToolDefinition(
        name="search",
        description="Search the web",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    openai_format = tool.to_openai_format()

    assert openai_format["type"] == "function"
    assert "function" in openai_format
    assert openai_format["function"]["name"] == "search"
    assert openai_format["function"]["description"] == "Search the web"
    assert openai_format["function"]["parameters"] == tool.parameters


def test_tool_definition_to_system_prompt():
    """to_system_prompt generates readable prompt."""
    tool = ToolDefinition(
        name="lookup",
        description="Look up information",
        parameters={
            "type": "object",
            "properties": {
                "term": {"type": "string", "description": "Term to look up"},
            },
            "required": ["term"],
        },
    )
    prompt = tool.to_system_prompt()

    assert "lookup" in prompt
    assert "Look up information" in prompt
    assert "term" in prompt
    assert "required" in prompt.lower()


def test_tool_definition_serialization():
    """ToolDefinition can be serialized to dict and JSON."""
    tool = ToolDefinition(
        name="test",
        description="Test tool",
        parameters={"type": "object", "properties": {}},
    )
    data = tool.model_dump()
    json_str = json.dumps(data)

    assert "test" in json_str
    restored = ToolDefinition.model_validate(data)
    assert restored.name == tool.name
