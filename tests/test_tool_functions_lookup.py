"""Test 16: Tests for tool functions lookup."""

from synkro.tools import (
    TOOL_DEFINITIONS,
    get_tool_by_name,
    get_tool_definitions,
    get_tool_names,
)


def test_tool_definitions_is_list():
    """TOOL_DEFINITIONS is a non-empty list."""
    assert isinstance(TOOL_DEFINITIONS, list)
    assert len(TOOL_DEFINITIONS) > 0


def test_tool_definitions_structure():
    """Each tool definition has type and function keys."""
    for tool in TOOL_DEFINITIONS:
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_get_tool_definitions_returns_copy():
    """get_tool_definitions returns a copy, not the original."""
    result = get_tool_definitions()
    assert result is not TOOL_DEFINITIONS
    assert result == TOOL_DEFINITIONS


def test_get_tool_names_returns_all_names():
    """get_tool_names returns list of all tool names."""
    names = get_tool_names()
    assert isinstance(names, list)
    assert len(names) == len(TOOL_DEFINITIONS)


def test_get_tool_names_contains_expected_tools():
    """get_tool_names includes expected core tools."""
    names = get_tool_names()
    expected_tools = ["extract_rules", "generate_scenarios", "verify_traces"]
    for expected in expected_tools:
        assert expected in names


def test_get_tool_by_name_finds_existing():
    """get_tool_by_name returns tool when found."""
    tool = get_tool_by_name("extract_rules")
    assert tool is not None
    assert tool["function"]["name"] == "extract_rules"


def test_get_tool_by_name_returns_none_for_missing():
    """get_tool_by_name returns None for non-existent tool."""
    tool = get_tool_by_name("non_existent_tool")
    assert tool is None


def test_get_tool_by_name_returns_copy():
    """get_tool_by_name returns a copy, not the original."""
    tool1 = get_tool_by_name("extract_rules")
    tool2 = get_tool_by_name("extract_rules")
    assert tool1 is not tool2
    assert tool1 == tool2


def test_tool_parameters_have_required_field():
    """Each tool's parameters has a required field."""
    for tool in TOOL_DEFINITIONS:
        params = tool["function"]["parameters"]
        assert "type" in params
        assert params["type"] == "object"
        assert "required" in params


def test_extract_rules_has_policy_required():
    """extract_rules tool requires 'policy' parameter."""
    tool = get_tool_by_name("extract_rules")
    assert tool is not None
    required = tool["function"]["parameters"]["required"]
    assert "policy" in required


def test_save_session_has_path_required():
    """save_session tool requires 'path' parameter."""
    tool = get_tool_by_name("save_session")
    assert tool is not None
    required = tool["function"]["parameters"]["required"]
    assert "path" in required


def test_tool_descriptions_are_non_empty():
    """All tools have non-empty descriptions."""
    for tool in TOOL_DEFINITIONS:
        desc = tool["function"]["description"]
        assert isinstance(desc, str)
        assert len(desc) > 10  # Reasonable minimum length
