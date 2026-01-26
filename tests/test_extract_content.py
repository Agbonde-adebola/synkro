"""Test 18: Tests for extracting content from various LLM response formats."""

from synkro.parsers import extract_content


def test_extract_from_string():
    """extract_content returns string as-is."""
    result = extract_content("Hello, world!")
    assert result == "Hello, world!"


def test_extract_from_gemini_format():
    """extract_content handles Gemini response format."""
    response = {"candidates": [{"content": {"parts": [{"text": "Gemini response text"}]}}]}
    result = extract_content(response)
    assert result == "Gemini response text"


def test_extract_from_openai_format():
    """extract_content handles OpenAI response format."""
    response = {"choices": [{"message": {"content": "OpenAI response text"}}]}
    result = extract_content(response)
    assert result == "OpenAI response text"


def test_extract_from_simple_content():
    """extract_content handles simple content field."""
    response = {"content": "Simple content"}
    result = extract_content(response)
    assert result == "Simple content"


def test_extract_from_text_field():
    """extract_content handles text field."""
    response = {"text": "Text field content"}
    result = extract_content(response)
    assert result == "Text field content"


def test_extract_from_output_field():
    """extract_content handles output field."""
    response = {"output": "Output field content"}
    result = extract_content(response)
    assert result == "Output field content"


def test_extract_from_dict_fallback():
    """extract_content falls back to JSON dump for unknown dict."""
    response = {"unknown_key": "value", "another": 123}
    result = extract_content(response)
    assert "unknown_key" in result
    assert "value" in result


def test_extract_handles_exception():
    """extract_content handles malformed data gracefully."""
    # Malformed Gemini format (missing parts)
    response = {"candidates": [{"content": {}}]}
    result = extract_content(response)
    # Should not raise, returns string representation
    assert isinstance(result, str)


def test_extract_from_empty_string():
    """extract_content handles empty string."""
    result = extract_content("")
    assert result == ""


def test_extract_from_none_like():
    """extract_content handles non-standard types."""
    result = extract_content(123)
    assert "123" in result


def test_extract_from_list():
    """extract_content handles list input."""
    response = [{"content": "item1"}, {"content": "item2"}]
    result = extract_content(response)
    # Should return JSON dump of list
    assert isinstance(result, str)
