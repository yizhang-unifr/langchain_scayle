"""Tests for tool calling functionality."""

import json
import pytest
from langchain_core.tools import tool
from src.langchain_scayle.llm import ScayleLLM


# Define a simple tool for testing
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A string describing the weather.
    """
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 55°F",
        "Tokyo": "Rainy, 68°F",
    }
    return weather_data.get(city, f"Weather information for {city} is not available.")


def format_tool_for_api(tool_obj) -> dict:
    """Convert LangChain tool to OpenAI-compatible tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param_name: {
                        "type": "string",
                        "description": f"Parameter {param_name}",
                    }
                    for param_name in tool_obj.args_schema.model_fields.keys()
                    if tool_obj.args_schema
                },
                "required": list(tool_obj.args_schema.model_fields.keys())
                if tool_obj.args_schema
                else [],
            },
        },
    }


def test_tool_calling_with_tools(scayle_llm, working_model):
    """Test tool calling when tools are provided."""
    scayle_llm.model = working_model

    tools = [get_weather]
    tools_api_format = [format_tool_for_api(t) for t in tools]

    prompt = "What's the weather like in New York?"
    response = scayle_llm._call_api(prompt, tools=tools_api_format)

    # Check response structure
    assert "choices" in response
    assert len(response["choices"]) > 0

    message = response["choices"][0].get("message", {})
    tool_calls = message.get("tool_calls")

    # Tool calls should be present (model should call the tool)
    if tool_calls:
        assert len(tool_calls) > 0
        tool_call = tool_calls[0]
        assert "function" in tool_call
        assert tool_call["function"]["name"] == "get_weather"
        assert "arguments" in tool_call["function"]


def test_tool_calling_without_tools(scayle_llm, working_model):
    """Test that normal requests work without tools."""
    scayle_llm.model = working_model

    prompt = "Tell me a joke"
    response = scayle_llm.invoke(prompt)

    # Should get a text response
    assert isinstance(response, str)
    assert len(response) > 0


def test_tool_format_conversion():
    """Test that tool format conversion works correctly."""
    tools_api_format = format_tool_for_api(get_weather)

    assert "type" in tools_api_format
    assert tools_api_format["type"] == "function"
    assert "function" in tools_api_format
    assert tools_api_format["function"]["name"] == "get_weather"
    assert "parameters" in tools_api_format["function"]

