"""Comprehensive usage examples for ScayleLLM.

This script demonstrates:
- Basic usage (simple text generation)
- Tool calling (function calling)
- Structured output (Pydantic models)
- Time logging decorator for inference
"""

import logging
import os
import json
import sys
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_scayle.llm import ScayleLLM
from langchain_scayle.utils import elapsed_time
from langchain_scayle.utils import format_tool_for_openai_api
from langchain_scayle.utils import load_configuration


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[33m",  # Yellow
        "INFO": "\033[36m",  # Cyan
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[31m",  # Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for this level
        color = self.COLORS.get(record.levelname, self.RESET)

        # Format time
        log_time = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Get module and function name
        # Check if extra dict has decorated_module/decorated_funcName (from decorator),
        # otherwise use record attributes
        decorated_module = getattr(record, "decorated_module", None)
        if decorated_module:
            module_name = decorated_module
        elif hasattr(record, "module") and record.module:
            module_name = record.module
        elif hasattr(record, "name") and record.name:
            module_name = record.name.split(".")[-1]
        else:
            module_name = "unknown"

        decorated_func_name = getattr(record, "decorated_funcName", None)
        if decorated_func_name:
            func_name = decorated_func_name
        elif hasattr(record, "funcName") and record.funcName:
            func_name = record.funcName
        else:
            func_name = "unknown"

        module_func = f"{module_name}.{func_name}"

        # Format message
        level_name = record.levelname
        message = record.getMessage()

        # Build formatted string
        formatted = f"{log_time} - {color}[{level_name}]{self.RESET} - {module_func}: {message}"

        return formatted


def setup_colored_logger(level: int = logging.INFO) -> logging.Logger:
    """Set up a colorful logger with custom formatting.

    Args:
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Set custom formatter
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


# Set up colorful logger
setup_colored_logger(logging.INFO)


# Define Pydantic models for structured output examples
class Joke(BaseModel):
    """A joke with setup and punchline."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")


# Define a tool for tool calling example
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
        "Paris": "Partly cloudy, 65°F",
    }
    return weather_data.get(city, f"Weather information for {city} is not available.")





def initialize_llm(config: dict) -> ScayleLLM:
    """Initialize and configure ScayleLLM instance.

    Args:
        config: Configuration dictionary from load_configuration().

    Returns:
        Configured ScayleLLM instance.
    """
    llm = ScayleLLM(
        scayle_username=config["username"],
        scayle_password=config["password"],
        base_url=config["base_url"],
        verify_ssl=config["verify_ssl"],
        timeout=30.0,
    )
    return llm


def setup_llm_and_get_working_models(llm: ScayleLLM) -> list[str]:
    """Set up LLM with connection check and get all working models.

    Args:
        llm: ScayleLLM instance to set up.

    Returns:
        List of working model IDs.

    Raises:
        RuntimeError: If connection fails or no working models are available.
    """
    print("Checking connection...")
    if not llm.check_connection():
        raise RuntimeError("Connection failed. Please check your VPN connection.")

    print("Fetching available models...")
    models = llm._get_models()
    print(f"✓ Found {len(models)} model(s): {', '.join(models)}")
    print()

    print("Checking model status...")
    status = llm.check_model_status(test_prompt="Hello")
    working_models = status["working_models"]
    print(f"✓ {len(working_models)} working model(s): {', '.join(working_models)}")
    print()

    if not working_models:
        raise RuntimeError("No working models available.")

    return working_models


@elapsed_time
def _invoke_with_timing(llm: ScayleLLM, prompt: str) -> str:
    """Helper function to invoke LLM with timing decorator.

    This function is decorated with elapsed_time to demonstrate
    the time logging functionality.

    Args:
        llm: ScayleLLM instance.
        prompt: Prompt to send to the model.

    Returns:
        Model response as string.
    """
    return llm.invoke(prompt)


def example_basic_usage(llm: ScayleLLM, model_id: str) -> bool:
    """Example: Basic text generation with ScayleLLM.

    This demonstrates the simplest use case - generating a text response
    from a prompt without any special formatting or tool calling.
    Also demonstrates the time logging decorator.

    Args:
        llm: Configured ScayleLLM instance.
        model_id: Model ID to use for this example.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[Model: {model_id}] Example 1: Basic Text Generation")
    print("-" * 60)
    prompt = "Hello! Can you tell me a short joke?"
    print(f"Prompt: {prompt}")
    print()

    try:
        llm.model = model_id
        # Use the decorated function to demonstrate time logging
        response = _invoke_with_timing(llm, prompt)
        print("Response:")
        print(response)
        print()
        print("✓ Basic usage example completed!")
        print("  (Check logs above for inference time)")
        print()
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return False

def example_tool_calling(llm: ScayleLLM, model_id: str) -> bool:
    """Example: Tool calling (function calling) with ScayleLLM.

    This demonstrates how to use tool calling, where the model can decide
    to call functions/tools based on the user's request. The model receives
    tool definitions and can request to call them with specific parameters.

    Args:
        llm: Configured ScayleLLM instance.
        model_id: Model ID to use for this example.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[Model: {model_id}] Example 2: Tool Calling (Function Calling)")
    print("-" * 60)

    # Create tools
    tools = [get_weather]
    print(f"Available tools: {[t.name for t in tools]}")
    print()

    # Convert tools to API format
    tools_api_format = [format_tool_for_openai_api(t) for t in tools]

    # Test tool calling
    prompt = "What's the weather like in New York?"
    print(f"Prompt: {prompt}")
    print()
    print(f"Tools API format: {tools_api_format}")
    print()

    try:
        llm.model = model_id
        response = llm._call_api(prompt, tools=tools_api_format)
        message = response.get("choices", [{}])[0].get("message", {})

        # Check if tool calls are present
        tool_calls = message.get("tool_calls")
        if tool_calls:
            print("✓ Tool calls detected!")
            for tool_call in tool_calls:
                func_name = tool_call.get("function", {}).get("name")
                func_args = tool_call.get("function", {}).get("arguments")
                print(f"  Function: {func_name}")
                print(f"  Arguments: {func_args}")
                print()
                # In a real application, you would execute the tool here
                # and send the result back to the model
        else:
            print("Note: Model responded directly without calling tools")
            print(f"Content: {message.get('content', 'N/A')}")
            print()

        print("✓ Tool calling example completed!")
        print()
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return False


def example_structured_output(llm: ScayleLLM, model_id: str) -> bool:
    """Example: Structured output with PydanticOutputParser.

    This demonstrates how to get structured, validated output from the model
    using Pydantic models. The model is instructed to format its response as
    JSON matching a specific schema, which is then parsed and validated.

    Args:
        llm: Configured ScayleLLM instance.
        model_id: Model ID to use for this example.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[Model: {model_id}] Example 3: Structured Output with Pydantic")
    print("-" * 60)

    # Create parser with Pydantic model
    parser = PydanticOutputParser(pydantic_object=Joke)
    prompt_template = PromptTemplate(
        template="""Tell me a joke about cats. Format your response as JSON matching this schema:
{format_instructions}

Joke:""",
        input_variables=[],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt = prompt_template.format()
    print("Prompt (with format instructions):")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    print()

    try:
        llm.model = model_id
        response = llm.invoke(prompt)
        print("Raw response:")
        print(response)
        print()

        # Parse structured output
        try:
            parsed = parser.parse(str(response))
            print("✓ Successfully parsed structured output:")
            print(f"  Setup: {parsed.setup}")
            print(f"  Punchline: {parsed.punchline}")
            print()
            print("Parsed object:")
            print(f"  {parsed}")
            print()

            print("✓ Structured output example completed!")
            print()
            return True
        except Exception as parse_error:
            print(f"⚠ Parsing error: {parse_error}")
            print("  Note: The model may need better prompting or JSON mode support.")
            print("  This is acceptable - the framework is working correctly.")
            print()
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return False


def main():
    """Main function demonstrating ScayleLLM usage examples.

    Tests all working models iteratively with three examples:
    1. Basic text generation
    2. Tool calling
    3. Structured output
    """
    print("=" * 60)
    print("ScayleLLM Usage Examples - Testing All Models")
    print("=" * 60)
    print()

    try:
        # Load configuration
        config = load_configuration()

        # Initialize LLM
        print("Initializing ScayleLLM...")
        llm = initialize_llm(config)
        print("✓ ScayleLLM initialized")
        print()

        # Get all working models
        working_models = setup_llm_and_get_working_models(llm)

        # Track results for each model
        results = {}

        # Test each model with all examples
        for i, model_id in enumerate(working_models, 1):
            print("=" * 60)
            print(f"Testing Model {i}/{len(working_models)}: {model_id}")
            print("=" * 60)
            print()

            model_results = {
                "basic_usage": False,
                "tool_calling": False,
                "structured_output": False,
            }

            # Run examples for this model
            model_results["basic_usage"] = example_basic_usage(llm, model_id)
            model_results["tool_calling"] = example_tool_calling(llm, model_id)
            model_results["structured_output"] = example_structured_output(llm, model_id)

            results[model_id] = model_results

            print("=" * 60)
            print(f"Model {model_id} testing completed")
            print("=" * 60)
            print()

        # Print summary
        print("=" * 60)
        print("Summary - Results by Model")
        print("=" * 60)
        for model_id, model_results in results.items():
            print(f"\nModel: {model_id}")
            print(f"  Basic Usage: {'✓' if model_results['basic_usage'] else '✗'}")
            print(f"  Tool Calling: {'✓' if model_results['tool_calling'] else '✗'}")
            print(f"  Structured Output: {'✓' if model_results['structured_output'] else '✗'}")

        print()
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    except RuntimeError as e:
        print(f"✗ Setup error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
