# langchain-scayle

LangChain integration for [Scayle Chat API](https://chat.scayle.es) - a hosted LLM service. This package provides a seamless interface to use Scayle's language models with LangChain (>=1.1.0), enabling you to build AI applications with tool calling, structured outputs, and more.

## Features

- ✅ **Full LangChain Compatibility**: Implements `BaseLLM` interface for seamless integration
- ✅ **LDAP Authentication**: Secure authentication with Scayle API
- ✅ **Tool Calling**: Support for function calling with LangChain tools
- ✅ **Structured Output**: Compatible with Pydantic models and `PydanticOutputParser`
- ✅ **Automatic Model Discovery**: Automatically retrieves and lists available models
- ✅ **Model Status Checking**: Check which models are available and working
- ✅ **SSL Handling**: Automatic SSL verification handling with retry logic
- ✅ **Connection Management**: VPN connection checking and error handling
- ✅ **Time Logging Utilities**: Decorators for monitoring inference performance

## Installation

```bash
pip install langchain-scayle
```

Or using `uv`:

```bash
uv pip install langchain-scayle
```

### Requirements

- Python >= 3.12
- langchain-core >= 1.1.2
- pydantic >= 2.12.5
- requests >= 2.32.5

## Quick Start

### Basic Usage

```python
from langchain_scayle import ScayleLLM

# Initialize the LLM
llm = ScayleLLM(
    scayle_username="your_username",
    scayle_password="your_password",
    base_url="https://chat.scayle.es/api",
    model="your-model-id",  # Optional: will auto-select if not provided
    temperature=0.7,
    timeout=30.0,
)

# Generate text
response = llm.invoke("Hello, how are you?")
print(response)
```

### Using Environment Variables

Create a `.env` file:

```env
SCAYLE_USERNAME=your_username
SCAYLE_PASSWORD=your_password
SCAYLE_BASE_URL=https://chat.scayle.es/api
SCAYLE_VERIFY_SSL=true
```

Then load it in your code:

```python
from dotenv import load_dotenv
import os
from langchain_scayle import ScayleLLM

load_dotenv()

llm = ScayleLLM(
    scayle_username=os.getenv("SCAYLE_USERNAME"),
    scayle_password=os.getenv("SCAYLE_PASSWORD"),
    base_url=os.getenv("SCAYLE_BASE_URL", "https://chat.scayle.es/api"),
    verify_ssl=os.getenv("SCAYLE_VERIFY_SSL", "true").lower() == "true",
)
```

## Configuration

### Connection Check

Before using the LLM, you may want to verify your connection (especially if using VPN):

```python
if llm.check_connection():
    print("Connection successful!")
else:
    print("Connection failed! Please check your VPN connection.")
```

### Model Discovery

Get a list of available models:

```python
models = llm._get_models()
print(f"Available models: {models}")
```

### Model Status Check

Check which models are working and which are unavailable:

```python
status = llm.check_model_status(test_prompt="Hello")
print(f"Working models: {status['working_models']}")
print(f"Unavailable models: {status['unavailable_models']}")
```

## Usage Examples

### 1. Basic Text Generation

```python
from langchain_scayle import ScayleLLM

llm = ScayleLLM(
    scayle_username="your_username",
    scayle_password="your_password",
    model="your-model-id",
)

# Single prompt
response = llm.invoke("What is artificial intelligence?")
print(response)

# Multiple prompts
prompts = ["Hello", "How are you?"]
results = llm.generate(prompts)
for i, result in enumerate(results.generations):
    print(f"Prompt {i+1}: {result[0].text}")
```

### 2. Tool Calling (Function Calling)

```python
from langchain_scayle import ScayleLLM
from langchain_scayle.utils import format_tool_for_openai_api
from langchain_core.tools import tool

# Define a tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to get weather for.
    
    Returns:
        A string describing the weather.
    """
    # Your weather API logic here
    return f"Weather in {city}: Sunny, 72°F"

# Initialize LLM
llm = ScayleLLM(
    scayle_username="your_username",
    scayle_password="your_password",
    model="your-model-id",
)

# Convert tools to API format
tools = [get_weather]
tools_api_format = [format_tool_for_openai_api(t) for t in tools]

# Call with tools
response = llm._call_api("What's the weather in New York?", tools=tools_api_format)

# Process tool calls from response
message = response["choices"][0]["message"]
if "tool_calls" in message:
    for tool_call in message["tool_calls"]:
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        # Execute the tool and continue conversation
```

### 3. Structured Output with Pydantic

```python
from langchain_scayle import ScayleLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Define your output schema
class Joke(BaseModel):
    """A joke with setup and punchline."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

# Create parser
parser = PydanticOutputParser(pydantic_object=Joke)

# Create prompt template
prompt_template = PromptTemplate(
    template="""Tell me a joke about cats. Format your response as JSON matching this schema:
{format_instructions}

Joke:""",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Initialize LLM
llm = ScayleLLM(
    scayle_username="your_username",
    scayle_password="your_password",
    model="your-model-id",
)

# Generate and parse
prompt = prompt_template.format()
response = llm.invoke(prompt)
parsed = parser.parse(response)

print(f"Setup: {parsed.setup}")
print(f"Punchline: {parsed.punchline}")
```

### 4. Time Logging for Inference

Monitor inference performance with built-in decorators:

```python
import logging
from langchain_scayle import ScayleLLM
from langchain_scayle.utils import elapsed_time, elapsed_time_with_params

# Configure logging
logging.basicConfig(level=logging.INFO)

# Simple time logging
@elapsed_time
def simple_inference(llm, prompt: str) -> str:
    return llm.invoke(prompt)

# Time logging with parameters
@elapsed_time_with_params
def inference_with_model(llm, model: str, prompt: str, temperature: float = 0.7) -> str:
    llm.model = model
    llm.temperature = temperature
    return llm.invoke(prompt)

# Use with custom logger
custom_logger = logging.getLogger("my_logger")
@elapsed_time(logger=custom_logger)
def custom_logged_inference(llm, prompt: str) -> str:
    return llm.invoke(prompt)
```

## API Reference

### ScayleLLM

Main class for interacting with Scayle Chat API.

#### Parameters

- `scayle_username` (str, required): Scayle username for authentication
- `scayle_password` (str, required): Scayle password for authentication
- `model` (str, optional): The Scayle model ID to use. If not provided, will auto-select the first available model
- `base_url` (str, default: `"https://chat.scayle.es/api"`): Base URL for the Scayle API
- `temperature` (float, default: `0.7`): Sampling temperature (0.0 to 1.0)
- `max_tokens` (int, optional): Maximum number of tokens to generate
- `top_p` (float, optional): Nucleus sampling parameter
- `verify_ssl` (bool, default: `True`): Whether to verify SSL certificates. Set to `False` for self-signed certs
- `timeout` (float, default: `30.0`): Request timeout in seconds

#### Methods

- `invoke(prompt: str) -> str`: Generate a response for a single prompt
- `generate(prompts: List[str]) -> LLMResult`: Generate responses for multiple prompts
- `check_connection() -> bool`: Check if the base_url is reachable (VPN connection check)
- `check_model_status(test_prompt: str = "Hello", test_timeout: Optional[float] = None) -> Dict[str, Any]`: Check the status of all available models
- `_get_models() -> List[str]`: Retrieve available models from Scayle API

### Utilities

#### `elapsed_time`

Decorator to log the elapsed time for any function.

```python
from langchain_scayle.utils import elapsed_time

@elapsed_time
def my_function():
    # Your code here
    pass
```

#### `elapsed_time_with_params`

Decorator to log inference time with function parameters (excluding sensitive data).

```python
from langchain_scayle.utils import elapsed_time_with_params

@elapsed_time_with_params
def my_inference(model, prompt):
    # Your code here
    pass
```

#### `format_tool_for_openai_api`

Convert LangChain tool to OpenAI-compatible tool format.

```python
from langchain_scayle.utils import format_tool_for_openai_api
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description."""
    return "result"

tool_dict = format_tool_for_openai_api(my_tool)
```

## Error Handling

The library includes comprehensive error handling:

- **Connection Errors**: Automatically detects VPN/network issues
- **SSL Errors**: Automatically retries with SSL verification disabled if needed
- **Authentication Errors**: Clear error messages for credential issues
- **Model Unavailability**: Detects and reports unavailable models
- **Timeout Handling**: Configurable timeouts with clear error messages

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run without integration tests
pytest -m "not integration"

# Run without VPN-required tests
pytest -m "not requires_vpn"
```

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

### Code Quality

The project uses:
- `ruff` for linting
- `black` for code formatting
- `mypy` for type checking

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy .
```

## License

MIT License

## Authors

- Yi Zhang (zhay@zhaw.ch)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- Check the [Scayle Chat API documentation](https://chat.scayle.es)
- Open an issue on the project repository

## Acknowledgments

This package is inspired by `langchain_community.llms.Ollama` and provides similar functionality for the Scayle Chat API.
