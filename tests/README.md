# ScayleLLM Test Suite

This directory contains comprehensive tests for the ScayleLLM LangChain integration.

## Test Structure

The tests are organized into several modules:

- **`conftest.py`**: Pytest fixtures and configuration
- **`test_connection.py`**: Connection checking and VPN connectivity tests
- **`test_model_status.py`**: Model status checking and discovery tests
- **`test_tool_calling.py`**: Tool/function calling functionality tests
- **`test_structured_output.py`**: Structured output with PydanticOutputParser tests
- **`test_basic_functionality.py`**: Basic ScayleLLM functionality tests

## Running Tests

### Run all tests:
```bash
uv run pytest tests/
```

### Run specific test file:
```bash
uv run pytest tests/test_connection.py
```

### Run with verbose output:
```bash
uv run pytest tests/ -v
```

### Run with coverage:
```bash
uv run pytest tests/ --cov=src/langchain_scayle --cov-report=html
```

## Test Requirements

Tests require environment variables to be set in a `.env` file:

```env
base_url=https://your-scayle-api-url
username=your_username
password=your_password
verify_ssl=true
```

## Test Coverage

### Connection Tests
- ✅ Connection check success
- ✅ verify_ssl property access
- ✅ Default verify_ssl=True
- ✅ Automatic SSL retry on SSLError

### Model Status Tests
- ✅ Status structure validation
- ✅ Working models identification
- ✅ Custom prompt support
- ✅ Model list retrieval

### Tool Calling Tests
- ✅ Tool calling with tools provided
- ✅ Normal requests without tools
- ✅ Tool format conversion

### Structured Output Tests
- ✅ Simple structured output (Joke)
- ✅ Complex structured output (PersonInfo)
- ✅ PydanticOutputParser format instructions

### Basic Functionality Tests
- ✅ LLM initialization
- ✅ Invoke method
- ✅ Generate method
- ✅ LLM type identification
- ✅ Identifying parameters
- ✅ Model selection

## Notes

- Tests that require VPN connection will be skipped if connection fails
- Tests that require working models will be skipped if no models are available
- Some structured output tests may skip if the model doesn't follow JSON format exactly (this is acceptable)

## Fixtures

The test suite provides several fixtures:

- `scayle_credentials`: Provides credentials from environment variables
- `scayle_llm`: Provides a configured ScayleLLM instance
- `working_model`: Provides a working model ID for testing

