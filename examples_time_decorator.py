"""Example demonstrating the time logging decorator for inference."""

import logging
import time
from langchain_scayle.utils import elapsed_time, elapsed_time_with_params

# Configure default logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a custom logger for demonstration
custom_logger = logging.getLogger("my_custom_logger")
custom_logger.setLevel(logging.INFO)
custom_handler = logging.StreamHandler()
custom_handler.setFormatter(logging.Formatter("[CUSTOM] %(message)s"))
custom_logger.addHandler(custom_handler)


@elapsed_time
def simple_inference(prompt: str) -> str:
    """Simple inference function with default logger."""
    time.sleep(0.2)  # Simulate processing time
    return f"Response to: {prompt}"


@elapsed_time(logger=custom_logger)
def inference_with_custom_logger(prompt: str) -> str:
    """Inference function with custom logger."""
    time.sleep(0.2)  # Simulate processing time
    return f"Response to: {prompt}"


@elapsed_time_with_params
def inference_with_model(model: str, prompt: str, temperature: float = 0.7) -> str:
    """Inference function with parameter logging (default logger)."""
    time.sleep(0.2)  # Simulate processing time
    return f"Model {model} response to: {prompt} (temp={temperature})"


@elapsed_time_with_params(logger=custom_logger)
def inference_with_model_custom_logger(
    model: str, prompt: str, temperature: float = 0.7
) -> str:
    """Inference function with parameter logging (custom logger)."""
    time.sleep(0.2)  # Simulate processing time
    return f"Model {model} response to: {prompt} (temp={temperature})"


def main():
    """Demonstrate time logging decorators."""
    print("=" * 60)
    print("Time Logging Decorator Examples")
    print("=" * 60)
    print()

    print("Example 1: Simple time logging (default logger)")
    print("-" * 60)
    result1 = simple_inference("What is AI?")
    print(f"Result: {result1}")
    print()

    print("Example 2: Simple time logging (custom logger)")
    print("-" * 60)
    result2 = inference_with_custom_logger("What is AI?")
    print(f"Result: {result2}")
    print()

    print("Example 3: Time logging with parameters (default logger)")
    print("-" * 60)
    result3 = inference_with_model("qwen-qwq", "Tell me a joke", temperature=0.8)
    print(f"Result: {result3}")
    print()

    print("Example 4: Time logging with parameters (custom logger)")
    print("-" * 60)
    result4 = inference_with_model_custom_logger(
        "llama-4-maverick", "Tell me a joke", temperature=0.8
    )
    print(f"Result: {result4}")
    print()

    print("=" * 60)
    print("Check the logs above for inference timing information!")
    print("Notice how custom logger messages are prefixed with [CUSTOM]")
    print("=" * 60)


if __name__ == "__main__":
    main()

