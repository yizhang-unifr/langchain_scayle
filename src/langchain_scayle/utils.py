"""Utility functions for langchain_scayle."""

import functools
import logging
import time
from typing import Callable, Optional, overload, ParamSpec, TypeVar, Union
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
import os

# Set up default logger
_default_logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
# F = TypeVar("F", bound=Callable[..., Any])


@overload
def elapsed_time(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def elapsed_time(
    *, logger: Optional[logging.Logger] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def elapsed_time(
    func: Optional[Callable[P, R]] = None, *, logger: Optional[logging.Logger] = None
) -> Union[Callable[P, R], Callable[[Callable[P, R]], Callable[P, R]]]:
    """Decorator to log the elapsed time for any function.

    This decorator measures and logs the execution time of decorated functions,
    logging the elapsed time at the INFO level with a clear message. Optionally,
    a custom logger can be specified.

    Usage:
        # With default logger
        @elapsed_time
        def my_function(*args, **kwargs):
            # ... function code ...
            return result

        # With custom logger
        custom_logger = logging.getLogger("my_logger")
        @elapsed_time(logger=custom_logger)
        def my_function(*args, **kwargs):
            # ... function code ...
            return result

    Args:
        func: The function to be decorated (when used without arguments).
        logger: Optional logger instance to use. If not provided, uses default logger.

    Returns:
        Decorated function that logs execution time.

    Example:
        >>> @elapsed_time
        ... def add(a, b):
        ...     return a + b
        ...
        >>> add(2, 3)
        INFO: Function call completed in 0.001 seconds

        >>> custom_logger = logging.getLogger("my_app")
        >>> @elapsed_time(logger=custom_logger)
        ... def multiply(a, b):
        ...     return a * b
    """
    # Determine which logger to use
    _log = logger if logger is not None else _default_logger

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            function_name = f.__name__
            # Get module and function info from the decorated function
            module_name = getattr(f, "__module__", "unknown")
            func_name = getattr(f, "__name__", "unknown")

            def _log_with_logger(level: int, message: str) -> None:
                """Log with correct module and function name."""
                # Use extra parameter to pass module and function info with custom keys
                # The formatter will use these if available
                extra = {
                    "decorated_module": module_name.split(".")[-1] if module_name else "unknown",
                    "decorated_funcName": func_name,
                }
                _log.log(level, message, extra=extra)

            try:
                result = f(*args, **kwargs)
                elapsed_time = time.time() - start_time
                _log_with_logger(
                    logging.INFO,
                    f"Function call completed in {elapsed_time:.3f} seconds",
                )
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                _log_with_logger(
                    logging.ERROR,
                    f"Function call failed after {elapsed_time:.3f} seconds: {e}",
                )
                raise

        return wrapper

    # Support both @elapsed_time and @elapsed_time(logger=...)
    if func is None:
        return decorator
    else:
        return decorator(func)


@overload
def elapsed_time_with_params(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def elapsed_time_with_params(
    *, logger: Optional[logging.Logger] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def elapsed_time_with_params(
    func: Optional[Callable[P, R]] = None, *, logger: Optional[logging.Logger] = None
) -> Union[Callable[P, R], Callable[[Callable[P, R]], Callable[P, R]]]:
    """Decorator to log inference time with function parameters.

    Similar to elapsed_time, but also logs function arguments
    (excluding sensitive data like passwords) for better debugging.

    Usage:
        # With default logger
        @elapsed_time_with_params
        def my_inference_function(prompt):
            # ... inference code ...
            return result

        # With custom logger
        custom_logger = logging.getLogger("my_logger")
        @elapsed_time_with_params(logger=custom_logger)
        def my_inference_function(prompt):
            # ... inference code ...
            return result

    Args:
        func: The function to be decorated (when used without arguments).
        logger: Optional logger instance to use. If not provided, uses default logger.

    Returns:
        Decorated function that logs execution time with parameters.

    Example:
        >>> @elapsed_time_with_params
        ... def generate_with_model(model, prompt):
        ...     return llm.invoke(prompt)
        ...
        >>> generate_with_model("qwen-qwq", "Hello")
        INFO: Inference 'generate_with_model' started with model='qwen-qwq'
        INFO: Inference 'generate_with_model' completed in 1.234 seconds
    """
    # Determine which logger to use
    log = logger if logger is not None else _default_logger

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            function_name = f.__name__
            # Get module and function info from the decorated function
            module_name = getattr(f, "__module__", "unknown")
            func_name = getattr(f, "__name__", "unknown")

            def _log_with_logger(level: int, message: str) -> None:
                """Log with correct module and function name."""
                # Use extra parameter to pass module and function info with custom keys
                # The formatter will use these if available
                extra = {
                    "decorated_module": module_name.split(".")[-1] if module_name else "unknown",
                    "decorated_funcName": func_name,
                }
                log.log(level, message, extra=extra)

            # Log function call with safe arguments (exclude sensitive data)
            sensitive_keys = {"password", "scayle_password", "token", "_token"}
            safe_kwargs = {k: v if k not in sensitive_keys else "***" for k, v in kwargs.items()}
            safe_args = [
                arg if not isinstance(arg, str) or "password" not in str(arg).lower() else "***"
                for arg in args
            ]

            _log_with_logger(
                logging.INFO,
                f"Inference '{function_name}' started with args={safe_args}, kwargs={safe_kwargs}",
            )

            try:
                result = f(*args, **kwargs)
                elapsed_time = time.time() - start_time
                _log_with_logger(
                    logging.INFO,
                    f"Inference '{function_name}' completed in {elapsed_time:.3f} seconds",
                )
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                _log_with_logger(
                    logging.ERROR,
                    f"Inference '{function_name}' failed after {elapsed_time:.3f} seconds: {e}",
                )
                raise

        return wrapper

    # Support both @elapsed_time_with_params and @elapsed_time_with_params(logger=...)
    if func is None:
        return decorator
    else:
        return decorator(func)


def format_tool_for_openai_api(tool_obj: BaseTool) -> dict:
    """Convert LangChain tool to OpenAI-compatible tool format.

    Args:
        tool_obj: LangChain tool object decorated with @tool.

    Returns:
        Dictionary in OpenAI-compatible tool format.
    """
    # Check if args_schema is a Pydantic model (has model_fields attribute)
    args_schema = tool_obj.args_schema
    # Use getattr to safely access model_fields, which exists on Pydantic models
    model_fields = getattr(args_schema, "model_fields", None) if args_schema else None
    if model_fields:
        param_names = list(model_fields.keys())
    else:
        param_names = []

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
                    for param_name in param_names
                },
                "required": param_names,
            },
        },
    }


def load_configuration() -> dict:
    """Load configuration from environment variables.

    Returns:
        Dictionary containing configuration:
        - base_url: Base URL for Scayle API
        - username: Scayle username
        - password: Scayle password
        - verify_ssl: Whether to verify SSL certificates

    Raises:
        ValueError: If required environment variables are missing.
    """
    load_dotenv()

    base_url = os.getenv("SCAYLE_BASE_URL", "https://chat.scayle.es/api")
    username = os.getenv("SCAYLE_USERNAME")
    password = os.getenv("SCAYLE_PASSWORD")
    verify_ssl = os.getenv("SCAYLE_VERIFY_SSL", "true").lower() == "true"

    # Ensure base_url ends with /api if needed
    if "/api" not in base_url and not base_url.rstrip("/").endswith("/v1"):
        if not base_url.endswith("/"):
            base_url = f"{base_url}/api"
        else:
            base_url = f"{base_url}api"

    if not username or not password:
        raise ValueError(
            "Missing required environment variables: 'username' and 'password' must be set in .env file"
        )

    return {
        "base_url": base_url,
        "username": username,
        "password": password,
        "verify_ssl": verify_ssl,
    }
