"""
langchain_scayle: LangChain integration for Scayle Chat API.

This module provides a Scayle LLM class similar to langchain_community.llms.Ollama,
allowing seamless integration with LangChain workflows.
"""

from typing import Any, Dict, List, Optional
import requests
import urllib3
from pydantic import Field, ConfigDict

# Suppress SSL warnings when verify_ssl=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult


class ScayleLLM(BaseLLM):
    """Scayle Chat API language model integration."""

    scayle_username: str = Field(..., description="Scayle username for authentication.")
    scayle_password: str = Field(..., description="Scayle password for authentication.")
    model: str = Field(default="", description="The Scayle model ID to use.")
    base_url: str = Field(
        default="https://chat.scayle.es/api", description="Base URL for the Scayle API."
    )
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum number of tokens to generate."
    )
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter.")
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates. Set to False for self-signed certs.",
    )
    timeout: Optional[float] = Field(default=30.0, description="Request timeout in seconds.")
    # Add other parameters as needed, e.g., top_k, frequency_penalty, etc.
    # These can be passed to the API if supported.

    _token: Optional[str] = None
    _models: Optional[List[str]] = None
    _verify_ssl_auto_disabled: bool = False  # Track if we auto-disabled SSL verification

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Do not auto-authenticate in __init__ to avoid blocking; defer to first use
        pass

    def _make_request_with_ssl_retry(
        self, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        """
        Make a request with automatic retry using verify_ssl=False if SSLError occurs.

        Args:
            method: HTTP method ('get', 'post', etc.)
            url: URL to request
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response object

        Raises:
            ValueError: If request fails after retry
        """
        # Get current verify_ssl setting from kwargs or use instance value
        verify = kwargs.pop("verify", self.verify_ssl)

        try:
            # Try with current verify_ssl setting
            response = getattr(requests, method.lower())(url, verify=verify, **kwargs)
            return response
        except requests.exceptions.SSLError as e:
            # If verify_ssl was True and we got SSLError, retry with verify_ssl=False
            if verify and not self._verify_ssl_auto_disabled:
                # Auto-disable SSL verification and retry
                # Update the Field value directly
                object.__setattr__(self, "verify_ssl", False)
                self._verify_ssl_auto_disabled = True
                try:
                    response = getattr(requests, method.lower())(url, verify=False, **kwargs)
                    return response
                except Exception as retry_error:
                    raise ValueError(
                        f"SSL verification failed and retry with verify_ssl=False also failed. "
                        f"Original error: {e}. Retry error: {retry_error}"
                    ) from retry_error
            else:
                # Already tried with verify=False or verify was already False
                raise ValueError(f"SSL verification failed. Error: {e}") from e

    def check_connection(self) -> bool:
        """
        Check if the base_url is reachable (VPN connection check).

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            # Try to connect to the base_url with a simple request
            # Use a shorter timeout for connection check
            check_timeout = min(10.0, self.timeout or 10.0)
            response = self._make_request_with_ssl_retry(
                "get",
                self.base_url,
                timeout=check_timeout,
                allow_redirects=True,
            )
            # Any response (even 404, 401, etc.) means the server is reachable
            print("Successfully connected")
            return True
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError):
            print("Connecting failed! Please check your VPN connection first!")
            return False
        except Exception:
            print("Connecting failed! Please check your VPN connection first!")
            return False

    def _authenticate(self) -> None:
        """Authenticate with Scayle API and retrieve token using LDAP endpoint."""
        if self._token is not None:
            return
        url = f"{self.base_url}/v1/auths/ldap"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        data = {
            "user": self.scayle_username,
            "password": self.scayle_password,
        }
        try:
            response = self._make_request_with_ssl_retry(
                "post",
                url,
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            raise ValueError(
                f"Connection timeout after {self.timeout} seconds. "
                f"Please check your network connection and base_url. Error: {e}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise ValueError(
                f"Failed to connect to {url}. Please check your base_url and network connection. "
                f"Error: {e}"
            ) from e
        if response.status_code != 200:
            raise ValueError(
                f"Failed to authenticate with Scayle API. Status code: {response.status_code}. "
                f"Response: {response.text}. Please check your credentials."
            )
        self._token = response.json()["token"]

    def _get_models(self) -> List[str]:
        """Retrieve available models from Scayle API."""
        if self._models is not None:
            return self._models
        self._authenticate()  # Ensure token is available
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            response = self._make_request_with_ssl_retry(
                "get",
                url,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            raise ValueError(
                f"Connection timeout after {self.timeout} seconds. "
                f"Please check your network connection and base_url. Error: {e}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise ValueError(
                f"Failed to connect to {url}. Please check your base_url and network connection. "
                f"Error: {e}"
            ) from e
        if response.status_code != 200:
            raise ValueError(
                f"Failed to retrieve models from Scayle API. Status code: {response.status_code}. "
                f"Response: {response.text}"
            )
        self._models = [model["id"] for model in response.json()["data"]]
        return self._models

    def check_model_status(
        self, test_prompt: str = "Hello", test_timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check the status of all available models by testing each one.

        Args:
            test_prompt: The prompt to use for testing each model. Defaults to "Hello".
            test_timeout: Optional timeout for testing each model. Uses self.timeout if not provided.

        Returns:
            A dictionary containing:
            - 'total_models': Total number of models found
            - 'working_models': List of model IDs that are working
            - 'unavailable_models': List of model IDs that are unavailable
            - 'models': Dictionary with detailed status for each model
                - Each model entry contains:
                  - 'status': 'working', 'unavailable', or 'error'
                  - 'error': Error message if status is not 'working'
        """
        self._authenticate()  # Ensure authentication
        models = self._get_models()
        timeout = test_timeout if test_timeout is not None else self.timeout

        result: Dict[str, Any] = {
            "total_models": len(models),
            "working_models": [],
            "unavailable_models": [],
            "models": {},
        }

        original_model = self.model
        for model_id in models:
            self.model = model_id
            model_status: Dict[str, Any] = {"status": "unknown", "model_id": model_id}

            try:
                # Try to generate a response with the test prompt
                response = self._call_api(test_prompt)
                model_status["status"] = "working"
                result["working_models"].append(model_id)
            except ValueError as e:
                error_msg = str(e)
                if "Connection error" in error_msg or "InternalServerError" in error_msg:
                    model_status["status"] = "unavailable"
                    model_status["error"] = "Connection/internal server error"
                    result["unavailable_models"].append(model_id)
                else:
                    model_status["status"] = "error"
                    model_status["error"] = error_msg
            except Exception as e:
                model_status["status"] = "error"
                model_status["error"] = str(e)

            result["models"][model_id] = model_status

        # Restore original model
        self.model = original_model

        return result

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "scayle_username": self.scayle_username,
            "model": self.model,
            "base_url": self.base_url,
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of the language model."""
        return "scayle"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for the given prompts."""
        self._authenticate()  # Ensure token is available

        if not self.model:
            available_models = self._get_models()
            if not available_models:
                raise ValueError("No models available from Scayle API.")
            self.model = available_models[0]  # Default to first model if not specified

        generations = []
        # Extract tools from kwargs if provided
        tools = kwargs.get("tools", None)
        for prompt in prompts:
            response = self._call_api(prompt, tools=tools)
            text = response["choices"][0]["message"]["content"]
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    def _call_api(
        self, message: str, tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Call the Scayle Chat Completions API."""
        self._authenticate()  # Ensure token is fresh

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            # Add other params if supported by Scayle API
        }
        # Add tools if provided
        if tools is not None:
            data["tools"] = tools
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        try:
            response = self._make_request_with_ssl_retry(
                "post",
                url,
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            raise ValueError(
                f"Connection timeout after {self.timeout} seconds. "
                f"Please check your network connection and base_url. Error: {e}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise ValueError(
                f"Failed to connect to {url}. Please check your base_url and network connection. "
                f"Error: {e}"
            ) from e
        if response.status_code != 200:
            error_detail = response.text
            # Provide more helpful error messages
            if response.status_code == 400:
                if "Connection error" in error_detail or "InternalServerError" in error_detail:
                    raise ValueError(
                        f"Model '{self.model}' is not available or experiencing connection issues. "
                        f"Status code: {response.status_code}. Response: {error_detail}"
                    )
            raise ValueError(
                f"Failed to generate response from Scayle API. Status code: {response.status_code}. "
                f"Response: {error_detail}"
            )
        return response.json()
