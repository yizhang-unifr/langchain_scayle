"""
langchain_scayle: LangChain integration for Scayle Chat API (OpenAI-compatible interface).

This module provides a ScayleOpenAI class that implements all functionalities
from ScayleLLM, providing an OpenAI-compatible interface for Scayle Chat API.
"""

from typing import Any, Dict, List, Optional
import requests
import urllib3
from pydantic import Field, ConfigDict

# Suppress SSL warnings when verify_ssl=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from langchain_openai import ChatOpenAI
except ImportError as e:
    raise ImportError(
        "langchain_openai is required for ScayleOpenAI. "
        "Please install it with: pip install langchain-openai"
    ) from e


class ScayleOpenAI(ChatOpenAI):
    """
    Scayle Chat API language model integration with OpenAI-compatible interface. This class is an alternative to ScayleLLM.
    
    This class extends the ChatOpenAI class from langchain_openai and provides a Scayle-specific
    implementation for the OpenAI-compatible interface.
    
    It handles authentication with Scayle API, model retrieval, and SSL verification.
    parameters:
        scayle_username: Scayle username for authentication.
        scayle_password: Scayle password for authentication.
        base_url: Base URL for the Scayle API.
        verify_ssl: Whether to verify SSL certificates. Set to False for self-signed certs.
        timeout: Request timeout in seconds.
    attributes:
        _token: Authentication token.
        _models: List of available models.
        _verify_ssl_auto_disabled: Track if we auto-disabled SSL verification.
    methods:
        check_connection: Check if the base_url is reachable (VPN connection check).
        check_model_status: Check the status of all available models by testing each one.
        _prepare_client_params: Prepare parameters for client initialization.
        _make_request_with_ssl_retry: Make a request with automatic retry using verify_ssl=False if SSLError occurs.
        _authenticate: Authenticate with Scayle API and retrieve token using LDAP endpoint.
        _get_models: Retrieve available models from Scayle API.
    """

    scayle_username: str = Field(..., description="Scayle username for authentication.")
    scayle_password: str = Field(..., description="Scayle password for authentication.")
    base_url: str = Field(
        default="https://chat.scayle.es/api", description="Base URL for the Scayle API."
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates. Set to False for self-signed certs.",
    )
    timeout: Optional[float] = Field(default=30.0, description="Request timeout in seconds.")

    _token: Optional[str] = None
    _models: Optional[List[str]] = None
    _verify_ssl_auto_disabled: bool = False  # Track if we auto-disabled SSL verification

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any):
        # Extract Scayle-specific parameters
        scayle_username = kwargs.pop("scayle_username", None)
        scayle_password = kwargs.pop("scayle_password", None)
        base_url = kwargs.pop("base_url", "https://chat.scayle.es/api")
        verify_ssl = kwargs.pop("verify_ssl", True)
        timeout = kwargs.pop("timeout", 30.0)

        # Set default model if not provided
        if "model" not in kwargs or not kwargs.get("model"):
            # Will be set after authentication
            kwargs["model"] = ""

        # Initialize ChatOpenAI with base_url and a placeholder api_key
        # We'll override the client to use our authentication token
        super().__init__(
            base_url=base_url,
            api_key=None,  # Will be replaced with token in client
            **kwargs,
        )

        # Set Scayle-specific attributes
        object.__setattr__(self, "scayle_username", scayle_username)
        object.__setattr__(self, "scayle_password", scayle_password)
        object.__setattr__(self, "verify_ssl", verify_ssl)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "_token", None)
        object.__setattr__(self, "_models", None)
        object.__setattr__(self, "_verify_ssl_auto_disabled", False)

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
                # Try to generate a response with the test prompt using invoke
                response = self.invoke(test_prompt)
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

    def _prepare_client_params(self) -> Dict[str, Any]:
        """Prepare parameters for client initialization."""
        # Ensure we're authenticated
        self._authenticate()

        # If no model is set, get the first available model
        if not self.model:
            available_models = self._get_models()
            if not available_models:
                raise ValueError("No models available from Scayle API.")
            self.model = available_models[0]

        return {
            "api_key": self._token,
            "base_url": self.base_url,
        }

    @property
    def _client(self) -> Any:
        """Override client property to use Scayle authentication token."""
        # Prepare client parameters
        client_params = self._prepare_client_params()

        # Get the parent client (ChatOpenAI has _client property)
        client = super()._client  # type: ignore[attr-defined]

        # Update the API key with our token
        if hasattr(client, "api_key"):
            client.api_key = client_params["api_key"]

        # Update base URL if needed
        if hasattr(client, "base_url"):
            client.base_url = client_params["base_url"]

        # Configure SSL verification for httpx client
        if hasattr(client, "_client"):
            # For OpenAI client, we need to configure the httpx client
            http_client = getattr(client, "_client", None)
            if http_client:
                # Try to set verify on the httpx client
                if hasattr(http_client, "verify"):
                    http_client.verify = self.verify_ssl
                # Also try to configure through transport if available
                if hasattr(http_client, "transport") and hasattr(http_client.transport, "verify"):
                    http_client.transport.verify = self.verify_ssl

        return client
