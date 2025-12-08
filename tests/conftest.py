"""Pytest configuration and fixtures for ScayleLLM tests."""

import os
import pytest
from dotenv import load_dotenv
from src.langchain_scayle.llm import ScayleLLM

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def scayle_credentials():
    """Get Scayle credentials from environment variables."""
    base_url = os.getenv("base_url", "https://chat.scayle.es/api")
    username = os.getenv("username")
    password = os.getenv("password")
    verify_ssl = os.getenv("verify_ssl", "true").lower() == "true"

    # Ensure base_url ends with /api
    if "/api" not in base_url and not base_url.rstrip("/").endswith("/v1"):
        if not base_url.endswith("/"):
            base_url = f"{base_url}/api"
        else:
            base_url = f"{base_url}api"

    if not username or not password:
        pytest.skip("Missing Scayle credentials in environment variables")

    return {
        "base_url": base_url,
        "username": username,
        "password": password,
        "verify_ssl": verify_ssl,
    }


@pytest.fixture(scope="session")
def scayle_llm(scayle_credentials):
    """Create a ScayleLLM instance for testing."""
    llm = ScayleLLM(
        scayle_username=scayle_credentials["username"],
        scayle_password=scayle_credentials["password"],
        base_url=scayle_credentials["base_url"],
        verify_ssl=scayle_credentials["verify_ssl"],
        timeout=30.0,
    )
    return llm


@pytest.fixture(scope="session")
def working_model(scayle_llm):
    """Get a working model for testing."""
    # Check connection first
    if not scayle_llm.check_connection():
        pytest.skip("Cannot connect to Scayle API - check VPN connection")

    # Get model status
    status = scayle_llm.check_model_status(test_prompt="Hello")
    if not status["working_models"]:
        pytest.skip("No working models available")

    model_id = status["working_models"][0]
    scayle_llm.model = model_id
    return model_id
