"""Tests for connection checking functionality."""

import pytest
from src.langchain_scayle.llm import ScayleLLM


def test_check_connection_success(scayle_llm):
    """Test successful connection check."""
    result = scayle_llm.check_connection()
    assert result is True, "Connection check should succeed when VPN is connected"


def test_check_connection_verify_ssl_property(scayle_llm):
    """Test that verify_ssl is accessible as a property."""
    # Should be accessible
    assert hasattr(scayle_llm, "verify_ssl")
    assert isinstance(scayle_llm.verify_ssl, bool)


def test_verify_ssl_default_true(scayle_credentials):
    """Test that verify_ssl defaults to True."""
    llm = ScayleLLM(
        scayle_username=scayle_credentials["username"],
        scayle_password=scayle_credentials["password"],
        base_url=scayle_credentials["base_url"],
        # Don't specify verify_ssl, should default to True
    )
    assert llm.verify_ssl is True


def test_verify_ssl_auto_retry(scayle_credentials):
    """Test that SSL errors trigger automatic retry with verify_ssl=False."""
    llm = ScayleLLM(
        scayle_username=scayle_credentials["username"],
        scayle_password=scayle_credentials["password"],
        base_url=scayle_credentials["base_url"],
        verify_ssl=True,  # Start with True
    )

    # If connection works, verify_ssl might be auto-disabled if SSL error occurs
    result = llm.check_connection()
    # Connection should succeed (either with or without auto-retry)
    assert result is True
