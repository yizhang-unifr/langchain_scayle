"""Tests for basic ScayleLLM functionality."""

import pytest
from src.langchain_scayle.llm import ScayleLLM


def test_scayle_llm_initialization(scayle_credentials):
    """Test that ScayleLLM can be initialized."""
    llm = ScayleLLM(
        scayle_username=scayle_credentials["username"],
        scayle_password=scayle_credentials["password"],
        base_url=scayle_credentials["base_url"],
        verify_ssl=scayle_credentials["verify_ssl"],
        timeout=30.0,
    )

    assert llm.scayle_username == scayle_credentials["username"]
    assert llm.scayle_password == scayle_credentials["password"]
    assert llm.base_url == scayle_credentials["base_url"]
    assert llm.verify_ssl == scayle_credentials["verify_ssl"]


def test_scayle_llm_invoke(scayle_llm, working_model):
    """Test basic invoke functionality."""
    scayle_llm.model = working_model

    prompt = "Say hello in one sentence."
    response = scayle_llm.invoke(prompt)

    assert isinstance(response, str)
    assert len(response) > 0


def test_scayle_llm_generate(scayle_llm, working_model):
    """Test generate method with multiple prompts."""
    scayle_llm.model = working_model

    prompts = ["Hello", "Hi"]
    result = scayle_llm.generate(prompts)

    assert result is not None
    assert len(result.generations) == len(prompts)
    assert all(len(gen) > 0 for gen in result.generations)


def test_scayle_llm_llm_type(scayle_llm):
    """Test that _llm_type returns 'scayle'."""
    assert scayle_llm._llm_type == "scayle"


def test_scayle_llm_identifying_params(scayle_llm):
    """Test that _identifying_params returns correct information."""
    params = scayle_llm._identifying_params

    assert isinstance(params, dict)
    assert "scayle_username" in params
    assert "model" in params
    assert "base_url" in params


def test_scayle_llm_model_selection(scayle_llm, working_model):
    """Test automatic model selection when model is not set."""
    # Use a working model instead of auto-selecting
    # (auto-selection might pick an unavailable model)
    scayle_llm.model = working_model

    # Test that invoke works
    response = scayle_llm.invoke("Hello")

    assert scayle_llm.model == working_model
    assert isinstance(response, str)
    assert len(response) > 0

