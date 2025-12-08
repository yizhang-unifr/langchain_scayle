"""Tests for model status checking functionality."""

import pytest
from src.langchain_scayle.llm import ScayleLLM


def test_check_model_status_structure(scayle_llm):
    """Test that check_model_status returns the correct structure."""
    if not scayle_llm.check_connection():
        pytest.skip("Cannot connect to Scayle API")

    status = scayle_llm.check_model_status(test_prompt="Hello")

    # Check structure
    assert "total_models" in status
    assert "working_models" in status
    assert "unavailable_models" in status
    assert "models" in status

    # Check types
    assert isinstance(status["total_models"], int)
    assert isinstance(status["working_models"], list)
    assert isinstance(status["unavailable_models"], list)
    assert isinstance(status["models"], dict)

    # Check consistency
    assert status["total_models"] == len(status["models"])
    assert len(status["working_models"]) + len(status["unavailable_models"]) <= status[
        "total_models"
    ]


def test_check_model_status_working_models(scayle_llm):
    """Test that working models are correctly identified."""
    if not scayle_llm.check_connection():
        pytest.skip("Cannot connect to Scayle API")

    status = scayle_llm.check_model_status(test_prompt="Hello")

    # Should have at least some models
    assert status["total_models"] > 0

    # Check each model entry
    for model_id, model_info in status["models"].items():
        assert "status" in model_info
        assert "model_id" in model_info
        assert model_info["model_id"] == model_id
        assert model_info["status"] in ["working", "unavailable", "error"]

        if model_info["status"] == "working":
            assert model_id in status["working_models"]
        elif model_info["status"] == "unavailable":
            assert model_id in status["unavailable_models"]


def test_check_model_status_custom_prompt(scayle_llm):
    """Test check_model_status with a custom prompt."""
    if not scayle_llm.check_connection():
        pytest.skip("Cannot connect to Scayle API")

    custom_prompt = "Test prompt for status check"
    status = scayle_llm.check_model_status(test_prompt=custom_prompt)

    assert status["total_models"] > 0


def test_get_models(scayle_llm):
    """Test getting available models."""
    if not scayle_llm.check_connection():
        pytest.skip("Cannot connect to Scayle API")

    models = scayle_llm._get_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(model, str) for model in models)

