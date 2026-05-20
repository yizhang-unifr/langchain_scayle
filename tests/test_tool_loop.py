"""Tests for ScayleLLM tool-call loop handling."""

import json
from unittest.mock import patch

from src.langchain_scayle.llm import ScayleLLM


def _llm() -> ScayleLLM:
    return ScayleLLM(
        scayle_username="user",
        scayle_password="pass",
        base_url="https://example.test/api",
        model="qwen3",
        max_tool_rounds=3,
    )


def test_complete_with_tool_loop_returns_final_text():
    llm = _llm()
    tool_call_response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "query_knowledge_bases",
                                "arguments": json.dumps(
                                    {"query": "temperature", "count": 5}
                                ),
                            },
                        }
                    ],
                },
            }
        ]
    }
    final_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"SQL_pattern": "SELECT 1"}',
                },
            }
        ]
    }

    with patch.object(llm, "_chat_completion", side_effect=[tool_call_response, final_response]):
        text = llm._complete_with_tool_loop([{"role": "user", "content": "test"}])

    assert text == '{"SQL_pattern": "SELECT 1"}'


def test_generate_never_passes_none_to_generation():
    llm = _llm()
    final_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ]
    }

    with patch.object(llm, "_authenticate"), patch.object(
        llm, "_chat_completion", return_value=final_response
    ):
        result = llm.generate(["Say hello"])

    assert result.generations[0][0].text == "hello"


def test_custom_tool_executor_is_used():
    llm = _llm()
    tool_call_response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Zurich"}),
                            },
                        }
                    ],
                },
            }
        ]
    }
    final_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
            }
        ]
    }
    seen: list[tuple[str, dict]] = []

    def executor(name: str, arguments: dict) -> str:
        seen.append((name, arguments))
        return "sunny"

    with patch.object(llm, "_chat_completion", side_effect=[tool_call_response, final_response]):
        llm._complete_with_tool_loop(
            [{"role": "user", "content": "weather?"}],
            tool_executor=executor,
        )

    assert seen == [("get_weather", {"city": "Zurich"})]
