from __future__ import annotations

import urllib.error

from nhs_care_access_agent.providers import (
    _anthropic_message,
    _gemini_content,
    _openai_tool_call,
    _retry_delay,
)
from nhs_care_access_agent.types import ChatMessage, ToolCall


def _http_error(headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example/x", code=429, msg="Too Many Requests", hdrs=headers or {}, fp=None
    )


def test_gemini_tool_result_preserves_call_id_and_structured_payload():
    content = _gemini_content(
        ChatMessage(
            role="tool",
            tool_name="trust_profile",
            tool_call_id="gemini-call-1",
            content='{"found": true, "wait_weeks": 12}',
        )
    )

    response = content["parts"][0]["functionResponse"]
    assert content["role"] == "user"
    assert response["id"] == "gemini-call-1"
    assert response["name"] == "trust_profile"
    assert response["response"]["result"] == {"found": True, "wait_weeks": 12}


def test_anthropic_tool_result_maps_to_tool_result_block():
    block = _anthropic_message(
        ChatMessage(
            role="tool",
            tool_name="trust_profile",
            tool_call_id="toolu_1",
            content='{"found": true}',
        )
    )

    assert block["role"] == "user"
    result = block["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_1"
    assert result["content"] == '{"found": true}'


def test_anthropic_assistant_tool_call_maps_to_tool_use_block():
    block = _anthropic_message(
        ChatMessage(
            role="assistant",
            content="Looking that up.",
            tool_calls=(ToolCall(id="toolu_2", name="lookup_wait_time", arguments={"provider": "X"}),),
        )
    )

    assert block["role"] == "assistant"
    types = [part["type"] for part in block["content"]]
    assert types == ["text", "tool_use"]
    call = block["content"][1]
    assert call["id"] == "toolu_2"
    assert call["name"] == "lookup_wait_time"
    assert call["input"] == {"provider": "X"}


def test_openai_tool_call_parses_json_arguments():
    call = _openai_tool_call(
        {
            "id": "call-1",
            "function": {
                "name": "trust_profile",
                "arguments": '{"identifier":"RGT","specialty":"Cardiology"}',
            },
        }
    )

    assert call.name == "trust_profile"
    assert call.arguments == {"identifier": "RGT", "specialty": "Cardiology"}


def test_retry_delay_prefers_body_retry_delay():
    body = '{"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "50s"}]}}'
    assert _retry_delay(_http_error(), body, attempt=0) == 50.0


def test_retry_delay_honours_retry_after_header():
    assert _retry_delay(_http_error({"Retry-After": "12"}), "", attempt=3) == 12.0


def test_retry_delay_falls_back_to_exponential_backoff():
    assert _retry_delay(_http_error(), "no hint here", attempt=3) == 8.0


def test_retry_delay_is_capped():
    body = '{"retryDelay": "9999s"}'
    assert _retry_delay(_http_error(), body, attempt=0) == 65.0
