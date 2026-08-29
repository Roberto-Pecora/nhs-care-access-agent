from __future__ import annotations

from nhs_care_access_agent.providers import _gemini_content, _openai_tool_call
from nhs_care_access_agent.types import ChatMessage


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
