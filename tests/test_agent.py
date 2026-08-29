from __future__ import annotations

import asyncio
from typing import Any

from nhs_care_access_agent.agent import CareAccessAgent
from nhs_care_access_agent.types import ChatMessage, ModelReply, ToolCall, ToolDefinition


class FakeTools:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def list_tools(self):
        return [
            ToolDefinition(
                name="trust_profile",
                description="joined profile",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        assert name == "trust_profile"
        assert arguments == {"identifier": "EX1", "specialty": "Cardiology"}
        return {"found": True, "current_wait_weeks": 12, "overall_rating": "Good"}


class TwoTurnModel:
    model_name = "fake"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="trust_profile",
                        arguments={"identifier": "EX1", "specialty": "Cardiology"},
                    ),
                ),
            )
        return ModelReply("Example Trust has a 12 weeks wait and a Good rating. Sources: trust_profile")


def test_agent_runs_allowed_tool_and_records_trace():
    agent = CareAccessAgent(TwoTurnModel(), FakeTools(), backend="test")
    run = asyncio.run(agent.run("What is the profile?"))

    assert "12 weeks" in run.answer
    assert run.trace.stop_reason == "final_answer"
    assert [tool.name for tool in run.trace.tools] == ["trust_profile"]
    assert run.trace.tools[0].blocked is False
