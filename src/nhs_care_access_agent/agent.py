"""The bounded, read-only agent loop."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .mcp_client import McpToolClient
from .providers import ModelProvider
from .tracing import AgentTrace, JsonlTraceSink, ToolTrace, elapsed_ms, new_trace
from .types import ChatMessage, ToolCall

SYSTEM_PROMPT = """You are an NHS care-access research assistant.

Use only the supplied NHS MCP tools for factual claims about waiting times,
trends, and CQC ratings. Do not invent provider codes, ratings, waits, dates, or
data coverage. Prefer trust_profile for a joined trust + specialty question.
If the tool data is insufficient, say what is missing and offer a safe next
step. This is research support, not clinical advice or a treatment recommendation.

In every factual answer, include a short `Sources:` line naming the tools you
actually used. Keep the answer concise and distinguish current waits from
historical trends."""


@dataclass(frozen=True)
class AgentRun:
    answer: str
    trace: AgentTrace


class CareAccessAgent:
    """Executes a model-selected sequence of permitted, read-only MCP tools."""

    def __init__(
        self,
        model: ModelProvider,
        tools: McpToolClient,
        *,
        backend: str,
        max_tool_calls: int = 8,
        max_turns: int = 12,
        trace_sink: JsonlTraceSink | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._backend = backend
        self._max_tool_calls = max_tool_calls
        self._max_turns = max_turns
        self._trace_sink = trace_sink

    async def run(self, question: str) -> AgentRun:
        started = time.perf_counter()
        trace = new_trace(uuid.uuid4().hex, self._backend, self._model.model_name, question)
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=question),
        ]

        async with self._tools as tool_client:
            tool_definitions = await tool_client.list_tools()
            allowed_tools = {tool.name for tool in tool_definitions}
            for _ in range(self._max_turns):
                trace.model_turns += 1
                reply = await self._model.complete(messages, tool_definitions)
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=reply.content,
                        tool_calls=reply.tool_calls,
                        provider_payload=reply.provider_payload,
                    )
                )
                if not reply.tool_calls:
                    trace.final_answer = reply.content
                    trace.stop_reason = "final_answer"
                    return self._finish(trace, started)

                for call in reply.tool_calls:
                    if len(trace.tools) >= self._max_tool_calls:
                        return self._limit_reached(trace, started, "tool_call_budget")
                    result, blocked = await self._invoke_tool(tool_client, allowed_tools, call)
                    duration_ms = result.pop("_duration_ms")
                    trace.tools.append(
                        ToolTrace(
                            call_id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                            result=result,
                            duration_ms=duration_ms,
                            blocked=blocked,
                        )
                    )
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps(result, sort_keys=True),
                            tool_name=call.name,
                            tool_call_id=call.id,
                        )
                    )

        return self._limit_reached(trace, started, "turn_budget")

    async def _invoke_tool(
        self, tool_client: McpToolClient, allowed_tools: set[str], call: ToolCall
    ) -> tuple[dict[str, Any], bool]:
        started = time.perf_counter()
        if call.name not in allowed_tools:
            return (
                {
                    "error": f"Tool '{call.name}' is not in the MCP allowlist.",
                    "_duration_ms": elapsed_ms(started),
                },
                True,
            )
        try:
            payload = await tool_client.call_tool(call.name, call.arguments)
        except Exception as error:  # noqa: BLE001 - tool errors must be returned to the model.
            payload = {"error": f"Tool execution failed: {error}"}
        payload = dict(payload)
        payload["_duration_ms"] = elapsed_ms(started)
        return payload, False

    def _limit_reached(self, trace: AgentTrace, started: float, reason: str) -> AgentRun:
        trace.final_answer = (
            "I could not complete this safely within the configured research-tool budget. "
            "Please narrow the question or review the recorded trace."
        )
        trace.stop_reason = reason
        return self._finish(trace, started)

    def _finish(self, trace: AgentTrace, started: float) -> AgentRun:
        trace.total_duration_ms = elapsed_ms(started)
        if self._trace_sink:
            self._trace_sink.write(trace)
        return AgentRun(answer=trace.final_answer, trace=trace)
