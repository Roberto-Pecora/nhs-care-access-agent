"""Provider-neutral types shared by the agent, model adapters, and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolDefinition:
    """A model-facing view of one MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A provider-neutral request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    """A normalised conversation message.

    ``provider_payload`` retains provider-specific response state when needed. In
    particular, Gemini thought signatures must be sent back unchanged in later
    function-calling turns.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str | None = None
    tool_call_id: str | None = None
    provider_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelReply:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    provider_payload: dict[str, Any] | None = None
    model_name: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    name: str
    call_id: str
    payload: dict[str, Any]
    is_error: bool = False
