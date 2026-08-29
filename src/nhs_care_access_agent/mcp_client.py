"""MCP transport adapters.

The agent only relies on this small protocol. The production adapter uses the
standard MCP Python client over stdio; tests can use an in-memory implementation.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

from .types import ToolDefinition


class McpToolClient(AbstractAsyncContextManager["McpToolClient"], Protocol):
    async def list_tools(self) -> list[ToolDefinition]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class StdioMcpToolClient:
    """Connect to ``nhs-intelligence-mcp`` over standard input/output."""

    def __init__(self, command: str, args: tuple[str, ...], cwd: Path | None = None) -> None:
        self._command = command
        self._args = args
        self._cwd = cwd
        self._session: Any = None
        self._stack: Any = None

    async def __aenter__(self) -> Self:
        # Imported lazily so deterministic evaluator unit tests do not need the
        # MCP package or a running NHS server.
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        transport = await self._stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=self._command,
                    args=list(self._args),
                    cwd=str(self._cwd) if self._cwd else None,
                )
            )
        )
        read_stream, write_stream = transport
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._session = None
        self._stack = None

    async def list_tools(self) -> list[ToolDefinition]:
        self._require_session()
        response = await self._session.list_tools()
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema),
            )
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._require_session()
        response = await self._session.call_tool(name, arguments)
        if getattr(response, "isError", False):
            return {"error": _content_to_text(response.content)}
        return _content_to_payload(response.content)

    def _require_session(self) -> None:
        if self._session is None:
            raise RuntimeError("MCP session is not connected")


def _content_to_payload(content: list[Any]) -> dict[str, Any]:
    """Keep structured content when available and degrade text safely."""
    if len(content) == 1:
        item = content[0]
        text = getattr(item, "text", None)
        if isinstance(text, str):
            try:
                import json

                parsed = json.loads(text)
            except ValueError:
                return {"text": text}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"content": _content_to_text(content)}


def _content_to_text(content: list[Any]) -> str:
    return "\n".join(str(getattr(item, "text", item)) for item in content)
