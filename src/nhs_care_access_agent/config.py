"""Environment-driven configuration with no provider secrets in source control."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when a selected backend is missing required settings."""


@dataclass(frozen=True)
class AgentSettings:
    backend: str
    mcp_command: str
    mcp_args: tuple[str, ...]
    mcp_cwd: Path | None
    max_tool_calls: int
    max_turns: int
    trace_path: Path | None
    model: str
    base_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> AgentSettings:
        backend = os.getenv("MODEL_BACKEND", "local").strip().lower()
        if backend not in {"local", "gemini", "hosted"}:
            raise ConfigurationError("MODEL_BACKEND must be one of: local, gemini, hosted")

        mcp_command = os.getenv("NHS_MCP_COMMAND", "uv")
        mcp_args = tuple(shlex.split(os.getenv("NHS_MCP_ARGS", "run nhs-intel-mcp")))
        cwd_value = os.getenv("NHS_MCP_CWD")
        mcp_cwd = Path(cwd_value).expanduser() if cwd_value else None
        trace_value = os.getenv("TRACE_PATH")
        trace_path = Path(trace_value).expanduser() if trace_value else None

        if backend == "gemini":
            model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ConfigurationError("GEMINI_API_KEY is required for MODEL_BACKEND=gemini")
            base_url = None
        elif backend == "hosted":
            model = os.getenv("HOSTED_MODEL", "")
            base_url = os.getenv("HOSTED_BASE_URL", "")
            api_key = os.getenv("HOSTED_API_KEY")
            if not model or not base_url or not api_key:
                raise ConfigurationError(
                    "HOSTED_MODEL, HOSTED_BASE_URL, and HOSTED_API_KEY are required "
                    "for MODEL_BACKEND=hosted"
                )
        else:
            model = os.getenv("LOCAL_MODEL", "Qwen/Qwen3-8B")
            base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8080/v1")
            api_key = os.getenv("LOCAL_API_KEY") or None

        return cls(
            backend=backend,
            mcp_command=mcp_command,
            mcp_args=mcp_args,
            mcp_cwd=mcp_cwd,
            max_tool_calls=_positive_int("MAX_TOOL_CALLS", 8),
            max_turns=_positive_int("MAX_AGENT_TURNS", 12),
            trace_path=trace_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value
