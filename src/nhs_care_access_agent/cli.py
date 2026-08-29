"""Command-line entrypoints for interactive research and repeatable evals."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .agent import CareAccessAgent
from .config import AgentSettings
from .evaluation import load_taskset, run_evaluation
from .mcp_client import StdioMcpToolClient
from .providers import GeminiModel, OpenAICompatibleModel
from .tracing import JsonlTraceSink


def main() -> None:
    parser = argparse.ArgumentParser(description="NHS care-access research agent")
    subcommands = parser.add_subparsers(dest="command", required=True)
    chat = subcommands.add_parser("chat", help="Answer one research question")
    chat.add_argument("question")
    evaluate = subcommands.add_parser("evaluate", help="Run the deterministic task set")
    evaluate.add_argument("--taskset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/evaluation-report.json"))
    arguments = parser.parse_args()
    settings = AgentSettings.from_env()

    if arguments.command == "chat":
        run = asyncio.run(_make_agent(settings).run(arguments.question))
        print(run.answer)
        print(f"\nTrace: {run.trace.run_id} ({run.trace.total_duration_ms} ms)")
    else:
        report = asyncio.run(
            run_evaluation(load_taskset(arguments.taskset), lambda: _make_agent(settings))
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport: {arguments.output}")


def _make_agent(settings: AgentSettings) -> CareAccessAgent:
    if settings.backend == "gemini":
        model = GeminiModel(settings.model, settings.api_key or "")
    else:
        model = OpenAICompatibleModel(settings.model, settings.base_url or "", settings.api_key)
    tools = StdioMcpToolClient(settings.mcp_command, settings.mcp_args, settings.mcp_cwd)
    trace_sink = JsonlTraceSink(settings.trace_path) if settings.trace_path else None
    return CareAccessAgent(
        model,
        tools,
        backend=settings.backend,
        max_tool_calls=settings.max_tool_calls,
        max_turns=settings.max_turns,
        trace_sink=trace_sink,
    )
