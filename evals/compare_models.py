"""Run the frozen task set across several model backends and tabulate results.

This does not launch any model itself: each backend must already be reachable
(a local OpenAI-compatible server running, or GEMINI_API_KEY / HOSTED_* set).
The script re-reads MODEL_BACKEND-style settings per named backend from a small
config file, runs the same frozen task set against each, writes every report to
JSON, and prints one markdown comparison table.

    uv run python evals/compare_models.py --taskset evals/frozen_tasks.jsonl \
        --config evals/backends.example.json --out artifacts/comparison

The heavy work (agent loop, MCP tools, scoring) is the same code the CLI uses;
this only sequences it over backends and formats the summary a reader wants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from nhs_care_access_agent.cli import _make_agent
from nhs_care_access_agent.config import AgentSettings
from nhs_care_access_agent.evaluation import EvaluationReport, load_taskset, run_evaluation


@dataclass(frozen=True)
class BackendRun:
    label: str
    report: EvaluationReport


def _resolve(value: str) -> str:
    """Expand a ``${VAR}`` reference from the environment, so a committed config
    can name a secret without holding it. A plain string is returned unchanged."""
    if value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        resolved = os.environ.get(name)
        if resolved is None:
            raise KeyError(f"config references ${{{name}}} but it is not set in the environment")
        return resolved
    return value


def _apply_env(overrides: dict[str, str]) -> dict[str, str | None]:
    """Set env vars for one backend, returning the prior values to restore."""
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = _resolve(value)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


async def _run_backend(label: str, env: dict[str, str], taskset_path: Path) -> BackendRun:
    previous = _apply_env(env)
    try:
        settings = AgentSettings.from_env()
        report = await run_evaluation(
            load_taskset(taskset_path), lambda: _make_agent(settings)
        )
    finally:
        _restore_env(previous)
    return BackendRun(label=label, report=report)


def _markdown_table(runs: list[BackendRun]) -> str:
    header = (
        "| Model | Task pass | Factual | Tool select | p50 ms | p95 ms |\n"
        "|---|--:|--:|--:|--:|--:|"
    )
    rows = [
        f"| {run.label} | {r.task_pass_rate}% | {r.factual_correct_rate}% | "
        f"{r.tool_selection_rate}% | {r.latency_p50_ms:.0f} | {r.latency_p95_ms:.0f} |"
        for run in runs
        for r in (run.report,)
    ]
    return "\n".join([header, *rows])


async def _main(config_path: Path, taskset_path: Path, out_dir: Path) -> None:
    backends: dict[str, dict[str, str]] = json.loads(config_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[BackendRun] = []
    for label, env in backends.items():
        print(f"Running backend: {label}")
        run = await _run_backend(label, env, taskset_path)
        (out_dir / f"{label}.json").write_text(
            json.dumps(run.report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        runs.append(run)

    table = _markdown_table(runs)
    (out_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nWrote reports and comparison.md to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", type=Path, default=Path("evals/frozen_tasks.jsonl"))
    parser.add_argument("--config", type=Path, required=True, help="JSON: {label: {ENV: value}}")
    parser.add_argument("--out", type=Path, default=Path("artifacts/comparison"))
    args = parser.parse_args()
    asyncio.run(_main(args.config, args.taskset, args.out))


if __name__ == "__main__":
    main()
