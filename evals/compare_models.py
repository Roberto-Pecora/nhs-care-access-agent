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
import statistics
from dataclasses import dataclass
from pathlib import Path

from nhs_care_access_agent.cli import _make_agent
from nhs_care_access_agent.config import AgentSettings
from nhs_care_access_agent.evaluation import EvaluationReport, load_taskset, run_evaluation


@dataclass(frozen=True)
class BackendRun:
    """Aggregate of ``repeats`` independent evaluations of one backend.

    Agent runs are stochastic even at temperature 0 (provider non-determinism),
    so a single pass over eight cases is a point estimate with no error bar. We
    run the frozen set ``repeats`` times and report mean and sample standard
    deviation of each rate, which is what a reader needs to tell a real
    difference from run-to-run noise.
    """

    label: str
    reports: tuple[EvaluationReport, ...]

    def spread(self, metric: str) -> tuple[float, float]:
        values = [getattr(r, metric) for r in self.reports]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std


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


async def _run_backend(
    label: str, env: dict[str, str], taskset_path: Path, repeats: int
) -> BackendRun:
    previous = _apply_env(env)
    reports: list[EvaluationReport] = []
    try:
        settings = AgentSettings.from_env()
        cases = load_taskset(taskset_path)
        for i in range(repeats):
            if repeats > 1:
                print(f"  {label}: run {i + 1}/{repeats}")
            reports.append(await run_evaluation(cases, lambda: _make_agent(settings)))
    finally:
        _restore_env(previous)
    return BackendRun(label=label, reports=tuple(reports))


def _cell(run: BackendRun, metric: str, suffix: str = "%", fmt: str = ".1f") -> str:
    mean, std = run.spread(metric)
    if len(run.reports) > 1:
        return f"{mean:{fmt}}{suffix} ± {std:{fmt}}"
    return f"{mean:{fmt}}{suffix}"


def _markdown_table(runs: list[BackendRun]) -> str:
    n = max((len(run.reports) for run in runs), default=1)
    caption = f"Mean ± sample sd over {n} runs; temperature 0." if n > 1 else "Single run."
    header = (
        "| Model | Task pass | Factual | Tool select | p50 ms | p95 ms |\n"
        "|---|--:|--:|--:|--:|--:|"
    )
    rows = [
        f"| {run.label} | {_cell(run, 'task_pass_rate')} | "
        f"{_cell(run, 'factual_correct_rate')} | {_cell(run, 'tool_selection_rate')} | "
        f"{_cell(run, 'latency_p50_ms', suffix='', fmt='.0f')} | "
        f"{_cell(run, 'latency_p95_ms', suffix='', fmt='.0f')} |"
        for run in runs
    ]
    return "\n".join([header, *rows, "", f"_{caption}_"])


async def _main(config_path: Path, taskset_path: Path, out_dir: Path, repeats: int) -> None:
    backends: dict[str, dict[str, str]] = json.loads(config_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[BackendRun] = []
    for label, env in backends.items():
        print(f"Running backend: {label}")
        run = await _run_backend(label, env, taskset_path, repeats)
        (out_dir / f"{label}.json").write_text(
            json.dumps({"runs": [r.to_dict() for r in run.reports]}, indent=2) + "\n",
            encoding="utf-8",
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
    parser.add_argument(
        "--repeats", type=int, default=1, help="Independent runs per backend for mean ± sd"
    )
    args = parser.parse_args()
    asyncio.run(_main(args.config, args.taskset, args.out, args.repeats))


if __name__ == "__main__":
    main()
