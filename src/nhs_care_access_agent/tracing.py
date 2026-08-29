"""Portable JSONL traces for local inspection or OpenTelemetry export later."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass
class ToolTrace:
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_ms: float
    blocked: bool = False


@dataclass
class AgentTrace:
    run_id: str
    started_at: str
    backend: str
    model: str
    question: str
    tools: list[ToolTrace] = field(default_factory=list)
    model_turns: int = 0
    final_answer: str = ""
    stop_reason: str = ""
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlTraceSink:
    """Append traces without coupling the core agent to a vendor platform."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, trace: AgentTrace) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")


def new_trace(run_id: str, backend: str, model: str, question: str) -> AgentTrace:
    return AgentTrace(
        run_id=run_id,
        started_at=datetime.now(UTC).isoformat(),
        backend=backend,
        model=model,
        question=question,
    )


def elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1_000, 2)
