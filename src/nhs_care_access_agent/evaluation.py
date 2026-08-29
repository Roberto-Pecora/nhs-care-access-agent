"""Deterministic, versionable agent evaluation.

The first evaluator deliberately avoids an LLM judge. It scores facts, required
tool use, tool budgets, and safe abstention against a frozen task set. An
LLM-as-judge can be added as a supplementary quality signal later, never as the
only correctness gate.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .agent import AgentRun


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    should_abstain: bool = False
    max_tool_calls: int = 8

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationCase:
        return cls(
            id=value["id"],
            question=value["question"],
            required_tools=tuple(value.get("required_tools", [])),
            forbidden_tools=tuple(value.get("forbidden_tools", [])),
            required_facts=tuple(value.get("required_facts", [])),
            should_abstain=bool(value.get("should_abstain", False)),
            max_tool_calls=int(value.get("max_tool_calls", 8)),
        )


@dataclass(frozen=True)
class CaseScore:
    id: str
    passed: bool
    factual_correct: bool
    tool_selection_correct: bool
    tool_budget_ok: bool
    abstention_correct: bool
    answer: str
    tool_names: tuple[str, ...]
    duration_ms: float


@dataclass(frozen=True)
class EvaluationReport:
    scores: tuple[CaseScore, ...]

    @property
    def task_pass_rate(self) -> float:
        return _rate([score.passed for score in self.scores])

    @property
    def factual_correct_rate(self) -> float:
        return _rate([score.factual_correct for score in self.scores])

    @property
    def tool_selection_rate(self) -> float:
        return _rate([score.tool_selection_correct for score in self.scores])

    @property
    def latency_p50_ms(self) -> float:
        return _percentile([score.duration_ms for score in self.scores], 50)

    @property
    def latency_p95_ms(self) -> float:
        return _percentile([score.duration_ms for score in self.scores], 95)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_pass_rate": self.task_pass_rate,
            "factual_correct_rate": self.factual_correct_rate,
            "tool_selection_rate": self.tool_selection_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "scores": [asdict(score) for score in self.scores],
        }


def load_taskset(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            cases.append(EvaluationCase.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid evaluation case at {path}:{line_number}") from error
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


# Phrases a model uses to decline rather than guess. Extend this list from real
# evaluation transcripts rather than guessing every paraphrase up front.
_ABSTENTION_PATTERN = re.compile(
    r"\b(cannot|can't|unable|insufficient|not (available|covered|listed)|"
    r"(do|does)n't (have|cover|include)|no data|not sure|don't know)\b"
)


def score_case(case: EvaluationCase, run: AgentRun) -> CaseScore:
    answer = run.answer.casefold()
    tool_names = tuple(tool.name for tool in run.trace.tools)
    facts_ok = all(_fact_present(fact, answer) for fact in case.required_facts)
    required_tools_ok = set(case.required_tools).issubset(tool_names)
    forbidden_tools_ok = not set(case.forbidden_tools).intersection(tool_names)
    tool_selection_ok = required_tools_ok and forbidden_tools_ok
    budget_ok = len(tool_names) <= case.max_tool_calls
    abstained = bool(_ABSTENTION_PATTERN.search(answer))
    abstention_ok = abstained if case.should_abstain else True
    passed = facts_ok and tool_selection_ok and budget_ok and abstention_ok
    return CaseScore(
        id=case.id,
        passed=passed,
        factual_correct=facts_ok,
        tool_selection_correct=tool_selection_ok,
        tool_budget_ok=budget_ok,
        abstention_correct=abstention_ok,
        answer=run.answer,
        tool_names=tool_names,
        duration_ms=run.trace.total_duration_ms,
    )


def _fact_present(fact: str, answer: str) -> bool:
    """Match a required fact, tolerating hyphenation and singular/plural wording.

    A verbatim substring check fails a correct answer over formatting alone,
    for example "12-week wait" against a required fact of "12 weeks". This
    only normalises whitespace, hyphens, and trailing "s"; it is not a
    semantic match.
    """
    fact_words = re.findall(r"\w+", fact.casefold())
    answer_words = re.findall(r"\w+", answer)
    if not fact_words:
        return True
    span = len(fact_words)
    return any(
        all(_word_matches(fact_word, answer_word) for fact_word, answer_word in zip(fact_words, window))
        for window in (answer_words[start : start + span] for start in range(len(answer_words) - span + 1))
    )


def _word_matches(fact_word: str, answer_word: str) -> bool:
    return fact_word == answer_word or fact_word.rstrip("s") == answer_word.rstrip("s")


async def run_evaluation(
    cases: list[EvaluationCase], agent_factory: Callable[[], Awaitable[Any] | Any]
) -> EvaluationReport:
    scores: list[CaseScore] = []
    for case in cases:
        candidate = agent_factory()
        agent = await candidate if asyncio.iscoroutine(candidate) else candidate
        scores.append(score_case(case, await agent.run(case.question)))
    return EvaluationReport(scores=tuple(scores))


def _rate(values: list[bool]) -> float:
    return round(mean(values) * 100, 1) if values else 0.0


def _percentile(values: list[float], p: int) -> float:
    """Nearest-rank percentile. quantiles needs two points, so short runs
    fall back to the sorted-position estimate that also handles n == 1."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(p / 100 * len(ordered)) - 1))
    return round(ordered[rank], 2)
