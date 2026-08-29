from __future__ import annotations

from nhs_care_access_agent.agent import AgentRun
from nhs_care_access_agent.evaluation import (
    EvaluationCase,
    EvaluationReport,
    score_case,
)
from nhs_care_access_agent.tracing import AgentTrace, ToolTrace


def _run(answer: str, tool_name: str = "trust_profile", duration_ms: float = 1.0) -> AgentRun:
    trace = AgentTrace(
        run_id="test",
        started_at="2026-01-01T00:00:00+00:00",
        backend="test",
        model="test",
        question="question",
        tools=[ToolTrace("1", tool_name, {}, {}, 1.0)],
        total_duration_ms=duration_ms,
    )
    return AgentRun(answer=answer, trace=trace)


def test_score_case_requires_facts_and_expected_tool():
    case = EvaluationCase(
        id="joined",
        question="question",
        required_tools=("trust_profile",),
        required_facts=("12 weeks", "good"),
        max_tool_calls=1,
    )

    score = score_case(case, _run("The wait is 12 weeks and the rating is Good."))

    assert score.passed is True
    assert score.factual_correct is True


def test_score_case_rejects_incorrect_tool_and_missing_fact():
    case = EvaluationCase(
        id="joined",
        question="question",
        required_tools=("trust_profile",),
        required_facts=("12 weeks",),
    )

    score = score_case(case, _run("The rating is Good.", tool_name="lookup_wait_time"))

    assert score.passed is False
    assert score.tool_selection_correct is False
    assert score.factual_correct is False


def test_score_case_tolerates_hyphenated_fact_wording():
    case = EvaluationCase(id="joined", question="question", required_facts=("12 weeks",))

    score = score_case(case, _run("Cardiology has a 12-week wait at this trust."))

    assert score.factual_correct is True


def test_score_case_recognises_paraphrased_abstention():
    case = EvaluationCase(id="missing-data", question="question", should_abstain=True)

    score = score_case(case, _run("That specialty is not covered in the current dataset."))

    assert score.abstention_correct is True


def test_report_summarises_run_latency():
    case = EvaluationCase(id="c", question="question")
    scores = tuple(score_case(case, _run("ok", duration_ms=ms)) for ms in (10, 20, 400))
    report = EvaluationReport(scores=scores)

    assert report.latency_p50_ms == 20.0
    assert report.latency_p95_ms == 400.0
