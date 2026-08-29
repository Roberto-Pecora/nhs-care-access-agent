# NHS Care Access Agent

An evaluation-first, model-agnostic research agent for NHS care access. It uses
the existing [`nhs-intelligence-mcp`](https://github.com/Roberto-Pecora/nhs-intelligence-mcp)
server as a **read-only** tool layer, then compares local open-weight, free hosted,
and low-cost hosted models on exactly the same grounded tasks.

```
Question → model adapter → bounded agent loop → NHS MCP tools → SQLite snapshot
                           ↓                         ↓
                        JSONL trace            deterministic evaluation
```

The agent is not clinical decision support. It researches published wait-time,
trend, and CQC-rating data and must say when its sources are insufficient.

## Why this exists

An MCP server makes trustworthy tools available; it does not prove a model will
choose the right tool, pass meaningful arguments, interpret a result correctly,
or abstain when data is missing. This project evaluates those behaviours
separately instead of treating a fluent response as evidence of reliability.

## What it demonstrates

- **Agent runtime:** a small, inspectable multi-turn tool-use loop with turn and
  tool-call budgets rather than a hidden multi-agent framework.
- **Open-model operation:** Qwen through `llama.cpp`, vLLM, or any compatible
  local server.
- **Model portability:** Gemini Flash and a pinned hosted Qwen/DeepSeek endpoint
  can run through the same MCP tools and task set.
- **Safe tool use:** only tool names exposed by the connected MCP server are
  allowed; all NHS tools are read-only.
- **Evaluation and traces:** every run produces portable JSONL traces; the
  evaluator scores known facts, tool selection, tool budget, and abstention, and
  reports p50/p95 run latency across the task set.

## Architecture

The model never connects directly to NHS data. The application owns the tool
loop and is responsible for executing calls and returning results to the model.

| Component | Responsibility |
|---|---|
| `nhs-intelligence-mcp` | Stable, typed, deterministic NHS data tools |
| `CareAccessAgent` | Tool allowlist, execution budget, prompt, trace creation |
| Model adapters | Translate a provider's function-calling API into common types |
| JSONL trace sink | Vendor-neutral run records for review or later OTEL export |
| Deterministic evaluator | Regression gates over a versioned, frozen task set |

## Quick start

### 1. Run the NHS MCP server locally

Clone and initialise [`nhs-intelligence-mcp`](https://github.com/Roberto-Pecora/nhs-intelligence-mcp)
first. Its normal development command is `uv run nhs-intel-mcp`.

### 2. Configure this agent

```bash
cp .env.example .env
# Export the values in .env with your preferred shell or environment manager.
uv sync --extra dev
```

Set `NHS_MCP_CWD` to the MCP repository checkout. The agent launches the server
over stdio, so it has no public listening port and no network call at NHS-query
time.

### 3. Choose a model backend

**Local Qwen** — start `llama-server`, vLLM, or another OpenAI-compatible server:

```bash
export MODEL_BACKEND=local
export LOCAL_MODEL=Qwen/Qwen3-8B
export LOCAL_BASE_URL=http://localhost:8080/v1
```

**Gemini Flash** — useful as a free development baseline:

```bash
export MODEL_BACKEND=gemini
export GEMINI_MODEL=gemini-3.7-flash
export GEMINI_API_KEY=replace-me
```

**Pinned hosted open-weight endpoint** — use a specific provider/model/region,
not a dynamic free router, for a reproducible published result:

```bash
export MODEL_BACKEND=hosted
export HOSTED_MODEL=provider-pinned-model-id
export HOSTED_BASE_URL=https://provider.example/v1
export HOSTED_API_KEY=replace-me
```

### 4. Ask a question

```bash
uv run nhs-care-access-agent chat \
  "Compare the current wait, trend, and quality for Guy's and St Thomas' cardiology."
```

The answer must include a `Sources:` line naming the MCP tools it used. Trace
records are written to `TRACE_PATH` when configured.

## Evaluation

```bash
uv run nhs-care-access-agent evaluate --taskset evals/seed_tasks.jsonl
```

`evals/seed_tasks.jsonl` is intentionally a **synthetic smoke test**, not a
benchmark. Before publishing portfolio metrics:

1. Freeze a known `nhs_intel.db` version and record its release checksum.
2. Build a task set from that snapshot, with expected facts and permitted tool
   paths reviewed by a human.
3. Include direct lookup, ranking, joined profiles, incomplete-data cases, and
   misleading/ambiguous prompts.
4. Run each model with fixed prompt, temperature, model revision, hardware or
   provider, and date.
5. Report task pass rate, factual correctness, tool-selection rate, p50/p95
   latency, and cost per completed task—plus a small manual review set. The
   evaluator emits the pass rates and p50/p95 latency directly; cost per task is
   still a manual step, as the trace carries no token or pricing data yet.

The deterministic evaluator deliberately does not use an LLM judge for factual
correctness. An LLM judge can later assess communication quality, but should be
calibrated against human labels and never replace the frozen-data checks.

## Privacy and data governance

Use only public/synthetic data with a free hosted API tier. Do not send personal
health information, patient data, or sensitive user questions to Gemini's free
tier or an unvetted model gateway. For a real deployment, pin the model provider
and region, document retention and data-use terms, and keep all audit traces
access-controlled.

## Development

```bash
make test
```

The core tests use fake models and fake MCP tools. They require no API key,
GPU, downloaded model, or live NHS data.
