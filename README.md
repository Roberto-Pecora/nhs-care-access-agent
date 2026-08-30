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

**Claude** — the frontier ceiling for the benchmark, via the native Messages API:

```bash
export MODEL_BACKEND=anthropic
export ANTHROPIC_MODEL=claude-sonnet-5
export ANTHROPIC_API_KEY=replace-me
```

### 4. Ask a question

```bash
uv run nhs-care-access-agent chat \
  "Compare the current wait, trend, and quality for Guy's and St Thomas' cardiology."
```

The answer must include a `Sources:` line naming the MCP tools it used. Trace
records are written to `TRACE_PATH` when configured.

## Evaluation

Two task files ship here. `evals/seed_tasks.jsonl` is a synthetic smoke test with
no real data behind it. `evals/frozen_tasks.jsonl` is the real benchmark: every
expected fact is a value the NHS MCP tools return against a **pinned**
`nhs_intel.db` snapshot (its sha256 is recorded in the file header, along with the
RTT months and My Planned Care date it was captured from). It spans direct lookup,
ranking, joined profile, a twelve-month waiting-time trend, a missing-data
abstention, a trend-unavailable abstention, and a misleading-premise correction.

```bash
# Point at the pinned DB and the MCP checkout, then run the frozen set:
export NHS_MCP_CWD=/path/to/nhs-intelligence-mcp
export NHS_INTEL_DB="$NHS_MCP_CWD/data/nhs_intel.db"
uv run nhs-care-access-agent evaluate --taskset evals/frozen_tasks.jsonl
```

A frozen fact is only trustworthy while its snapshot is unchanged.
`evals/verify_facts.py` checks the DB checksum and re-asserts every fact against
the live tools; run it before trusting a result—a mismatch means the snapshot
moved, not that a model failed:

```bash
uv run python evals/verify_facts.py
```

To compare models, run the same frozen set across each backend and get one table
(pass rate, factual correctness, tool-selection rate, p50/p95 latency). Backends
are named in a JSON config; copy `evals/backends.example.json`, fill in your
endpoints, and keep secrets in the shell via `${VAR}` references:

```bash
uv run python evals/compare_models.py \
  --taskset evals/frozen_tasks.jsonl \
  --config evals/backends.json \
  --out artifacts/comparison \
  --repeats 5
```

`--repeats` runs the set that many times per backend and reports mean ± sample
standard deviation, so a real difference can be told from run-to-run noise; the
default of 1 is a single point estimate.

### Results

Measured on the pinned snapshot (`0e5eec15…`, 12 RTT months, 8 frozen cases),
**five runs per model** (mean ± sample sd). Claude via the `anthropic` backend,
Nemotron and MiniMax via OpenRouter (`hosted`), the 3B locally through Ollama.

| Model | Task pass | Factual | Tool select | p50 ms | p95 ms |
|---|--:|--:|--:|--:|--:|
| Claude Sonnet 5 | 97.5 ± 5.6 | 100 ± 0 | 100 ± 0 | 4897 | 7479 |
| nvidia/nemotron-3-super-120b | 77.5 ± 10.5 | 95.0 ± 6.8 | 100 ± 0 | 6873 | 15733 |
| minimax/minimax-m3 | 77.5 ± 5.6 | 87.5 ± 0 | 97.5 ± 5.6 | 16029 | 29815 |
| llama3.2:3b (local) | 37.5 ± 0 | 50.0 ± 0 | 62.5 ± 0 | 4865 | 7297 |

Running five times, rather than once, changes the reading:

- **Point estimates were optimistic.** A single run put Nemotron at 87.5%; over
  five it is 77.5 ± 10.5 — the widest spread of the four, and the one run had
  caught a good day. Reporting n=1 would have overstated it by ten points.
- **Claude Sonnet 5 sets a clean ceiling at 97.5%,** and its one recurring miss
  is the evaluator's keyword abstention check failing to match a correctly hedged
  refusal — instrumentation, not capability.
- **Accuracy ties hide a latency gap.** Nemotron and MiniMax match on pass rate,
  but MiniMax is ~3× slower (p50 16s vs. 7s) — invisible without the latency axis.
- **The 3B is deterministic at 37.5%** (σ=0 on every quality metric): it fails the
  same five cases every run — emitting tool calls as literal text without invoking
  them, passing malformed arguments (`by_name` as the string `"True"`, invented
  identifiers), and confirming a false premise by agreeing a 31-week wait was one
  of the shortest in England. Systematic failures, not unlucky sampling.

The most useful output is what the harness caught about itself. Two **scoring**
gaps are reported rather than tuned away. Abstention is detected by keyword
(`cannot`, `no data`, `does not have`), so a naturally phrased refusal such as
"does not currently have a reported waiting time" is under-counted — the single
largest distortion, and the reason even Sonnet drops below 100%. And pinning one
`required_tool` penalises a correct answer reached by a valid alternate path. A
more robust abstention check (or an LLM judge for abstention only, calibrated
against human labels) and equivalent-path tool scoring are future work.

Runs are at temperature 0 where the provider allows it; Claude runs at its default
(temperature is deprecated on current Claude models). Cost per completed task is
recorded manually; the trace carries no token or pricing data yet.

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
