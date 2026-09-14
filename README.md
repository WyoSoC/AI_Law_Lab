# AI Law Lab

An experiment platform for legal AI research, running entirely on local infrastructure:
a shared DGX Spark cluster serving `gemma4` via Ollama, orchestrated with LangGraph.

Faculty, students, and external partners configure experiments through a web interface.
Every run is traced — model calls, tool calls, retrievals, reasoning, and citations — so
results are auditable and reproducible.

```
Users ──▶ Web interface ──▶ Experiment manager
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     Document analysis    Agentic workflow    Role-play simulation
              └───────────────────┼───────────────────┘
                                  ▼
                    LLM router (bounded queue)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              test-spark3                  test-spark4
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
         Legal RAG          External tools      Memory / state
              └───────────────────┼───────────────────┘
                                  ▼
                  Logging and evaluation engine
```

## Quick start

```bash
docker compose up -d          # Postgres 17 + pgvector on 127.0.0.1:5433
uv venv && uv pip install -e ".[dev]"

.venv/bin/python -m ailawlab.cli status    # check the Spark cluster
.venv/bin/python -m ailawlab.cli serve     # http://localhost:8088
```

## Infrastructure

Two DGX Spark units (NVIDIA GB10, 20 cores, 121 GB unified memory, 3.7 TB disk each)
serve `gemma4:latest` — 8B parameters, Q4_K_M, 128K context, with vision, audio, tools,
and thinking capabilities.

**Reach them by Tailscale hostname only.** `test-spark3` and `test-spark4` are routable
from the orchestrator; their lab-subnet addresses `10.99.252.31/.32` are not.

### The concurrency ceiling

Benchmarking (`scripts/bench_sparks.py`, 2026-07-23) found:

| concurrency | aggregate tok/s | p50 latency |
|---:|---:|---:|
| 1 | 18 | 0.66 s |
| 16 | 160 | 1.87 s |
| 32 | 131 | 3.29 s |
| 50 | 160 | 5.22 s |

50 concurrent requests complete without error, but **throughput plateaus at ~16**. Past
that, requests queue inside Ollama: latency grows linearly while throughput stays flat.

The router therefore admits at most `slots_per_host` (default 16) per host — ~32 across
the cluster — and queues the rest itself, recording queue-wait separately from model-eval
time. Without that split, a run that was merely waiting looks like a run that was slow.

Re-run the benchmark after any hardware or Ollama change and update
`AILAWLAB_SLOTS_PER_HOST`.

### A note on `gemma4` and thinking

gemma4 reasons before answering **by default**. That is useful — the reasoning trace is
captured to `run_events.thinking` and is exactly what a research setting wants — but it
costs roughly an order of magnitude more output tokens on short answers (measured: 2
tokens with `think=False` vs 32 with it on, for the same reply).

Consequences worth knowing:

- A small `num_predict` can be consumed entirely by the thinking block, returning empty
  content with `done_reason="length"`. `LLMResult.truncated` makes that detectable rather
  than a silent empty string.
- Nodes doing mechanical work (planning, turn-taking, summarizing) pass `think=False`.
  Nodes doing substantive legal reasoning pass `think=True`.

## Experiment modes

**`document_analysis`** — `plan → analyze → ground → synthesize`. The planner splits the
question into sub-questions, which are analyzed concurrently against the document (128K
context usually swallows a whole contract), grounded against the corpus, and synthesized
with numbered citations. Citation markers pointing past the retrieved passage list are
recorded as `unsupported` rather than dropped — catching fabrication is the point.

**`agentic_workflow`** — a ReAct loop over gemma4's native function calling. Tools arrive
as structured `message.tool_calls`, not scraped JSON. Built-in tools are corpus search and
a safe arithmetic evaluator. Network-touching tools would be opt-in per experiment via
`allow_network`, but none are installed yet, so that switch currently changes nothing.

**`roleplay`** — several agents with distinct roles, goals, and **private memory scoped to
`(run_id, agent_id)`**. A moderator picks the next speaker or ends the scene; an evaluator
assesses the outcome. The memory separation is the whole point: a shared buffer would leak
one side's private reasoning into the other's context and quietly invalidate the exercise.

Each turn, the speaking agent first updates its **private negotiation notes** (concessions
made and received, where things stand, its read on the others, open issues, arguments
already made, next move), then speaks with a short character reminder at the end of its
prompt. The moderator returns structured JSON: whether the last speaker asked someone a
direct question (that person answers next), whether talks have stalled (it intervenes by
reframing, narrowing to one issue, or reality testing), and whether the scene is over (not
before everyone has spoken twice). Code enforces turn balance so no agent monopolizes the
floor. Transcripts too long for the 128K window are summarized in sections, concurrently,
before assessment, and the assessor sees each party's bottom line so it can check whether
anyone gave in past it or leaked a confidential fact.

These rules, and the studies they come from, are in `graphs/roleplay_policy.py`. Defaults
are 100 turns and 1000 words a turn. Each turn is three calls; on a live run they took
about 26-30 s together (the reply ~17 s, private notes ~4 s, the moderator ~2.5 s), rising
as prompts grow, so a full scene takes about an hour. The moderator ends it sooner on
agreement or a clear walk-away.

### Agent files

Agents can be uploaded and downloaded as plain Markdown, written for lawyers rather than
programmers. A file holds one agent or a whole cast; each person starts at `# Name`:

```markdown
# Dana Reyes

## Role
Lead counsel for the Provider

## Objective
Cap the Provider's indemnification at 12 months of fees.

## Tendencies
- Anchors hard early, then concedes slowly
- Reframes every risk as a dollar figure

## Bottom line
Will go as far as 24 months of fees; walks away from uncapped indemnification.

## Confidential information
The Provider's insurer refuses to cover uncapped indemnities.
```

The other sections are `Background`, `Demeanor` and `Priorities`. Only the name is
required. Headings match case-insensitively and by common synonyms ("Goal", "Walk-away
point", "Interests"); ids are generated from names; text before the first name or inside
`<!-- -->` is ignored. Unrecognized sections are kept under "Additional notes" and reported,
never dropped. The builder offers a commented blank file, bulk upload (several files, or
several people per file), per-agent and whole-cast download, an AI-drafted cast from a
scenario, and a cast check (rules plus an optional AI review against the scenario). The
format lives in `agent_spec.py`.

### Drafting a cast from real material

"Draft with AI" can start from a web link (a news story, a court page, an opinion), an
uploaded PDF, HTML or text file, or pasted text instead of the scenario box. The source is
previewed first so you can confirm the right text came through; the AI then writes the
scenario and the cast. `source_material.py` does the reading: it picks the main article out
of a page without an HTML library, caps a source at 12,000 words, and fetches a link only if
its host resolves to public addresses (re-checked on every redirect), because this server
can reach the Sparks, Postgres and the university network.

Real people and private organizations in the source are renamed before the model drafts
anything, and the same substitution is applied to its output. A draft invents bottom lines
and confidential facts, and telling the model to use invented names was not enough: on a
Casper Mountain gravel-pit story it kept the real mining company and gave it an invented
secret. The builder lists every name it changed; courts, agencies and places keep their
real names. The experiment records the source's title, link, retrieval time and a SHA-256
of the text that was used.

## Memory

Ported from `ollama-chat-agent/memory.py`, preserving its two-tier design — a verbatim
short-term buffer bounded by a token budget, with overflow compacted into model-written
summaries, embedded, and retrieved by cosine similarity above a floor.

Four things changed for Lab scale:

1. **Postgres + pgvector** instead of SQLite. The original held one connection without
   `check_same_thread` and scanned every row computing cosine in Python; retrieval is now
   an indexed ANN query (HNSW), and concurrent agents no longer serialize.
2. **Scoped to `(run_id, agent_id)`** instead of a single `session_id`.
3. **Provenance** — long-term rows carry the trace event or corpus chunk they came from.
4. **Compaction off the request path**, rather than blocking a turn on summarization.

## Evaluation

`run_events` is an append-only trace: one row per LLM call, tool call, retrieval, and node
transition, with `queue_wait_ms` and `eval_ms` stored separately, plus reasoning in its own
column. `citations` links asserted citations back to chunks with a `grounded` /
`unsupported` / `unchecked` verdict.

`run_metrics()` aggregates this into performance, grounding rate, and host distribution.
`queue_share` answers the first question you have about a slow run: was it the model, or
the cluster being busy?

## Reproducibility

Each run snapshots its experiment's `config` at launch into `runs.config_snapshot`. Editing
an experiment later never rewrites the conditions a past result was produced under.

## Layout

```
src/ailawlab/
  config.py            settings (env / .env)
  router.py            bounded-queue LLM router across the Sparks
  db.py                async Postgres pool + pgvector binding
  memory.py            two-tier agent memory
  rag.py               corpus ingestion and grounded retrieval
  sources.py           online legal databases (CourtListener, Federal Register, eCFR, govinfo, EDGAR)
  source_material.py   reading a web page, PDF or text to draft a cast from
  agent_spec.py        Markdown agent files: reading, writing, cast checks
  cast_assistant.py    AI-drafted casts, real-name replacement, AI cast review
  tools.py             tool registry for agentic workflows
  tracing.py           trace writer + run metrics
  experiments.py       experiment/run lifecycle
  graphs/              LangGraph definitions per mode; roleplay_policy.py holds the moderation rules
  web/                 FastAPI app, page views (views.py), templates, static
db/schema.sql          Postgres schema
docs/                  method notes and the script that builds them
scripts/               benchmark and utilities
tests/                 offline unit tests and integration tests against real hardware
```

The role-play moderation logic is written up as a short method note with references:
[`src/ailawlab/web/static/docs/roleplay-moderation.pdf`](src/ailawlab/web/static/docs/roleplay-moderation.pdf)
(also as Word), linked from the web portal's About page and from each role-play experiment.
`docs/roleplay-moderation/build.py` regenerates both files.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The graph and router tests hit the real Sparks and a real Postgres — they are integration
tests by design, since the failures worth catching here (admission control, pgvector
binding, tool dispatch, memory scoping) are exactly the ones a mock would hide. The tests
for agent files, moderation rules, source reading, cast drafting and the experiment page
are pure and run offline:

```bash
.venv/bin/python -m pytest -q tests/test_agent_spec.py tests/test_roleplay_policy.py \
  tests/test_source_material.py tests/test_cast_assistant.py tests/test_experiment_page.py
```

## Configuration

See `.env.example`. Everything is overridable by environment variable with the
`AILAWLAB_` prefix.
