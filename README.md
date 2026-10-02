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

A full architecture diagram, with a component reference covering each tier, the queue, the
trace and what does and does not leave campus, is at
[`src/ailawlab/web/static/docs/architecture.pdf`](src/ailawlab/web/static/docs/architecture.pdf),
linked from the portal's About page (the landing page, reached by the logo). `docs/architecture/build.py` regenerates it.

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
question into sub-questions. With no document, the question is asked of the run's libraries
directly: each sub-question (and the question itself) is a search, run concurrently, and the
answer is written from the passages found. With a document, the sub-questions are analyzed
concurrently against it (128K context usually swallows a whole contract) and grounded
against the libraries. Either way the answer cites numbered passages; markers pointing past
the passage list are recorded as `unsupported` rather than dropped — catching fabrication is
the point. A run needs a document, a library, or both.

Answers are written as answers (`graphs/answer_style.py`): the prompts ask for no memo or
letter format, no persona and no placeholders, and a header or sign-off that slips through
anyway ("To: / From: / Date:", "[Your Name]") is removed from the stored answer, with a note
in the trace, which keeps the reply as written.

**`agentic_workflow`** — a ReAct loop over gemma4's native function calling, given a task and
optionally a document (put in its first message, up to 200,000 characters). Tools arrive
as structured `message.tool_calls`, not scraped JSON. Built-in tools are corpus search and
a safe arithmetic evaluator. With `allow_network` the agent also gets two network tools
(`network_tools.py`): `search_online` searches CourtListener, the Federal Register, the eCFR,
govinfo and SEC EDGAR and lists results as [W1], [W2] (leads, not citable), and `read_online`
reads a result or a public web page. Reading is ingesting: the document is saved to a library
(`fetch_library`, by default "Fetched: <experiment name>"), which records a new version, and
the passages come back numbered like any other, so a citation to something read online names
a document, library and version and survives the page changing. A run reads at most
`network_max_reads` (8) documents; addresses must be public, as for every outbound fetch.

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
are 100 turns and 2000 words a turn (the most a turn may be). Each turn is three calls; at
1000 words they took about 26-30 s together on a live run (the reply ~17 s, private notes
~4 s, the moderator ~2.5 s), rising as prompts grow, so a full scene at 1000 words takes
about an hour and at 2000 words the pages estimate about 1 h 40 min. The moderator ends it
sooner on agreement or a clear walk-away.

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
never dropped. The builder offers an agent file template, bulk upload (several files, or
several people per file), per-agent and whole-cast download, "Generate cast with AI" from a
scenario, and "Validate cast" (rules plus an optional AI review against the scenario). The
format lives in `agent_spec.py`.

### Experiment files

An experiment file is an agent file with the rest of the role-play above the cast, so one
file holds everything needed to recreate it:

```markdown
# Scenario
A mediation session over the renewal of state gravel leases...

# Settings
Max turns: 100
Words per turn: 1000

# Source
Title: Wyoming's high court hears arguments over gravel pit controversy
Link: https://oilcity.news/...

# Evelyn Reed
## Role
Mediator
```

`# Source` and a `# Cast` line before the people are optional. The same upload button takes
both kinds of file: an agent file adds people to the cast, while an experiment file fills
in the scenario, settings and cast (asking first if any are already filled in). Settings
outside the builder's limits are brought within them with a note. The builder, and every
saved role-play experiment (`/experiments/{id}/experiment.md`), can download one, and there
is an experiment file template alongside the agent file template.

### Drafting a cast from real material

"Generate cast with AI" can start from a web link (a news story, a court page, an opinion), an
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

### Legal sources in a role-play

A role-play can be given a corpus, when it is designed or for a single run. Before each turn
the speaker's objective and what was just said are used as a query, and the closest four
passages (each cut to about 220 words) are put in front of that speaker as [S1]..[S4], with
an instruction to cite only what a passage says. The retrieval is traced, and the transcript
records which passages each turn was given and which it cited. A failed search costs that turn
its sources, not the run. With no corpus, turns are exactly as before.

### Reading a run

The run page leads with the result summary (the assessment, or the answer), rendered from the
model's Markdown by `web/markdown.py`, which escapes everything before formatting so model
output cannot inject markup. A role-play's transcript follows, each person in their own colour,
with the assessor's "Turn 17" references and each reply's [S2] markers linked. Trace events
show their full text. **Export PDF** (`web/report_pdf.py`, reportlab) writes the summary, the
scenario and cast, and the full transcript in the same colours; private notes and reasoning
are included only when asked for.

## Accounts

Everything but the About page needs a sign-in, through a self-hosted Keycloak at `/sso/`
(UW sign-on, Google, Microsoft, or an AI Law Lab account). Verified `@uwyo.edu` addresses are
approved automatically as researchers; others wait for an admin on the Users page. Roles are
viewer, researcher and admin; everything is shared lab-wide, with experiments, runs, documents
and library versions attributed to the person who made them. Setup, providers, email, backups
and upgrades: [`docs/keycloak.md`](docs/keycloak.md). The sign-in itself is in `web/auth.py`,
the approval rules in `accounts.py`.

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

## The portal

The logo leads to the About page; the tabs are **Legal Sources**, **New experiment** and
**Dashboard**, and inner pages carry breadcrumbs back to them. A switch in the header picks
the light or dark theme (saved per browser; otherwise the system setting is followed).

**Legal Sources** lists each corpus with its size and the experiments that use it. Opening one
shows its documents, lets you read any document as the passages experiments retrieve, and
has a "Try a search" box that shows what retrieval would return for a question, with
similarity scores. Documents are added from the online databases, from web links, or by
uploading several files at once, into an existing corpus picked from a drop-down or a new one.

**Web links** can be pasted in any layout (one per line, or inside a paragraph), up to 20 at a
time. Each page's main text is kept as a document, up to 100,000 words, fetched with the same
public-address and redirect checks as cast sources (`web_links.py`). Because pages change, a
link remembers its address: **Check for updates**, per link or for the whole corpus, re-reads
it and rebuilds its passages only if the text changed, recording when it was last checked and
last changed. Any document can be renamed or moved to another corpus. A document can be
removed on its own; deleting a whole corpus asks you to type its name. Neither is allowed while
a run that retrieves from that corpus is in progress.

**Experiments are deleted in two steps.** Moving one to the trash (from the dashboard or the
experiment's page, with an Undo) only hides it: its runs and results stay, it cannot be run,
and it can be restored. Deleting it for good from the trash removes its runs, traces and
citations. `experiments.deleted_at` marks trashed experiments; `db/schema.sql` adds the column
to an existing database.

## Reproducibility

Each run snapshots its experiment's `config` at launch into `runs.config_snapshot`. Editing
an experiment later never rewrites the conditions a past result was produced under.

The library it searched is versioned the same way. After every change (documents added,
removed, moved, or a web page re-read with new text) `rag.record_version()` stores the set of
document ids the library then held in `corpus_versions`, with a note of what changed. Documents
are never edited or deleted while their library exists: removing one sets `removed_at`, and
re-reading or moving one adds a new row that the old one points to through `replaced_by`, so
every version can still be searched exactly as it was. Deleting a whole library is the one
operation that removes its history. The same text may sit in several libraries, but only once
in each library's current contents.

### Several libraries per experiment

All three modes can retrieve from several libraries at once. The experiment config names them
as a list, `"libraries": ["case_law", "regulations"]` (an empty list means no retrieval, a
baseline); configs from before hold one name in `corpus`, which `rag.library_names()` still
reads. A run may change the set at launch. When it starts, each library is pinned to a
version (`rag.pin_libraries()`): the one the run names in `library_versions` (`{name: n}`),
or its current contents. What it searched is stored in `run_libraries`, one row per library
with its version; the run page's "run again with these versions" link repeats exactly that.
`runs.corpus` / `runs.corpus_version` are the older single-library columns, copied into
`run_libraries` by `db/schema.sql` and no longer written.

The pinned libraries are searched as one: passages are ranked against each other by
similarity, not taken in turns from each library, and text present in two libraries is
returned once. Every passage put in a prompt is labelled with its library, and an agent can
confine a search to one library (`search_libraries(..., library=...)`).

In a role-play each agent can also hold **case files**: libraries only it can search (the
agent's `libraries` list, the "Case files" section of an agent file). Before each turn the
speaker sees up to three passages from its own case files, marked as unseen by anyone else,
then the closest from the shared libraries. Citing a case-file passage discloses it: it goes
on the record as an exhibit, [E1], [E2], which every participant sees from then on and may
cite to rely on it or answer it. Another agent's case file never reaches a prompt except as
an exhibit. The run stores the exhibits with who disclosed each and in which turn; the
assessor sees them and each side's case files, and the run page and PDF list them.

Citations are traced to the source in every mode (`grounding.py`). Document analysis numbers
its one retrieval [1], [2]; an agent's passages keep one number across all of its searches
(`SourceLedger`), so a [3] in its answer names one passage whichever search found it; a
role-play speaker cites the passages shown that turn as [S1], [S2], and exhibits as [E1]. Each marker becomes a
`citations` row holding the chunk, document, library and version, and where it appeared
("Answer", "Turn 7 · Dana Reyes"); a marker naming no passage the model was given is recorded
as unsupported. The run page lists the sources given to the model and a **References**
section, the cited documents in the library citation form with the library version each was
found in and every place it was cited, also as a `.txt` download and in the PDF report.

**Crawling a page** ("Everything linked from one page, on the same website" under Add web
links, `crawler.py`) reads an index page and adds the documents it links to on the same
website, one level deep. A preview, which reads only the page and its robots.txt, shows how
many links there are, what robots.txt allows, the pause it will use and the licence notice it
found, before anything is collected. Its rules are shown on the page and enforced in code:
robots.txt is obeyed (a failing one stops the crawl), requests go one at a time at least
`crawl_delay_s` (2 s) apart or slower if robots.txt asks, at most `crawl_max_pages` (100) a
crawl, it stops on HTTP 429/503 or three failures in a row, it identifies itself with
`crawl_user_agent`, and only public addresses and web pages, PDFs and text are read. Each
document keeps the licence notice found, the page it was linked from and when it was read;
each crawl is a row in `crawls` (and the audit log) with its rules and every link's outcome.
It runs in the background, one crawl per website at a time, and can be stopped.

Several documents can be removed at once from a library's page (one new version for the
lot), and a library can be reverted to any earlier version from that version's page
(`rag.revert_to`): documents added since are marked removed and documents removed since are
put back, recorded as a new version, so a revert never loses anything and can itself be
reverted. The page shows what a revert would put back and take out before it is done.

A library can be renamed from its page (`rag.rename_library`). The name is a label: one
transaction moves its documents, versions, run records, citations, crawls and hidden setting to
the new name and updates every experiment that names it (libraries, case files, fetch library),
while runs keep the settings they ran with as a record; `library_renames` maps old names to
current ones. A library cannot be renamed, changed or deleted while a run searches it or a crawl
is adding to it.

A library can be hidden from the lists and pickers (its page has "Hide from lists"); hidden
ones sit under a "Hidden libraries" drop-down and stay usable. With no choice recorded in
`library_settings`, a library whose name contains "test" starts hidden.

Each library also has a citation list (`citations.py`): one reference per document, built
from what its source recorded, for the current contents or any version, to copy or download.

## Layout

```
src/ailawlab/
  config.py            settings (env / .env)
  router.py            bounded-queue LLM router across the Sparks
  db.py                async Postgres pool + pgvector binding
  memory.py            two-tier agent memory
  rag.py               corpus ingestion and grounded retrieval, across several libraries
  grounding.py         citation markers in model output, traced to document, library and version
  sources.py           online legal databases (CourtListener, Federal Register, eCFR, govinfo, EDGAR)
  source_material.py   reading a web page, PDF or text to draft a cast from
  agent_spec.py        Markdown agent files: reading, writing, cast checks
  cast_assistant.py    AI-drafted casts, real-name replacement, AI cast review
  tools.py             tool registry for agentic workflows
  network_tools.py     online search and reading for agents, saved into a library
  crawler.py           polite one-level crawl of a page's same-site links into a library
  tracing.py           trace writer + run metrics
  experiments.py       experiment/run lifecycle
  graphs/              LangGraph definitions per mode; roleplay_policy.py holds the moderation rules
  web/                 FastAPI app, page views (views.py), templates, static
db/schema.sql          Postgres schema
docs/                  architecture and method notes, and the scripts that build them
scripts/               benchmark and utilities
tests/                 offline unit tests and integration tests against real hardware
```

Two notes live in `docs/`, each with a script that regenerates it into
`src/ailawlab/web/static/docs/`, where the portal links to it:

- **[System architecture](src/ailawlab/web/static/docs/architecture.pdf)** --
  the one-page diagram above in full, plus a component reference
  (`docs/architecture/build.py`).
- **[Moderating multi-agent legal role-play](src/ailawlab/web/static/docs/roleplay-moderation.pdf)** --
  the moderation logic with references, also as Word, linked from each role-play experiment
  (`docs/roleplay-moderation/build.py`).

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
.venv/bin/python -m pytest -q tests/test_agent_spec.py tests/test_experiment_file.py \
  tests/test_roleplay_policy.py tests/test_source_material.py tests/test_cast_assistant.py \
  tests/test_experiment_page.py
```

## Configuration

See `.env.example`. Everything is overridable by environment variable with the
`AILAWLAB_` prefix.
