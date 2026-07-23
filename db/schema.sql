-- AI Law Lab schema.
--
-- Design notes:
--   * Everything an experiment produces is addressable by run_id so a result can be
--     reproduced and audited after the fact.
--   * run_events is the append-only trace backing the logging/evaluation tier: one row
--     per LLM call, tool call, or retrieval, with timings split into queue-wait and
--     model-eval (the Spark benchmark showed those diverge sharply under load).
--   * Memory keeps the two-tier structure of the original SQLite agent (verbatim
--     short-term, summarized long-term) but is scoped to (run_id, agent_id) so several
--     agents can hold private memory inside one shared scenario.
--   * Embeddings are nomic-embed-text, 768 dimensions.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------- experiments

CREATE TABLE IF NOT EXISTS experiments (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT NOT NULL,
    mode         TEXT NOT NULL CHECK (mode IN ('document_analysis', 'agentic_workflow', 'roleplay')),
    description  TEXT NOT NULL DEFAULT '',
    -- Full experiment configuration: model, temperature, scenario, agent roster,
    -- retrieval settings. Snapshotted into each run so later edits don't rewrite history.
    config       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by   TEXT NOT NULL DEFAULT 'unknown',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id  UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    status         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')),
    -- Immutable copy of experiments.config at launch time -> reproducibility.
    config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    inputs         JSONB NOT NULL DEFAULT '{}'::jsonb,
    result         JSONB,
    error          TEXT,
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status) WHERE status IN ('pending', 'running');

-- ---------------------------------------------------------------- trace / eval

CREATE TABLE IF NOT EXISTS run_events (
    id            BIGSERIAL PRIMARY KEY,
    run_id        UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    agent_id      TEXT,
    node          TEXT,
    event_type    TEXT NOT NULL
                  CHECK (event_type IN ('llm_call', 'tool_call', 'retrieval', 'node_enter',
                                        'node_exit', 'memory_write', 'error', 'note')),
    payload       JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Reasoning trace from gemma4's `thinking` capability, kept separate from the answer.
    thinking      TEXT,
    -- Timing split: queue_wait_ms is time spent waiting for a Spark slot, eval_ms is
    -- actual model time. Conflating them makes throughput numbers meaningless.
    queue_wait_ms INTEGER,
    eval_ms       INTEGER,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    host          TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_run ON run_events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_type ON run_events(run_id, event_type);

-- Citations asserted by a model, linked back to the chunk they should be grounded in.
-- verified/verdict are filled by the citation checker in the evaluation tier.
CREATE TABLE IF NOT EXISTS citations (
    id           BIGSERIAL PRIMARY KEY,
    run_id       UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    event_id     BIGINT REFERENCES run_events(id) ON DELETE CASCADE,
    quoted_text  TEXT NOT NULL,
    source_label TEXT,
    chunk_id     BIGINT,
    similarity   REAL,
    verdict      TEXT CHECK (verdict IN ('grounded', 'unsupported', 'unchecked')) DEFAULT 'unchecked',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_citations_run ON citations(run_id);

-- ---------------------------------------------------------------- RAG corpus

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    corpus      TEXT NOT NULL DEFAULT 'default',
    title       TEXT NOT NULL,
    source_uri  TEXT,
    doc_type    TEXT NOT NULL DEFAULT 'other',
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    sha256      TEXT UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_documents_corpus ON documents(corpus);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    content      TEXT NOT NULL,
    -- Page/section anchors so a citation can point at a location a human can check.
    page_start   INTEGER,
    page_end     INTEGER,
    embedding    vector(768),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
-- HNSW beats IVFFlat here: the corpus grows incrementally and HNSW needs no retraining.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------- memory

CREATE TABLE IF NOT EXISTS memory_messages (
    id          BIGSERIAL PRIMARY KEY,
    run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    agent_id    TEXT NOT NULL DEFAULT 'default',
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    archived    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_memmsg_scope ON memory_messages(run_id, agent_id, archived, id);

CREATE TABLE IF NOT EXISTS memory_long_term (
    id          BIGSERIAL PRIMARY KEY,
    run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    agent_id    TEXT NOT NULL DEFAULT 'default',
    kind        TEXT NOT NULL DEFAULT 'summary' CHECK (kind IN ('summary', 'fact', 'observation')),
    content     TEXT NOT NULL,
    embedding   vector(768),
    -- Provenance: which trace event or document this memory came from, so a claim
    -- recalled from memory can still be traced to a source.
    source_event_id BIGINT REFERENCES run_events(id) ON DELETE SET NULL,
    source_chunk_id BIGINT REFERENCES chunks(id) ON DELETE SET NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_memlt_scope ON memory_long_term(run_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_memlt_embedding ON memory_long_term
    USING hnsw (embedding vector_cosine_ops);
