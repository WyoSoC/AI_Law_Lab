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
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Set when the experiment is moved to the trash; NULL while it is in use. Trashed
    -- experiments keep their runs until they are deleted for good.
    deleted_at   TIMESTAMPTZ
);
-- For databases created before the trash existed.
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

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
-- The library a run searched and which version of it (NULL for runs without one, and for
-- runs from before libraries were versioned).
ALTER TABLE runs ADD COLUMN IF NOT EXISTS corpus TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS corpus_version INTEGER;
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
-- Where a citation points, kept on the row itself: the document, the library it sits in and
-- the version the run searched, and where in the output the marker appeared ("Answer",
-- "Turn 7 · Dana Reyes"). A chunk id alone stops resolving once its library is deleted.
ALTER TABLE citations ADD COLUMN IF NOT EXISTS document_id    BIGINT;
ALTER TABLE citations ADD COLUMN IF NOT EXISTS corpus         TEXT;
ALTER TABLE citations ADD COLUMN IF NOT EXISTS corpus_version INTEGER;
ALTER TABLE citations ADD COLUMN IF NOT EXISTS context        TEXT NOT NULL DEFAULT '';

-- ---------------------------------------------------------------- RAG corpus

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    corpus      TEXT NOT NULL DEFAULT 'default',
    title       TEXT NOT NULL,
    source_uri  TEXT,
    doc_type    TEXT NOT NULL DEFAULT 'other',
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    sha256      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Documents are never edited or deleted while their library exists, so every past
    -- version of a library can still be searched exactly as it was. Removing a document
    -- sets removed_at; re-reading a changed web page or moving a document to another
    -- library adds a new row and points the old one at it through replaced_by.
    removed_at  TIMESTAMPTZ,
    replaced_by BIGINT REFERENCES documents(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_corpus ON documents(corpus);
-- For databases created before versioning: the same text may now sit in several libraries,
-- but only once in each library's current contents.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS removed_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS replaced_by BIGINT REFERENCES documents(id) ON DELETE SET NULL;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_sha256_key;
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_live_sha ON documents(corpus, sha256)
    WHERE removed_at IS NULL;

-- A library version is the exact set of documents it held after a change. Runs record the
-- version they searched, so a result can be traced to, and re-run against, that set.
CREATE TABLE IF NOT EXISTS corpus_versions (
    corpus        TEXT NOT NULL,
    version       INTEGER NOT NULL,
    document_ids  BIGINT[] NOT NULL,
    change        TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus, version)
);

-- The libraries a run searched, each pinned to the version it searched. A run may search
-- several; `position` keeps the order they were chosen in. Rows are written when the run is
-- created (version NULL) so a library in use can be told apart, and the version is filled
-- in when the run starts. runs.corpus / runs.corpus_version are the single-library columns
-- this replaced: older runs are copied over here, and nothing writes them any more.
CREATE TABLE IF NOT EXISTS run_libraries (
    run_id     UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    position   SMALLINT NOT NULL DEFAULT 0,
    corpus     TEXT NOT NULL,
    version    INTEGER,
    documents  INTEGER,
    PRIMARY KEY (run_id, corpus)
);
CREATE INDEX IF NOT EXISTS idx_run_libraries_version ON run_libraries(corpus, version);
INSERT INTO run_libraries (run_id, corpus, version)
    SELECT id, corpus, corpus_version FROM runs WHERE corpus IS NOT NULL
    ON CONFLICT DO NOTHING;

-- Whether a library is shown in the lists and pickers. A library has no table of its own,
-- so this holds only libraries someone has hidden or shown; with no row, a library whose
-- name contains "test" is hidden and any other is shown (rag.library_hidden).
CREATE TABLE IF NOT EXISTS library_settings (
    corpus      TEXT PRIMARY KEY,
    hidden      BOOLEAN NOT NULL,
    changed_by  UUID REFERENCES users(id) ON DELETE SET NULL,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- "Add everything linked from this page" (crawler.py): one row per crawl, with the rules it
-- ran under and what happened to every link, so a library's crawled documents can be traced
-- to the crawl that took them and to who started it.
CREATE TABLE IF NOT EXISTS crawls (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus       TEXT NOT NULL,
    start_url    TEXT NOT NULL,
    host         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running'
                 CHECK (status IN ('running', 'stopping', 'finished', 'stopped', 'failed', 'interrupted')),
    max_pages    INTEGER NOT NULL,
    rules        JSONB NOT NULL DEFAULT '{}'::jsonb,
    total        INTEGER NOT NULL DEFAULT 0,
    done         INTEGER NOT NULL DEFAULT 0,
    added        INTEGER NOT NULL DEFAULT 0,
    results      JSONB NOT NULL DEFAULT '[]'::jsonb,
    message      TEXT NOT NULL DEFAULT '',
    version      INTEGER,
    started_by   UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_crawls_created ON crawls(created_at DESC);

-- Every renaming of a library: the name is a label, so renaming updates every reference to
-- it (rag.rename_library) and this log keeps the old name findable. Runs' stored settings
-- keep the names they ran under; the log maps those to the current name.
CREATE TABLE IF NOT EXISTS library_renames (
    id          BIGSERIAL PRIMARY KEY,
    old_name    TEXT NOT NULL,
    new_name    TEXT NOT NULL,
    renamed_by  UUID REFERENCES users(id) ON DELETE SET NULL,
    renamed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

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

-- ---------------------------------------------------------------- accounts
--
-- People sign in through Keycloak (see docs/keycloak.md); this table holds only what the
-- app decides about them. `subject` is Keycloak's stable id for the account. Everything in
-- the lab is shared among approved users, so ownership columns below are attribution --
-- who created, launched or changed something -- not access control.

CREATE TABLE IF NOT EXISTS users (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject           TEXT NOT NULL UNIQUE,
    email             TEXT NOT NULL DEFAULT '',
    email_verified    BOOLEAN NOT NULL DEFAULT FALSE,
    name              TEXT NOT NULL DEFAULT '',
    identity_provider TEXT NOT NULL DEFAULT '',     -- uwyo, google, microsoft, or '' (Keycloak account)
    status            TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'disabled')),
    role              TEXT NOT NULL DEFAULT 'researcher' CHECK (role IN ('admin', 'researcher', 'viewer')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at     TIMESTAMPTZ,
    approved_by       UUID REFERENCES users(id) ON DELETE SET NULL,
    approved_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);

-- Security-relevant actions: approvals, role changes, and anything deleted for good.
CREATE TABLE IF NOT EXISTS audit_log (
    id        BIGSERIAL PRIMARY KEY,
    at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id   UUID REFERENCES users(id) ON DELETE SET NULL,
    action    TEXT NOT NULL,
    target    TEXT NOT NULL DEFAULT '',
    detail    JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_audit_at ON audit_log(at DESC);

ALTER TABLE experiments     ADD COLUMN IF NOT EXISTS owner_id    UUID REFERENCES users(id) ON DELETE SET NULL;
ALTER TABLE runs            ADD COLUMN IF NOT EXISTS launched_by UUID REFERENCES users(id) ON DELETE SET NULL;
ALTER TABLE documents       ADD COLUMN IF NOT EXISTS added_by    UUID REFERENCES users(id) ON DELETE SET NULL;
ALTER TABLE corpus_versions ADD COLUMN IF NOT EXISTS changed_by  UUID REFERENCES users(id) ON DELETE SET NULL;
