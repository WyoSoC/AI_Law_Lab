"""Two-tier agent memory: verbatim short-term buffer + summarized long-term store.

This is a port of the design in `ollama-chat-agent/memory.py`, preserving its structure
(bounded short-term buffer, overflow compacted into model-written summaries, embedded and
retrieved by cosine similarity) while changing four things needed at Lab scale:

  1. Postgres + pgvector instead of SQLite. The original held one sqlite3 connection
     without check_same_thread, which serializes or raises under concurrent agents.
     Retrieval also scanned every row and computed cosine in Python; here it is an
     indexed ANN query.
  2. Scoped to (run_id, agent_id) rather than a single session_id, so a role-play
     scenario can host several agents with private memory inside one shared run.
  3. Provenance. Long-term rows carry the trace event or corpus chunk they came from,
     so a claim recalled from memory can still be traced to a source during cite-check.
  4. Compaction runs off the request path. The original blocked a user turn on a
     summarization round-trip; here it is awaited by the caller only when it chooses to.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import settings
from .db import fetch_all, fetch_one, get_pool, vec
from .router import LLMRouter

log = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Summarize the following conversation excerpt into a concise, factual memory note (120 words max). Preserve names, decisions, legal positions, and concrete facts. Write in third person. Do not add commentary or preamble, just the summary text.

--- CONVERSATION EXCERPT ---
{excerpt}
--- END EXCERPT ---

Summary:"""


def estimate_tokens(text: str) -> int:
    """Rough token estimate.

    Legal prose runs denser than the usual 4 chars/token rule because citations and
    statutory references tokenize badly, so this uses 3.5 to avoid under-budgeting and
    silently overflowing the context window.
    """
    return max(1, int(len(text) / 3.5))


@dataclass
class MemoryScope:
    run_id: str
    agent_id: str = "default"


class Memory:
    """Memory for one (run, agent) pair."""

    def __init__(self, scope: MemoryScope, router: LLMRouter):
        self.scope = scope
        self.router = router
        self.cfg = settings

    # ---------------------------------------------------------------- short-term

    async def add_message(self, role: str, content: str) -> None:
        pool = await get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO memory_messages (run_id, agent_id, role, content) "
                "VALUES (%s, %s, %s, %s)",
                (self.scope.run_id, self.scope.agent_id, role, content),
            )

    async def short_term(self) -> list[dict[str, str]]:
        rows = await fetch_all(
            "SELECT role, content FROM memory_messages "
            "WHERE run_id=%s AND agent_id=%s AND archived=FALSE ORDER BY id",
            (self.scope.run_id, self.scope.agent_id),
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    async def short_term_tokens(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in await self.short_term())

    # ---------------------------------------------------------------- long-term

    async def store_long_term(
        self,
        content: str,
        kind: str = "summary",
        *,
        source_event_id: int | None = None,
        source_chunk_id: int | None = None,
    ) -> int:
        embedding = await self.router.embed_one(content)
        row = await fetch_one(
            "INSERT INTO memory_long_term "
            "(run_id, agent_id, kind, content, embedding, source_event_id, source_chunk_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (self.scope.run_id, self.scope.agent_id, kind, content, vec(embedding),
             source_event_id, source_chunk_id),
        )
        return row["id"]

    async def add_fact(self, content: str, **kw) -> int:
        """Record a durable fact independent of conversation flow."""
        return await self.store_long_term(content, kind="fact", **kw)

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Nearest long-term memories above the similarity floor.

        pgvector's `<=>` is cosine *distance*, so similarity is 1 - distance. The
        threshold is applied in SQL to keep weak matches out of the prompt entirely.
        """
        top_k = top_k or self.cfg.retrieval_top_k
        qvec = vec(await self.router.embed_one(query))
        return await fetch_all(
            "SELECT id, kind, content, source_event_id, source_chunk_id, "
            "       1 - (embedding <=> %s) AS similarity "
            "FROM memory_long_term "
            "WHERE run_id=%s AND agent_id=%s AND embedding IS NOT NULL "
            "  AND 1 - (embedding <=> %s) >= %s "
            "ORDER BY embedding <=> %s LIMIT %s",
            (qvec, self.scope.run_id, self.scope.agent_id, qvec,
             self.cfg.retrieval_min_similarity, qvec, top_k),
        )

    async def list_long_term(self) -> list[dict]:
        return await fetch_all(
            "SELECT id, kind, content, created_at FROM memory_long_term "
            "WHERE run_id=%s AND agent_id=%s ORDER BY id",
            (self.scope.run_id, self.scope.agent_id),
        )

    # ---------------------------------------------------------------- compaction

    async def maybe_compact(self) -> bool:
        """Fold the oldest half of an over-budget short-term buffer into a summary."""
        if await self.short_term_tokens() <= self.cfg.short_term_token_budget:
            return False

        rows = await fetch_all(
            "SELECT id, role, content FROM memory_messages "
            "WHERE run_id=%s AND agent_id=%s AND archived=FALSE ORDER BY id",
            (self.scope.run_id, self.scope.agent_id),
        )
        if len(rows) < 2:
            return False

        n_fold = max(2, int(len(rows) * self.cfg.compact_fraction))
        chunk = rows[:n_fold]
        excerpt = "\n".join(f"{r['role']}: {r['content']}" for r in chunk)

        # think=False: this is mechanical compression, not reasoning worth tracing.
        res = await self.router.chat(
            [{"role": "user", "content": SUMMARIZE_PROMPT.format(excerpt=excerpt)}],
            think=False,
            temperature=0.1,
            options={"num_predict": 400},
        )
        summary = res.text.strip()
        if not summary:
            log.warning("compaction produced an empty summary; keeping buffer intact")
            return False

        await self.store_long_term(summary, kind="summary")
        pool = await get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "UPDATE memory_messages SET archived=TRUE WHERE id = ANY(%s)",
                ([r["id"] for r in chunk],),
            )
        return True

    # ---------------------------------------------------------------- prompt assembly

    async def build_messages(self, user_query: str, system_prompt: str = "") -> list[dict]:
        """system + retrieved long-term memory + short-term buffer + the new message."""
        relevant = await self.retrieve(user_query)
        parts: list[str] = []
        if system_prompt:
            parts.append(system_prompt)
        if relevant:
            parts.append(
                "Relevant memory from earlier in this matter:\n"
                + "\n".join(f"- {m['content']}" for m in relevant)
            )
        messages: list[dict] = []
        if parts:
            messages.append({"role": "system", "content": "\n\n".join(parts)})
        messages.extend(await self.short_term())
        messages.append({"role": "user", "content": user_query})
        return messages
