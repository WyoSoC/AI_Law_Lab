"""Logging and evaluation tier: the append-only trace behind every run.

One row per LLM call, tool call, retrieval, and node transition. Timings are stored
split -- queue_wait_ms (waiting for a Spark slot) apart from eval_ms (actual model
time) -- because the benchmark showed those diverge by 3x under load. Reporting them
combined would make an experiment look slow when the cluster was merely busy.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .db import fetch_all, fetch_one, get_pool, jsonb
from .router import LLMResult

log = logging.getLogger(__name__)


class Tracer:
    """Sequenced event writer for one run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._seq = 0
        self._lock = asyncio.Lock()

    async def _next_seq(self) -> int:
        async with self._lock:
            self._seq += 1
            return self._seq

    async def event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        agent_id: str | None = None,
        node: str | None = None,
        thinking: str | None = None,
        queue_wait_ms: int | None = None,
        eval_ms: int | None = None,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        host: str | None = None,
    ) -> int:
        seq = await self._next_seq()
        row = await fetch_one(
            "INSERT INTO run_events (run_id, seq, agent_id, node, event_type, payload, "
            " thinking, queue_wait_ms, eval_ms, prompt_tokens, output_tokens, host) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (self.run_id, seq, agent_id, node, event_type, jsonb(payload or {}), thinking,
             queue_wait_ms, eval_ms, prompt_tokens, output_tokens, host),
        )
        return row["id"]

    async def llm_call(
        self, res: LLMResult, *, node: str | None = None, agent_id: str | None = None,
        prompt_preview: str = "",
    ) -> int:
        return await self.event(
            "llm_call",
            node=node,
            agent_id=agent_id,
            payload={
                "prompt_preview": prompt_preview[:2000],
                "response": res.text[:8000],
                "tool_calls": res.tool_calls,
                "done_reason": res.done_reason,
                "truncated": res.truncated,
                "tokens_per_s": round(res.tokens_per_s, 1),
            },
            thinking=res.thinking,
            queue_wait_ms=res.queue_wait_ms,
            eval_ms=res.eval_ms,
            prompt_tokens=res.prompt_tokens,
            output_tokens=res.output_tokens,
            host=res.host,
        )

    async def retrieval(self, query: str, passages: list, *, node: str | None = None,
                        agent_id: str | None = None) -> int:
        return await self.event(
            "retrieval",
            node=node,
            agent_id=agent_id,
            payload={
                "query": query,
                "hits": [
                    {"chunk_id": p.chunk_id, "label": p.cite_label(),
                     "similarity": round(p.similarity, 4)}
                    for p in passages
                ],
            },
        )

    async def tool_call(self, name: str, args: dict, result: Any, *, node: str | None = None,
                        agent_id: str | None = None, eval_ms: int | None = None) -> int:
        return await self.event(
            "tool_call",
            node=node,
            agent_id=agent_id,
            eval_ms=eval_ms,
            payload={"tool": name, "args": args, "result": str(result)[:4000]},
        )

    async def error(self, message: str, *, node: str | None = None,
                    agent_id: str | None = None) -> int:
        log.error("run %s node=%s: %s", self.run_id, node, message)
        return await self.event("error", node=node, agent_id=agent_id,
                                payload={"message": message})

    async def note(self, message: str, **kw) -> int:
        return await self.event("note", payload={"message": message}, **kw)

    # ------------------------------------------------------------------ citations

    async def record_citations(self, event_id: int, citations: list[dict]) -> None:
        if not citations:
            return
        pool = await get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            for c in citations:
                await cur.execute(
                    "INSERT INTO citations (run_id, event_id, quoted_text, source_label, "
                    "chunk_id, similarity, verdict) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (self.run_id, event_id, c.get("quoted_text", ""), c.get("source_label"),
                     c.get("chunk_id"), c.get("similarity"), c.get("verdict", "unchecked")),
                )


async def run_metrics(run_id: str) -> dict[str, Any]:
    """Aggregate performance and grounding metrics for a finished run."""
    perf = await fetch_one(
        "SELECT COUNT(*) FILTER (WHERE event_type='llm_call')   AS llm_calls, "
        "       COUNT(*) FILTER (WHERE event_type='tool_call')  AS tool_calls, "
        "       COUNT(*) FILTER (WHERE event_type='retrieval')  AS retrievals, "
        "       COUNT(*) FILTER (WHERE event_type='error')      AS errors, "
        "       COALESCE(SUM(prompt_tokens),0)  AS prompt_tokens, "
        "       COALESCE(SUM(output_tokens),0)  AS output_tokens, "
        "       COALESCE(SUM(eval_ms),0)        AS total_eval_ms, "
        "       COALESCE(SUM(queue_wait_ms),0)  AS total_queue_wait_ms, "
        "       COALESCE(MAX(queue_wait_ms),0)  AS max_queue_wait_ms "
        "FROM run_events WHERE run_id=%s",
        (run_id,),
    ) or {}

    cites = await fetch_one(
        "SELECT COUNT(*) AS total, "
        "       COUNT(*) FILTER (WHERE verdict='grounded')    AS grounded, "
        "       COUNT(*) FILTER (WHERE verdict='unsupported') AS unsupported, "
        "       COUNT(*) FILTER (WHERE verdict='unchecked')   AS unchecked "
        "FROM citations WHERE run_id=%s",
        (run_id,),
    ) or {}

    hosts = await fetch_all(
        "SELECT host, COUNT(*) AS calls FROM run_events "
        "WHERE run_id=%s AND host IS NOT NULL GROUP BY host ORDER BY host",
        (run_id,),
    )

    total = cites.get("total") or 0
    grounded = cites.get("grounded") or 0
    return {
        "performance": {
            **perf,
            # The ratio that matters when reading a slow run: was it the model, or the queue?
            "queue_share": (
                round(perf.get("total_queue_wait_ms", 0)
                      / max(1, perf.get("total_queue_wait_ms", 0) + perf.get("total_eval_ms", 0)), 3)
            ),
        },
        "citations": {**cites, "grounded_rate": round(grounded / total, 3) if total else None},
        "host_distribution": {h["host"]: h["calls"] for h in hosts},
    }
