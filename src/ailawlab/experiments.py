"""Experiment manager: configure, launch, and record experiment runs.

A run snapshots its experiment's config at launch, so editing an experiment later never
rewrites the conditions a past result was produced under. That property is what makes
results in this system citable in a paper.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from .db import fetch_all, fetch_one, get_pool, jsonb
from .graphs.agentic_workflow import build_agentic_graph
from .graphs.document_analysis import build_document_graph
from .graphs.roleplay import build_roleplay_graph
from .graphs.state import RunContext
from .rag import Corpus
from .router import get_router
from .tools import default_registry
from .tracing import Tracer, run_metrics

log = logging.getLogger(__name__)

MODES = ("document_analysis", "agentic_workflow", "roleplay")

# Graphs are stateless and compiled once at import; only RunContext varies per run.
_GRAPHS = {
    "document_analysis": build_document_graph(),
    "agentic_workflow": build_agentic_graph(),
    "roleplay": build_roleplay_graph(),
}


# ---------------------------------------------------------------- experiments


async def create_experiment(name: str, mode: str, description: str = "",
                            config: dict | None = None, created_by: str = "unknown") -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    return await fetch_one(
        "INSERT INTO experiments (name, mode, description, config, created_by) "
        "VALUES (%s,%s,%s,%s,%s) RETURNING *",
        (name, mode, description, jsonb(config or {}), created_by),
    )


async def list_experiments() -> list[dict]:
    return await fetch_all(
        "SELECT e.*, "
        "  (SELECT COUNT(*) FROM runs r WHERE r.experiment_id = e.id) AS run_count, "
        "  (SELECT MAX(r.created_at) FROM runs r WHERE r.experiment_id = e.id) AS last_run "
        "FROM experiments e ORDER BY e.created_at DESC"
    )


async def get_experiment(experiment_id: str) -> dict | None:
    return await fetch_one("SELECT * FROM experiments WHERE id=%s", (experiment_id,))


# ---------------------------------------------------------------- runs


async def create_run(experiment_id: str, inputs: dict | None = None) -> dict:
    exp = await get_experiment(experiment_id)
    if exp is None:
        raise ValueError(f"no such experiment: {experiment_id}")
    return await fetch_one(
        "INSERT INTO runs (experiment_id, config_snapshot, inputs, status) "
        "VALUES (%s,%s,%s,'pending') RETURNING *",
        (experiment_id, jsonb(exp["config"]), jsonb(inputs or {})),
    )


async def get_run(run_id: str) -> dict | None:
    return await fetch_one(
        "SELECT r.*, e.name AS experiment_name, e.mode "
        "FROM runs r JOIN experiments e ON e.id = r.experiment_id WHERE r.id=%s",
        (run_id,),
    )


async def list_runs(experiment_id: str | None = None, limit: int = 50) -> list[dict]:
    if experiment_id:
        return await fetch_all(
            "SELECT r.*, e.name AS experiment_name, e.mode FROM runs r "
            "JOIN experiments e ON e.id = r.experiment_id "
            "WHERE r.experiment_id=%s ORDER BY r.created_at DESC LIMIT %s",
            (experiment_id, limit),
        )
    return await fetch_all(
        "SELECT r.*, e.name AS experiment_name, e.mode FROM runs r "
        "JOIN experiments e ON e.id = r.experiment_id ORDER BY r.created_at DESC LIMIT %s",
        (limit,),
    )


async def _set_status(run_id: str, status: str, **fields) -> None:
    sets = ["status=%s"]
    params: list[Any] = [status]
    for k, v in fields.items():
        sets.append(f"{k}=%s")
        params.append(v)
    params.append(run_id)
    pool = await get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=%s", params)


def _initial_state(mode: str, config: dict, inputs: dict) -> dict:
    """Build the graph's entry state from experiment config + per-run inputs."""
    merged = {**config, **inputs}
    if mode == "document_analysis":
        return {
            "document_title": merged.get("document_title", "(untitled)"),
            "document_text": merged.get("document_text", ""),
            "document_id": merged.get("document_id"),
            "question": merged.get("question", "Summarize this document and flag legal risks."),
            "findings": [],
        }
    if mode == "agentic_workflow":
        return {
            "task": merged.get("task", ""),
            "scratchpad": [],
            "tool_results": [],
            "iterations": 0,
            "max_iterations": int(merged.get("max_iterations", 8)),
            "done": False,
        }
    if mode == "roleplay":
        return {
            "scenario": merged.get("scenario", ""),
            "agents": merged.get("agents", []),
            "transcript": [],
            "turn": 0,
            "max_turns": int(merged.get("max_turns", 12)),
            "done": False,
        }
    raise ValueError(f"unknown mode {mode!r}")


async def execute_run(run_id: str) -> dict:
    """Execute a run to completion, recording trace and result.

    Any failure is written to the run row and re-raised context-free to the caller as a
    status, not an exception -- a failed experiment is a result, and the trace up to the
    failure point is still worth keeping.
    """
    run = await get_run(run_id)
    if run is None:
        raise ValueError(f"no such run: {run_id}")

    mode = run["mode"]
    config = run["config_snapshot"] or {}
    inputs = run["inputs"] or {}

    router = await get_router()
    corpus = Corpus(router, name=config.get("corpus", "default"))
    tracer = Tracer(run_id)
    ctx = RunContext(run_id=run_id, router=router, tracer=tracer, corpus=corpus, config=config)

    if mode == "agentic_workflow":
        ctx.config = {**config,
                      "registry": default_registry(corpus,
                                                   allow_network=config.get("allow_network", False))}

    await _set_status(run_id, "running", started_at=datetime.now(UTC))
    await tracer.note(f"run started (mode={mode})")

    try:
        graph = _GRAPHS[mode]
        state = _initial_state(mode, config, inputs)
        # recursion_limit must exceed 2x max_turns for roleplay's moderator/speak cycle.
        limit = int(config.get("max_turns", 12)) * 3 + 20
        final = await graph.ainvoke(
            state,
            config={"configurable": {"ctx": ctx}, "recursion_limit": limit},
        )
        result = _summarize(mode, final)
        metrics = await run_metrics(run_id)
        result["metrics"] = metrics

        await _set_status(run_id, "succeeded", result=jsonb(result),
                          finished_at=datetime.now(UTC))
        await tracer.note("run completed")
        return result

    except Exception as e:
        log.exception("run %s failed", run_id)
        await tracer.error(f"{type(e).__name__}: {e}")
        await _set_status(run_id, "failed", error=f"{type(e).__name__}: {e}",
                          finished_at=datetime.now(UTC))
        raise


def _summarize(mode: str, final: dict) -> dict:
    if mode == "document_analysis":
        return {
            "answer": final.get("answer", ""),
            "plan": final.get("plan", []),
            "findings": final.get("findings", []),
            "citations": final.get("citations", []),
            "passages_used": len(final.get("passages", [])),
        }
    if mode == "agentic_workflow":
        return {
            "answer": final.get("answer", ""),
            "iterations": final.get("iterations", 0),
            "tool_results": final.get("tool_results", []),
        }
    if mode == "roleplay":
        return {
            "outcome": final.get("outcome", ""),
            "transcript": final.get("transcript", []),
            "turns": final.get("turn", 0),
        }
    return dict(final)


# Strong references to in-flight background runs. asyncio only holds a weak reference
# to a task, so without this a long run can be garbage-collected mid-execution.
_background_runs: set[asyncio.Task] = set()


async def launch(experiment_id: str, inputs: dict | None = None) -> str:
    """Create a run and execute it in the background. Returns the run id immediately."""
    run = await create_run(experiment_id, inputs)
    run_id = str(run["id"])

    async def _bg() -> None:
        try:
            await execute_run(run_id)
        except Exception:
            # execute_run already wrote the failure to the run row and the trace;
            # log here so it also surfaces in server output rather than vanishing.
            log.warning("background run %s failed; see runs.error", run_id)

    task = asyncio.create_task(_bg())
    _background_runs.add(task)
    task.add_done_callback(_background_runs.discard)
    return run_id
