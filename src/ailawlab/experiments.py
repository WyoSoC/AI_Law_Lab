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

from .agent_spec import check_cast, normalize_agent
from .config import settings
from .db import fetch_all, fetch_one, get_pool, jsonb
from .graphs.agentic_workflow import build_agentic_graph
from .graphs.document_analysis import build_document_graph
from .graphs.roleplay import build_roleplay_graph
from .graphs.state import RunContext
from .grounding import SourceLedger
from .network_tools import OnlineReader, fetch_library_name, given_pages, sources_wanted
from .rag import Libraries, all_run_library_names, pin_libraries, run_library_names, wanted_versions
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
                            config: dict | None = None, created_by: str = "unknown",
                            owner_id: Any = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    return await fetch_one(
        "INSERT INTO experiments (name, mode, description, config, created_by, owner_id) "
        "VALUES (%s,%s,%s,%s,%s,%s) RETURNING *",
        (name, mode, description, jsonb(config or {}), created_by, owner_id),
    )


_LIST_SQL = (
    "SELECT e.*, "
    "  (SELECT COUNT(*) FROM runs r WHERE r.experiment_id = e.id) AS run_count, "
    "  (SELECT MAX(r.created_at) FROM runs r WHERE r.experiment_id = e.id) AS last_run "
    "FROM experiments e "
)


async def list_experiments() -> list[dict]:
    """Experiments in use, newest first. Trashed ones are listed by list_trash()."""
    return await fetch_all(_LIST_SQL + "WHERE e.deleted_at IS NULL ORDER BY e.created_at DESC")


async def get_experiment(experiment_id: str) -> dict | None:
    """An experiment whether or not it is in the trash, so links to it keep working."""
    return await fetch_one("SELECT * FROM experiments WHERE id=%s", (experiment_id,))


# ---------------------------------------------------------------- trash
#
# Deleting an experiment is two steps. Moving it to the trash only hides it: its runs,
# traces and results stay, and it can be restored. Deleting it for good removes all of
# that (runs cascade), which is why only an experiment already in the trash can be.


class ExperimentBusy(Exception):
    """The experiment has a run in progress, so it cannot be trashed or deleted."""


async def _active_runs(experiment_id: str) -> int:
    row = await fetch_one(
        "SELECT COUNT(*) AS n FROM runs WHERE experiment_id=%s "
        "AND status IN ('pending', 'running')", (experiment_id,))
    return row["n"] if row else 0


async def list_trash() -> list[dict]:
    return await fetch_all(_LIST_SQL + "WHERE e.deleted_at IS NOT NULL ORDER BY e.deleted_at DESC")


async def trash_count() -> int:
    row = await fetch_one("SELECT COUNT(*) AS n FROM experiments WHERE deleted_at IS NOT NULL")
    return row["n"] if row else 0


async def trash_experiment(experiment_id: str) -> dict | None:
    """Move an experiment to the trash. Returns it, or None if there is no such experiment."""
    if await _active_runs(experiment_id):
        raise ExperimentBusy("A run of this experiment is still in progress. "
                             "Wait for it to finish before moving the experiment to the trash.")
    return await fetch_one(
        "UPDATE experiments SET deleted_at = COALESCE(deleted_at, now()) WHERE id=%s RETURNING *",
        (experiment_id,))


async def restore_experiment(experiment_id: str) -> dict | None:
    return await fetch_one(
        "UPDATE experiments SET deleted_at = NULL WHERE id=%s RETURNING *", (experiment_id,))


async def delete_experiment(experiment_id: str) -> dict | None:
    """Delete a trashed experiment for good, with its runs, traces and citations.
    Returns None if it does not exist or is not in the trash."""
    if await _active_runs(experiment_id):
        raise ExperimentBusy("A run of this experiment is still in progress.")
    return await fetch_one(
        "DELETE FROM experiments WHERE id=%s AND deleted_at IS NOT NULL RETURNING id, name",
        (experiment_id,))


async def empty_trash() -> int:
    """Delete every trashed experiment that has no run in progress. Returns how many."""
    rows = await fetch_all(
        "DELETE FROM experiments e WHERE e.deleted_at IS NOT NULL AND NOT EXISTS ("
        "  SELECT 1 FROM runs r WHERE r.experiment_id = e.id "
        "  AND r.status IN ('pending', 'running')) RETURNING e.id")
    return len(rows)


# ---------------------------------------------------------------- runs


async def create_run(experiment_id: str, inputs: dict | None = None,
                     launched_by: Any = None) -> dict:
    exp = await get_experiment(experiment_id)
    if exp is None:
        raise ValueError(f"no such experiment: {experiment_id}")
    if exp.get("deleted_at"):
        raise ValueError("this experiment is in the trash; restore it before running it")
    names = all_run_library_names(exp["mode"], exp["config"] or {}, inputs or {})
    merged = {**(exp["config"] or {}), **(inputs or {})}
    if exp["mode"] == "document_analysis":
        if not str(merged.get("question") or "").strip():
            raise ValueError("Ask a question for this run.")
        if not str(merged.get("document_text") or "").strip() and not names:
            raise ValueError("Give a document to analyze, or choose at least one library to ask.")
    pool = await get_pool()
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO runs (experiment_id, config_snapshot, inputs, status, launched_by) "
            "VALUES (%s,%s,%s,'pending',%s) RETURNING *",
            (experiment_id, jsonb(exp["config"]), jsonb(inputs or {}), launched_by))
        run = await cur.fetchone()
        # Recorded now, before the run starts, so its libraries count as in use (see
        # rag.check_idle); the versions are filled in when it starts.
        for position, name in enumerate(names):
            await cur.execute("INSERT INTO run_libraries (run_id, position, corpus) "
                              "VALUES (%s, %s, %s)", (run["id"], position, name))
    return run


_RUN_LIBRARIES = (
    "COALESCE((SELECT jsonb_agg(jsonb_build_object('name', rl.corpus, 'version', rl.version, "
    "  'documents', rl.documents) ORDER BY rl.position) FROM run_libraries rl "
    "  WHERE rl.run_id = r.id), '[]'::jsonb) AS libraries")


async def get_run(run_id: str) -> dict | None:
    """A run, with `libraries`: the libraries it searches, in order, as {name, version,
    documents} (version None until the run starts, or for a library that was empty)."""
    return await fetch_one(
        "SELECT r.*, e.name AS experiment_name, e.mode, "
        "       COALESCE(NULLIF(u.name, ''), u.email) AS launched_by_name, "
        f"      {_RUN_LIBRARIES} "
        "FROM runs r JOIN experiments e ON e.id = r.experiment_id "
        "LEFT JOIN users u ON u.id = r.launched_by WHERE r.id=%s",
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
        "JOIN experiments e ON e.id = r.experiment_id WHERE e.deleted_at IS NULL "
        "ORDER BY r.created_at DESC LIMIT %s",
        (limit,),
    )


async def _set_fields(run_id: str, **fields) -> None:
    sets = ", ".join(f"{k}=%s" for k in fields)
    pool = await get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(f"UPDATE runs SET {sets} WHERE id=%s", [*fields.values(), run_id])


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
            "document_title": merged.get("document_title", ""),
            "document_text": merged.get("document_text", ""),
            "scratchpad": [],
            "tool_results": [],
            "iterations": 0,
            # Not the experiment's: older experiments set a small number, and the agent
            # is no longer limited by one. It stops when it has an answer.
            "max_iterations": settings.agent_max_steps,
            "done": False,
        }
    if mode == "roleplay":
        max_turns = int(merged.get("max_turns", settings.default_max_turns))
        word_limit = int(merged.get("word_limit", settings.default_word_limit))
        return {
            "scenario": merged.get("scenario", ""),
            "agents": [normalize_agent(a) for a in merged.get("agents", [])],
            "transcript": [],
            "turn": 0,
            "max_turns": max(2, min(max_turns, settings.max_turns_limit)),
            "word_limit": max(40, min(word_limit, settings.word_limit_max)),
            "directive": "",
            "last_intervention": 0,
            "ledgers": {},
            # Legal sources are optional for a role-play; with none, turns are as before.
            # These are the shared ones; an agent's own case files travel with the agent.
            "libraries": run_library_names(mode, config, inputs),
            "exhibits": [],
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
    tracer = Tracer(run_id)
    await _set_status(run_id, "running", started_at=datetime.now(UTC))
    await tracer.note(f"run started (mode={mode})")

    try:
        libraries = await _libraries_for_run(run_id, mode, config, inputs, router, tracer)
        ctx = RunContext(run_id=run_id, router=router, tracer=tracer, libraries=libraries,
                         config=config)
        pages: list[str] = []
        if mode == "agentic_workflow":
            ledger = SourceLedger()
            settings_ = {**config, **inputs}     # a run's own choices win over the experiment's
            allow = bool(settings_.get("allow_network", False))
            # Web pages given to read are part of the network tools: off, none are read.
            pages = given_pages(settings_) if allow else []
            reader = None
            if allow:
                save_to = fetch_library_name(settings_, run["experiment_name"])
                reader = OnlineReader(router, libraries, ledger, tracer, run_id, save_to,
                                      added_by=run.get("launched_by"),
                                      wanted=sources_wanted(settings_))
                row = await fetch_one("SELECT count(*) AS n FROM documents "
                                      "WHERE corpus=%s AND removed_at IS NULL", (save_to,))
                held = int(row["n"]) if row else 0
                where = (f"“{save_to}” ({held} document{'' if held == 1 else 's'} already)" if held
                         else f"a new library, “{save_to}”, created with the first document read")
                await tracer.note(
                    f"network tools on: reads up to {reader.wanted} sources online"
                    + (f"; {len(pages)} web page{'' if len(pages) == 1 else 's'} given to read first"
                       if pages else "") + f". What is read is added to {where}")
            ctx.config = {**config, "ledger": ledger, "reader": reader, "registry": default_registry(
                libraries, ledger, allow_network=allow, reader=reader)}

        graph = _GRAPHS[mode]
        state = _initial_state(mode, config, inputs)
        if mode == "document_analysis" and not state["document_text"].strip() and not libraries:
            raise ValueError("there is no document and the run's libraries hold nothing, so "
                             "there is nothing to answer the question from")
        if mode == "roleplay":
            await _check_roleplay(state, tracer)
        if mode == "agentic_workflow" and pages:
            # The pages the researcher gave are read before the agent's first step, so they
            # are in front of it whatever it decides; it cites their passages like any other.
            await tracer.note(f"reading the {len(pages)} web page{'' if len(pages) == 1 else 's'} "
                              "given with the task")
            state["given_pages"] = await reader.read_given(pages, look_for=state["task"])
        # recursion_limit must exceed 2x max_turns for roleplay's moderator/speak cycle. Read
        # it from the built state, since run inputs may override the experiment's max_turns.
        limit = int(state.get("max_turns", settings.default_max_turns)) * 3 + 20
        if mode == "agentic_workflow":      # each step is a think node and an act node
            limit = 2 * int(state["max_iterations"]) + 10
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


async def _libraries_for_run(run_id: str, mode: str, config: dict, inputs: dict,
                             router, tracer: Tracer) -> Libraries:
    """The libraries this run searches, each pinned to one version and recorded on the run.

    In a role-play these are the shared libraries and every agent's case files; which agent
    may search which is decided per turn (graphs/roleplay.py). A run may choose its own
    shared libraries at launch, and may name a version of any library, to
    repeat an earlier result against a library as it was then. A library with no version
    named is searched as it is now: its current contents are recorded as a version if they
    changed since the last one, and that version is used.
    """
    names = all_run_library_names(mode, config, inputs)
    libraries = await pin_libraries(router, names, wanted_versions(names, inputs))
    pool = await get_pool()
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        # Replaced rather than updated, so a run created some other way still ends up with
        # exactly the libraries it searched.
        await cur.execute("DELETE FROM run_libraries WHERE run_id=%s", (run_id,))
        for position, pin in enumerate(libraries.pins):
            await cur.execute(
                "INSERT INTO run_libraries (run_id, position, corpus, version, documents) "
                "VALUES (%s, %s, %s, %s, %s)",
                (run_id, position, pin.name, pin.version, len(pin.document_ids)))
    for pin in libraries.pins:
        if pin.version is None:
            await tracer.note(f"library “{pin.name}” is empty; nothing can be retrieved from it")
    if libraries:
        n = sum(len(p.document_ids) for p in libraries.pins)
        await tracer.note(f"searching {libraries.describe()} "
                          f"({n} document{'' if n == 1 else 's'} in all)")
    elif not names:
        await tracer.note("no library: nothing is retrieved in this run")
    return libraries


async def _check_roleplay(state: dict, tracer: Tracer) -> None:
    """Fail fast on a cast that cannot run; record lesser problems in the trace.

    Without this, a duplicate id or a moderator pick that matches no agent surfaces mid-run
    as a bare StopIteration, long after the cause was knowable.
    """
    issues = check_cast(state["agents"])
    errors = [i["message"] for i in issues if i["level"] == "error"]
    if errors:
        raise ValueError("the cast cannot run: " + " ".join(errors))
    if not state["scenario"].strip():
        raise ValueError("the run has no scenario; add one to the experiment or the run inputs")
    for i in issues:
        await tracer.note(f"cast check ({i['level']}): {i['message']}")


def _summarize(mode: str, final: dict) -> dict:
    """The run's stored result. `sources` lists every passage the model was given, by the
    marker it could cite it with, with its document, library and version."""
    if mode == "document_analysis":
        return {
            "answer": final.get("answer", ""),
            "plan": final.get("plan", []),
            "findings": final.get("findings", []),
            "citations": final.get("citations", []),
            "sources": final.get("sources", []),
            "passages_used": len(final.get("passages", [])),
        }
    if mode == "agentic_workflow":
        return {
            "answer": final.get("answer", ""),
            "iterations": final.get("iterations", 0),
            "tool_results": final.get("tool_results", []),
            "citations": final.get("citations", []),
            "sources": final.get("sources", []),
            "fetched": final.get("fetched", []),
        }
    if mode == "roleplay":
        return {
            "outcome": final.get("outcome", ""),
            "transcript": final.get("transcript", []),
            "turns": final.get("turn", 0),
            "exhibits": final.get("exhibits", []),
        }
    return dict(final)


# Strong references to in-flight background runs. asyncio only holds a weak reference
# to a task, so without this a long run can be garbage-collected mid-execution.
_background_runs: set[asyncio.Task] = set()


async def launch(experiment_id: str, inputs: dict | None = None, launched_by: Any = None) -> str:
    """Create a run and execute it in the background. Returns the run id immediately."""
    run = await create_run(experiment_id, inputs, launched_by)
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
