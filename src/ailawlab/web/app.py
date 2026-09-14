"""AI Law Lab web interface: FastAPI + server-rendered templates + SSE for live runs."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .. import agent_spec, cast_assistant, experiments, source_material, sources
from ..config import settings
from ..db import close_pool, fetch_all, fetch_one, get_pool
from ..graphs.roleplay_policy import ESTIMATE
from ..rag import Corpus
from ..router import get_router
from ..tracing import run_metrics
from . import views

log = logging.getLogger(__name__)
BASE = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()
    await get_router()
    log.info("AI Law Lab ready on %s:%s", settings.host, settings.port)
    yield
    await close_pool()


# root_path makes FastAPI strip the proxy's mount path off incoming request paths, so
# the routes below stay written as if the app owned the root. It does NOT rewrite outgoing
# URLs, which is what `prefix` (below) is for.
app = FastAPI(title="AI Law Lab", lifespan=lifespan, root_path=settings.url_prefix)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def _prefix(request: Request) -> dict[str, str]:
    """Expose the mount path to templates as `prefix`.

    Taken from the live request scope rather than settings so that a mismatch between
    the proxy path and configured url_prefix shows up as broken links immediately,
    instead of links that work only until the proxy is moved.
    """
    return {"prefix": request.scope.get("root_path", "")}


templates = Jinja2Templates(directory=BASE / "templates", context_processors=[_prefix])
P = settings.url_prefix


# ---------------------------------------------------------------- pages


@app.get("/", response_class=HTMLResponse)
async def about(request: Request):
    """Landing page: what the lab is and what it can do. Dashboard lives at /dashboard."""
    corpus_stats = await fetch_all(
        "SELECT d.corpus, COUNT(DISTINCT d.id) AS documents, COUNT(c.id) AS chunks "
        "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.corpus"
    )
    return templates.TemplateResponse(request, "about.html", {
        "modes": experiments.MODES,
        "topics": sources.by_topic(),
        "corpora": corpus_stats,
        "exp_count": len(await experiments.list_experiments()),
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    router = await get_router()
    corpus_stats = await fetch_all(
        "SELECT d.corpus, COUNT(DISTINCT d.id) AS documents, COUNT(c.id) AS chunks "
        "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.corpus"
    )
    return templates.TemplateResponse(request, "dashboard.html", {
        "cluster": router.stats(),
        "experiments": await experiments.list_experiments(),
        "runs": await experiments.list_runs(limit=15),
        "corpora": corpus_stats,
    })


@app.get("/experiments/new", response_class=HTMLResponse)
async def new_experiment_form(request: Request):
    corpora = await fetch_all("SELECT DISTINCT corpus FROM documents ORDER BY corpus")
    return templates.TemplateResponse(request, "new_experiment.html", {
        "modes": experiments.MODES,
        "corpora": [c["corpus"] for c in corpora],
        "default_max_turns": settings.default_max_turns,
        "max_turns_limit": settings.max_turns_limit,
        "default_word_limit": settings.default_word_limit,
        "word_limit_max": settings.word_limit_max,
    })


@app.post("/experiments")
async def create_experiment(
    name: str = Form(...),
    mode: str = Form(...),
    description: str = Form(""),
    config_json: str = Form("{}"),
    created_by: str = Form("unknown"),
):
    """Create an experiment from the config the form assembled.

    The builder UI serializes its fields to config_json client-side, so this endpoint
    stays a single JSON sink regardless of mode. Raw-JSON power users hit the same path.
    """
    try:
        config = json.loads(config_json or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"config must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise HTTPException(400, "config must be a JSON object")
    exp = await experiments.create_experiment(name, mode, description, config, created_by)
    return RedirectResponse(f"{P}/experiments/{exp['id']}", status_code=303)


@app.get("/experiments/{experiment_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, experiment_id: str):
    exp = await experiments.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, "no such experiment")
    runs = await experiments.list_runs(experiment_id)

    # A role-play run can take hours, so its page says how far along it is.
    progress: dict[str, int] = {}
    active = [str(r["id"]) for r in runs if r["status"] in ("pending", "running")]
    if active and exp["mode"] == "roleplay":
        rows = await fetch_all(
            "SELECT run_id::text AS run_id, COUNT(*) AS turns FROM run_events "
            "WHERE run_id = ANY(%s::uuid[]) AND node = 'speak' AND event_type = 'llm_call' "
            "GROUP BY run_id", (active,))
        progress = {r["run_id"]: r["turns"] for r in rows}

    corpus_documents = None
    if exp["mode"] != "roleplay":
        row = await fetch_one("SELECT COUNT(*) AS n FROM documents WHERE corpus = %s",
                              ((exp["config"] or {}).get("corpus") or "default",))
        corpus_documents = row["n"] if row else 0

    return templates.TemplateResponse(request, "experiment.html", {
        "exp": exp,
        "view": views.experiment_view(exp, runs, progress, corpus_documents),
        "config_pretty": json.dumps(exp["config"], indent=2),
        "estimate": ESTIMATE,
        "limits": {"max_turns": settings.max_turns_limit, "word_limit": settings.word_limit_max},
    })


@app.get("/experiments/{experiment_id}/cast.md")
async def experiment_cast_file(experiment_id: str):
    """The experiment's cast as an agent file, to reuse in another experiment."""
    exp = await experiments.get_experiment(experiment_id)
    agents = [a for a in ((exp or {}).get("config") or {}).get("agents") or [] if isinstance(a, dict)]
    if not agents:
        raise HTTPException(404, "this experiment has no cast")
    return _markdown_download(agent_spec.to_markdown(agents), f"{exp['name']} cast")


@app.post("/experiments/{experiment_id}/launch")
async def launch_run(experiment_id: str, inputs_json: str = Form("{}")):
    try:
        inputs = json.loads(inputs_json or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"inputs must be valid JSON: {e}") from e
    run_id = await experiments.launch(experiment_id, inputs)
    return RedirectResponse(f"{P}/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    run = await experiments.get_run(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    events = await fetch_all(
        "SELECT * FROM run_events WHERE run_id=%s ORDER BY seq", (run_id,)
    )
    return templates.TemplateResponse(request, "run.html", {
        "run": run,
        "events": events,
        "metrics": await run_metrics(run_id),
        "result_pretty": json.dumps(run["result"], indent=2) if run.get("result") else None,
        "citations": await fetch_all(
            "SELECT * FROM citations WHERE run_id=%s ORDER BY id", (run_id,)
        ),
    })


async def _corpus_documents():
    """Every ingested document with its chunk count, newest first."""
    return await fetch_all(
        "SELECT d.*, COUNT(c.id) AS chunk_count FROM documents d "
        "LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id ORDER BY d.created_at DESC"
    )


@app.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request):
    return templates.TemplateResponse(request, "sources.html", {
        "documents": await _corpus_documents(),
        "topics": sources.by_topic(),
    })


@app.get("/api/sources/documents", response_class=HTMLResponse)
async def api_sources_documents(request: Request):
    """The "In the corpus" table as an HTML fragment, so the page can refresh it in
    place after an ingest without a full reload."""
    return templates.TemplateResponse(request, "_documents.html", {
        "documents": await _corpus_documents(),
    })


# Old bookmark: /corpus was the page's name before it became Legal Sources.
@app.get("/corpus", include_in_schema=False)
async def corpus_redirect():
    return RedirectResponse(f"{P}/sources", status_code=301)


@app.get("/api/sources/search")
async def api_sources_search(provider: str, q: str):
    if not q.strip():
        raise HTTPException(400, "empty query")
    try:
        hits = await sources.search(provider, q)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except (RuntimeError, PermissionError) as e:
        raise HTTPException(502, str(e)) from e
    return JSONResponse({"provider": provider, "query": q,
                         "hits": [h.as_dict() for h in hits]})


@app.post("/api/sources/ingest")
async def api_sources_ingest(request: Request):
    """Fetch the selected hits and add them to a corpus. Body: {provider, corpus, hits}."""
    body = await request.json()
    provider = body.get("provider", "")
    hits = body.get("hits") or []
    corpus_name = (body.get("corpus") or "").strip()
    if not hits:
        raise HTTPException(400, "no hits selected")
    router = await get_router()
    try:
        report = await sources.ingest(provider, hits, corpus_name, router)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return JSONResponse(report)


@app.post("/corpus/upload")
async def corpus_upload(file: UploadFile, corpus: str = Form("default")):
    router = await get_router()
    store = Corpus(router, name=corpus)
    raw = await file.read()
    name = file.filename or "upload"

    if name.lower().endswith(".pdf"):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            # Keep the original filename as the title; the temp path is meaningless.
            doc_id = await store.add_pdf(Path(tmp.name), title=Path(name).stem,
                                         source_uri=name)
    else:
        text = raw.decode("utf-8", errors="replace")
        doc_id = await store.add_document(title=Path(name).stem, text=text,
                                          doc_type="text", source_uri=name)
    if doc_id is None:
        return RedirectResponse(f"{P}/sources?msg=already+ingested+or+no+text", status_code=303)
    return RedirectResponse(f"{P}/sources", status_code=303)


# ---------------------------------------------------------------- role-play cast files

# Agent files are a few kilobytes of prose; anything far larger is the wrong file.
MAX_AGENT_FILE_BYTES = 1_000_000
AGENT_FILE_SUFFIXES = (".md", ".markdown", ".txt")


def _markdown_download(text: str, filename: str) -> Response:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename or "cast").stem).strip("-.") or "cast"
    return Response(text, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.md"'})


def _taken_ids(raw: object) -> list[str]:
    return [str(i) for i in raw] if isinstance(raw, list) else []


@app.get("/api/agents/template")
async def api_agents_template():
    """A blank, commented agent file to fill in by hand."""
    return _markdown_download(agent_spec.template(), "agent-template")


@app.post("/api/agents/upload")
async def api_agents_upload(files: list[UploadFile] = File(...),  # noqa: B008 - FastAPI idiom
                            taken: str = Form("[]")):
    """Read one or more agent files into agents. `taken` lists ids already in the builder,
    so uploads never collide with agents that are already there."""
    readable: list[tuple[str, str]] = []
    skipped: list[str] = []
    for f in files:
        name = f.filename or "upload"
        raw = await f.read()
        if not name.lower().endswith(AGENT_FILE_SUFFIXES):
            skipped.append(f"{name}: skipped. Agent files must be plain text saved as .md or "
                           ".txt (in Word, use Save As and choose Plain Text).")
        elif len(raw) > MAX_AGENT_FILE_BYTES:
            skipped.append(f"{name}: skipped, because it is too large to be an agent file.")
        else:
            readable.append((name, raw.decode("utf-8-sig", errors="replace")))
    try:
        taken_ids = _taken_ids(json.loads(taken or "[]"))
    except json.JSONDecodeError:
        taken_ids = []
    result = agent_spec.parse_files(readable, taken=taken_ids)
    return JSONResponse({"agents": result.agents, "warnings": skipped + result.warnings})


@app.post("/api/agents/download")
async def api_agents_download(request: Request):
    """Agents as one Markdown file. Body: {agents, filename}."""
    body = await request.json()
    agents = [a for a in body.get("agents") or [] if isinstance(a, dict)]
    if not agents:
        raise HTTPException(400, "no agents to download")
    return _markdown_download(agent_spec.to_markdown(agents), body.get("filename") or "cast")


@app.post("/api/agents/check")
async def api_agents_check(request: Request):
    """Problems with a cast. Body: {agents, scenario, ai}. The rule-based checks always run;
    the AI review runs only when asked for and a scenario is given, since it costs a call."""
    body = await request.json()
    agents = [a for a in body.get("agents") or [] if isinstance(a, dict)]
    scenario = (body.get("scenario") or "").strip()
    issues = agent_spec.check_cast(agents)
    if body.get("ai") and scenario and agents:
        try:
            issues += await cast_assistant.review_cast(await get_router(), agents, scenario)
        except Exception as e:  # noqa: BLE001 - the rule-based results are still worth showing
            log.warning("AI cast review failed: %s", e)
            issues.append({"level": "warning", "agent": None,
                           "message": f"The AI review could not run ({type(e).__name__})."})
    return JSONResponse({"issues": issues})


@app.post("/api/source/read")
async def api_source_read(request: Request):
    """Read source material to draft a cast from: a JSON body {url} or {text, title}, or a
    multipart upload named `file`. Returns the text with its title, site, date and
    warnings, for the browser to preview and send back with a draft request."""
    try:
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise HTTPException(400, "Choose a file to read.")
            data = await upload.read(settings.source_max_bytes + 1)
            if len(data) > settings.source_max_bytes:
                raise HTTPException(413, "That file is too large to read.")
            # PDF extraction is CPU-bound; keep it off the event loop.
            doc = await asyncio.to_thread(source_material.from_bytes, data,
                                          content_type=upload.content_type or "",
                                          filename=upload.filename or "upload")
        else:
            body = await request.json()
            if (body.get("url") or "").strip():
                doc = await source_material.fetch_url(body["url"])
            elif (body.get("text") or "").strip():
                doc = source_material.from_text(body["text"], title=body.get("title") or "")
            else:
                raise HTTPException(400, "Give a link, a file, or some text to read.")
    except source_material.SourceError as e:
        raise HTTPException(422, str(e)) from e
    return JSONResponse(doc.as_dict())


@app.post("/api/agents/draft")
async def api_agents_draft(request: Request):
    """Draft agents from a scenario, or a scenario and agents from a source.

    Body: {scenario | source, count, notes, taken}, where `source` is what /api/source/read
    returned. For a source, the response adds `source` (what the experiment records about
    it, without the text) and `renamed` (the real names the draft replaced).
    """
    body = await request.json()
    try:
        count = max(2, min(int(body.get("count") or 2), 8))
    except (TypeError, ValueError):
        count = 2
    notes, taken = body.get("notes") or "", _taken_ids(body.get("taken"))
    router = await get_router()

    if isinstance(body.get("source"), dict):
        try:
            source = source_material.from_client(body["source"])
        except source_material.SourceError as e:
            raise HTTPException(422, str(e)) from e
        draft = await cast_assistant.draft_from_source(router, source, count, notes=notes,
                                                       taken=taken)
        return JSONResponse({"scenario": draft.scenario, "agents": draft.agents,
                             "warnings": draft.warnings, "renamed": draft.renamed,
                             "source": source.meta()})

    scenario = (body.get("scenario") or "").strip()
    if not scenario:
        raise HTTPException(400, "describe the scenario first")
    result = await cast_assistant.draft_cast(router, scenario, count, notes=notes, taken=taken)
    return JSONResponse({"agents": result.agents, "warnings": result.warnings})


# ---------------------------------------------------------------- api / streaming


@app.get("/api/cluster")
async def api_cluster():
    router = await get_router()
    await router.health_check()
    return router.stats()


@app.get("/api/runs/{run_id}")
async def api_run(run_id: str):
    run = await experiments.get_run(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return JSONResponse({"run": json.loads(json.dumps(run, default=str)),
                         "metrics": await run_metrics(run_id)})


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request):
    """Push new trace events as they are written, then close when the run finishes.

    Polls the events table rather than using LISTEN/NOTIFY: runs emit events on the
    order of seconds, so a 1s poll is well within budget and avoids holding a dedicated
    connection open per viewer.
    """
    async def gen():
        last_seq = 0
        while True:
            if await request.is_disconnected():
                break
            rows = await fetch_all(
                "SELECT seq, event_type, node, agent_id, host, queue_wait_ms, eval_ms, "
                "       output_tokens, payload FROM run_events "
                "WHERE run_id=%s AND seq>%s ORDER BY seq",
                (run_id, last_seq),
            )
            for r in rows:
                last_seq = r["seq"]
                yield {"event": "trace", "data": json.dumps(r, default=str)}

            run = await fetch_one("SELECT status FROM runs WHERE id=%s", (run_id,))
            if run and run["status"] in ("succeeded", "failed", "cancelled"):
                yield {"event": "done", "data": json.dumps(
                    {"status": run["status"], "metrics": await run_metrics(run_id)}, default=str)}
                break
            await asyncio.sleep(1.0)

    return EventSourceResponse(gen())


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("ailawlab.web.app:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
