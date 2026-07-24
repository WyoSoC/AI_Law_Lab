"""AI Law Lab web interface: FastAPI + server-rendered templates + SSE for live runs."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .. import experiments, sources
from ..config import settings
from ..db import close_pool, fetch_all, fetch_one, get_pool
from ..rag import Corpus
from ..router import get_router
from ..tracing import run_metrics

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
    return templates.TemplateResponse(request, "experiment.html", {
        "exp": exp,
        "runs": await experiments.list_runs(experiment_id),
        "config_pretty": json.dumps(exp["config"], indent=2),
    })


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


@app.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request):
    docs = await fetch_all(
        "SELECT d.*, COUNT(c.id) AS chunk_count FROM documents d "
        "LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id ORDER BY d.created_at DESC"
    )
    return templates.TemplateResponse(request, "sources.html", {
        "documents": docs,
        "topics": sources.by_topic(),
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
