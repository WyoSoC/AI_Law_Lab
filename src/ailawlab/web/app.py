"""AI Law Lab web interface: FastAPI + server-rendered templates + SSE for live runs."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .. import (
    agent_spec,
    cast_assistant,
    citations,
    experiments,
    rag,
    source_material,
    sources,
    web_links,
)
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
    return {"prefix": request.scope.get("root_path", ""), "asset_version": _asset_version()}


def _asset_version() -> str:
    """The newest modification time among the stylesheet and scripts, appended to their URLs
    so a browser fetches new ones after a deploy instead of reusing stale cached copies."""
    static = BASE / "static"
    try:
        return str(int(max(f.stat().st_mtime for f in (static / "style.css", *static.glob("*.js")))))
    except (OSError, ValueError):
        return "0"


templates = Jinja2Templates(directory=BASE / "templates", context_processors=[_prefix])
P = settings.url_prefix


# ---------------------------------------------------------------- pages


@app.get("/", response_class=HTMLResponse)
async def about(request: Request):
    """Landing page: what the lab is and what it can do. Dashboard lives at /dashboard."""
    return templates.TemplateResponse(request, "about.html", {
        "modes": experiments.MODES,
        "topics": sources.by_topic(),
        "corpora": await rag.list_corpora(),
        "exp_count": len(await experiments.list_experiments()),
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    router = await get_router()
    trashed_id = request.query_params.get("trashed")
    return templates.TemplateResponse(request, "dashboard.html", {
        "cluster": router.stats(),
        "experiments": await experiments.list_experiments(),
        "runs": await experiments.list_runs(limit=15),
        "corpora": await rag.list_corpora(),
        "trash_count": await experiments.trash_count(),
        "trashed": await experiments.get_experiment(trashed_id) if _is_uuid(trashed_id) else None,
    })


# ---------------------------------------------------------------- trash

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _is_uuid(value: str | None) -> bool:
    return bool(value and _UUID.match(value))


def _experiment_id(experiment_id: str) -> str:
    """Reject a malformed id as a 404 rather than letting Postgres raise on the cast."""
    if not _is_uuid(experiment_id):
        raise HTTPException(404, "no such experiment")
    return experiment_id


def _run_id(run_id: str) -> str:
    if not _is_uuid(run_id):
        raise HTTPException(404, "no such run")
    return run_id


@app.get("/trash", response_class=HTMLResponse)
async def trash_page(request: Request):
    return templates.TemplateResponse(request, "trash.html", {
        "experiments": await experiments.list_trash(),
    })


@app.post("/experiments/{experiment_id}/trash")
async def trash_experiment(experiment_id: str, next: str = Form("dashboard")):
    try:
        exp = await experiments.trash_experiment(_experiment_id(experiment_id))
    except experiments.ExperimentBusy as e:
        raise HTTPException(409, str(e)) from e
    if exp is None:
        raise HTTPException(404, "no such experiment")
    if next == "experiment":
        return RedirectResponse(f"{P}/experiments/{experiment_id}", status_code=303)
    return RedirectResponse(f"{P}/dashboard?trashed={experiment_id}", status_code=303)


@app.post("/experiments/{experiment_id}/restore")
async def restore_experiment(experiment_id: str, next: str = Form("trash")):
    exp = await experiments.restore_experiment(_experiment_id(experiment_id))
    if exp is None:
        raise HTTPException(404, "no such experiment")
    target = {"dashboard": "/dashboard", "experiment": f"/experiments/{experiment_id}"}
    return RedirectResponse(P + target.get(next, "/trash?msg=" + quote(f"Restored “{exp['name']}”.")),
                            status_code=303)


@app.post("/experiments/{experiment_id}/delete")
async def delete_experiment(experiment_id: str):
    try:
        gone = await experiments.delete_experiment(_experiment_id(experiment_id))
    except experiments.ExperimentBusy as e:
        raise HTTPException(409, str(e)) from e
    if gone is None:
        raise HTTPException(404, "no such experiment in the trash")
    return RedirectResponse(f"{P}/trash?msg=" + quote(f"Deleted “{gone['name']}” for good."),
                            status_code=303)


@app.post("/trash/empty")
async def empty_trash():
    n = await experiments.empty_trash()
    return RedirectResponse(
        f"{P}/trash?msg=" + quote(f"Deleted {n} experiment{'' if n == 1 else 's'} for good."),
        status_code=303)


@app.get("/experiments/new", response_class=HTMLResponse)
async def new_experiment_form(request: Request):
    corpora = await fetch_all("SELECT DISTINCT corpus FROM documents WHERE removed_at IS NULL "
                              "ORDER BY corpus")
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


async def _library_versions() -> dict[str, list[dict]]:
    """Every library's versions, newest first, for the launch form's version picker."""
    rows = await fetch_all("SELECT corpus, version, cardinality(document_ids) AS documents, created_at "
                           "FROM corpus_versions ORDER BY corpus, version DESC")
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["corpus"], []).append(
            {"version": r["version"], "documents": r["documents"],
             "date": f"{r['created_at']:%b} {r['created_at'].day}, {r['created_at']:%Y}"})
    return out


@app.get("/experiments/{experiment_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, experiment_id: str):
    exp = await experiments.get_experiment(_experiment_id(experiment_id))
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
        row = await fetch_one("SELECT COUNT(*) AS n FROM documents WHERE corpus = %s "
                              "AND removed_at IS NULL",
                              ((exp["config"] or {}).get("corpus") or "default",))
        corpus_documents = row["n"] if row else 0

    return templates.TemplateResponse(request, "experiment.html", {
        "exp": exp,
        "view": views.experiment_view(exp, runs, progress, corpus_documents),
        "config_pretty": json.dumps(exp["config"], indent=2),
        "estimate": ESTIMATE,
        "limits": {"max_turns": settings.max_turns_limit, "word_limit": settings.word_limit_max},
        "corpora": [c["corpus"] for c in await rag.list_corpora()],
        "library_versions": await _library_versions(),
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
    try:
        run_id = await experiments.launch(experiment_id, inputs)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return RedirectResponse(f"{P}/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    run = await experiments.get_run(_run_id(run_id))
    if run is None:
        raise HTTPException(404, "no such run")
    events = await fetch_all(
        "SELECT * FROM run_events WHERE run_id=%s ORDER BY seq", (run_id,)
    )
    return templates.TemplateResponse(request, "run.html", {
        "run": run,
        "view": views.run_view(run, request.scope.get("root_path", "")),
        "events": events,
        "metrics": await run_metrics(run_id),
        "result_pretty": json.dumps(run["result"], indent=2) if run.get("result") else None,
        "citations": await fetch_all(
            "SELECT * FROM citations WHERE run_id=%s ORDER BY id", (run_id,)
        ),
    })


@app.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request):
    return templates.TemplateResponse(request, "sources.html", {
        "corpora": await rag.list_corpora(),
        "topics": sources.by_topic(),
        "link_limit": settings.web_link_max_per_request,
    })


@app.get("/api/sources/corpora")
async def api_sources_corpora(request: Request):
    """The corpus cards as an HTML fragment plus the names for the corpus pickers, so the
    page can refresh both in place after an ingest without a full reload."""
    corpora = await rag.list_corpora()
    html = templates.get_template("_corpora.html").render(
        corpora=corpora, **_prefix(request))
    return JSONResponse({"html": html,
                         "corpora": [{"name": c["corpus"], "documents": c["documents"]}
                                     for c in corpora]})


def _corpus_url(name: str) -> str:
    return f"{P}/sources/corpora/{quote(name, safe='')}"


@app.get("/sources/corpora/{name}", response_class=HTMLResponse)
async def corpus_page(request: Request, name: str):
    documents = await rag.corpus_documents(name)
    versions = await rag.list_versions(name)
    if not documents and not versions:
        raise HTTPException(404, f"There is no library named “{name}”.")
    if documents and (not versions or sorted(versions[0]["document_ids"]) != sorted(d["id"] for d in documents)):
        # Changed some other way since the last version (the command line, or before
        # libraries were versioned): record it now, so what the page shows has a number.
        await rag.record_version(name)
        versions = await rag.list_versions(name)
    return templates.TemplateResponse(request, "corpus.html", {
        "name": name,
        "documents": documents,
        "chunks": sum(d["chunk_count"] for d in documents),
        "experiments": await rag.corpus_experiments(name),
        "versions": versions,
        "citations": citations.citation_list(documents),
    })


@app.get("/sources/corpora/{name}/versions/{version}", response_class=HTMLResponse)
async def corpus_version_page(request: Request, name: str, version: int):
    v = await rag.get_version(name, version)
    if v is None:
        raise HTTPException(404, f"The library “{name}” has no version {version}.")
    documents = await rag.corpus_documents(ids=list(v["document_ids"]))
    latest = await rag.latest_version(name)
    runs = await fetch_all(
        "SELECT r.id, r.status, r.created_at, e.name AS experiment_name, e.id AS experiment_id "
        "FROM runs r JOIN experiments e ON e.id = r.experiment_id "
        "WHERE r.corpus=%s AND r.corpus_version=%s ORDER BY r.created_at DESC", (name, version))
    return templates.TemplateResponse(request, "corpus_version.html", {
        "name": name, "v": v, "latest": latest["version"] if latest else None,
        "documents": documents, "runs": runs,
        "citations": citations.citation_list(documents),
    })


@app.get("/sources/corpora/{name}/citations.txt")
async def corpus_citations(name: str, version: int | None = None):
    """A library's citation list as plain text: its current documents, or a version's."""
    if version is not None:
        v = await rag.get_version(name, version)
        if v is None:
            raise HTTPException(404, f"The library “{name}” has no version {version}.")
        documents = await rag.corpus_documents(ids=list(v["document_ids"]))
        label = f"version {version}"
    else:
        documents = await rag.corpus_documents(name)
        latest = await rag.latest_version(name)
        label = f"version {latest['version']}" if latest else "current contents"
    text = citations.as_text(citations.citation_list(documents),
                             f"Citation list: {name} ({label}), AI Law Lab")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{name} {label} citations").strip("-.")
    return Response(text, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.txt"'})


@app.get("/api/sources/corpora/{name}/search")
async def api_corpus_search(name: str, q: str):
    """What an experiment would retrieve from this corpus for a query, with similarity."""
    if not q.strip():
        raise HTTPException(400, "empty query")
    passages = await Corpus(await get_router(), name=name).search(q)
    return JSONResponse({"passages": [
        {"document_id": p.document_id, "label": p.cite_label(), "content": p.content,
         "similarity": round(p.similarity, 3)} for p in passages]})


@app.post("/sources/corpora/{name}/delete")
async def corpus_delete(name: str, confirm_name: str = Form("")):
    if confirm_name.strip() != name:
        return RedirectResponse(_corpus_url(name) + "?msg=" + quote(
            "The name you typed did not match, so nothing was deleted."), status_code=303)
    # Deleting a library takes its versions with it, so past runs can no longer be repeated.
    try:
        n = await rag.delete_corpus(name)
    except rag.CorpusBusy as e:
        raise HTTPException(409, str(e)) from e
    return RedirectResponse(f"{P}/sources?msg=" + quote(
        f"Deleted the library “{name}”, its {n} document{'' if n == 1 else 's'} and all its versions."),
        status_code=303)


@app.get("/sources/documents/{document_id}", response_class=HTMLResponse)
async def document_page(request: Request, document_id: int):
    doc = await rag.get_document(document_id)
    if doc is None:
        raise HTTPException(404, "no such document")
    replacement = await rag.get_document(doc["replaced_by"]) if doc["replaced_by"] else None
    in_versions = await fetch_all(
        "SELECT version FROM corpus_versions WHERE corpus=%s AND %s = ANY(document_ids) "
        "ORDER BY version", (doc["corpus"], document_id))
    return templates.TemplateResponse(request, "document.html", {
        "doc": doc,
        "replacement": replacement,
        "in_versions": [r["version"] for r in in_versions],
        "citation": citations.citation(doc),
        "chunks": await rag.document_chunks(document_id),
        "corpora": [c["corpus"] for c in await rag.list_corpora()],
        "metadata_pretty": json.dumps(doc["metadata"], indent=2) if doc["metadata"] else None,
    })


@app.post("/api/sources/links")
async def api_sources_links(request: Request):
    """Read web links into a corpus. Body: {corpus, text}, where text holds the links in any
    layout (one per line, separated by spaces or commas, or inside a paragraph)."""
    body = await request.json()
    links = web_links.parse_links(str(body.get("text") or ""))
    if not links:
        raise HTTPException(400, "No web links were found. Paste addresses such as "
                                 "https://www.wyoleg.gov/... , one per line.")
    corpus = str(body.get("corpus") or "").strip()
    if not corpus:
        raise HTTPException(400, "Choose a corpus, or name a new one.")
    try:
        report = await web_links.add_links(await get_router(), corpus, links)
    except rag.CorpusBusy as e:
        raise HTTPException(409, str(e)) from e
    skipped = len(links) - len(report["results"])
    if skipped:
        report["note"] = (f"Only the first {settings.web_link_max_per_request} links were read; "
                          f"add the other {skipped} in another batch.")
    return JSONResponse(report)


@app.post("/api/sources/documents/{document_id}/refresh")
async def api_document_refresh(document_id: int):
    """Re-read a web-link document and rebuild its passages if the page changed."""
    try:
        return JSONResponse(await web_links.refresh(await get_router(), document_id))
    except rag.CorpusBusy as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/sources/corpora/{name}/refresh")
async def api_corpus_refresh(name: str):
    """Check every web link in a corpus for changes."""
    try:
        return JSONResponse(await web_links.refresh_corpus(await get_router(), name))
    except rag.CorpusBusy as e:
        raise HTTPException(409, str(e)) from e


@app.post("/sources/documents/{document_id}/rename")
async def document_rename(document_id: int, title: str = Form("")):
    try:
        doc = await rag.rename_document(document_id, title)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if doc is None:
        raise HTTPException(404, "no such document")
    return RedirectResponse(f"{P}/sources/documents/{document_id}?msg=" + quote("Renamed."),
                            status_code=303)


@app.post("/sources/documents/{document_id}/move")
async def document_move(document_id: int, corpus: str = Form("")):
    try:
        doc = await rag.move_document(document_id, corpus)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except rag.CorpusBusy as e:
        raise HTTPException(409, str(e)) from e
    if doc is None:
        raise HTTPException(404, "no such document")
    return RedirectResponse(f"{P}/sources/documents/{doc['id']}?msg=" + quote(
        f"Moved to “{doc['corpus']}”."), status_code=303)


@app.post("/sources/documents/{document_id}/delete")
async def document_delete(document_id: int):
    try:
        gone = await rag.delete_document(document_id)
    except rag.CorpusBusy as e:
        raise HTTPException(409, str(e)) from e
    if gone is None:
        raise HTTPException(404, "no such document")
    msg = quote(f"Removed “{gone['title']}” from the library. Earlier versions still include "
                "it, so runs that searched them can be repeated.")
    return RedirectResponse(f"{_corpus_url(gone['corpus'])}?msg={msg}", status_code=303)


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
async def corpus_upload(files: list[UploadFile] = File(...),  # noqa: B008 - FastAPI idiom
                        corpus: str = Form("default")):
    """Add one or more uploaded PDFs or text files to a corpus."""
    corpus = corpus.strip() or "default"
    store = Corpus(await get_router(), name=corpus)
    added, refused = [], []
    for upload in files:
        name = upload.filename or "upload"
        raw = await upload.read()
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
        (added if doc_id is not None else refused).append(name)
    if added:
        await rag.record_version(corpus)

    parts = []
    if added:
        parts.append(f"Added {len(added)} document{'' if len(added) == 1 else 's'}.")
    if refused:
        parts.append("Not added, because it has no readable text or the same text is already "
                     "in this library: " + ", ".join(f"“{n}”" for n in refused) + ".")
    msg = quote(" ".join(parts))
    if not added:
        return RedirectResponse(f"{P}/sources?tab=upload&msg={msg}", status_code=303)
    return RedirectResponse(f"{_corpus_url(corpus)}?msg={msg}", status_code=303)


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


@app.get("/api/experiment-file/template")
async def api_experiment_file_template():
    """A blank, commented experiment file: scenario, settings, and a person to copy."""
    return _markdown_download(
        agent_spec.experiment_template(settings.default_max_turns, settings.default_word_limit,
                                       settings.max_turns_limit, settings.word_limit_max),
        "experiment-template")


@app.post("/api/agents/upload")
async def api_agents_upload(files: list[UploadFile] = File(...),  # noqa: B008 - FastAPI idiom
                            taken: str = Form("[]")):
    """Read agent or experiment files. `taken` lists ids already in the builder, so uploads
    never collide with agents that are already there.

    `experiment` says whether any file had a Scenario, Settings or Source section; if so,
    `scenario`, `settings` (brought within the builder's limits) and `source` carry them.
    """
    readable: list[tuple[str, str]] = []
    skipped: list[str] = []
    for f in files:
        name = f.filename or "upload"
        raw = await f.read()
        if not name.lower().endswith(AGENT_FILE_SUFFIXES):
            skipped.append(f"{name}: skipped. Agent and experiment files must be plain text saved "
                           "as .md or .txt (in Word, use Save As and choose Plain Text).")
        elif len(raw) > MAX_AGENT_FILE_BYTES:
            skipped.append(f"{name}: skipped, because it is too large to be an agent file.")
        else:
            readable.append((name, raw.decode("utf-8-sig", errors="replace")))
    try:
        taken_ids = _taken_ids(json.loads(taken or "[]"))
    except json.JSONDecodeError:
        taken_ids = []
    result = agent_spec.parse_files(readable, taken=taken_ids)
    run_settings, limit_warnings = agent_spec.apply_limits(
        result.settings, settings.max_turns_limit, settings.word_limit_max)
    return JSONResponse({"agents": result.agents,
                         "warnings": skipped + result.warnings + limit_warnings,
                         "experiment": result.is_experiment, "scenario": result.scenario,
                         "settings": run_settings, "source": result.source})


@app.post("/api/agents/download")
async def api_agents_download(request: Request):
    """Agents as one Markdown file. Body: {agents, filename}."""
    body = await request.json()
    agents = [a for a in body.get("agents") or [] if isinstance(a, dict)]
    if not agents:
        raise HTTPException(400, "no agents to download")
    return _markdown_download(agent_spec.to_markdown(agents), body.get("filename") or "cast")


def _experiment_file(scenario: str, raw_settings: dict, agents: list, source: object) -> str:
    wanted = {}
    for key in ("max_turns", "word_limit"):
        try:
            wanted[key] = int(raw_settings[key])
        except (KeyError, TypeError, ValueError):
            continue
    run_settings, _ = agent_spec.apply_limits(wanted, settings.max_turns_limit,
                                              settings.word_limit_max)
    return agent_spec.to_experiment_markdown(
        scenario or "", run_settings, [a for a in agents or [] if isinstance(a, dict)],
        source if isinstance(source, dict) else None,
        max_turns_limit=settings.max_turns_limit, word_limit_max=settings.word_limit_max)


@app.post("/api/experiment-file")
async def api_experiment_file(request: Request):
    """A role-play being built, as one experiment file.
    Body: {name, scenario, max_turns, word_limit, agents, source}."""
    body = await request.json()
    text = _experiment_file(body.get("scenario") or "", body, body.get("agents"), body.get("source"))
    return _markdown_download(text, f"{body.get('name') or 'role-play'} experiment")


@app.get("/experiments/{experiment_id}/experiment.md")
async def experiment_file(experiment_id: str):
    """A saved role-play experiment as one experiment file, to reuse or share."""
    exp = await experiments.get_experiment(experiment_id)
    if exp is None or exp["mode"] != "roleplay":
        raise HTTPException(404, "no such role-play experiment")
    config = exp["config"] or {}
    raw_settings = {"max_turns": config.get("max_turns", settings.default_max_turns),
                    "word_limit": config.get("word_limit", settings.default_word_limit)}
    text = _experiment_file(config.get("scenario") or "", raw_settings, config.get("agents"),
                            config.get("source"))
    return _markdown_download(text, f"{exp['name']} experiment")


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


@app.get("/runs/{run_id}/report.pdf")
async def run_report_pdf(run_id: str, private: str = ""):
    """The run as a PDF: summary, scenario and cast, transcript. `private=1` adds each
    speaker's private notes and reasoning, which the default report leaves out."""
    from .report_pdf import run_report

    run = await experiments.get_run(_run_id(run_id))
    if run is None:
        raise HTTPException(404, "no such run")
    if not run.get("result"):
        raise HTTPException(409, "This run has no result to export yet.")
    pdf = await asyncio.to_thread(run_report, run, views.run_view(run), bool(private))
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{run['experiment_name']} run {str(run['id'])[:8]}").strip("-.")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.pdf"'})


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
