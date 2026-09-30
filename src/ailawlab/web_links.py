"""Web links kept in a corpus: add pages by their address, then keep them current.

A web page, unlike an opinion pulled from CourtListener, can change after it is added: an
agency updates guidance, a court page posts a new order, a statute page is amended. So a
link is stored as a document that remembers where it came from (`metadata.link`) and can be
checked again. Checking re-reads the page; if the text is the same only the check time is
recorded, and if it changed the document's passages are rebuilt from the new text.

Reading goes through source_material.fetch_url, so the same guards apply as for cast
sources: public addresses only, every redirect re-checked, size and time capped.

Metadata on a link document:
  link          the address as added (normalized); what "check again" fetches
  site, published, words, truncated
  retrieved_at  when the text currently stored was read
  checked_at    when the page was last checked, changed or not
  changed_at    when a check last found different text (absent until one does)
  check_error   why the last check failed, if it did (cleared by a successful one)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import UTC, datetime
from typing import Any

from . import source_material
from .config import settings
from .db import fetch_all, fetch_one, get_pool, jsonb
from .rag import Corpus, check_idle, get_document
from .router import LLMRouter

log = logging.getLogger(__name__)

_LINK = re.compile(r"(?:https?://|www\.)[^\s<>\"'()]+|[\w-]+(?:\.[\w-]+)+/[^\s<>\"'()]*", re.IGNORECASE)


def parse_links(text: str) -> list[str]:
    """The links in whatever someone pasted: one per line, or separated by spaces or
    commas, or scattered through a paragraph. Order kept, repeats dropped."""
    seen: list[str] = []
    for m in _LINK.finditer(text or ""):
        link = source_material.normalize_link(m.group(0).rstrip(".,;:!?]"))
        if link.startswith("www."):
            link = "https://" + link
        if link not in seen:
            seen.append(link)
    return seen


# The reader's messages were written for the cast drafter, which has a box to paste text
# into; a corpus takes an uploaded file instead.
_PASTE = re.compile(r"(?:copy the\s+text and\s+paste(?:\s+it)?|paste the\s+text)\s+instead", re.IGNORECASE)


def _for_corpus(message: str) -> str:
    def upload(m: re.Match) -> str:
        advice = "save the text as a file and upload it instead"
        return advice.capitalize() if m.group(0)[0].isupper() else advice
    return _PASTE.sub(upload, message)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    # Same digest Corpus.add_document stores, so a link and an upload of the same text match.
    return hashlib.sha256(text.encode()).hexdigest()


def _meta(doc: source_material.SourceDoc, link: str, *, previous: dict | None = None) -> dict:
    now = _now()
    meta = {k: v for k, v in (previous or {}).items() if k != "check_error"}
    meta.update({"link": link, "site": doc.site, "published": doc.published,
                 "words": doc.words, "truncated": doc.truncated,
                 "retrieved_at": doc.retrieved_at, "checked_at": now})
    return meta


async def _read(link: str) -> source_material.SourceDoc:
    return await source_material.fetch_url(link, max_words=settings.web_link_max_words)


async def _holder(sha: str) -> dict | None:
    """The document that already holds this exact text, if any (sha256 is unique)."""
    return await fetch_one("SELECT id, title, corpus FROM documents WHERE sha256=%s", (sha,))


async def add_links(router: LLMRouter, corpus: str, links: list[str]) -> dict[str, Any]:
    """Read each link and add it to `corpus`. Returns a per-link report.

    Pages are fetched concurrently (network-bound, different hosts) and embedded one at a
    time (they share the cluster's embedding slots), as sources.ingest does.
    """
    corpus = " ".join(corpus.split())[:120] or "default"
    links = links[:settings.web_link_max_per_request]
    await check_idle(corpus)
    known = {r["link"]: r for r in await fetch_all(
        "SELECT id, title, metadata->>'link' AS link FROM documents "
        "WHERE corpus=%s AND metadata ? 'link'", (corpus,))}

    async def read(link: str) -> dict[str, Any]:
        if link in known:
            return {"link": link, "status": "exists", "document_id": known[link]["id"],
                    "title": known[link]["title"],
                    "detail": "Already in this corpus. Use “Check for updates” to re-read it."}
        try:
            return {"link": link, "doc": await _read(link)}
        except source_material.SourceError as e:
            return {"link": link, "status": "error", "detail": _for_corpus(str(e))}
        except Exception as e:  # noqa: BLE001 - one bad link must not sink the rest
            log.warning("reading %s failed: %s", link, e)
            return {"link": link, "status": "error", "detail": f"Could not read it ({type(e).__name__})."}

    store = Corpus(router, name=corpus)
    report: list[dict[str, Any]] = []
    for item in await asyncio.gather(*(read(link) for link in links)):
        doc = item.pop("doc", None)
        if doc is None:
            report.append(item)
            continue
        holder = await _holder(_sha(doc.text))
        if holder:
            where = "this corpus" if holder["corpus"] == corpus else f"the corpus “{holder['corpus']}”"
            report.append({"link": item["link"], "status": "exists", "document_id": holder["id"],
                           "title": holder["title"],
                           "detail": f"The same text is already in {where}, as “{holder['title']}”."})
            continue
        try:
            doc_id = await store.add_document(
                title=doc.title, text=doc.text, source_uri=doc.url or item["link"],
                doc_type=doc.kind, metadata=_meta(doc, item["link"]))
        except Exception as e:
            log.exception("adding %s failed", item["link"])
            report.append({"link": item["link"], "status": "error", "title": doc.title,
                           "detail": f"Read, but could not be added ({type(e).__name__})."})
            continue
        report.append({"link": item["link"], "status": "added", "document_id": doc_id,
                       "title": doc.title, "detail": _for_corpus(" ".join(doc.warnings))})
    return {"corpus": corpus, "results": report,
            "added": sum(1 for r in report if r["status"] == "added")}


async def refresh(router: LLMRouter, document_id: int) -> dict[str, Any]:
    """Check a link document for changes. Status: unchanged, updated, error, or not_a_link."""
    doc = await get_document(document_id)
    if doc is None:
        return {"document_id": document_id, "status": "error", "detail": "No such document."}
    meta = dict(doc["metadata"] or {})
    link = meta.get("link")
    if not link:
        return {"document_id": document_id, "status": "not_a_link", "title": doc["title"],
                "detail": "This document was not added from a web link."}
    await check_idle(doc["corpus"])

    async def record(status: str, **fields: Any) -> dict[str, Any]:
        return {"document_id": document_id, "title": doc["title"], "link": link,
                "status": status, **fields}

    try:
        page = await _read(link)
    except Exception as e:  # noqa: BLE001 - record why, keep the stored text
        detail = (_for_corpus(str(e)) if isinstance(e, source_material.SourceError)
                  else f"Could not read it ({type(e).__name__}).")
        meta.update({"checked_at": _now(), "check_error": detail})
        await fetch_one("UPDATE documents SET metadata=%s WHERE id=%s RETURNING id",
                        (jsonb(meta), document_id))
        return await record("error", detail=f"{detail} The text already stored is kept.")

    sha = _sha(page.text)
    if sha == doc["sha256"]:
        meta.update({"checked_at": _now()})
        meta.pop("check_error", None)
        await fetch_one("UPDATE documents SET metadata=%s WHERE id=%s RETURNING id",
                        (jsonb(meta), document_id))
        return await record("unchanged", detail="No change since it was last read.")

    holder = await _holder(sha)
    if holder:
        meta.update({"checked_at": _now(),
                     "check_error": f"The page now matches “{holder['title']}” in "
                                    f"“{holder['corpus']}”, so it was not re-read."})
        await fetch_one("UPDATE documents SET metadata=%s WHERE id=%s RETURNING id",
                        (jsonb(meta), document_id))
        return await record("error", detail=meta["check_error"])

    n = await Corpus(router, name=doc["corpus"]).store_chunks(document_id, page.text)
    meta = _meta(page, link, previous=meta)
    meta["changed_at"] = meta["checked_at"]
    pool = await get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "UPDATE documents SET sha256=%s, metadata=%s, source_uri=%s, doc_type=%s WHERE id=%s",
            (sha, jsonb(meta), page.url or link, page.kind, document_id))
    return await record("updated", passages=n,
                        detail=f"The page had changed; re-read it into {n} passages.")


async def refresh_corpus(router: LLMRouter, corpus: str) -> dict[str, Any]:
    """Check every link in a corpus, one at a time, and summarize."""
    rows = await fetch_all("SELECT id FROM documents WHERE corpus=%s AND metadata ? 'link' "
                           "ORDER BY created_at", (corpus,))
    results = [await refresh(router, r["id"]) for r in rows]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"corpus": corpus, "results": results, "counts": counts}
