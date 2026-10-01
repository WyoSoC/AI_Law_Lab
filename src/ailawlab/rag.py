"""Legal RAG: corpus ingestion and grounded retrieval.

Retrieval returns chunks carrying document title and page anchors, because a citation a
lawyer cannot check is worthless. Every passage handed to a model comes with the
metadata needed to point a human at the page it came from.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings
from .db import fetch_all, fetch_one, get_pool, jsonb, vec
from .router import LLMRouter

log = logging.getLogger(__name__)


@dataclass
class Passage:
    chunk_id: int
    document_id: int
    title: str
    content: str
    page_start: int | None
    page_end: int | None
    similarity: float
    corpus: str = ""                # the library the document sits in
    version: int | None = None      # the version of it the run searched, when pinned

    def cite_label(self) -> str:
        if self.page_start and self.page_end and self.page_start != self.page_end:
            return f"{self.title}, pp. {self.page_start}-{self.page_end}"
        if self.page_start:
            return f"{self.title}, p. {self.page_start}"
        return self.title

    def library_label(self) -> str:
        """“Case law” v3, or "" for a passage not tied to a library."""
        if not self.corpus:
            return ""
        return f"“{self.corpus}”" + (f" v{self.version}" if self.version else "")

    def source(self, marker: str, cited: bool = False) -> dict[str, Any]:
        """What a run keeps about a passage it was given, enough to find it again: the
        document, its library and version, and the passage itself."""
        return {"marker": marker, "label": self.cite_label(), "corpus": self.corpus,
                "version": self.version, "document_id": self.document_id,
                "chunk_id": self.chunk_id, "similarity": round(self.similarity, 3),
                "cited": cited}


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split on paragraph boundaries, packing up to `size` chars with `overlap` carry-over.

    Paragraph-aware rather than fixed-width: splitting mid-sentence in a statute or
    holding tends to sever the subject from its qualifier, which wrecks both the
    embedding and any quote taken from the chunk.
    """
    size = size or settings.chunk_chars
    overlap = overlap or settings.chunk_overlap
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + para).strip()
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        chunks.append(current)

    # A single paragraph longer than `size` still needs splitting.
    out: list[str] = []
    for c in chunks:
        while len(c) > size * 1.5:
            out.append(c[:size])
            c = c[size - overlap:]
        out.append(c)
    return [c for c in out if c.strip()]


class Corpus:
    """A library: search it, add to it.

    With `document_ids` (a recorded version's contents) search is confined to exactly those
    documents, removed or replaced since or not, which is what lets a run be repeated
    against the library as it was. Without, it searches the library's current contents.
    """

    def __init__(self, router: LLMRouter, name: str = "default",
                 document_ids: list[int] | None = None, added_by: Any = None):
        self.router = router
        self.name = name
        self.document_ids = document_ids
        self.added_by = added_by            # user id recorded on documents this adds

    async def add_document(
        self,
        title: str,
        text: str,
        *,
        source_uri: str | None = None,
        doc_type: str = "other",
        metadata: dict | None = None,
        page_map: list[tuple[int, int]] | None = None,
    ) -> int | None:
        """Ingest one document. Returns None if this library already holds the same text.

        `page_map` is an optional list of (char_offset, page_number) pairs used to anchor
        chunks to pages; PDF ingestion supplies it.
        """
        sha = hashlib.sha256(text.encode()).hexdigest()
        existing = await fetch_one("SELECT id FROM documents WHERE sha256=%s AND corpus=%s "
                                   "AND removed_at IS NULL", (sha, self.name))
        if existing:
            log.info("document already ingested: %s", title)
            return None

        row = await fetch_one(
            "INSERT INTO documents (corpus, title, source_uri, doc_type, metadata, sha256, added_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (self.name, title, source_uri, doc_type, jsonb(metadata or {}), sha, self.added_by),
        )
        doc_id = row["id"]
        n = await self.store_chunks(doc_id, text, page_map)
        log.info("ingested %s: %d chunks", title, n)
        return doc_id

    async def store_chunks(self, doc_id: int, text: str,
                           page_map: list[tuple[int, int]] | None = None) -> int:
        """Chunk and embed `text` as the document's passages, replacing any it had.

        Embeds first and swaps the passages in one transaction, so a document being
        re-read keeps its old passages if embedding fails partway.
        """
        chunks = chunk_text(text)

        # Embed in batches so a large document does not occupy a slot indefinitely.
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), 16):
            vectors.extend(await self.router.embed(chunks[i:i + 16]))

        offset = 0
        pool = await get_pool()
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute("DELETE FROM chunks WHERE document_id=%s", (doc_id,))
            for ordinal, (content, vector) in enumerate(zip(chunks, vectors)):
                page = _page_for_offset(page_map, offset) if page_map else None
                end_page = _page_for_offset(page_map, offset + len(content)) if page_map else None
                await cur.execute(
                    "INSERT INTO chunks (document_id, ordinal, content, page_start, page_end, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (doc_id, ordinal, content, page, end_page, vec(vector)),
                )
                offset += len(content)
        return len(chunks)

    async def add_pdf(self, path: Path, **kw) -> int | None:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        parts, page_map, offset = [], [], 0
        for n, page in enumerate(reader.pages, start=1):
            t = page.extract_text() or ""
            page_map.append((offset, n))
            parts.append(t)
            offset += len(t) + 2
        text = "\n\n".join(parts)
        if not text.strip():
            log.warning("no extractable text in %s (scanned image?)", path.name)
            return None
        kw.setdefault("title", path.stem)
        kw.setdefault("doc_type", "pdf")
        kw.setdefault("source_uri", str(path))
        return await self.add_document(text=text, page_map=page_map, **kw)

    async def search(self, query: str, top_k: int | None = None) -> list[Passage]:
        if self.document_ids is not None:
            scope, arg = "d.id = ANY(%s)", self.document_ids
        else:
            scope, arg = "d.corpus = %s AND d.removed_at IS NULL", self.name
        return await _search(self.router, query, scope, arg, top_k or settings.rag_top_k)

    async def stats(self) -> dict:
        row = await fetch_one(
            "SELECT COUNT(DISTINCT d.id) AS documents, COUNT(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "WHERE d.corpus=%s AND d.removed_at IS NULL", (self.name,),
        )
        return {"corpus": self.name, **(row or {})}


async def _search(router: LLMRouter, query: str, scope: str, arg: Any, top_k: int) -> list[Passage]:
    qvec = vec(await router.embed_one(query))
    rows = await fetch_all(
        "SELECT c.id AS chunk_id, c.document_id, d.title, c.content, d.corpus, "
        "       c.page_start, c.page_end, 1 - (c.embedding <=> %s) AS similarity "
        "FROM chunks c JOIN documents d ON d.id = c.document_id "
        f"WHERE {scope} AND c.embedding IS NOT NULL "
        "  AND 1 - (c.embedding <=> %s) >= %s "
        "ORDER BY c.embedding <=> %s LIMIT %s",
        (qvec, arg, qvec, settings.rag_min_similarity, qvec, top_k),
    )
    return [Passage(**r) for r in rows]


# ---------------------------------------------------------------- a run's libraries
#
# An experiment names the libraries it retrieves from as a list, `libraries`. Experiments
# from before a run could search several hold one name in `corpus` instead (none, for a
# role-play without legal sources; "default" for the other modes when it is missing), and
# library_names() reads both. A run pins each library to one version and searches the
# documents of all of them together, so passages are ranked against each other by how
# well they match, not taken in turns from each library.


def library_names(config: dict[str, Any], mode: str = "") -> list[str]:
    """The libraries a config names, in order, each once. Pure."""
    raw = config.get("libraries")
    if raw is None:
        legacy = config.get("corpus")
        legacy = legacy.strip() if isinstance(legacy, str) else ""
        raw = [legacy] if legacy else ([] if mode == "roleplay" else ["default"])
    elif isinstance(raw, str):
        raw = [raw]
    names: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        name = " ".join(item.split())[:120] if isinstance(item, str) else ""
        if name and name not in names:
            names.append(name)
    return names


def run_library_names(mode: str, config: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    """The libraries a run searches: the run's own choice if it made one, else the
    experiment's. Decided per source rather than on the merged dict, so a run naming one
    library the old way (`corpus`) is not overridden by the experiment's `libraries`."""
    if "libraries" in inputs or "corpus" in inputs:
        return library_names(inputs, mode)
    return library_names(config, mode)


def case_files(agent: dict[str, Any]) -> list[str]:
    """The libraries only this role-play agent may search (its "Case files"). Pure."""
    return library_names({"libraries": agent.get("libraries") or []})


def all_run_library_names(mode: str, config: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    """Every library a run must pin: its shared libraries and, in a role-play, each agent's
    case files (from the run's own cast if it brought one). Pure."""
    names = run_library_names(mode, config, inputs)
    if mode == "roleplay":
        agents = inputs.get("agents") if "agents" in inputs else config.get("agents")
        for agent in agents if isinstance(agents, list) else []:
            if isinstance(agent, dict):
                names += [n for n in case_files(agent) if n not in names]
    return names


def wanted_versions(names: list[str], inputs: dict[str, Any]) -> dict[str, int]:
    """Versions a run asks for, by library: `library_versions` ({name: n}), or the older
    single `corpus_version`, which applies when the run searches one library. Pure."""
    raw = inputs.get("library_versions")
    wanted: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    if not wanted and inputs.get("corpus_version") not in (None, "") and len(names) == 1:
        wanted = {names[0]: inputs["corpus_version"]}
    out: dict[str, int] = {}
    for name, value in wanted.items():
        if name not in names or value in (None, ""):
            continue
        try:
            out[name] = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"version {value!r} of the library “{name}” is not a number") from e
    return out


@dataclass
class LibraryPin:
    """One library as a run searches it: a recorded version and the documents it held.
    `version` is None for a library that is empty, which contributes nothing."""
    name: str
    version: int | None
    document_ids: list[int]


class Libraries:
    """The libraries one run searches, each pinned to a version, searched as one."""

    def __init__(self, router: LLMRouter, pins: list[LibraryPin] | None = None):
        self.router = router
        self.pins = list(pins or [])

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.pins]

    def searchable(self) -> list[str]:
        """Libraries holding at least one document."""
        return [p.name for p in self.pins if p.document_ids]

    def __bool__(self) -> bool:
        return bool(self.searchable())

    def subset(self, names: list[str]) -> Libraries:
        """The same pins, narrowed to `names`: what one role-play agent may search."""
        return Libraries(self.router, [p for p in self.pins if p.name in names])

    def describe(self) -> str:
        return ", ".join(f"“{p.name}”" + (f" v{p.version}" if p.version else " (empty)")
                         for p in self.pins)

    async def search(self, query: str, top_k: int | None = None,
                     library: str | None = None) -> list[Passage]:
        """The passages closest to `query` across every library, or only in `library`.

        The same text can sit in two libraries (a document copied from one to another);
        it is returned once, from the library where it matched best, which is why more
        than `top_k` rows are fetched when several libraries are searched.
        """
        top_k = top_k or settings.rag_top_k
        pins = [p for p in self.pins if p.document_ids and (library is None or p.name == library)]
        if not pins:
            return []
        ids = [i for p in pins for i in p.document_ids]
        found = await _search(self.router, query, "d.id = ANY(%s)", ids,
                              top_k * 2 if len(pins) > 1 else top_k)
        versions = {p.name: p.version for p in pins}
        out: list[Passage] = []
        seen: set[str] = set()
        for p in found:
            key = hashlib.sha256(p.content.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            p.version = versions.get(p.corpus)
            out.append(p)
        return out[:top_k]


async def pin_libraries(router: LLMRouter, names: list[str],
                        versions: dict[str, int] | None = None) -> Libraries:
    """Each named library pinned to a version: the one asked for, or its current contents
    (recorded as a new version first if they changed since the last)."""
    versions = versions or {}
    pins: list[LibraryPin] = []
    for name in names:
        if name in versions:
            v = await get_version(name, versions[name])
            if v is None:
                raise ValueError(f"the library “{name}” has no version {versions[name]}")
        else:
            v = await record_version(name)
        pins.append(LibraryPin(name, v["version"] if v else None,
                               list(v["document_ids"]) if v else []))
    return Libraries(router, pins)


def _page_for_offset(page_map: list[tuple[int, int]], offset: int) -> int | None:
    page = None
    for start, n in page_map:
        if start <= offset:
            page = n
        else:
            break
    return page


def format_passages(passages: list[Passage], numbers: list[int] | None = None) -> str:
    """Render passages for a prompt, numbered so the model can cite [1], [2], ...

    `numbers` replaces 1, 2, 3 when the numbering runs across several searches. Each
    passage names its library, so a model working from several can tell, say, a statute
    from commentary on it.
    """
    numbers = numbers or list(range(1, len(passages) + 1))
    return "\n\n".join(
        f"[{n}] {p.cite_label()}" + (f" (library {p.library_label()})" if p.corpus else "")
        + f"\n{p.content}" for n, p in zip(numbers, passages)
    )


# ---------------------------------------------------------------- managing libraries
#
# A library (a "corpus" in the code) has no table of its own: it is the set of current
# documents sharing a `corpus` name, so it exists while it holds a document. Its history
# lives in corpus_versions: after every change, record_version() stores the set of document
# ids it then held. Documents are never edited in place or deleted while the library
# exists, so every recorded version can still be searched exactly as it was.


class CorpusBusy(Exception):
    """A run that retrieves from this library is in progress."""


_LIVE = "d.removed_at IS NULL"

# Whether experiment `e` retrieves from the library {name}: listed in `libraries`, or, for an
# experiment from before that, named in `corpus` (with "default" standing in for a missing
# one, except in a role-play, which then has no legal sources), or held as a case file by
# one of a role-play's agents. See library_names() and case_files().
_USES = ("((CASE WHEN e.config ? 'libraries' THEN COALESCE(e.config->'libraries' ? {name}, FALSE) "
         "ELSE COALESCE(e.config->>'corpus', CASE WHEN e.mode <> 'roleplay' THEN 'default' END) "
         "= {name} END) "
         # ... or a role-play agent holds it as a case file.
         "OR EXISTS (SELECT 1 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(e.config->'agents') "
         "= 'array' THEN e.config->'agents' ELSE '[]'::jsonb END) a "
         "WHERE jsonb_typeof(a->'libraries') = 'array' AND a->'libraries' ? {name}))")


async def list_corpora() -> list[dict]:
    """Every library with its size, when it last changed, its current version, and how many
    experiments use it. A library whose documents have all been removed is still listed
    (empty) while its versions exist, since past runs point at them."""
    live = await fetch_all(
        "SELECT d.corpus, COUNT(DISTINCT d.id) AS documents, COUNT(c.id) AS chunks, "
        "       MAX(d.created_at) AS last_added, "
        "       (SELECT MAX(v.version) FROM corpus_versions v WHERE v.corpus = d.corpus) AS version, "
        "       (SELECT MAX(v.created_at) FROM corpus_versions v WHERE v.corpus = d.corpus) AS changed, "
        "       (SELECT COUNT(*) FROM experiments e WHERE e.deleted_at IS NULL "
        f"          AND {_USES.format(name='d.corpus')}) AS experiments "
        "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
        f"WHERE {_LIVE} GROUP BY d.corpus ORDER BY lower(d.corpus)"
    )
    names = {c["corpus"] for c in live}
    empty = [{"corpus": r["corpus"], "documents": 0, "chunks": 0, "last_added": r["changed"],
              "changed": r["changed"],
              "version": r["version"], "experiments": 0}
             for r in await fetch_all("SELECT corpus, MAX(version) AS version, MAX(created_at) AS changed "
                                      "FROM corpus_versions GROUP BY corpus")
             if r["corpus"] not in names]
    return sorted(live + empty, key=lambda c: c["corpus"].casefold())


async def corpus_documents(name: str | None = None,
                           ids: list[int] | None = None) -> list[dict]:
    """Documents with their chunk counts, newest first: a library's current documents, or
    exactly the documents `ids` names (a recorded version, current or not)."""
    if ids is not None:
        where, params = "WHERE d.id = ANY(%s) ", (ids,)
    elif name is not None:
        where, params = f"WHERE d.corpus = %s AND {_LIVE} ", (name,)
    else:
        where, params = f"WHERE {_LIVE} ", ()
    return await fetch_all(
        "SELECT d.*, COUNT(c.id) AS chunk_count, "
        "       COALESCE(SUM(LENGTH(c.content)), 0) AS chars FROM documents d "
        f"LEFT JOIN chunks c ON c.document_id = d.id {where}"
        "GROUP BY d.id ORDER BY d.created_at DESC", params)


async def corpus_experiments(name: str) -> list[dict]:
    """Experiments (not in the trash) that retrieve from this library."""
    return await fetch_all(
        "SELECT e.id, e.name, e.mode FROM experiments e WHERE e.deleted_at IS NULL "
        f"AND {_USES.format(name='%s')} ORDER BY e.created_at DESC", (name, name))


async def get_document(document_id: int) -> dict | None:
    return await fetch_one("SELECT * FROM documents WHERE id=%s", (document_id,))


async def document_chunks(document_id: int) -> list[dict]:
    return await fetch_all(
        "SELECT id, ordinal, content, page_start, page_end FROM chunks "
        "WHERE document_id=%s ORDER BY ordinal", (document_id,))


async def check_idle(name: str) -> None:
    row = await fetch_one(
        "SELECT COUNT(*) AS n FROM run_libraries rl JOIN runs r ON r.id = rl.run_id "
        "WHERE r.status IN ('pending', 'running') AND rl.corpus = %s", (name,))
    if row and row["n"]:
        raise CorpusBusy(f"A run that retrieves from “{name}” is in progress. "
                         "Wait for it to finish first.")


# ---------------------------------------------------------------- versions


def describe_change(before: dict[int, str], after: dict[int, str],
                    replaced: dict[int, int] | None = None,
                    moved: dict[int, str] | None = None) -> str:
    """A version's change note from the documents (id -> title) before and after it.

    `replaced` maps an old document id to the id that replaced it, so a re-read web page
    reads as "updated" rather than as one document removed and another added; `moved` maps
    an old id to the library it was moved to. Pure.
    """
    replaced, moved = replaced or {}, moved or {}
    updated = [after[new] for old, new in replaced.items() if old in before and new in after
               and old not in after]
    skip_old = {old for old, new in replaced.items() if new in after and old not in after}
    skip_new = {replaced[o] for o in skip_old}
    added = [t for i, t in after.items() if i not in before and i not in skip_new]
    removed = [t for i, t in before.items() if i not in after and i not in skip_old and i not in moved]
    moves: dict[str, list[str]] = {}
    for i, target in moved.items():
        if i in before and i not in after:
            moves.setdefault(target, []).append(before[i])

    def part(verb: str, titles: list[str], where: str = "") -> str:
        if not titles:
            return ""
        shown = "; ".join(f"“{t}”" for t in sorted(titles)[:3])
        more = f"; and {len(titles) - 3} more" if len(titles) > 3 else ""
        return f"{verb} {len(titles)}{where}: {shown}{more}"

    parts = [p for p in (part("Added", added), part("Updated", updated),
                         *(part("Moved", titles, f" to “{t}”") for t, titles in sorted(moves.items())),
                         part("Removed", removed)) if p]
    return (". ".join(parts) + ".") if parts else "No change."


async def latest_version(name: str) -> dict | None:
    return await fetch_one("SELECT * FROM corpus_versions WHERE corpus=%s "
                           "ORDER BY version DESC LIMIT 1", (name,))


async def get_version(name: str, version: int) -> dict | None:
    return await fetch_one("SELECT * FROM corpus_versions WHERE corpus=%s AND version=%s",
                           (name, version))


async def list_versions(name: str) -> list[dict]:
    """A library's versions, newest first, each with how many runs searched it."""
    return await fetch_all(
        "SELECT v.*, cardinality(v.document_ids) AS documents, "
        "       COALESCE(NULLIF(u.name, ''), u.email) AS changed_by_name, "
        "       (SELECT COUNT(*) FROM run_libraries rl WHERE rl.corpus = v.corpus "
        "          AND rl.version = v.version) AS runs "
        "FROM corpus_versions v LEFT JOIN users u ON u.id = v.changed_by "
        "WHERE v.corpus=%s ORDER BY v.version DESC", (name,))


async def record_version(name: str, changed_by: Any = None) -> dict | None:
    """Record the library's current contents as a new version, if they differ from the last
    recorded version. Returns the current version (new or not), or None for an empty or
    unknown library.

    Called after every change, and again when a run starts, so a change made some other
    way (the command line, a script) is still recorded before anything searches it.
    """
    pool = await get_pool()
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        # One writer per library at a time, so two changes cannot claim the same number.
        await cur.execute("SELECT pg_advisory_xact_lock(hashtext('corpus_version:' || %s))", (name,))
        await cur.execute("SELECT id, title FROM documents WHERE corpus=%s AND removed_at IS NULL "
                          "ORDER BY id", (name,))
        after = {r["id"]: r["title"] for r in await cur.fetchall()}
        await cur.execute("SELECT * FROM corpus_versions WHERE corpus=%s "
                          "ORDER BY version DESC LIMIT 1", (name,))
        last = await cur.fetchone()
        if last and sorted(last["document_ids"]) == sorted(after):
            return last
        if not after and not last:
            return None
        before_ids = list(last["document_ids"]) if last else []
        before: dict[int, str] = {}
        replaced: dict[int, int] = {}
        if before_ids:
            await cur.execute("SELECT d.id, d.title, d.replaced_by, n.corpus AS new_corpus "
                              "FROM documents d LEFT JOIN documents n ON n.id = d.replaced_by "
                              "WHERE d.id = ANY(%s)", (before_ids,))
            moved: dict[int, str] = {}
            for r in await cur.fetchall():
                before[r["id"]] = r["title"]
                if r["replaced_by"] and r["new_corpus"] and r["new_corpus"] != name:
                    moved[r["id"]] = r["new_corpus"]
                elif r["replaced_by"]:
                    replaced[r["id"]] = r["replaced_by"]
        change = describe_change(before, after, replaced, moved if before_ids else None) if last else (
            f"First recorded version: {len(after)} document{'' if len(after) == 1 else 's'}.")
        await cur.execute(
            "INSERT INTO corpus_versions (corpus, version, document_ids, change, changed_by) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING *",
            (name, (last["version"] + 1) if last else 1, list(after), change, changed_by))
        return await cur.fetchone()


# ---------------------------------------------------------------- changing a library


async def delete_document(document_id: int, by: Any = None) -> dict | None:
    """Take a document out of its library. It stays stored for the versions that hold it."""
    doc = await get_document(document_id)
    if doc is None:
        return None
    await check_idle(doc["corpus"])
    gone = await fetch_one("UPDATE documents SET removed_at = COALESCE(removed_at, now()) "
                           "WHERE id=%s RETURNING id, title, corpus", (document_id,))
    await record_version(doc["corpus"], by)
    return gone


async def rename_document(document_id: int, title: str) -> dict | None:
    """Retitle a document. A title is a label, not content, so this makes no new version."""
    title = " ".join(title.split())[:300]
    if not title:
        raise ValueError("A document needs a title.")
    return await fetch_one("UPDATE documents SET title=%s WHERE id=%s RETURNING id, title, corpus",
                           (title, document_id))


async def copy_document(doc: dict, corpus: str, by: Any = None, **changes) -> int:
    """A new document row in `corpus`, carrying `doc`'s passages and embeddings over."""
    fields = {k: doc[k] for k in ("title", "source_uri", "doc_type", "metadata", "sha256")}
    fields.update(changes)
    pool = await get_pool()
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO documents (corpus, title, source_uri, doc_type, metadata, sha256, added_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (corpus, fields["title"], fields["source_uri"], fields["doc_type"],
             jsonb(fields["metadata"] or {}), fields["sha256"], by))
        new_id = (await cur.fetchone())["id"]
        await cur.execute(
            "INSERT INTO chunks (document_id, ordinal, content, page_start, page_end, embedding) "
            "SELECT %s, ordinal, content, page_start, page_end, embedding FROM chunks "
            "WHERE document_id=%s", (new_id, doc["id"]))
    return new_id


async def move_document(document_id: int, corpus: str, by: Any = None) -> dict | None:
    """File a document under another library (created if it does not exist).

    The document is copied into the other library and taken out of this one, so each
    library's earlier versions still hold what they held.
    """
    corpus = " ".join(corpus.split())[:120]
    if not corpus:
        raise ValueError("Choose a library to move the document to.")
    doc = await get_document(document_id)
    if doc is None:
        return None
    if doc["removed_at"]:
        raise ValueError("This document is no longer in its library, so it cannot be moved.")
    if corpus == doc["corpus"]:
        return {"id": doc["id"], "title": doc["title"], "corpus": corpus}
    await check_idle(doc["corpus"])
    await check_idle(corpus)
    if await fetch_one("SELECT id FROM documents WHERE corpus=%s AND sha256=%s AND removed_at IS NULL",
                       (corpus, doc["sha256"])):
        raise ValueError(f"“{corpus}” already holds the same text.")
    new_id = await copy_document(doc, corpus, by)
    await fetch_one("UPDATE documents SET removed_at=now(), replaced_by=%s WHERE id=%s RETURNING id",
                    (new_id, document_id))
    await record_version(doc["corpus"], by)
    await record_version(corpus, by)
    return {"id": new_id, "title": doc["title"], "corpus": corpus}


async def delete_corpus(name: str) -> int:
    """Delete a library outright: every document it ever held, and all its versions.
    Returns how many current documents it had.

    Past runs keep their results and citation text, but can no longer be repeated against
    this library.
    """
    await check_idle(name)
    current = await fetch_one("SELECT COUNT(*) AS n FROM documents WHERE corpus=%s "
                              "AND removed_at IS NULL", (name,))
    await fetch_all("DELETE FROM documents WHERE corpus=%s RETURNING id", (name,))
    await fetch_all("DELETE FROM corpus_versions WHERE corpus=%s RETURNING version", (name,))
    log.info("deleted library %s", name)
    return current["n"] if current else 0
