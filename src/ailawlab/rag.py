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

    def cite_label(self) -> str:
        if self.page_start and self.page_end and self.page_start != self.page_end:
            return f"{self.title}, pp. {self.page_start}-{self.page_end}"
        if self.page_start:
            return f"{self.title}, p. {self.page_start}"
        return self.title


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
    def __init__(self, router: LLMRouter, name: str = "default"):
        self.router = router
        self.name = name

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
        """Ingest one document. Returns None if it is already present (by sha256).

        `page_map` is an optional list of (char_offset, page_number) pairs used to anchor
        chunks to pages; PDF ingestion supplies it.
        """
        sha = hashlib.sha256(text.encode()).hexdigest()
        existing = await fetch_one("SELECT id FROM documents WHERE sha256=%s", (sha,))
        if existing:
            log.info("document already ingested: %s", title)
            return None

        row = await fetch_one(
            "INSERT INTO documents (corpus, title, source_uri, doc_type, metadata, sha256) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (self.name, title, source_uri, doc_type, jsonb(metadata or {}), sha),
        )
        doc_id = row["id"]

        chunks = chunk_text(text)
        if not chunks:
            return doc_id

        # Embed in batches so a large document does not occupy a slot indefinitely.
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), 16):
            vectors.extend(await self.router.embed(chunks[i:i + 16]))

        offset = 0
        pool = await get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            for ordinal, (content, vector) in enumerate(zip(chunks, vectors)):
                page = _page_for_offset(page_map, offset) if page_map else None
                end_page = _page_for_offset(page_map, offset + len(content)) if page_map else None
                await cur.execute(
                    "INSERT INTO chunks (document_id, ordinal, content, page_start, page_end, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (doc_id, ordinal, content, page, end_page, vec(vector)),
                )
                offset += len(content)
        log.info("ingested %s: %d chunks", title, len(chunks))
        return doc_id

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
        top_k = top_k or settings.rag_top_k
        qvec = vec(await self.router.embed_one(query))
        rows = await fetch_all(
            "SELECT c.id AS chunk_id, c.document_id, d.title, c.content, "
            "       c.page_start, c.page_end, 1 - (c.embedding <=> %s) AS similarity "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.corpus = %s AND c.embedding IS NOT NULL "
            "  AND 1 - (c.embedding <=> %s) >= %s "
            "ORDER BY c.embedding <=> %s LIMIT %s",
            (qvec, self.name, qvec, settings.rag_min_similarity, qvec, top_k),
        )
        return [Passage(**r) for r in rows]

    async def stats(self) -> dict:
        row = await fetch_one(
            "SELECT COUNT(DISTINCT d.id) AS documents, COUNT(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id WHERE d.corpus=%s",
            (self.name,),
        )
        return {"corpus": self.name, **(row or {})}


def _page_for_offset(page_map: list[tuple[int, int]], offset: int) -> int | None:
    page = None
    for start, n in page_map:
        if start <= offset:
            page = n
        else:
            break
    return page


def format_passages(passages: list[Passage]) -> str:
    """Render passages for a prompt, numbered so the model can cite [1], [2], ..."""
    return "\n\n".join(
        f"[{i}] {p.cite_label()}\n{p.content}" for i, p in enumerate(passages, start=1)
    )
