"""Network tools for the agentic workflow: search public legal databases, read what is found.

An agent that reads the web mid-run would normally leave nothing to check: the page can
change or vanish, and a citation to it points nowhere fixed. So reading here is ingesting.
Every document an agent reads is saved into a library (by default "Fetched: <experiment>"),
which records a new version, and the passages handed back are that library's passages,
numbered in the run's ledger like any other. A citation to something read online therefore
names a document, a library and a version, exactly as a citation to a curated library does,
and the document stays readable after the page changes.

Searching is not reading: a search returns candidates ([W1], [W2], ...) with snippets, which
are leads, not sources, and cannot be cited. Both tools need the network and are offered
only to experiments that allow it (tools.ToolRegistry). Fetches of a bare address go through
source_material.fetch_url, which refuses anything that is not a public http(s) host.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any

from . import source_material, sources
from .config import settings
from .db import fetch_one, get_pool
from .grounding import SourceLedger
from .rag import Corpus, Libraries, LibraryPin, Passage, format_passages, record_version
from .router import LLMRouter

log = logging.getLogger(__name__)

_HIT = re.compile(r"^\s*\[?W(\d+)\]?\s*$", re.IGNORECASE)


def sources_wanted(config: dict[str, Any]) -> int:
    """How many sources an agent should read online: the experiment's (or run's)
    `network_sources`, within 1 and settings.network_sources_max. Pure."""
    try:
        n = int(config.get("network_sources") or settings.network_sources_default)
    except (TypeError, ValueError):
        n = settings.network_sources_default
    return max(1, min(n, settings.network_sources_max))


def fetch_library_name(config: dict[str, Any], experiment_name: str) -> str:
    """The library a run saves what it reads into: the experiment's choice, or one named for
    the experiment. Pure."""
    chosen = config.get("fetch_library")
    name = chosen if isinstance(chosen, str) and chosen.strip() else f"Fetched: {experiment_name}"
    return " ".join(name.split())[:120]


_W_MARK = re.compile(r"\[(W\d{1,3}(?:\s*,\s*W?\d{1,3})*)\]")


def resolve_result_markers(text: str, numbers_for) -> tuple[str, list[int]]:
    """An answer with search-result markers ([W2], [W1, W3]) turned into the passage numbers
    of the documents read from them, which are what can be cited. Returns the text and the
    result numbers cited that were never read (left as they are, to be recorded unsupported).
    `numbers_for(n)` gives result n's passage numbers, or [] if it was not read. Pure."""
    unread: list[int] = []

    def swap(m: re.Match) -> str:
        ns = [int(x.strip().lstrip("W")) for x in m.group(1).split(",")]
        passages = [numbers_for(n)[0] for n in ns if numbers_for(n)]
        unread.extend(n for n in ns if not numbers_for(n))
        if not passages or len(passages) < len(ns):
            return m.group(0)
        return "[" + ", ".join(str(p) for p in dict.fromkeys(passages)) + "]"

    return _W_MARK.sub(swap, text), unread


class OnlineReader:
    """One run's searches and reads of public sources, and the library it saves them into.

    `libraries` is the run's own set: once something is saved, the fetch library is pinned
    to its new version there, so the agent's later library searches cover it too.
    """

    def __init__(self, router: LLMRouter, libraries: Libraries, ledger: SourceLedger, tracer,
                 run_id: str, library: str, added_by: Any = None,
                 max_reads: int | None = None, wanted: int | None = None):
        self.router, self.libraries, self.ledger, self.tracer = router, libraries, ledger, tracer
        self.run_id, self.library, self.added_by = run_id, library, added_by
        # The agent is asked for `wanted` sources; its budget leaves room for two that turn out
        # to be unreadable or empty.
        self.wanted = wanted or settings.network_sources_default
        self.max_reads = max_reads or (self.wanted + 2 if wanted else settings.network_max_reads)
        self.hits: list[tuple[str, dict[str, Any]]] = []      # [W<n>] -> (provider id, hit)
        self.fetched: list[dict[str, Any]] = []               # what was read, for the result
        self.read_hits: dict[int, list[int]] = {}             # W<n> read -> its passage numbers

    @staticmethod
    def databases() -> dict[str, str]:
        """Databases that can be searched now (those needing a key report themselves off)."""
        return {pid: p.name for pid, p in sources.PROVIDERS.items() if p.status()[0]}

    # ------------------------------------------------------------------ searching

    async def search(self, query: str, database: str = "") -> str:
        dbs = self.databases()
        database = (database or "").strip()
        if database and database not in dbs:
            return f"ERROR: unknown database {database!r}. Choose one of: {', '.join(dbs)}."
        targets = [database] if database else list(dbs)

        async def one(pid: str) -> tuple[str, list | str]:
            try:
                return pid, await sources.search(pid, query, settings.network_hits_per_source)
            except Exception as e:  # noqa: BLE001 - one database down must not sink the rest
                return pid, f"{type(e).__name__}: {e}"

        lines: list[str] = []
        for pid, found in await asyncio.gather(*(one(p) for p in targets)):
            if isinstance(found, str):
                lines.append(f"({dbs[pid]} could not be searched: {found})")
                continue
            for hit in found:
                self.hits.append((pid, hit.as_dict()))
                n = len(self.hits)
                meta = ", ".join(x for x in (hit.badge, hit.date) if x)
                lines.append(f"[W{n}] {hit.title} ({dbs[pid]}{', ' + meta if meta else ''})"
                             + ("" if hit.full_text else " -- only its snippet can be read")
                             + (f"\n{hit.snippet[:300]}" if hit.snippet else ""))
        if not any(line.startswith("[W") for line in lines):
            lines.insert(0, "No results.")
        return ("\n\n".join(lines) + "\n\nThese are search results, not sources: do not cite "
                "them. To use one, call read_online with its [W] number; what you read is saved "
                "and handed back as numbered passages you can cite.")

    # ------------------------------------------------------------------ reading

    # Results one read_online call may take: enough to read the best few of a search at once.
    PER_CALL = 5

    async def read(self, source: str, look_for: str = "") -> str:
        """Read one or several sources: "W2", "W1, W3, W4", or a single address."""
        source = (source or "").strip()
        refs = ([source] if source.lower().startswith(("http://", "https://"))
                else [r for r in re.split(r"[\s,;]+", source) if r])
        if not refs:
            return "ERROR: give a search result's number, like W2, or a web address."
        out = [await self._read_one(ref, look_for) for ref in refs[:self.PER_CALL]]
        if len(refs) > self.PER_CALL:
            out.append(f"(Only the first {self.PER_CALL} were read; ask again for the rest.)")
        return "\n\n".join(out)

    def cite_numbers(self, n: int) -> list[int]:
        """The passage numbers handed back when search result W<n> was read ([] if never read)."""
        return self.read_hits.get(n, [])

    async def _read_one(self, source: str, look_for: str = "") -> str:
        if len(self.fetched) >= self.max_reads:
            return (f"ERROR: this run has already read {self.max_reads} documents online, the "
                    "most it may. Work with what you have, or search the libraries.")
        try:
            doc = await self._fetch(source)
        except LookupError as e:
            return f"ERROR: {e}"
        except Exception as e:  # noqa: BLE001 - a failed read is information for the agent
            log.warning("run %s could not read %s: %s", self.run_id, source, e)
            return f"ERROR: could not read {source}: {e}"
        if not doc["text"].strip():
            return f"ERROR: {source} has no readable text."

        try:
            doc_id, version, reused = await self._save(doc)
        except Exception as e:  # noqa: BLE001 - one failed save must not sink the others
            log.warning("run %s could not save %s: %s", self.run_id, source, e)
            return f"ERROR: {source} was read but could not be saved ({type(e).__name__})."
        passages = await self._passages(doc_id, version, look_for or doc["title"])
        record = {"source": source, "title": doc["title"], "url": doc["source_uri"],
                  "provider": doc["metadata"].get("provider") or "web", "document_id": doc_id,
                  "corpus": self.library, "version": version}
        self.fetched.append(record)
        numbers = self.ledger.number(passages)
        if m := _HIT.match(source):
            self.read_hits[int(m.group(1))] = numbers
        await self.tracer.note(f"read “{doc['title']}” online; "
                               + (f"the same text was already in “{self.library}”, so that copy "
                                  f"is used (v{version})" if reused
                                  else f"saved it to “{self.library}” v{version}"), node="act")
        return (f"Read “{doc['title']}” and saved it to the library “{self.library}” "
                f"(version {version}). The passages most relevant to "
                f"“{look_for or doc['title']}”, which you can cite by number:\n\n"
                + format_passages(passages, numbers))

    async def _fetch(self, source: str) -> dict[str, Any]:
        """The document behind a [W] search result or a public address."""
        if m := _HIT.match(source):
            n = int(m.group(1))
            if not 1 <= n <= len(self.hits):
                raise LookupError(f"there is no search result W{n} in this run.")
            pid, hit = self.hits[n - 1]
            fetched = await sources.get_provider(pid).fetch(hit["ref"], hit)
            return {"title": fetched.title, "text": fetched.text, "source_uri": fetched.source_uri,
                    "doc_type": fetched.doc_type, "metadata": dict(fetched.metadata)}
        if not source.lower().startswith(("http://", "https://")):
            raise LookupError("give a search result's number, like W2, or a web address "
                              "starting with https://.")
        page = await source_material.fetch_url(source, max_words=settings.web_link_max_words)
        return {"title": page.title, "text": page.text, "source_uri": page.url or source,
                "doc_type": page.kind, "page_map": page.page_map or None,
                "metadata": {"link": source, "site": page.site, "published": page.published,
                             "words": page.words, "truncated": page.truncated,
                             "retrieved_at": page.retrieved_at}}

    async def _save(self, doc: dict[str, Any]) -> tuple[int, int, bool]:
        """Add the document to the fetch library (once: the same text is reused) and record
        the library's new version, pinned for the rest of this run. Returns the document id,
        the version, and whether an earlier copy was reused."""
        meta = {**doc["metadata"], "fetched_by_run": self.run_id}
        doc_id = await Corpus(self.router, name=self.library, added_by=self.added_by).add_document(
            doc["title"], doc["text"], source_uri=doc["source_uri"], doc_type=doc["doc_type"],
            metadata=meta, page_map=doc.get("page_map"))
        reused = doc_id is None
        if reused:
            sha = hashlib.sha256(doc["text"].encode()).hexdigest()
            row = await fetch_one("SELECT id FROM documents WHERE corpus=%s AND sha256=%s "
                                  "AND removed_at IS NULL", (self.library, sha))
            doc_id = row["id"]
        v = await record_version(self.library, self.added_by)
        pin = LibraryPin(self.library, v["version"], list(v["document_ids"]))
        others = [p for p in self.libraries.pins if p.name != self.library]
        position = next((i for i, p in enumerate(self.libraries.pins) if p.name == self.library),
                        len(self.libraries.pins))
        self.libraries.pins = others[:position] + [pin] + others[position:]
        pool = await get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO run_libraries (run_id, position, corpus, version, documents) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (run_id, corpus) DO UPDATE "
                "SET version = EXCLUDED.version, documents = EXCLUDED.documents",
                (self.run_id, position, self.library, pin.version, len(pin.document_ids)))
        return doc_id, pin.version, reused

    async def _passages(self, doc_id: int, version: int, query: str) -> list[Passage]:
        """The document's passages closest to what the agent is looking for; its opening
        passages if nothing matches closely."""
        only = Libraries(self.router, [LibraryPin(self.library, version, [doc_id])])
        found = await only.search(query, top_k=4)
        if found:
            return found
        rows = await fetch_one(
            "SELECT json_agg(x ORDER BY x.ordinal) AS rows FROM (SELECT c.id AS chunk_id, "
            "c.document_id, d.title, c.content, c.page_start, c.page_end, c.ordinal "
            "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.document_id=%s "
            "ORDER BY c.ordinal LIMIT 2) x", (doc_id,))
        return [Passage(chunk_id=r["chunk_id"], document_id=r["document_id"], title=r["title"],
                        content=r["content"], page_start=r["page_start"], page_end=r["page_end"],
                        similarity=0.0, corpus=self.library, version=version)
                for r in (rows or {}).get("rows") or []]
