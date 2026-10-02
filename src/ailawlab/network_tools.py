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
import contextvars
import hashlib
import html
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from . import source_material, sources
from .config import settings
from .db import fetch_one, get_pool
from .grounding import SourceLedger
from .rag import Corpus, Libraries, LibraryPin, Passage, format_passages, record_version
from .router import LLMRouter

log = logging.getLogger(__name__)

# Which research agent is reading, so the trace says who read what (agents share a reader).
current_agent: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_agent", default=None)

_HIT = re.compile(r"^\s*\[?W(\d+)\]?\s*$", re.IGNORECASE)


def permissions(config: dict[str, Any]) -> tuple[bool, bool]:
    """What a run may reach outside its libraries: (legal databases, open web). Experiments
    from before the two were separate say `allow_network`, which meant the databases (and
    pages given by address). Pure."""
    legacy = bool(config.get("allow_network"))
    databases = config.get("use_databases")
    web = config.get("use_web")
    return (legacy if databases is None else bool(databases),
            bool(web) if web is not None else False)


_URL = re.compile(r"https?://[^\s<>\"'()\[\]]+", re.IGNORECASE)


def urls_in(text: str, limit: int = 10) -> list[str]:
    """Web addresses written in a text (a brief), each once, without trailing punctuation. Pure."""
    out: list[str] = []
    for m in _URL.finditer(text or ""):
        url = m.group(0).rstrip(".,;:!?")
        if url not in out:
            out.append(url)
    return out[:limit]


def clean_urls(raw: Any, limit: int = 10) -> list[str]:
    """http(s) addresses from a list or one-per-line text, each once. Pure."""
    items = raw.splitlines() if isinstance(raw, str) else [str(x) for x in raw or [] if x]
    out: list[str] = []
    for item in items:
        url = item.strip()
        if url.lower().startswith(("http://", "https://")) and url not in out:
            out.append(url)
    return out[:limit]


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


class WebSearchUnavailable(RuntimeError):
    """Open-web search is not configured (no Brave key)."""


def web_search_ready() -> bool:
    return bool(settings.brave_api_key.strip())


_brave_lock = asyncio.Lock()
_brave_last = 0.0


async def brave_search(query: str, count: int | None = None) -> list[dict[str, Any]]:
    """Open-web results from the Brave Search API: [{title, url, snippet, age}]. Requests are
    spaced at least a second apart (the API's per-second limit), and one 429 is retried."""
    global _brave_last
    if not web_search_ready():
        raise WebSearchUnavailable("open-web search needs AILAWLAB_BRAVE_API_KEY")
    params = {"q": query[:400], "count": max(1, min(count or settings.web_hits, 20))}
    headers = {"Accept": "application/json", "X-Subscription-Token": settings.brave_api_key.strip()}
    async with _brave_lock, httpx.AsyncClient(timeout=settings.source_timeout_s) as client:
        for attempt in range(2):
            wait = 1.1 - (time.monotonic() - _brave_last)
            if wait > 0:
                await asyncio.sleep(wait)
            r = await client.get(BRAVE_URL, params=params, headers=headers)
            _brave_last = time.monotonic()
            if r.status_code == 429 and attempt == 0:
                await asyncio.sleep(2.0)
                continue
            r.raise_for_status()
            break
    data = r.json()
    return [{"title": strip_tags(x.get("title") or x.get("url") or ""), "url": x.get("url") or "",
             "snippet": strip_tags(x.get("description") or ""), "age": x.get("age") or "",
             "site": (x.get("profile") or {}).get("name") or ""}
            for x in (data.get("web") or {}).get("results") or [] if x.get("url")]


BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_TAG = re.compile(r"<[^>]+>")


def strip_tags(text: str) -> str:
    """Brave marks matched words with <strong>; plain text is wanted. Pure."""
    return html.unescape(_TAG.sub("", text or "")).strip()


class OnlineReader:
    """One run's searches and reads of outside sources, and the library it saves them into.

    Two permissions, each the researcher's to grant: `databases` (the public legal databases
    with their own APIs) and `web` (open-web search, and reading any public page). Reading is
    not budgeted: an agent reads what the question needs. `libraries` is the run's own set:
    once something is saved, the fetch library is pinned to its new version there, so later
    library searches by any of the run's agents cover it too. Agents work in parallel, so
    saving (which records a library version) is serialized.
    """

    def __init__(self, router: LLMRouter, libraries: Libraries, ledger: SourceLedger, tracer,
                 run_id: str, library: str, added_by: Any = None,
                 databases: bool = True, web: bool = False):
        self.router, self.libraries, self.ledger, self.tracer = router, libraries, ledger, tracer
        self.run_id, self.library, self.added_by = run_id, library, added_by
        self.allow_databases, self.allow_web = databases, web
        self.hits: list[tuple[str, dict[str, Any]]] = []      # [W<n>] -> (provider id or "web", hit)
        self.fetched: list[dict[str, Any]] = []               # what was read, for the result
        self.read_hits: dict[int, list[int]] = {}             # W<n> read -> its passage numbers
        self.searches: list[dict[str, Any]] = []              # every search, with what it returned
        self._save_lock = asyncio.Lock()
        self._read_urls: dict[str, int] = {}                  # address -> its index in fetched

    async def read_given(self, urls: list[str], look_for: str = "") -> str:
        """Read pages named in the brief, before the agents start, and say what came of each:
        its passages to cite, or why it could not be read."""
        out = []
        for url in urls:
            before = len(self.fetched)
            out.append(await self._read_one(url, look_for))
            if len(self.fetched) > before:
                self.fetched[-1]["given"] = True
        return "\n\n".join(out)

    @staticmethod
    def databases() -> dict[str, str]:
        """Databases that can be searched now (those needing a key report themselves off)."""
        return {pid: p.name for pid, p in sources.PROVIDERS.items() if p.status()[0]}

    def _record_search(self, kind: str, query: str, where: str, numbers: list[int]) -> None:
        self.searches.append({"kind": kind, "query": query, "where": where,
                              "results": [{"n": n, "title": self.hits[n - 1][1].get("title"),
                                           "url": self.hits[n - 1][1].get("url")
                                           or self.hits[n - 1][1].get("source_uri")}
                                          for n in numbers]})

    # ------------------------------------------------------------------ searching

    async def search(self, query: str, database: str = "") -> str:
        """Search the public legal databases."""
        if not self.allow_databases:
            return "ERROR: searching the legal databases is not allowed in this run."
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
        numbers: list[int] = []
        for pid, found in await asyncio.gather(*(one(p) for p in targets)):
            if isinstance(found, str):
                lines.append(f"({dbs[pid]} could not be searched: {found})")
                continue
            for hit in found:
                self.hits.append((pid, hit.as_dict()))
                n = len(self.hits)
                numbers.append(n)
                meta = ", ".join(x for x in (hit.badge, hit.date) if x)
                lines.append(f"[W{n}] {hit.title} ({dbs[pid]}{', ' + meta if meta else ''})"
                             + ("" if hit.full_text else " -- only its snippet can be read")
                             + (f"\n{hit.snippet[:300]}" if hit.snippet else ""))
        self._record_search("databases", query, database or "all databases", numbers)
        return self._results_text(lines)

    async def search_web(self, query: str) -> str:
        """Search the open web (Brave)."""
        if not self.allow_web:
            return "ERROR: searching the open web is not allowed in this run."
        try:
            found = await brave_search(query)
        except Exception as e:  # noqa: BLE001 - a failed search is information for the agent
            log.warning("run %s web search failed: %s", self.run_id, e)
            return f"ERROR: the web search failed ({type(e).__name__}). Try again or another wording."
        lines: list[str] = []
        numbers: list[int] = []
        for hit in found:
            self.hits.append(("web", hit))
            n = len(self.hits)
            numbers.append(n)
            meta = ", ".join(x for x in (hit["site"] or urlparse(hit["url"]).netloc, hit["age"]) if x)
            lines.append(f"[W{n}] {hit['title']} ({meta})\n{hit['url']}"
                         + (f"\n{hit['snippet'][:300]}" if hit["snippet"] else ""))
        self._record_search("web", query, "the open web", numbers)
        return self._results_text(lines)

    @staticmethod
    def _results_text(lines: list[str]) -> str:
        if not any(line.startswith("[W") for line in lines):
            lines.insert(0, "No results.")
        return ("\n\n".join(lines) + "\n\nThese are search results, not sources: do not cite "
                "them. To use one, call read with its [W] number (several at once: W1, W3, W4); "
                "what you read is saved and handed back as numbered passages you can cite.")

    # ------------------------------------------------------------------ reading

    # Results one read call may take: enough to read the best few of a search at once.
    PER_CALL = 6

    async def read(self, source: str, look_for: str = "") -> str:
        """Read one or several sources: "W2", "W1, W3, W4", or a single address."""
        source = (source or "").strip()
        refs = ([source] if source.lower().startswith(("http://", "https://"))
                else [r for r in re.split(r"[\s,;]+", source) if r])
        if not refs:
            return "ERROR: give a search result's number, like W2, or a web address."
        out = await asyncio.gather(*(self._read_one(ref, look_for) for ref in refs[:self.PER_CALL]))
        out = list(out)
        if len(refs) > self.PER_CALL:
            out.append(f"(Only the first {self.PER_CALL} were read; ask again for the rest.)")
        return "\n\n".join(out)

    def cite_numbers(self, n: int) -> list[int]:
        """The passage numbers handed back when search result W<n> was read ([] if never read)."""
        return self.read_hits.get(n, [])

    def _key(self, source: str) -> str:
        """What a source is, whichever way it is named: a result's address, or the address."""
        if m := _HIT.match(source):
            n = int(m.group(1))
            if 1 <= n <= len(self.hits):
                pid, hit = self.hits[n - 1]
                return hit.get("url") or hit.get("source_uri") or f"{pid}:{hit.get('ref')}"
        return source.strip()

    async def _read_one(self, source: str, look_for: str = "") -> str:
        # Several agents may want the same document: it is read and saved once, and each gets
        # the passages that fit what it is looking for.
        key = self._key(source)
        if key in self._read_urls:
            record = self.fetched[self._read_urls[key]]
            passages = await self._passages(record["document_id"], record["version"],
                                            look_for or record["title"])
            numbers = self.ledger.number(passages)
            if m := _HIT.match(source):
                self.read_hits[int(m.group(1))] = numbers
            return (f"“{record['title']}” was already read in this run (library “{self.library}” "
                    f"v{record['version']}). The passages most relevant to “{look_for or record['title']}”:"
                    "\n\n" + format_passages(passages, numbers))
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
        self._read_urls[key] = len(self.fetched) - 1
        numbers = self.ledger.number(passages)
        if m := _HIT.match(source):
            self.read_hits[int(m.group(1))] = numbers
        await self.tracer.note(f"read “{doc['title']}” online; "
                               + (f"the same text was already in “{self.library}”, so that copy "
                                  f"is used (v{version})" if reused
                                  else f"saved it to “{self.library}” v{version}"), node="act",
                               agent_id=current_agent.get())
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
            if pid == "web":
                return await self._fetch(hit["url"])
            if not self.allow_databases:
                raise LookupError("reading the legal databases is not allowed in this run.")
            fetched = await sources.get_provider(pid).fetch(hit["ref"], hit)
            return {"title": fetched.title, "text": fetched.text, "source_uri": fetched.source_uri,
                    "doc_type": fetched.doc_type, "metadata": dict(fetched.metadata)}
        if not source.lower().startswith(("http://", "https://")):
            raise LookupError("give a search result's number, like W2, or a web address "
                              "starting with https://.")
        if not self.allow_web:
            raise LookupError("reading web pages is not allowed in this run (the researcher did "
                              "not allow the open web).")
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
        async with self._save_lock:
            return await self._save_locked(doc)

    async def _save_locked(self, doc: dict[str, Any]) -> tuple[int, int, bool]:
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
