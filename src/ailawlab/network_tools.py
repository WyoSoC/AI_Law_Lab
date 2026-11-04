"""Outside sources for research agents: the legal databases, the open web, reading, keeping.

Two permissions, each the researcher's to grant: the public legal databases (CourtListener,
the Federal Register, the eCFR, govinfo, SEC EDGAR, through their own interfaces) and the open
web (Brave Search, and reading any public page).

Reading and keeping are separate. An agent may read anything its permissions and the site's
AI rules allow; the passages it gets back come from a copy held in memory for the run, and it
may cite them. Keeping saves a full copy into a library (by default "Fetched: <experiment>"),
so the document stays readable after the page changes. An agent asks to keep a source when it
judges it reliable and useful, and keep_rules decides whether the source's rights allow it:
public legal databases, listed federal sites and openly licensed pages pass; anything else is
not kept. A passage of a page that was not kept is cited by its address, the time it was read
and a fingerprint of the text, and only the passages actually cited are stored, as quotations.
A document is read once per run, however many agents ask for it.

Searching is not reading: a search returns candidates ([W1], [W2], ...) with snippets, which
are leads, not sources, and cannot be cited. Every search is kept with the results it gave.
Fetches of a bare address go through source_material.fetch_page, which refuses anything that is
not a public http(s) host. A web page is read only if its site lets AI agents read it
(site_rules: robots.txt, "noai", TDM reservations); a page that does not is not read, the agent
is told why, and the run keeps a list of what was turned away.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import html
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from . import keep_rules, site_rules, source_material, sources
from .config import settings
from .db import fetch_all, fetch_one, get_pool
from .grounding import SourceLedger
from .rag import (
    Corpus,
    Libraries,
    LibraryPin,
    Passage,
    _page_for_offset,
    chunk_text,
    format_passages,
    record_version,
)
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


# Research agents call the databases far more often than a person on the Legal Sources page,
# and several agents at once. Each database's requests from agents are spaced out, and one
# that answers "too many requests" (429) is left alone for a while, across all runs.
DATABASE_SPACING_S = 1.0
DATABASE_COOLDOWN_S = 180.0
_db_locks: dict[str, asyncio.Lock] = {}
_db_last: dict[str, float] = {}
_db_cooling: dict[str, float] = {}


def cooling_for(pid: str) -> float:
    """Seconds left before database `pid` is used again (0 if it can be used now)."""
    return max(0.0, _db_cooling.get(pid, 0.0) - time.monotonic())


def _rate_limited(error: BaseException) -> bool:
    return "429" in str(error) or "Too Many Requests" in str(error)


async def paced(pid: str, call):
    """Run one request to database `pid`, spaced from the last and noting a 429."""
    lock = _db_locks.setdefault(pid, asyncio.Lock())
    async with lock:
        wait = DATABASE_SPACING_S - (time.monotonic() - _db_last.get(pid, 0.0))
        if wait > 0:
            await asyncio.sleep(wait)
        _db_last[pid] = time.monotonic()
    try:
        return await call()
    except Exception as e:
        if _rate_limited(e):
            _db_cooling[pid] = time.monotonic() + DATABASE_COOLDOWN_S
            log.warning("%s is limiting requests; agents leave it alone for %.0f s", pid, DATABASE_COOLDOWN_S)
        raise


class OnlineReader:
    """One run's searches and reads of outside sources, and the library it saves them into.

    Two permissions, each the researcher's to grant: `databases` (the public legal databases
    with their own APIs) and `web` (open-web search, and reading any public page). Reading is
    not budgeted: an agent reads what the question needs. Reading keeps nothing; `keep` saves
    a source the agent chose, if keep_rules allows it. `libraries` is the run's own set: once
    something is kept, the fetch library is pinned to its new version there, so later library
    searches by any of the run's agents cover it too. Agents work in parallel, so saving
    (which records a library version) is serialized.
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
        self.refused: list[dict[str, Any]] = []               # pages whose site does not allow it
        self._save_lock = asyncio.Lock()
        self._read_urls: dict[str, int] = {}                  # address -> its index in fetched
        # Per document read (same index as fetched): its text, passages and their embeddings,
        # held for the run only; and, once kept, its passages' chunk ids in the library.
        self._docs: list[dict[str, Any]] = []

    async def read_given(self, urls: list[str], look_for: str = "") -> str:
        """Read pages named in the brief, before the agents start, and say what came of each:
        its passages to cite, or why it could not be read."""
        out = []
        for url in urls:
            before = len(self.fetched)
            text = await self._read_one(url, look_for)
            if len(self.fetched) > before:
                record = self.fetched[-1]
                record["given"] = True
                # The researcher chose these pages, so they are kept whenever their rights allow.
                if record["keep_allowed"]:
                    text += "\n\n" + await self._keep_one(url, "given with the research brief")
            out.append(text)
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
            if (wait := cooling_for(pid)):
                return pid, (f"it is limiting requests, so it is not used for another {wait:.0f} s; "
                             "search the other databases or the web meanwhile")
            try:
                return pid, await paced(pid, lambda: sources.search(pid, query, settings.network_hits_per_source))
            except Exception as e:  # noqa: BLE001 - one database down must not sink the rest
                if _rate_limited(e):
                    return pid, ("it is limiting requests, so it is not used for the next few "
                                 "minutes; search the other databases or the web meanwhile")
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
                "what you read comes back as numbered passages you can cite.")

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
        # Several agents may want the same document: it is read once, and each gets the
        # passages that fit what it is looking for.
        key = self._key(source)
        if key in self._read_urls:
            index = self._read_urls[key]
            record = self.fetched[index]
            passages = await self._best_passages(index, look_for or record["title"])
            numbers = self.ledger.number(passages)
            if m := _HIT.match(source):
                self.read_hits[int(m.group(1))] = numbers
            return (f"“{record['title']}” was already read in this run. {self._status(record)} "
                    f"The passages most relevant to “{look_for or record['title']}”:\n\n"
                    + format_passages(passages, numbers))
        refused = len(self.refused)
        try:
            doc = await self._fetch(source)
        except LookupError as e:
            if len(self.refused) > refused:
                await self.tracer.note(f"did not read {self.refused[-1]['url']}: "
                                       f"{self.refused[-1]['reason']}", node="act",
                                       agent_id=current_agent.get())
            return f"ERROR: {e}"
        except Exception as e:  # noqa: BLE001 - a failed read is information for the agent
            log.warning("run %s could not read %s: %s", self.run_id, source, e)
            return f"ERROR: could not read {source}: {e}"
        doc["text"] = doc["text"].replace("\x00", "")
        if not doc["text"].strip():
            return f"ERROR: {source} has no readable text."
        try:
            held = await self._hold(doc)
        except Exception as e:  # noqa: BLE001 - one failed read must not sink the others
            log.warning("run %s could not prepare %s: %s", self.run_id, source, e)
            return f"ERROR: {source} was read but its passages could not be prepared ({type(e).__name__})."

        rights: keep_rules.KeepCheck = doc.pop("keep")
        record = {"source": source, "title": doc["title"], "url": doc["source_uri"],
                  "provider": doc["metadata"].get("provider") or "web",
                  "retrieved_at": held["retrieved_at"], "fingerprint": held["fingerprint"],
                  "keep_allowed": rights.allowed, "keep_reason": rights.reason,
                  "license_url": rights.license_url, "kept": False,
                  "document_id": None, "corpus": None, "version": None, "kept_because": ""}
        self.fetched.append(record)
        self._docs.append({**held, "doc": doc})
        index = len(self.fetched) - 1
        self._read_urls[key] = index
        passages = await self._best_passages(index, look_for or doc["title"])
        numbers = self.ledger.number(passages)
        if m := _HIT.match(source):
            self.read_hits[int(m.group(1))] = numbers
        await self.tracer.note(f"read “{doc['title']}” online (not kept; rights check "
                               f"{'positive' if rights.allowed else 'not positive'}: {rights.reason})",
                               node="act", agent_id=current_agent.get())
        return (f"Read “{doc['title']}”. {self._status(record)} The passages most relevant to "
                f"“{look_for or doc['title']}”, which you can cite by number:\n\n"
                + format_passages(passages, numbers))

    def _status(self, record: dict[str, Any]) -> str:
        """What the agent is told about keeping a document it read."""
        if record["kept"]:
            return f"A copy is kept in the library “{record['corpus']}” (version {record['version']})."
        if record["keep_allowed"]:
            return (f"No copy is kept yet. Rights check: positive ({record['keep_reason']}). If it "
                    "is reliable and useful for the study, call keep with "
                    f"“{record['source']}” to save a copy to the library.")
        return (f"No copy will be kept: {record['keep_reason']}. You may still cite its passages; "
                "they are recorded as read online, with the address and the time it was read.")

    async def _hold(self, doc: dict[str, Any]) -> dict[str, Any]:
        """The document's passages and their embeddings, in memory: the same passages a library
        would make of it, so keeping it later gives the same text."""
        chunks = chunk_text(doc["text"])
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), 16):
            vectors.extend(await self.router.embed(chunks[i:i + 16]))
        pages, offset = [], 0
        for content in chunks:
            page_map = doc.get("page_map")
            pages.append((_page_for_offset(page_map, offset), _page_for_offset(page_map, offset + len(content)))
                         if page_map else (None, None))
            offset += len(content)
        return {"chunks": chunks, "vectors": vectors, "pages": pages, "chunk_ids": {},
                "retrieved_at": doc["metadata"].get("retrieved_at")
                or datetime.now(UTC).isoformat(timespec="seconds"),
                "fingerprint": hashlib.sha256(doc["text"].encode()).hexdigest()}

    def _passage(self, index: int, i: int, similarity: float) -> Passage:
        """Passage i of document `index` read in this run: its library chunk once kept, else a
        run-only id (negative, so it can never collide with a library chunk)."""
        record, held = self.fetched[index], self._docs[index]
        start, end = held["pages"][i]
        if record["kept"] and i in held["chunk_ids"]:
            return Passage(chunk_id=held["chunk_ids"][i], document_id=record["document_id"],
                           title=record["title"], content=held["chunks"][i], page_start=start,
                           page_end=end, similarity=similarity, corpus=record["corpus"],
                           version=record["version"])
        return Passage(chunk_id=-(index * 100_000 + i + 1), document_id=0, title=record["title"],
                       content=held["chunks"][i], page_start=start, page_end=end,
                       similarity=similarity, kept=False, url=record["url"] or "",
                       retrieved_at=record["retrieved_at"], fingerprint=record["fingerprint"])

    async def _best_passages(self, index: int, query: str, top_k: int = 4) -> list[Passage]:
        """The document's passages closest to what the agent is looking for."""
        held = self._docs[index]
        q = await self.router.embed_one(query)
        scored = sorted(((_cosine(q, v), i) for i, v in enumerate(held["vectors"])), reverse=True)
        return [self._passage(index, i, sim) for sim, i in scored[:top_k]]

    # ------------------------------------------------------------------ keeping

    async def keep(self, source: str, why: str = "") -> str:
        """Keep a copy of one or several documents read in this run, as the rights allow."""
        refs = ([source.strip()] if source.strip().lower().startswith(("http://", "https://"))
                else [r for r in re.split(r"[\s,;]+", source or "") if r])
        if not refs:
            return "ERROR: give the source as you read it: a result number like W2, or its address."
        return "\n\n".join([await self._keep_one(ref, why) for ref in refs[:self.PER_CALL]])

    async def _keep_one(self, source: str, why: str) -> str:
        key = self._key(source)
        if key not in self._read_urls:
            return f"ERROR: {source} has not been read in this run; read it first."
        index = self._read_urls[key]
        record, held = self.fetched[index], self._docs[index]
        if record["kept"]:
            return f"“{record['title']}” is already kept in “{record['corpus']}” (version {record['version']})."
        if not record["keep_allowed"]:
            await self.tracer.note(f"asked to keep “{record['title']}”; not kept: {record['keep_reason']}",
                                   node="act", agent_id=current_agent.get())
            return (f"Not kept: {record['keep_reason']}. Its passages you read can still be cited; "
                    "they are recorded as read online.")
        try:
            doc_id, version, reused = await self._save(held["doc"])
        except Exception as e:  # noqa: BLE001 - a failed save is information for the agent
            log.warning("run %s could not keep %s: %s", self.run_id, source, e)
            return f"ERROR: {source} could not be kept ({type(e).__name__})."
        rows = await fetch_all("SELECT id, ordinal FROM chunks WHERE document_id=%s", (doc_id,))
        held["chunk_ids"] = {r["ordinal"]: r["id"] for r in rows
                             if r["ordinal"] < len(held["chunks"])}
        record.update({"kept": True, "document_id": doc_id, "corpus": self.library,
                       "version": version, "kept_because": why.strip()})
        # Passages already handed out keep their numbers, and now point to the library copy.
        for p in self.ledger.passages:
            if not p.kept and p.fingerprint == record["fingerprint"]:
                i = -p.chunk_id - 1 - index * 100_000
                if i in held["chunk_ids"]:
                    self.ledger.rekey(p.chunk_id, held["chunk_ids"][i])
                    p.chunk_id = held["chunk_ids"][i]
                p.document_id, p.corpus, p.version, p.kept = doc_id, self.library, version, True
        await self.tracer.note(f"kept “{record['title']}”"
                               + (f" ({why.strip()})" if why.strip() else "")
                               + (f": the same text was already in “{self.library}”, so that copy "
                                  f"is used (v{version})" if reused
                                  else f": saved to “{self.library}” v{version}")
                               + f"; rights: {record['keep_reason']}", node="act",
                               agent_id=current_agent.get())
        return (f"Kept “{record['title']}” in the library “{self.library}” (version {version}); "
                f"rights: {record['keep_reason']}. Its passages keep their numbers.")

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
            if (wait := cooling_for(pid)):
                raise LookupError(f"{self.databases().get(pid, pid)} is limiting requests; try W{n} "
                                  f"again in {wait:.0f} s, or read another result meanwhile.")
            fetched = await paced(pid, lambda: sources.get_provider(pid).fetch(hit["ref"], hit))
            return {"title": fetched.title, "text": fetched.text, "source_uri": fetched.source_uri,
                    "doc_type": fetched.doc_type, "metadata": dict(fetched.metadata),
                    "keep": keep_rules.check(pid, fetched.source_uri or "")}
        if not source.lower().startswith(("http://", "https://")):
            raise LookupError("give a search result's number, like W2, or a web address "
                              "starting with https://.")
        if not self.allow_web:
            raise LookupError("reading web pages is not allowed in this run (the researcher did "
                              "not allow the open web).")
        await source_material.check_public_url(source_material.normalize_link(source))
        await self._check_site(source)
        final, headers, body = await source_material.fetch_page(source, user_agent=settings.agent_user_agent)
        if urlparse(final).netloc != urlparse(source).netloc:
            await self._check_site(final)               # it moved to another site, with its own rules
        if refusal := site_rules.page_refusal(headers, body, site_rules.token(settings.agent_user_agent)):
            self._refuse(source, refusal)
        page = await asyncio.to_thread(source_material.from_bytes, body,
                                       content_type=headers.get("content-type", ""), url=final,
                                       max_words=settings.web_link_max_words)
        rights = keep_rules.check("web", final, body, headers.get("content-type", ""),
                                  sites=settings.keep_sites)
        return {"title": page.title, "text": page.text, "source_uri": page.url or source,
                "doc_type": page.kind, "page_map": page.page_map or None, "keep": rights,
                "metadata": {"link": source, "site": page.site, "published": page.published,
                             "words": page.words, "truncated": page.truncated,
                             "retrieved_at": page.retrieved_at,
                             "permission": "The site's robots.txt allows AI agents to read it, and "
                                           "it carries no AI or text-and-data-mining opt-out.",
                             "keep_reason": rights.reason,
                             **({"license": rights.license_url} if rights.license_url else {})}}

    async def _check_site(self, url: str) -> None:
        """Raise LookupError, and note the page as turned away, if its site's robots.txt or
        TDM reservations do not let AI agents read it."""
        robots = await site_rules.for_page(url, settings.agent_user_agent)
        if refusal := robots.refusal(url):
            self._refuse(url, refusal)

    def _refuse(self, url: str, reason: str) -> None:
        if not any(r["url"] == url for r in self.refused):
            self.refused.append({"url": url, "reason": reason, "agent": current_agent.get()})
        raise LookupError(f"{url} was not read: {reason}. Find another source for this.")

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


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
