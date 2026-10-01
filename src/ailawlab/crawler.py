"""Add everything linked from a page, on the same website, to a library -- politely.

A person gives one page (an index of opinions, a list of PDFs) and a library. The crawler
reads that page, collects the links on it that stay on the same website, and reads each one
into the library like an added web link. It goes one level deep: the pages it links to are
read, links on those pages are not followed.

The rules it follows are in RULES below, written for the people who start a crawl. The
Legal Sources page shows them before anything is fetched, every crawl stores the values it
ran under, and the code below is what enforces each one:

* the site's robots.txt is read first and obeyed, including any Crawl-delay; a robots.txt
  that cannot be read because the site is failing stops the crawl (RFC 9309, section 2.3.1);
* requests go one at a time, with a pause between them, and identify the lab;
* a crawl takes at most settings.crawl_max_pages pages, and stops when the site says it is
  overloaded (429, 503) or when requests keep failing;
* only public http(s) addresses on the same website are fetched (source_material's guard),
  and only web pages, PDFs and text are kept;
* the licence or copyright notice found is stored with every document, with its address,
  the page it was linked from, and when it was read.

A crawl runs in the background and records its progress in the `crawls` table, so a long one
outlives the request that started it and the page can show how far it has got.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from . import source_material
from .config import settings
from .db import fetch_all, fetch_one, jsonb
from .rag import Corpus, check_idle, record_version
from .router import LLMRouter

log = logging.getLogger(__name__)


def rules() -> list[dict[str, str]]:
    """The crawler's rules, in plain words, with the values in force."""
    delay = f"{settings.crawl_delay_s:g}"
    return [
        {"title": "Same website, one level deep",
         "text": "Only links on the page you give that lead to the same website are read (the same "
                 "host, with or without “www.”). The pages they lead to are read, but links on "
                 "those pages are not followed. You can narrow it further, to some kinds of file "
                 "or to links under the page's own folder."},
        {"title": "robots.txt is obeyed",
         "text": "The site's robots.txt is read before anything else, and every page it disallows "
                 "for this crawler is skipped. If it asks for a longer pause between requests, "
                 f"that pause is used (up to {settings.crawl_max_delay_s:g} seconds). If robots.txt "
                 "cannot be read because the site is failing, nothing is crawled."},
        {"title": "One page at a time, with a pause",
         "text": f"Pages are fetched one after another, at least {delay} seconds apart, never in "
                 "parallel, so a crawl puts no more load on a site than a person reading it."},
        {"title": "A limit on every crawl",
         "text": f"One crawl reads at most {settings.crawl_max_pages} pages; you can set a lower "
                 "limit. Links already in the library are not read again."},
        {"title": "It stops when asked",
         "text": "If the site answers “too many requests” (429) or “service unavailable” (503), "
                 "or three requests in a row get no answer or an error, the crawl stops. A file "
                 "that downloads but cannot be read (a damaged PDF, a scanned image) is skipped "
                 "and does not count. You can also stop a crawl at any time, and continue it "
                 "later; what was read until then is kept."},
        {"title": "It says who it is",
         "text": f"Every request carries the name “{settings.crawl_user_agent}”, so the site's "
                 "owner can see who is reading and how to reach the lab."},
        {"title": "Public pages and readable files only",
         "text": "Only public http(s) addresses are fetched; pages behind a login are not. Web "
                 "pages, PDFs and plain text are kept; images, archives, audio, video and scripts "
                 "are skipped."},
        {"title": "The licence and the source are kept",
         "text": "Each document is named as the page you started from lists it (or by its own "
                 "title). The licence or copyright notice found on each page (or, failing that, on "
                 "the page you started from) is stored with every document, along with its "
                 "address, the page it was linked from and when it was read. Check that the licence permits "
                 "your use before relying on what was collected."},
        {"title": "Every crawl is recorded",
         "text": "Who started it, the rules it ran under, and every link it read or skipped, with "
                 "the reason."},
    ]


# ---------------------------------------------------------------- links and licences (pure)

# What a link's path ends in decides whether it is worth fetching: documents yes, media and
# code no. A path with no extension is usually a web page.
_KEEP = {"", ".html", ".htm", ".xhtml", ".php", ".asp", ".aspx", ".jsp", ".cfm", ".shtml",
         ".pdf", ".txt", ".md"}


def same_site(a: str, b: str) -> bool:
    def host(u: str) -> str:
        h = (urlsplit(u).hostname or "").lower()
        return h.removeprefix("www.")
    return bool(host(a)) and host(a) == host(b)


def folder_of(url: str) -> str:
    """The folder a page sits in: "/lib/" for /lib/list-pdf.html."""
    path = urlsplit(url).path or "/"
    return path if path.endswith("/") else path.rsplit("/", 1)[0] + "/"


def narrow(links: list[str], start_url: str, kinds: list[str] | None = None,
           same_folder: bool = False) -> list[str]:
    """The links a person chose to take: of the kinds named (all if none), and with
    `same_folder` only those under the start page's folder. Pure."""
    folder = folder_of(start_url)
    return [u for u in links
            if (not kinds or kind_of(u) in kinds)
            and (not same_folder or (urlsplit(u).path or "/").startswith(folder))]


def kind_of(url: str) -> str:
    """"pdf", "page", "text", or "" for a link that is not kept."""
    path = urlsplit(url).path.lower()
    name = path.rsplit("/", 1)[-1]
    ext = name[name.rfind("."):] if "." in name else ""
    if ext not in _KEEP:
        return ""
    return {"pdf": "pdf", ".pdf": "pdf", ".txt": "text", ".md": "text"}.get(ext, "page")


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.licence_links: list[str] = []
        self.base = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "base" and a.get("href"):
            self.base = a["href"]
        if tag in ("a", "area", "link") and a.get("href"):
            if "license" in a.get("rel", "").lower().split():
                self.licence_links.append(a["href"])
            if tag != "link":
                self.hrefs.append(a["href"])


def _bare(url: str) -> str:
    """A page's address without "www.", a trailing slash or ".html", to spot a link back to
    the page itself written another way."""
    p = urlsplit(url)
    host = (p.hostname or "").removeprefix("www.")
    path = re.sub(r"(/|\.html?)$", "", p.path)
    return f"{host}{path}?{p.query}"


# Elements whose text describes the links inside them: a list item, a table cell, a paragraph.
_BLOCKS = {"li", "td", "th", "dd", "dt", "p", "h1", "h2", "h3", "h4", "h5", "h6",
           "caption", "figcaption"}


class _Named(HTMLParser):
    """Each link's own text, and the text of the block it sits in."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, Any]] = []        # {href, text, block}
        self.blocks: dict[int, list[str]] = {}
        self._open: list[int] = []
        self._a: dict[str, Any] | None = None
        self.base = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "base" and a.get("href"):
            self.base = a["href"]
        if tag in _BLOCKS:
            self._open.append(len(self.blocks))
            self.blocks[len(self.blocks)] = []
        if tag == "a" and a.get("href"):
            self._a = {"href": a["href"], "text": [], "block": self._open[-1] if self._open else None}
            self.links.append(self._a)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._a = None
        if tag in _BLOCKS and self._open:
            self._open.pop()

    def handle_data(self, data: str) -> None:
        if self._a is not None:
            self._a["text"].append(data)
        for b in self._open:
            self.blocks[b].append(data)


def _stem(url: str) -> str:
    """An address without its file extension: warder-key.pdf and warder-key are one document."""
    p = urlsplit(url)
    return f"{(p.hostname or '').removeprefix('www.')}{re.sub(r'\.[A-Za-z0-9]{1,5}$', '', p.path).rstrip('/')}"


def link_names(html: str, page_url: str) -> dict[str, str]:
    """What the page calls each document it links to, by address. Pure.

    In order: the link's own words; the text of the list item or table cell it sits in, when
    that item is about this document alone (an index often puts the title, author and year
    there, with only an icon on the PDF link); the words of another link in the same item
    that leads to the same document.
    Generic link words ("PDF", "download", an icon) do not count as a name.
    """
    parser = _Named()
    parser.feed(html)
    base = urljoin(page_url, parser.base) if parser.base else page_url
    links = []
    for link in parser.links:
        href = link["href"].strip()
        if href.startswith("#"):
            continue
        url = urldefrag(urljoin(base, href)).url
        links.append({"url": url, "stem": _stem(url), "block": link["block"],
                      "text": source_material.usable_title("".join(link["text"]))})
    names: dict[str, str] = {}
    for link in links:
        if link["url"] in names:
            continue
        name = link["text"]
        if not name and link["block"] is not None:
            mates = [m for m in links if m["block"] == link["block"]]
            if len({m["stem"] for m in mates}) == 1:          # the item is about this document
                name = source_material.usable_title("".join(parser.blocks[link["block"]]))
            name = name or next((m["text"] for m in mates if m["stem"] == link["stem"] and m["text"]), "")
        if name:
            names[link["url"]] = _unsort(name)[:300]
    return names


def _unsort(title: str) -> str:
    """An index's sorting order undone: "Autobiography of Ajaan Lee, The (…)" reads
    "The Autobiography of Ajaan Lee (…)"."""
    m = re.match(r"^(.+?), (The|A|An)(\s*\(.*)?$", title)
    return f"{m.group(2)} {m.group(1)}{m.group(3) or ''}" if m else title


def extract_links(html: str, page_url: str) -> list[str]:
    """The distinct links on a page that stay on its website and lead to something worth
    reading, in page order, without fragments, and not the page itself."""
    parser = _Links()
    parser.feed(html)
    base = urljoin(page_url, parser.base) if parser.base else page_url
    out: list[str] = []
    for href in parser.hrefs:
        if href.strip().startswith("#"):            # a place on this page, not another page
            continue
        url = urldefrag(urljoin(base, href.strip())).url
        if (urlsplit(url).scheme in ("http", "https") and same_site(url, page_url)
                and kind_of(url) and _bare(url) != _bare(page_url) and url not in out):
            out.append(url)
    return out


# A sentence runs to a full stop followed by a space (or the end), so "4.0" stays whole.
_NOTICE = re.compile(
    r"(?:[^.\n]|\.(?=\S)){0,160}(?:licen[cs]ed under|creative commons|public domain|"
    r"all rights reserved|for free distribution|copyright|©)(?:[^.\n]|\.(?=\S)){0,220}\.?",
    re.IGNORECASE)


def licence_notice(text: str, html: str = "") -> str:
    """The licence or copyright statement a page carries, as one short line, or "".

    Prefers a link marked rel="license" (how Creative Commons marks its licences), then the
    first sentence that states a licence or copyright.
    """
    if html:
        parser = _Links()
        parser.feed(html)
        if parser.licence_links:
            link = parser.licence_links[0]
            sentence = _NOTICE.search(text or "")
            said = _tidy(sentence.group(0)) if sentence else ""
            return (f"{said} ({link})" if said and link not in said else said or f"Licence: {link}")[:400]
    m = _NOTICE.search(text or "")
    return _tidy(m.group(0))[:400] if m else ""


def _tidy(text: str) -> str:
    return re.sub(r"\s+([.,;:])", r"\1", " ".join(text.split()))


# ---------------------------------------------------------------- robots.txt


class Robots:
    """A site's robots.txt as it applies to this crawler."""

    def __init__(self, status: str, parser: RobotFileParser | None = None, note: str = ""):
        self.status, self.parser, self.note = status, parser, note

    def allows(self, url: str) -> bool:
        if self.status == "unreachable":
            return False
        return self.parser is None or self.parser.can_fetch(settings.crawl_user_agent, url)

    def delay(self) -> float:
        asked = self.parser.crawl_delay(settings.crawl_user_agent) if self.parser else None
        return min(max(settings.crawl_delay_s, float(asked or 0)), settings.crawl_max_delay_s)


def parse_robots(text: str) -> Robots:
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    return Robots("read", parser, "robots.txt was read and is obeyed.")


async def read_robots(page_url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> Robots:
    """The site's robots.txt (RFC 9309): absent (4xx) means no rules; unreachable or failing
    (5xx, no answer) means crawl nothing."""
    parts = urlsplit(page_url)
    url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        _, _, body = await source_material.fetch_raw(url, transport=transport,
                                                     user_agent=settings.crawl_user_agent)
    except source_material.SourceError as e:
        if e.status is not None and 400 <= e.status < 500:
            return Robots("none", None, "The site has no robots.txt, so it sets no rules for crawlers.")
        return Robots("unreachable", None, "The site's robots.txt could not be read, so nothing "
                                           f"will be crawled ({e}).")
    if body[:600].lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml")):
        return Robots("none", None, "The site has no robots.txt (that address shows a web page "
                                    "instead), so it sets no rules for crawlers.")
    return parse_robots(body.decode("utf-8", errors="replace"))


# ---------------------------------------------------------------- preview


async def _start_page(url: str) -> tuple[str, str, str]:
    """(final url, html, text) of the page a crawl starts from."""
    final, ctype, body = await source_material.fetch_raw(url, user_agent=settings.crawl_user_agent)
    if "html" not in ctype.lower() and not body[:1000].lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise source_material.SourceError("That address is not a web page with links on it. Give "
                                          "the page that lists the documents.")
    html = body.decode("utf-8", errors="replace")
    return final, html, source_material.strip_markup(html)


KINDS = ("pdf", "page", "text")


async def preview(url: str, max_pages: int | None = None, kinds: list[str] | None = None,
                  same_folder: bool = False) -> dict[str, Any]:
    """What a crawl of `url` would do, without reading any of the linked pages."""
    plan = await _plan(url, max_pages, kinds, same_folder)
    names = plan.pop("names", {})
    plan.pop("links", None)
    # Each sample link with the name it would be saved under.
    plan["sample"] = [{"url": u, "name": names.get(u, "")} for u in plan["sample"]]
    return plan


async def _plan(url: str, max_pages: int | None = None, kinds: list[str] | None = None,
                same_folder: bool = False) -> dict[str, Any]:
    url = source_material.normalize_link(url)
    await source_material.check_public_url(url)
    robots = await read_robots(url)
    if robots.status == "unreachable":
        return {"url": url, "ok": False, "robots": robots.note, "rules": rules()}
    if not robots.allows(url):
        return {"url": url, "ok": False, "rules": rules(),
                "robots": "The site's robots.txt does not allow crawlers to read this page, so it "
                          "will not be crawled."}
    final, html, text = await _start_page(url)
    links = extract_links(html, final)
    allowed = [u for u in links if robots.allows(u)]
    kinds = [k for k in kinds or [] if k in KINDS]
    chosen = narrow(allowed, final, kinds, same_folder)
    limit = max(1, min(int(max_pages or settings.crawl_max_pages), settings.crawl_max_pages))
    take = chosen[:limit]
    delay = robots.delay()

    def count(urls: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for u in urls:
            out[kind_of(u)] = out.get(kind_of(u), 0) + 1
        return out

    return {
        "url": final, "ok": bool(take), "host": urlsplit(final).hostname,
        "found": len(links), "allowed": len(allowed), "disallowed": len(links) - len(allowed),
        # What the choices would keep, so a person can see the effect before choosing.
        "allowed_kinds": count(allowed), "folder": folder_of(final),
        "in_folder": len(narrow(allowed, final, None, True)),
        "chosen": len(chosen), "kinds_chosen": kinds, "same_folder": same_folder,
        "take": len(take), "limit": limit, "kinds": count(take), "delay": delay,
        "minutes": max(1, round(len(take) * (delay + 3) / 60)),
        "robots": robots.note, "licence": licence_notice(text, html),
        "sample": take[:8], "rules": rules(), "links": take, "names": link_names(html, final),
    }


# ---------------------------------------------------------------- the crawl

_tasks: dict[str, asyncio.Task] = {}


async def start(router: LLMRouter, url: str, corpus: str, max_pages: int | None = None,
                by: Any = None, kinds: list[str] | None = None,
                same_folder: bool = False) -> dict[str, Any]:
    """Begin a crawl in the background and return its row. One crawl per website at a time."""
    corpus = " ".join(corpus.split())[:120]
    if not corpus:
        raise ValueError("Choose a library, or name a new one.")
    await check_idle(corpus)
    plan = await _plan(url, max_pages, kinds, same_folder)
    if not plan["ok"]:
        raise ValueError(plan.get("robots") if plan.get("found") is None
                         else "There are no links on that page this crawler may read.")
    busy = await fetch_one("SELECT id FROM crawls WHERE host=%s AND status IN ('running', 'stopping')",
                           (plan["host"],))
    if busy:
        raise ValueError(f"A crawl of {plan['host']} is already running. Wait for it to finish, "
                         "so the site is read by one crawl at a time.")
    links = plan["links"]
    row = await fetch_one(
        "INSERT INTO crawls (corpus, start_url, host, max_pages, rules, total, started_by) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
        (corpus, plan["url"], plan["host"], plan["limit"],
         jsonb({"delay_s": plan["delay"], "robots": plan["robots"], "user_agent": settings.crawl_user_agent,
                "licence": plan["licence"], "found": plan["found"], "disallowed": plan["disallowed"],
                "kinds": plan["kinds_chosen"] or list(KINDS), "same_folder": same_folder,
                "folder": plan["folder"], "rules": [r["title"] for r in rules()]}),
         len(links), by))
    cid = str(row["id"])
    task = asyncio.create_task(_crawl(router, cid, plan["url"], links, corpus, plan["delay"],
                                      plan["licence"], by, plan["names"]))
    _tasks[cid] = task
    task.add_done_callback(lambda _t: _tasks.pop(cid, None))
    return row


async def _note(cid: str, result: dict[str, Any], *, added: bool = False) -> str:
    """Record one link's outcome; return the crawl's status (a person may have stopped it)."""
    row = await fetch_one(
        "UPDATE crawls SET done = done + 1, added = added + %s, results = results || %s::jsonb "
        "WHERE id=%s RETURNING status", (1 if added else 0, jsonb([result]), cid))
    return row["status"] if row else "stopped"


async def _crawl(router: LLMRouter, cid: str, start_url: str, links: list[str], corpus: str,
                 delay: float, licence: str, by: Any, names: dict[str, str] | None = None) -> None:
    names = names or {}
    store = Corpus(router, name=corpus, added_by=by)
    known = {r["link"] for r in await fetch_all(
        "SELECT metadata->>'link' AS link FROM documents WHERE corpus=%s AND metadata ? 'link' "
        "AND removed_at IS NULL", (corpus,))}
    status, message, failures, added = "finished", "", 0, 0
    try:
        for i, link in enumerate(links):
            if link in known:
                state = await _note(cid, {"link": link, "status": "exists",
                                          "detail": "Already in this library."})
                if state == "stopping":
                    status, message = "stopped", "Stopped by request."
                    break
                continue
            began = asyncio.get_running_loop().time()
            if i:
                await asyncio.sleep(delay)
            result, ok = await _read_one(store, link, start_url, licence, corpus, cid, names.get(link, ""))
            # Seconds this link took, pause included: what the time-left estimate is made from.
            result["secs"] = round(asyncio.get_running_loop().time() - began, 1)
            failures = 0 if ok or result["status"] == "exists" else failures + 1
            added += result["status"] == "added"
            state = await _note(cid, result, added=result["status"] == "added")
            if result.get("halt"):
                status, message = "stopped", result["halt"]
                break
            if failures >= 3:
                status, message = "stopped", ("Three requests in a row got no answer or an error, "
                                               "so the crawl stopped.")
                break
            if state == "stopping":
                status, message = "stopped", "Stopped by request."
                break
    except Exception as e:
        log.exception("crawl %s failed", cid)
        status, message = "failed", f"{type(e).__name__}: {e}"
    version = await record_version(corpus, by) if added else None
    await fetch_one(
        "UPDATE crawls SET status=%s, message=%s, version=%s, finished_at=now() WHERE id=%s RETURNING id",
        (status, message, version["version"] if version else None, cid))


async def _read_one(store: Corpus, link: str, start_url: str, licence: str, corpus: str,
                    cid: str, listed_as: str = "") -> tuple[dict[str, Any], bool]:
    """Read one link into the library. Returns its result and whether the request worked.

    A request that gets no answer or an HTTP error is a failed request, which the stop rule
    counts. A file that downloads but cannot be read (a damaged PDF, a scanned image) is the
    file's problem, not the site's: it is marked unreadable and the crawl goes on.
    """
    try:
        final, ctype, body = await source_material.fetch_raw(link, user_agent=settings.crawl_user_agent)
    except source_material.SourceError as e:
        result = {"link": link, "status": "error", "detail": str(e)}
        if e.status in (429, 503):
            result["halt"] = (f"The site answered HTTP {e.status} (it is busy or asking crawlers "
                              "to slow down), so the crawl stopped.")
        return result, False
    try:
        doc = await asyncio.to_thread(source_material.from_bytes, body, content_type=ctype,
                                      url=final, max_words=settings.web_link_max_words)
    except source_material.SourceError as e:
        return {"link": link, "status": "unreadable",
                "detail": f"Downloaded, but no text could be read from it: {e}"}, True
    if not doc.text.strip():
        return {"link": link, "status": "unreadable", "detail": "Downloaded, but it has no text."}, True
    sha = hashlib.sha256(doc.text.encode()).hexdigest()
    holder = await fetch_one("SELECT id, title FROM documents WHERE corpus=%s AND sha256=%s "
                             "AND removed_at IS NULL", (corpus, sha))
    if holder:
        return {"link": link, "status": "exists", "document_id": holder["id"], "title": holder["title"],
                "detail": "The same text is already in this library."}, True
    html = body.decode("utf-8", errors="replace") if doc.kind == "web page" else ""
    notice = licence_notice(doc.text[:20000], html) or licence
    now = datetime.now(UTC).isoformat(timespec="seconds")
    meta = {"link": link, "site": doc.site, "published": doc.published, "words": doc.words,
            "truncated": doc.truncated, "retrieved_at": doc.retrieved_at, "checked_at": now,
            "crawled_from": start_url, "crawl_id": cid, "licence": notice, "listed_as": listed_as,
            "licence_from": "this page" if notice and notice != licence else ("the starting page" if notice else "")}
    # The name the index page gives it, else the document's own title (a web page's heading,
    # a PDF's recorded title), else one made from its file name.
    title = listed_as or doc.title
    doc_id = await store.add_document(title, doc.text, source_uri=doc.url or link,
                                      doc_type=doc.kind, metadata=meta)
    return {"link": link, "status": "added", "document_id": doc_id, "title": title,
            "detail": ""}, True


async def stop(cid: str) -> dict | None:
    return await fetch_one("UPDATE crawls SET status='stopping' WHERE id=%s AND status='running' "
                           "RETURNING id", (cid,))


def timing(row: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Elapsed time and an estimate of the time left, from the pace the crawl has kept: the
    average time of the links it actually fetched, pause included (links already in the
    library take no time and are left out of the average). Pure."""
    now = now or datetime.now(UTC)
    end = row.get("finished_at") or now
    elapsed = max(0.0, (end - row["created_at"]).total_seconds())
    timed = [r["secs"] for r in row.get("results") or [] if r.get("secs") is not None
             and r.get("status") != "exists"]
    pace = sum(timed) / len(timed) if timed else (row.get("rules") or {}).get("delay_s", settings.crawl_delay_s) + 3
    left = max(0, int(row.get("total", 0)) - int(row.get("done", 0)))
    live = row.get("status") in ("running", "stopping")
    eta = round(left * pace) if live else 0
    counts: dict[str, int] = {}
    for r in row.get("results") or []:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
    return {"elapsed_s": round(elapsed), "eta_s": eta, "pace_s": round(pace, 1),
            "estimated": bool(timed), "counts": counts}


async def get(cid: str) -> dict | None:
    """A crawl, marked interrupted if the server restarted while it ran."""
    row = await fetch_one("SELECT c.*, COALESCE(NULLIF(u.name, ''), u.email) AS started_by_name "
                          "FROM crawls c LEFT JOIN users u ON u.id = c.started_by WHERE c.id=%s", (cid,))
    if row and row["status"] in ("running", "stopping") and cid not in _tasks:
        row = await fetch_one(
            "UPDATE crawls SET status='interrupted', finished_at=now(), message=%s WHERE id=%s "
            "RETURNING *", ("The server restarted while this crawl ran; what was read is kept.", cid))
        if row["added"]:
            await record_version(row["corpus"])
    return row


async def resume(router: LLMRouter, cid: str, by: Any = None) -> dict[str, Any]:
    """Start a new crawl of the same page, into the same library, with the same choices.
    Links the earlier crawl added are already in the library and are passed over without a
    request, so it picks up where the other left off."""
    row = await fetch_one("SELECT * FROM crawls WHERE id=%s", (cid,))
    if row is None:
        raise ValueError("There is no such crawl.")
    if row["status"] in ("running", "stopping"):
        raise ValueError("That crawl is still running.")
    rules_ = row["rules"] or {}
    return await start(router, row["start_url"], row["corpus"], row["max_pages"], by,
                       rules_.get("kinds"), bool(rules_.get("same_folder")))


async def recent(limit: int = 8) -> list[dict]:
    return await fetch_all(
        "SELECT c.id, c.corpus, c.start_url, c.host, c.status, c.total, c.done, c.added, c.version, "
        "c.created_at, COALESCE(NULLIF(u.name, ''), u.email) AS started_by_name FROM crawls c "
        "LEFT JOIN users u ON u.id = c.started_by ORDER BY c.created_at DESC LIMIT %s", (limit,))
