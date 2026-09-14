"""Source material for AI-drafted casts: a web page, an uploaded document, or pasted text.

A lawyer building a role-play often starts from something real -- a news story about a
case, an opinion, a complaint. This module turns such a source into plain text the cast
assistant can read. Two things make that harder than it looks:

* Fetching a user-supplied link from this server is a server-side request forgery risk.
  datahive can reach the Spark cluster over Tailscale, Postgres on localhost, and the
  university network. Only http(s) links whose host resolves exclusively to public
  addresses are fetched, every redirect is re-checked the same way, and responses are
  capped in size and time. A DNS answer could in principle change between the check and
  the connection; closing that fully would mean pinning the resolved address, which this
  feature's exposure (signed-in lab users on a university server) does not justify.
* A news page is mostly not the article. Menus, share buttons, related stories and
  footers can outweigh the story, and a model fed all of it may build a cast around a
  sidebar. extract_html() takes the smallest element holding most of the page's paragraph
  text, then drops navigation and boilerplate inside it. No HTML library is required.
  The first version skipped any element whose class looked like boilerplate, and a news
  site's <body class="... has-sidebar ..."> hid the entire article; boilerplate is now
  only removed below the chosen container.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from .config import settings
from .sources import pdf_to_text, strip_markup

MAX_REDIRECTS = 5


class SourceError(ValueError):
    """A source that cannot be read, explained in words a lawyer can act on."""


@dataclass
class SourceDoc:
    title: str
    text: str
    kind: str                       # "web page", "PDF", "text file", "pasted text"
    url: str = ""
    site: str = ""
    published: str = ""
    words: int = 0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def meta(self) -> dict[str, Any]:
        """What an experiment records about its source: enough to trace it, not the text."""
        return {"title": self.title, "kind": self.kind, "url": self.url, "site": self.site,
                "published": self.published, "words": self.words, "truncated": self.truncated,
                "retrieved_at": self.retrieved_at,
                "sha256": hashlib.sha256(self.text.encode()).hexdigest()}

    def as_dict(self) -> dict[str, Any]:
        return {**self.meta(), "text": self.text, "warnings": self.warnings}


# ---------------------------------------------------------------- article extraction

_HARD_SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head"})
_SOFT_SKIP = frozenset({"nav", "header", "footer", "aside", "form", "iframe", "button",
                        "select", "figure", "figcaption", "dialog", "menu", "table"})
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
                   "param", "source", "track", "wbr"})
_BLOCKS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dd",
                     "dt", "td", "th", "caption"})
_KEEP = frozenset({"p", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dd", "dt"})
# Start tags that implicitly close an open <p>, as browsers do. Without this, pages that
# leave paragraphs unclosed (common, and valid) nest every paragraph inside the first.
_CLOSES_P = frozenset({"address", "article", "aside", "blockquote", "details", "dialog", "div",
                       "dl", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
                       "h3", "h4", "h5", "h6", "header", "hr", "main", "menu", "nav", "ol", "p",
                       "pre", "section", "table", "ul"})
_SKIP_ROLES = frozenset({"navigation", "complementary", "banner", "contentinfo", "search",
                         "dialog", "alert"})
_BOILERPLATE = re.compile(
    r"(^|[\s_-])(share|sharing|social|related|comments?|newsletter|advert\w*|ads?|promo|"
    r"subscribe|sidebar|menu|breadcrumbs?|cookie|popup|modal|signup|toc|navbox|reflist|"
    r"references|mw-editsection|hatnote|infobox|metadata|byline-share)($|[\s_-])", re.IGNORECASE)
_STOP_HEADINGS = re.compile(
    r"^(related( stories| articles| posts| coverage)?|see also|references|external links|"
    r"further reading|notes|sources|citations|bibliography|share( this)?( story| article| post)?|"
    r"more stories|recommended( for you)?|you (may|might) also like|about the author|comments?|"
    r"leave a (comment|reply)|read more)$", re.IGNORECASE)


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.skippable: dict[int, bool] = {}
        self.blocks: list[tuple[str, list[int], str]] = []   # (text, element chain, block tag)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self._buf: list[str] = []
        self._n = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"].strip())
            return
        if tag in _VOID:
            if tag == "br":
                self._buf.append("\n")
            return
        if tag == "body" or tag in _CLOSES_P:
            self._close({"head"}, barrier=set())     # pages that never close <head>
        if tag in _CLOSES_P:
            self._close({"p"}, barrier={"button", "table", "td", "th", "caption"})
        if tag == "li":
            self._close({"li"}, barrier={"ul", "ol", "menu"})
        if tag in ("dd", "dt"):
            self._close({"dd", "dt"}, barrier={"dl"})
        if tag in _BLOCKS:
            self._flush()
        self._n += 1
        hint = f"{a.get('class', '')} {a.get('id', '')}"
        self.skippable[self._n] = (tag in _SOFT_SKIP or a.get("role") in _SKIP_ROLES
                                   or a.get("aria-hidden") == "true" or "hidden" in a
                                   or bool(_BOILERPLATE.search(hint)))
        self.stack.append((tag, self._n))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                if tag in _BLOCKS or tag in _CLOSES_P:
                    self._flush()
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        tags = {t for t, _ in self.stack}
        if "title" in tags and "svg" not in tags:
            self.title_parts.append(data)
        if not _HARD_SKIP & tags:
            self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _close(self, names: set[str], barrier: set[str]) -> None:
        for i in range(len(self.stack) - 1, -1, -1):
            tag = self.stack[i][0]
            if tag in barrier:
                return
            if tag in names:
                self._flush()
                del self.stack[i:]
                return

    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split())
        self._buf.clear()
        if text:
            kind = next((t for t, _ in reversed(self.stack) if t in _BLOCKS), "")
            self.blocks.append((text, [n for _, n in self.stack], kind))


@dataclass
class Page:
    title: str
    text: str
    site: str = ""
    published: str = ""


def _article_blocks(p: _PageParser) -> list[str]:
    score: dict[int, int] = {}
    depth: dict[int, int] = {}
    for text, chain, kind in p.blocks:
        if kind == "p":
            for d, n in enumerate(chain):
                score[n] = score.get(n, 0) + len(text)
                depth[n] = d
    if score:
        best = max(score.values())
        # The deepest element that still holds 80% of all paragraph text is the article.
        target = max((n for n, s in score.items() if s >= 0.8 * best), key=depth.__getitem__)
        candidates = [(t, ch[ch.index(target) + 1:], k) for t, ch, k in p.blocks if target in ch]
    else:
        candidates = [(t, ch[2:], k) for t, ch, k in p.blocks]

    out: list[str] = []
    for text, below, kind in candidates:
        if kind not in _KEEP or any(p.skippable[n] for n in below):
            continue
        if len(out) >= 2 and len(text.split()) <= 6 and _STOP_HEADINGS.match(text.rstrip(": ")):
            break                                    # "Related", "External links", ...
        out.append(text)
    return out


def extract_html(html: str) -> Page:
    """The main article text of a page, with its title, site name and publication date."""
    p = _PageParser()
    p.feed(html)
    p.close()
    m = p.meta
    title = (m.get("og:title") or m.get("twitter:title")
             or " ".join("".join(p.title_parts).split()))
    published = (m.get("article:published_time") or m.get("datepublished")
                 or m.get("date") or "")
    return Page(title=title, text="\n\n".join(_article_blocks(p)),
                site=m.get("og:site_name", ""), published=published)


# ---------------------------------------------------------------- reading sources


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").removeprefix("www.")
    except ValueError:
        return ""


def _title_from_url(url: str) -> str:
    try:
        segment = PurePosixPath(unquote(urlsplit(url).path)).stem
    except ValueError:
        return ""
    return re.sub(r"[-_]+", " ", segment).strip().capitalize()


def _charset(content_type: str) -> str:
    m = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
    return m.group(1) if m else ""


def _decode(data: bytes, content_type: str) -> str:
    try:
        return data.decode(_charset(content_type) or "utf-8", errors="replace")
    except LookupError:                              # a charset Python does not know
        return data.decode("utf-8", errors="replace")


def _first_words(text: str, limit: int) -> str:
    """The first `limit` words, keeping the original paragraph breaks."""
    for i, m in enumerate(re.finditer(r"\S+", text), start=1):
        if i == limit:
            return text[:m.end()]
    return text


def _finish(doc: SourceDoc) -> SourceDoc:
    text = doc.text.replace("\r\n", "\n").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n\s*", "\n\n", text).strip()
    total = len(text.split())
    if not total:
        raise SourceError("No readable text was found. If this is a web page that needs a "
                          "login or loads its text with JavaScript, copy the text and paste "
                          "it instead.")
    limit = settings.source_max_words
    if total > limit:
        text = _first_words(text, limit)
        doc.truncated = True
        doc.warnings.append(f"This source is long ({total:,} words), so only the first "
                            f"{limit:,} words are used.")
    doc.text = text
    doc.words = min(total, limit)
    if doc.words < 80:
        doc.warnings.append(f"Only {doc.words} words came through. If that is not the whole "
                            "source, the page may need a login or load its text with "
                            "JavaScript; copy the text and paste it instead.")
    doc.title = " ".join(doc.title.split())[:300] or "Untitled source"
    return doc


def from_bytes(data: bytes, *, content_type: str = "", filename: str = "",
               url: str = "") -> SourceDoc:
    """Read a fetched response or an uploaded file: PDF, HTML, or plain text."""
    ctype = content_type.split(";")[0].strip().lower()
    name = filename.lower()
    fallback_title = PurePosixPath(filename).stem if filename else _title_from_url(url)

    if ctype == "application/pdf" or name.endswith(".pdf") or data[:5] == b"%PDF-":
        text = pdf_to_text(data)
        if not text.strip():
            raise SourceError("No text could be read from that PDF. It may be a scanned "
                              "image; if so, copy the text and paste it instead.")
        return _finish(SourceDoc(title=fallback_title, text=text, kind="PDF", url=url,
                                 site=_host(url)))

    decoded = _decode(data, content_type)
    head = decoded[:1000].lstrip().lower()
    if (ctype in ("text/html", "application/xhtml+xml") or name.endswith((".html", ".htm"))
            or head.startswith(("<!doctype html", "<html"))):
        page = extract_html(decoded)
        doc = SourceDoc(title=page.title or fallback_title, text=page.text, kind="web page",
                        url=url, site=page.site or _host(url), published=page.published)
        if not page.text.strip():
            doc.text = strip_markup(decoded)
            doc.warnings.append("The main article could not be picked out, so all of the "
                                "page's text was used. Check the preview.")
        return _finish(doc)

    if ctype.startswith("text/") or name.endswith((".txt", ".md", ".markdown")) or not (ctype or name):
        return _finish(SourceDoc(title=fallback_title or "Text", text=decoded, kind="text file",
                                 url=url, site=_host(url)))
    raise SourceError("That kind of file cannot be read. Use a web page, a PDF, or plain text.")


def from_text(text: str, title: str = "") -> SourceDoc:
    """Text a person pasted in. Its first line stands in for a title if none is given."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return _finish(SourceDoc(title=title or first[:120] or "Pasted text", text=text,
                             kind="pasted text"))


def from_client(data: dict[str, Any]) -> SourceDoc:
    """A source the browser read earlier and sent back with a draft request.

    The text is re-cleaned and re-capped here: it came from the browser, so its size and
    shape are not trusted just because this server produced it a minute ago.
    """
    text = str(data.get("text") or "")
    if not text.strip():
        raise SourceError("The source has no text. Read the link or paste the text again.")
    doc = SourceDoc(title=str(data.get("title") or "Source")[:300], text=text,
                    kind=str(data.get("kind") or "pasted text")[:40],
                    url=str(data.get("url") or "")[:2000], site=str(data.get("site") or "")[:200],
                    published=str(data.get("published") or "")[:40])
    if data.get("retrieved_at"):
        doc.retrieved_at = str(data["retrieved_at"])[:40]
    return _finish(doc)


# ---------------------------------------------------------------- fetching links


def _normalize_link(raw: str) -> str:
    link = raw.strip()
    if link and "://" not in link and re.match(r"^[\w-]+(\.[\w-]+)+(:\d+)?(/|$)", link):
        link = f"https://{link}"                     # "oilcity.news/..." pasted without a scheme
    return link


async def check_public_url(url: str) -> None:
    """Refuse anything but an http(s) link whose host resolves only to public addresses."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as e:
        raise SourceError("That does not look like a web link.") from e
    if parts.scheme not in ("http", "https"):
        raise SourceError("Only web links that start with http:// or https:// can be read.")
    if not parts.hostname:
        raise SourceError("That link has no website address in it.")
    if parts.username or parts.password:
        raise SourceError("Links with a user name or password in them are not read.")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            parts.hostname, port or (443 if parts.scheme == "https" else 80),
            type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as e:
        raise SourceError(f"Could not find the website {parts.hostname}. Check the link.") from e
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        # is_global is False for loopback, private ranges, link-local (cloud metadata),
        # and 100.64.0.0/10 -- the range Tailscale, and so the Spark cluster, lives in.
        if not ip.is_global:
            raise SourceError("That link points to a private or internal network address, "
                              "which this server will not fetch.")


async def fetch_url(raw_url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> SourceDoc:
    """Fetch a link and read it. `transport` exists so tests can stand in for the network."""
    url = _normalize_link(raw_url)
    headers = {"User-Agent": settings.sec_user_agent,
               "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5"}
    async with httpx.AsyncClient(timeout=settings.source_timeout_s, follow_redirects=False,
                                 transport=transport, headers=headers) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await check_public_url(url)
            try:
                async with client.stream("GET", url) as r:
                    if r.is_redirect and r.headers.get("location"):
                        url = str(httpx.URL(url).join(r.headers["location"]))
                        continue
                    _raise_for_status(r.status_code)
                    body = bytearray()
                    async for chunk in r.aiter_bytes():
                        body += chunk
                        if len(body) > settings.source_max_bytes:
                            raise SourceError(
                                f"That page is too large to read (over "
                                f"{settings.source_max_bytes // 1_000_000} MB).")
                    # PDF extraction is CPU-bound; keep it off the event loop.
                    return await asyncio.to_thread(
                        from_bytes, bytes(body), content_type=r.headers.get("content-type", ""),
                        url=url)
            except httpx.RequestError as e:
                raise SourceError(f"Could not reach {_host(url) or 'that website'}: it did not "
                                  "respond. Check the link, or paste the text instead.") from e
    raise SourceError("That link redirects too many times to follow.")


def _raise_for_status(code: int) -> None:
    if code in (401, 402, 403, 451):
        raise SourceError("The website would not let this server read that page (it may need "
                          "a login or subscription, or block automated readers). Copy the "
                          "text and paste it instead.")
    if code == 404:
        raise SourceError("That page was not found. Check the link.")
    if code >= 400:
        raise SourceError(f"The website returned an error (HTTP {code}). Try again later, or "
                          "paste the text instead.")
