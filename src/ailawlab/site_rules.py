"""Whether a website lets an AI agent read a page, decided before the lab keeps it.

Every page the lab reads on its own initiative -- a page a research agent reads during a run,
a page a crawl collects -- is checked against what the site itself has published about
automated and AI use. Three signals are obeyed, and a page any of them rules out is not read
(or, if the signal only shows once the page arrives, not kept):

1. robots.txt (RFC 9309), read for the lab's own name. It is also read for the well-known AI
   agents (AI_AGENTS): a site that turns GPTBot, ClaudeBot, ChatGPT-User and the like away
   from a page has said it does not want AI systems reading it, and the lab is one. A
   robots.txt that cannot be read because the site is failing (5xx, no answer) means nothing
   is read (RFC 9309, section 2.3.1.4); a missing one (4xx) sets no rules.
2. An AI opt-out on the page: "noai" in an X-Robots-Tag header or a robots <meta> tag.
3. A text-and-data-mining reservation (W3C TDM Reservation Protocol): /.well-known/tdmrep.json
   on the site, a "tdm-reservation: 1" header, or the same in a <meta> tag.

Python's own robots.txt parser ignores the "*" and "$" wildcards that RFC 9309 defines, and
which sites use ("Disallow: /*.pdf$"), so rules are matched here: the most specific matching
rule wins, and "Allow" wins a tie.

What a person adds by hand (a link on the Legal Sources page, a source for a cast) is that
person reading the page, as they would in a browser; it is not checked here. The legal
databases are reached through the APIs their publishers offer for programmatic use, under
those APIs' terms, so robots.txt does not govern them either.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import httpx

from . import source_material
from .config import settings

# Agents whose exclusion from a page the lab treats as excluding AI systems in general.
# Assistants that read pages for a person, and the crawlers AI companies send out.
AI_AGENTS = (
    "ChatGPT-User", "Claude-User", "Perplexity-User", "MistralAI-User",
    "GPTBot", "OAI-SearchBot", "ClaudeBot", "Claude-SearchBot", "anthropic-ai",
    "PerplexityBot", "CCBot", "Google-Extended", "Applebot-Extended", "Bytespider",
    "meta-externalagent", "cohere-ai",
)


def token(user_agent: str) -> str:
    """The name robots.txt knows a crawler by: "AILawLab-agent" for "AILawLab-agent/1.0 (...)"."""
    return user_agent.split("/", 1)[0].split()[0].strip().lower() if user_agent.strip() else ""


# ---------------------------------------------------------------- robots.txt (pure)


@dataclass
class _Group:
    agents: list[str] = field(default_factory=list)
    rules: list[tuple[bool, str]] = field(default_factory=list)     # (allow, pattern)
    delay: float | None = None


def _pattern(path: str) -> re.Pattern:
    """A robots.txt path pattern as a regex: "*" is any run of characters, a final "$" ends it."""
    anchored = path.endswith("$")
    body = re.escape(path[:-1] if anchored else path).replace(r"\*", ".*")
    return re.compile(body + ("$" if anchored else ""))


def _target(url: str) -> str:
    p = urlsplit(url)
    path = unquote(p.path or "/")
    return path + (f"?{unquote(p.query)}" if p.query else "")


class RobotsTxt:
    """A parsed robots.txt."""

    def __init__(self, text: str):
        self.groups: list[_Group] = []
        group: _Group | None = None
        in_agents = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (x.strip() for x in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if not in_agents:
                    group = _Group()
                    self.groups.append(group)
                group.agents.append(value.split("/", 1)[0].strip().lower())
                in_agents = True
                continue
            in_agents = False
            if group is None:
                continue
            if key in ("allow", "disallow") and value:
                group.rules.append((key == "allow", value))
            elif key == "crawl-delay":
                try:
                    group.delay = float(value)
                except ValueError:
                    pass

    def _groups_for(self, agent: str, *, fallback: bool = True) -> list[_Group]:
        agent = agent.lower()
        named = [g for g in self.groups if agent in g.agents]
        if named or not fallback:
            return named
        return [g for g in self.groups if "*" in g.agents]

    def names(self, agent: str) -> bool:
        """Whether some group is written for this agent by name."""
        return bool(self._groups_for(agent, fallback=False))

    def allows(self, agent: str, url: str) -> bool:
        target = _target(url)
        if target == "/robots.txt":
            return True
        best: tuple[int, bool] | None = None          # (pattern length, allow)
        for g in self._groups_for(agent):
            for allow, path in g.rules:
                if _pattern(path).match(target):
                    candidate = (len(path), allow)
                    if best is None or candidate > best:   # longer wins; Allow (True) wins a tie
                        best = candidate
        return True if best is None else best[1]

    def crawl_delay(self, agent: str) -> float | None:
        delays = [g.delay for g in self._groups_for(agent) if g.delay is not None]
        return max(delays) if delays else None


# ---------------------------------------------------------------- TDMRep (pure)


def tdm_reservation(entries: list, url: str) -> str:
    """The TDM reservation /.well-known/tdmrep.json makes for this page: "" if none, else a
    sentence saying so. The first entry whose location matches decides (TDMRep, section 3.2)."""
    target = _target(url)
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        location = str(entry.get("location") or "")
        if not location or not _pattern(location if location.startswith("/") else "/" + location).match(target):
            continue
        if str(entry.get("tdm-reservation")) == "1":
            policy = entry.get("tdm-policy")
            return ("the site reserves its text-and-data-mining rights for this page "
                    "(/.well-known/tdmrep.json)" + (f"; its licensing terms are at {policy}" if policy else ""))
        return ""
    return ""


# ---------------------------------------------------------------- page signals (pure)

_META = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR = re.compile(r"""([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")


def _directives(value: str, agent: str) -> set[str]:
    """X-Robots-Tag directives that apply to everyone or to `agent` ("noai, noindex",
    "otherbot: noai" for another crawler only)."""
    out: set[str] = set()
    m = re.match(r"^\s*([\w.-]+)\s*:\s*(.*)$", value)
    if m and m.group(1).lower() not in ("unavailable_after",):
        if m.group(1).lower() not in (agent, "*"):
            return out
        value = m.group(2)
    for part in value.split(","):
        out.add(part.strip().lower())
    return out


def page_refusal(headers: httpx.Headers | dict, body: bytes, agent: str) -> str:
    """Why this page, as it arrived, may not be kept by an AI agent ("" if it may): a "noai"
    robots directive or a TDM reservation, in its headers or its <meta> tags."""
    headers = httpx.Headers(headers)
    for value in headers.get_list("x-robots-tag"):
        if "noai" in _directives(value, agent):
            return "the page says it is not for AI use (X-Robots-Tag: noai)"
    if headers.get("tdm-reservation", "").strip() == "1":
        return "the site reserves its text-and-data-mining rights for this page (tdm-reservation header)"
    ctype = headers.get("content-type", "").lower()
    if "html" not in ctype and not body[:1000].lstrip().lower().startswith((b"<!doctype", b"<html")):
        return ""
    head = body[:200_000].decode("utf-8", errors="replace")
    head = head.split("</head>", 1)[0] if "</head>" in head.lower() else head
    for tag in _META.findall(head):
        attrs = {m.group(1).lower(): (m.group(2) or m.group(3) or m.group(4) or "")
                 for m in _ATTR.finditer(tag)}
        name, content = attrs.get("name", "").lower(), attrs.get("content", "").lower()
        if name in ("robots", agent) and "noai" in {c.strip() for c in content.split(",")}:
            return "the page says it is not for AI use (robots meta tag: noai)"
        if name == "tdm-reservation" and content.strip() == "1":
            return "the site reserves its text-and-data-mining rights for this page (tdm-reservation meta tag)"
    return ""


# ---------------------------------------------------------------- a site's rules


class Robots:
    """What a site's robots.txt and tdmrep.json say about one agent reading its pages."""

    def __init__(self, status: str, parsed: RobotsTxt | None = None, note: str = "",
                 agent: str = "", tdm: list | None = None):
        self.status, self.parsed, self.note = status, parsed, note
        self.agent = agent or token(settings.crawl_user_agent)
        self.tdm = tdm or []

    def refusal(self, url: str) -> str:
        """Why `url` may not be read ("" if it may)."""
        if self.status == "unreachable":
            return "the site's robots.txt could not be read (the site did not answer properly)"
        if self.parsed is not None:
            if not self.parsed.allows(self.agent, url):
                return "the site's robots.txt does not allow this lab's agents to read it"
            refused = [a for a in AI_AGENTS if self.parsed.names(a) and not self.parsed.allows(a, url)]
            if refused:
                return ("the site's robots.txt turns AI agents away from it ("
                        + ", ".join(refused[:4]) + (", …" if len(refused) > 4 else "") + ")")
        return tdm_reservation(self.tdm, url)

    def allows(self, url: str) -> bool:
        return not self.refusal(url)

    def delay(self) -> float:
        asked = self.parsed.crawl_delay(self.agent) if self.parsed else None
        return min(max(settings.crawl_delay_s, float(asked or 0)), settings.crawl_max_delay_s)


def parse_robots(text: str, agent: str = "") -> Robots:
    return Robots("read", RobotsTxt(text), "robots.txt was read and is obeyed.", agent)


async def _tdmrep(origin: str, user_agent: str, transport) -> list:
    try:
        _, _, body = await source_material.fetch_raw(f"{origin}/.well-known/tdmrep.json",
                                                     transport=transport, user_agent=user_agent)
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (source_material.SourceError, ValueError):
        return []                      # no file (or not JSON): no reservation is made there
    return data if isinstance(data, list) else []


async def read_robots(page_url: str, *, user_agent: str | None = None,
                      transport: httpx.AsyncBaseTransport | None = None) -> Robots:
    """The site's robots.txt (RFC 9309) and TDM reservations, as they apply to `user_agent`
    (the crawler's, unless given): absent (4xx) means no rules; unreachable or failing (5xx,
    no answer) means read nothing."""
    user_agent = user_agent or settings.crawl_user_agent
    agent = token(user_agent)
    parts = urlsplit(page_url)
    origin = f"{parts.scheme}://{parts.netloc}"
    tdm = await _tdmrep(origin, user_agent, transport)
    try:
        _, _, body = await source_material.fetch_raw(f"{origin}/robots.txt", transport=transport,
                                                     user_agent=user_agent)
    except source_material.SourceError as e:
        if e.status is not None and 400 <= e.status < 500:
            return Robots("none", None, "The site has no robots.txt, so it sets no rules for crawlers.",
                          agent, tdm)
        return Robots("unreachable", None, "The site's robots.txt could not be read, so nothing "
                                           f"will be read from it ({e}).", agent, tdm)
    if body[:600].lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml")):
        return Robots("none", None, "The site has no robots.txt (that address shows a web page "
                                    "instead), so it sets no rules for crawlers.", agent, tdm)
    robots = parse_robots(body.decode("utf-8", errors="replace"), agent)
    robots.tdm = tdm
    return robots


# A site's rules are read once an hour at most, across all runs (RFC 9309 allows a day).
CACHE_S = 3600.0
_cache: dict[tuple[str, str], tuple[float, Robots]] = {}


async def for_page(url: str, user_agent: str) -> Robots:
    """The rules of the site `url` is on, for `user_agent`, from the cache when fresh. A failing
    site is asked again next time rather than remembered."""
    parts = urlsplit(url)
    key = (f"{parts.scheme}://{parts.netloc}".lower(), token(user_agent))
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_S:
        return hit[1]
    robots = await read_robots(url, user_agent=user_agent)
    if robots.status != "unreachable":
        _cache[key] = (time.monotonic(), robots)
    return robots
