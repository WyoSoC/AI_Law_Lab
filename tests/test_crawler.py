"""The crawler's rules, each checked where the code enforces it. Pure: the network is stood in for."""
from __future__ import annotations

import asyncio

import httpx

from ailawlab import crawler, source_material
from ailawlab.config import settings
from ailawlab.crawler import (
    extract_links,
    kind_of,
    licence_notice,
    parse_robots,
    read_robots,
    same_site,
)

PAGE = """<html><head><base href="/lib/"></head><body>
<a href="authors/a/one.pdf">One</a> <a href='two.html#part'>Two</a>
<a href="https://www.example.org/lib/three">Three</a> <a href="two.html">Two again</a>
<a href="https://other.org/x.pdf">elsewhere</a> <a href="pic.jpg">image</a> <a href="all.zip">zip</a>
<a href="mailto:a@b.c">mail</a> <a href="javascript:void(0)">js</a> <a href="#top">top</a>
<a href="/lib/list.html">this page</a> <a href="/lib/list">this page again</a>
<a rel="license" href="https://creativecommons.org/licenses/by/4.0/">CC BY</a>
</body></html>"""


def test_only_same_site_documents_are_followed_one_each():
    links = extract_links(PAGE, "https://example.org/lib/list.html")
    assert links == ["https://example.org/lib/authors/a/one.pdf", "https://example.org/lib/two.html",
                     "https://www.example.org/lib/three"]
    assert same_site("https://www.example.org/a", "https://example.org/b")
    assert not same_site("https://example.org.evil.com/", "https://example.org/")
    assert [kind_of(u) for u in ("https://x/a.pdf", "https://x/a", "https://x/a.txt", "https://x/a.png")] == \
        ["pdf", "page", "text", ""]


def test_choices_narrow_to_kinds_of_file_and_the_page_folder():
    from ailawlab.crawler import folder_of, narrow

    links = ["https://e.org/lib/a.pdf", "https://e.org/lib/b.html", "https://e.org/glossary",
             "https://e.org/x.pdf"]
    assert folder_of("https://e.org/lib/list-pdf.html") == "/lib/" and folder_of("https://e.org/lib/") == "/lib/"
    assert narrow(links, "https://e.org/lib/list.html", ["pdf"]) == ["https://e.org/lib/a.pdf", "https://e.org/x.pdf"]
    assert narrow(links, "https://e.org/lib/list.html", None, True) == links[:2]
    assert narrow(links, "https://e.org/lib/list.html", ["pdf"], True) == links[:1]


def test_a_web_page_at_the_robots_address_is_not_rules():
    soft404 = asyncio.run(read_robots("https://example.org/", transport=_transport(200, "<!DOCTYPE html><html>Lost")))
    assert soft404.status == "none" and "shows a web page" in soft404.note


def test_the_licence_notice_is_found_and_kept_short():
    text = ("Home. The text of this page is licensed under a Creative Commons Attribution 4.0 "
            "International License. Other words.")
    assert licence_notice(text, PAGE) == ("The text of this page is licensed under a Creative Commons "
                                          "Attribution 4.0 International License. "
                                          "(https://creativecommons.org/licenses/by/4.0/)")
    assert licence_notice("Copyright © 2001 Buddhist Publication Society. For free distribution only.") \
        == "Copyright © 2001 Buddhist Publication Society."
    assert licence_notice("Nothing about rights here.") == ""


def test_robots_rules_and_crawl_delay_are_obeyed():
    r = parse_robots("User-agent: *\nDisallow: /private/\nCrawl-delay: 7\n")
    assert r.allows("https://example.org/lib/a.pdf") and not r.allows("https://example.org/private/x")
    assert r.delay() == 7
    assert parse_robots("User-agent: *\nCrawl-delay: 1\n").delay() == settings.crawl_delay_s   # never faster
    assert parse_robots("User-agent: *\nCrawl-delay: 999\n").delay() == settings.crawl_max_delay_s
    blocked = parse_robots("User-agent: AILawLab-crawler\nDisallow: /\n")
    assert not blocked.allows("https://example.org/anything")


def _transport(status: int, body: str = ""):
    return httpx.MockTransport(lambda req: httpx.Response(status, text=body))


def test_no_robots_means_no_rules_but_a_failing_site_means_no_crawl():
    none = asyncio.run(read_robots("https://example.org/lib/list.html", transport=_transport(404)))
    assert none.status == "none" and none.allows("https://example.org/x")
    down = asyncio.run(read_robots("https://example.org/lib/list.html", transport=_transport(503)))
    assert down.status == "unreachable" and not down.allows("https://example.org/x")
    rules = asyncio.run(read_robots("https://example.org/", transport=_transport(200, "User-agent: *\nDisallow: /")))
    assert not rules.allows("https://example.org/lib/a.pdf")


def test_the_crawl_stops_when_the_site_says_it_is_busy(monkeypatch):
    for status in (429, 503):
        async def busy(*a, status=status, **k):
            raise source_material.SourceError("busy", status=status)
        monkeypatch.setattr(source_material, "fetch_raw", busy)
        result, ok = asyncio.run(crawler._read_one(None, "https://example.org/a.pdf",
                                                   "https://example.org/", "", "lib", "c1"))
        assert not ok and f"HTTP {status}" in result["halt"]

    async def gone(*a, **k):
        raise source_material.SourceError("not found", status=404)
    monkeypatch.setattr(source_material, "fetch_raw", gone)
    result, ok = asyncio.run(crawler._read_one(None, "https://example.org/a.pdf",
                                               "https://example.org/", "", "lib", "c1"))
    assert not ok and "halt" not in result                     # one missing page is not a reason to stop


def test_the_rules_shown_are_the_rules_in_force():
    text = " ".join(r["text"] for r in crawler.rules())
    assert f"at least {settings.crawl_delay_s:g} seconds apart" in text
    assert f"at most {settings.crawl_max_pages} pages" in text and settings.crawl_user_agent in text
    assert "AILawLab-crawler" in settings.crawl_user_agent and "+https://" in settings.crawl_user_agent
