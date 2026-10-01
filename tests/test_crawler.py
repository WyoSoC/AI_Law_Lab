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


def _run_crawl(monkeypatch, outcomes):
    """Run the crawl loop over fake links whose reads give `outcomes`; return how it ended."""
    calls, ended = iter(outcomes), {}

    async def read_one(*a, **k):
        status = next(calls)
        return {"link": "x", "status": status}, status != "error"

    async def nothing(*a, **k):
        return []

    async def note(cid, result, added=False):
        return "running"

    async def finish(sql, params=()):
        if "UPDATE crawls SET status" in sql:
            ended.update(status=params[0], message=params[1])
        return {"id": 1}

    async def no_sleep(*a):
        return None

    async def version(*a, **k):
        return {"version": 1}

    monkeypatch.setattr(crawler, "_read_one", read_one)
    monkeypatch.setattr(crawler, "fetch_all", nothing)
    monkeypatch.setattr(crawler, "_note", note)
    monkeypatch.setattr(crawler, "fetch_one", finish)
    monkeypatch.setattr(crawler, "record_version", version)
    monkeypatch.setattr(crawler.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(crawler, "Corpus", lambda *a, **k: None)
    links = [f"https://e.org/{i}.pdf" for i in range(len(outcomes))]
    asyncio.run(crawler._crawl(None, "c1", "https://e.org/", links, "lib", 0, "", None))
    return ended


def test_unreadable_files_do_not_stop_a_crawl_but_failed_requests_do(monkeypatch):
    # The 2026-10-01 crawl of 90 PDFs stopped after three damaged PDFs in a row.
    assert _run_crawl(monkeypatch, ["unreadable"] * 6 + ["added"])["status"] == "finished"
    ended = _run_crawl(monkeypatch, ["added", "error", "error", "error", "added"])
    assert ended["status"] == "stopped" and "no answer or an error" in ended["message"]
    assert _run_crawl(monkeypatch, ["error", "error", "unreadable", "error", "added"])["status"] == "finished"


def test_a_downloaded_file_that_cannot_be_read_is_marked_unreadable(monkeypatch):
    async def fine(*a, **k):
        return "https://e.org/a.pdf", "application/pdf", b"%PDF-1.5 broken"
    monkeypatch.setattr(source_material, "fetch_raw", fine)
    result, ok = asyncio.run(crawler._read_one(None, "https://e.org/a.pdf", "https://e.org/", "", "lib", "c1"))
    assert ok and result["status"] == "unreadable" and result["detail"].startswith("Downloaded, but")


def test_time_left_comes_from_the_pace_kept():
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 10, 1, 20, 10, tzinfo=UTC)
    row = {"status": "running", "created_at": now - timedelta(seconds=60), "total": 90, "done": 12,
           "rules": {"delay_s": 2.0},
           "results": [{"status": "exists"}] * 4 + [{"status": "added", "secs": 6.0}] * 4
                      + [{"status": "unreadable", "secs": 4.0}] * 4}
    t = crawler.timing(row, now)
    assert t["elapsed_s"] == 60 and t["pace_s"] == 5.0 and t["eta_s"] == 78 * 5
    assert t["counts"] == {"exists": 4, "added": 4, "unreadable": 4} and t["estimated"]
    first = crawler.timing({**row, "results": []}, now)
    assert not first["estimated"] and first["eta_s"] == 78 * 5      # delay + 3 s until measured
    done = crawler.timing({**row, "status": "finished", "finished_at": now}, now)
    assert done["eta_s"] == 0
