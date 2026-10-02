"""Web links for a corpus: reading what people paste. Pure: no database or network."""
from __future__ import annotations

import httpx

from ailawlab.config import settings
from ailawlab.source_material import fetch_url
from ailawlab.web_links import _for_corpus, parse_links


def test_links_are_found_however_they_are_pasted():
    text = """https://wyoleg.gov/statutes/compress/title35.pdf
    deq.wyoming.gov/water-quality/ , www.courtlistener.com/opinion/1/x/
    See also (https://example.com/a?b=1). And https://example.com/a?b=1 again."""
    assert parse_links(text) == [
        "https://wyoleg.gov/statutes/compress/title35.pdf",
        "https://deq.wyoming.gov/water-quality/",
        "https://www.courtlistener.com/opinion/1/x/",
        "https://example.com/a?b=1",
    ]


def test_text_without_links_gives_none():
    assert parse_links("no links here, just e.g. prose.") == []
    assert parse_links("") == []


async def test_corpus_links_keep_more_than_a_drafting_source(monkeypatch):
    monkeypatch.setattr(settings, "source_max_words", 5)
    body = "<html><head><title>T</title></head><body><article>" + "<p>word " * 40 + "</article></body></html>"
    page = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, text=body))
    drafted = await fetch_url("http://93.184.216.34/p", transport=page)
    kept = await fetch_url("http://93.184.216.34/p", transport=page, max_words=1000)
    assert drafted.words == 5 and drafted.truncated
    assert kept.words == 40 and not kept.truncated


def test_advice_to_paste_text_becomes_advice_to_upload():
    assert _for_corpus("It may need a login. Copy the text and paste it instead.") == \
        "It may need a login. Save the text as a file and upload it instead."
    assert _for_corpus("Check the link, or paste the text instead.").endswith("upload it instead.")
    assert _for_corpus("copy the text and paste\nit instead").endswith("upload it instead")


def test_checking_a_whole_library_records_one_version(monkeypatch):
    """2026-10-02: re-reading 45 PDFs recorded 44 versions, because the per-document flag
    was shadowed by a helper of the same name. Now refresh_corpus records exactly one."""
    import asyncio

    from ailawlab import web_links

    recorded, refreshed = [], []

    async def fake_fetch_all(sql, params=()):
        return [{"id": 1}, {"id": 2}, {"id": 3}]

    async def fake_refresh(router, document_id, by=None, new_version=True):
        refreshed.append(new_version)
        return {"status": "updated", "document_id": document_id}

    async def fake_record(name, by=None, note=""):
        recorded.append(name)
        return {"version": 7}

    monkeypatch.setattr(web_links, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(web_links, "refresh", fake_refresh)
    monkeypatch.setattr(web_links, "record_version", fake_record)
    out = asyncio.run(web_links.refresh_corpus(None, "lib"))
    assert refreshed == [False, False, False] and recorded == ["lib"] and out["version"] == 7


def test_the_one_version_flag_is_not_shadowed():
    import inspect

    from ailawlab import web_links

    src = inspect.getsource(web_links.refresh)
    assert "if new_version else None" in src and "def record(" in src   # both present, distinct names
