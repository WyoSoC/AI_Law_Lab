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
