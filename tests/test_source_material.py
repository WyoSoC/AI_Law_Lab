"""Reading source material for AI-drafted casts. Offline: the network is a mock transport."""
from __future__ import annotations

import httpx
import pytest

from ailawlab.config import settings
from ailawlab.source_material import (
    SourceError,
    check_public_url,
    extract_html,
    fetch_url,
    from_bytes,
    from_client,
    from_text,
)

PAGE = """<!doctype html><html><head><title>Fallback title</title>
<meta property="og:title" content="Court hears gravel pit case">
<meta property="og:site_name" content="Example News">
<meta property="article:published_time" content="2026-09-12T21:45:00+00:00">
<script>var ad = "<p>script text</p>";</script><style>p { color: red }</style>
</head>
<body class="single-post has-sidebar menu-open">
<header><nav><ul><li><a href="/">Home</a></li><li>Crime</li></ul></nav></header>
<div class="layout"><main><article><div class="entry-content">
<p>A controversial gravel mine remains in legal limbo after the state high court heard arguments.
<p>The state board denied the lease renewals in 2024, and the <em>company</em> sued.
<div class="share-buttons"><p>Share this on Facebook and share it widely with everyone you know</p></div>
<h2>What are state trust lands?</h2>
<p>Upon statehood in 1890, the federal government granted Wyoming 4.2 million acres.</p>
<p>The board manages them to fund public schools.</p>
<h3>Related</h3><ul><li>Another story about something else entirely</li></ul>
</div></article></main>
<aside class="widget-area"><p>Subscribe to our newsletter for the very latest news every single morning</p></aside>
</div>
<footer><p>Copyright Example News. All rights reserved, and a few more words to pad this.</p></footer>
</body></html>"""


def test_the_article_is_found_and_the_page_furniture_left_out():
    page = extract_html(PAGE)
    assert page.title == "Court hears gravel pit case"
    assert page.site == "Example News"
    assert page.published == "2026-09-12T21:45:00+00:00"
    # Unclosed <p> tags must not swallow the lead, and a "has-sidebar" body class must not
    # hide the article -- both happened on a real news page.
    assert page.text.startswith("A controversial gravel mine")
    for kept in ("The state board denied the lease renewals in 2024, and the company sued.",
                 "What are state trust lands?", "fund public schools"):
        assert kept in page.text
    for dropped in ("Home", "Share this", "Subscribe", "Copyright", "Related",
                    "Another story", "script text", "color: red"):
        assert dropped not in page.text


def test_a_page_without_head_or_paragraph_tags_still_yields_text():
    page = from_bytes(b"<html><body><div>Just a short note about a dispute over water rights "
                      b"between two ranches.</div></body></html>", content_type="text/html")
    assert "water rights" in page.text
    assert any("could not be picked out" in w for w in page.warnings)


def test_long_sources_are_capped_and_short_ones_flagged(monkeypatch):
    monkeypatch.setattr(settings, "source_max_words", 10)
    doc = from_text("Casper Mountain.\n\n" + "word " * 50)
    assert doc.truncated and doc.words == 10 and len(doc.text.split()) == 10
    assert doc.title == "Casper Mountain."
    assert any("only the first 10 words" in w for w in doc.warnings)
    assert any("Only 10 words came through" in w for w in doc.warnings)


def test_empty_and_unreadable_sources_explain_what_to_do():
    with pytest.raises(SourceError, match="paste"):
        from_text("   \n  ")
    with pytest.raises(SourceError, match="cannot be read"):
        from_bytes(b"\x89PNG....", content_type="image/png", filename="photo.png")


def test_text_sent_back_by_the_browser_is_recapped(monkeypatch):
    monkeypatch.setattr(settings, "source_max_words", 5)
    doc = from_client({"title": "T", "kind": "web page", "url": "https://example.com/a",
                       "text": "one two three four five six seven"})
    assert doc.text == "one two three four five" and doc.truncated
    assert doc.meta()["sha256"] and "text" not in doc.meta()


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8088/", "http://10.99.252.31/",
    "http://100.100.10.1/",                    # Tailscale's range, where the Sparks live
    "http://169.254.169.254/latest/meta-data/", "http://192.168.1.10/", "http://0.0.0.0/",
    "http://[::1]/", "http://[fd7a:115c:a1e0::1]/", "http://[::ffff:127.0.0.1]/",
])
async def test_internal_addresses_are_refused(url):
    with pytest.raises(SourceError, match="private or internal"):
        await check_public_url(url)


@pytest.mark.parametrize("url", ["ftp://example.com/a", "file:///etc/passwd", "javascript:alert(1)"])
async def test_only_web_links_are_read(url):
    with pytest.raises(SourceError, match="http"):
        await check_public_url(url)


async def test_a_public_address_is_allowed():
    await check_public_url("http://93.184.216.34/story")


async def test_redirects_are_rechecked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1:5433/"})

    with pytest.raises(SourceError, match="private or internal"):
        await fetch_url("http://93.184.216.34/story", transport=httpx.MockTransport(handler))


async def test_a_page_is_fetched_and_read():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/story"})
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=PAGE)

    doc = await fetch_url("http://93.184.216.34/old", transport=httpx.MockTransport(handler))
    assert doc.url == "http://93.184.216.34/story"
    assert (doc.title, doc.kind, doc.site) == ("Court hears gravel pit case", "web page", "Example News")
    assert "gravel mine" in doc.text


async def test_oversized_and_refused_pages_explain_themselves(monkeypatch):
    monkeypatch.setattr(settings, "source_max_bytes", 1000)
    big = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/html"},
                                                       content=b"x" * 5000))
    with pytest.raises(SourceError, match="too large"):
        await fetch_url("http://93.184.216.34/big", transport=big)
    refused = httpx.MockTransport(lambda r: httpx.Response(403))
    with pytest.raises(SourceError, match="paste it instead"):
        await fetch_url("http://93.184.216.34/paywalled", transport=refused)
