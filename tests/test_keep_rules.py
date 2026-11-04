"""Reading versus keeping what research agents find online. No database or network: the
reader's fetch, the embedding model and the tracer are stand-ins."""
from __future__ import annotations

from datetime import UTC, datetime

from ailawlab import keep_rules
from ailawlab.grounding import SourceLedger, link_citations
from ailawlab.network_tools import OnlineReader
from ailawlab.rag import Libraries
from ailawlab.web.views import cited_references, references_text

SITES = keep_rules.DEFAULT_KEEP_SITES


# ---------------------------------------------------------------- the rights check

def test_legal_databases_may_be_kept():
    for pid in ("courtlistener", "federal_register", "ecfr", "govinfo", "sec_edgar"):
        assert keep_rules.check(pid, "https://example.com/x").allowed, pid


def test_listed_federal_sites_and_their_subdomains_may_be_kept():
    assert keep_rules.check("web", "https://www.supremecourt.gov/opinions/23pdf/x.pdf", sites=SITES).allowed
    assert keep_rules.check("web", "https://www.ca10.uscourts.gov/opinion/1", sites=SITES).allowed
    # A look-alike host is not the site, nor is a state government site.
    assert not keep_rules.check("web", "https://supremecourt.gov.example.com/x", sites=SITES).allowed
    assert not keep_rules.check("web", "https://wyoleg.gov/statutes/title35.pdf", sites=SITES).allowed


def test_a_page_marking_an_open_license_may_be_kept():
    page = b'<html><head><link rel="license" href="https://creativecommons.org/licenses/by/4.0/"></head></html>'
    got = keep_rules.check("web", "https://blog.example.org/post", page, "text/html; charset=utf-8", SITES)
    assert got.allowed and "CC BY 4.0" in got.reason
    assert got.license_url == "https://creativecommons.org/licenses/by/4.0/"
    # Attribute order and the a-tag form, as Creative Commons' own license chooser writes it.
    page = b'<p>Text is available under <a href="https://creativecommons.org/publicdomain/zero/1.0/" rel="license noopener">CC0</a></p>'
    assert "CC0 1.0" in keep_rules.check("web", "https://x.org/", page, "text/html", SITES).reason
    page = b'<script type="application/ld+json">{"license": "https://creativecommons.org/licenses/by-sa/4.0/"}</script>'
    assert keep_rules.check("web", "https://x.org/", page, "text/html", SITES).allowed


def test_unclear_rights_mean_no_copy():
    # No marking at all; a license only mentioned in the text; a link to a license that is
    # not marked as the page's own; a non-HTML file outside the listed sites.
    for body, ctype in [
        (b"<html><body>All rights reserved.</body></html>", "text/html"),
        (b"<p>Licensed under CC BY 4.0.</p>", "text/html"),
        (b'<a href="https://creativecommons.org/licenses/by/4.0/">about licenses</a>', "text/html"),
        (b'<link rel="license" href="https://example.com/terms">', "text/html"),
        (b"%PDF-1.7 ...", "application/pdf"),
    ]:
        got = keep_rules.check("web", "https://lawreview.example.edu/article", body, ctype, SITES)
        assert not got.allowed and got.reason == keep_rules.NOT_CONFIRMED, body


def test_license_names_read_like_people_write_them():
    assert keep_rules.license_name("https://creativecommons.org/licenses/by-nc-nd/4.0/") == "CC BY-NC-ND 4.0"
    assert keep_rules.license_name("https://creativecommons.org/publicdomain/mark/1.0/") == "Public Domain Mark 1.0"


# ---------------------------------------------------------------- the reader

class FakeRouter:
    """Embeds a text as counts of a few words, so passages about the query rank first."""
    WORDS = ("search", "phone", "warrant", "weather")

    async def embed(self, texts):
        return [[t.lower().count(w) + 0.01 for w in self.WORDS] for t in texts]

    async def embed_one(self, text):
        return (await self.embed([text]))[0]


class FakeTracer:
    def __init__(self):
        self.notes = []

    async def note(self, message, **kw):
        self.notes.append(message)


PAGE_TEXT = ("The weather was fine that day.\n\n" * 3
             + "Officers generally must secure a warrant before conducting a search of a cell phone.\n\n"
             + "Unrelated closing remarks about the weather.")


def reader_with(page_rights: keep_rules.KeepCheck) -> OnlineReader:
    reader = OnlineReader(FakeRouter(), Libraries(FakeRouter(), []), SourceLedger(), FakeTracer(),
                          run_id="r1", library="Fetched: test", web=True)

    async def fake_fetch(source):
        return {"title": "A law review post", "text": PAGE_TEXT,
                "source_uri": "https://lawreview.example.edu/post", "doc_type": "html",
                "metadata": {"retrieved_at": "2026-11-05T10:00:00+00:00"}, "keep": page_rights}

    reader._fetch = fake_fetch
    return reader


async def test_reading_keeps_no_copy_and_tells_the_agent_why(monkeypatch):
    reader = reader_with(keep_rules.KeepCheck(False, keep_rules.NOT_CONFIRMED))
    out = await reader.read("https://lawreview.example.edu/post", look_for="warrant to search a phone")
    assert "No copy will be kept" in out and "You may still cite" in out
    record = reader.fetched[0]
    assert record["kept"] is False and record["document_id"] is None and not record["keep_allowed"]
    # The passage about the query comes back first, numbered, and marked as not kept.
    first = reader.ledger.get(1)
    assert "warrant" in first.content and first.kept is False and first.chunk_id < 0
    assert "(read online, not kept)" in out
    # Asking to keep it is refused, and says citing is still fine.
    kept = await reader.keep("https://lawreview.example.edu/post", "a clear statement of the rule")
    assert kept.startswith("Not kept") and reader.fetched[0]["kept"] is False
    assert any("not kept" in n for n in reader.tracer.notes)


async def test_a_second_read_reuses_the_first(monkeypatch):
    reader = reader_with(keep_rules.KeepCheck(True, "a page on supremecourt.gov, a listed US federal site"))
    first = await reader.read("https://lawreview.example.edu/post", look_for="phone search")
    assert "Rights check: positive" in first and "call keep" in first
    calls = []
    reader._fetch = lambda s: calls.append(s)       # must not be fetched again
    again = await reader.read("https://lawreview.example.edu/post", look_for="phone search")
    assert "already read" in again and not calls and len(reader.fetched) == 1


async def test_keeping_needs_a_read_first():
    reader = reader_with(keep_rules.KeepCheck(True, "x"))
    assert "read it first" in await reader.keep("https://elsewhere.example.com/", "why")


async def test_a_cited_page_that_was_not_kept_is_traced_by_address_and_quotation():
    reader = reader_with(keep_rules.KeepCheck(False, keep_rules.NOT_CONFIRMED))
    await reader.read("https://lawreview.example.edu/post", look_for="warrant to search a phone")
    cites = link_citations("Police need a warrant [1].", reader.ledger.get, context="Answer")
    c = cites[0]
    assert c["verdict"] == "grounded" and c["chunk_id"] is None and c["document_id"] is None
    assert c["source_url"] == "https://lawreview.example.edu/post"
    assert len(c["fingerprint"]) == 64 and "warrant" in c["excerpt"]
    # The run keeps an uncited passage only by where it came from, a cited one as a quotation.
    sources = reader.ledger.sources([1])
    assert sources[0]["excerpt"] and all("excerpt" not in s for s in sources[1:])


def test_references_list_a_page_that_was_not_kept():
    rows = [{"quoted_text": "[1]", "context": "Answer", "verdict": "grounded", "doc_id": None,
             "source_label": "A law review post", "source_url": "https://lawreview.example.edu/post",
             "retrieved_at": datetime(2026, 11, 5, 10, tzinfo=UTC), "fingerprint": "ab" * 32,
             "excerpt": "Officers generally must secure a warrant."},
            {"quoted_text": "[1]", "context": "Findings S1", "verdict": "grounded", "doc_id": None,
             "source_label": "A law review post", "source_url": "https://lawreview.example.edu/post",
             "retrieved_at": datetime(2026, 11, 5, 10, tzinfo=UTC), "excerpt": "Officers generally must secure a warrant."}]
    refs = cited_references(rows, {})
    assert len(refs["references"]) == 1 and not refs["unsupported"]
    ref = refs["references"][0]
    assert ref["read_online"] and ref["retrieved"] == "November 5, 2026" and len(ref["uses"]) == 2
    assert refs["by_library"] == [{"library": "read online, not kept", "documents": 1}]
    text = references_text({"experiment_name": "E", "id": "abcdef12"}, refs)
    assert "Read online on November 5, 2026; no copy kept" in text
