"""Network tools and hidden libraries. Pure: the databases and the store are stood in for."""
from __future__ import annotations

import asyncio

from ailawlab import sources
from ailawlab.grounding import SourceLedger
from ailawlab.network_tools import OnlineReader, fetch_library_name
from ailawlab.rag import Libraries, Passage, default_hidden, library_hidden
from ailawlab.tools import default_registry


class Notes:
    def __init__(self):
        self.messages = []

    async def note(self, message, **kw):
        self.messages.append(message)


def reader(**kw) -> OnlineReader:
    return OnlineReader(None, Libraries(None, []), SourceLedger(), Notes(), "run-1",
                        "Fetched: Test", **kw)


def test_the_fetch_library_is_named_for_the_experiment_unless_chosen():
    assert fetch_library_name({}, "Termination study") == "Fetched: Termination study"
    assert fetch_library_name({"fetch_library": "  web   finds "}, "x") == "web finds"
    assert fetch_library_name({"fetch_library": ""}, "x") == "Fetched: x"


def test_network_tools_are_offered_only_when_allowed():
    r = reader()
    off = default_registry(Libraries(None, []), SourceLedger(), allow_network=False, reader=r)
    on = default_registry(Libraries(None, []), SourceLedger(), allow_network=True, reader=r)
    assert "search_online" not in off.names() and "read_online" not in off.names()
    assert {"search_online", "read_online", "search_libraries"} <= set(on.names())
    refused, _ = asyncio.run(off.call("read_online", {"source": "W1"}))
    assert refused.startswith("ERROR: tool 'read_online' needs network access")


def test_search_lists_leads_numbered_across_searches(monkeypatch):
    class Hit:
        def __init__(self, title, full=True):
            self.title, self.badge, self.date, self.snippet, self.full_text = title, "Rule", "2026-01-05", "About notice.", full

        def as_dict(self):
            return {"ref": self.title, "title": self.title}

    async def fake_search(pid, query, limit):
        if pid == "ecfr":
            raise RuntimeError("eCFR is unreachable")
        return [Hit(f"{pid} one"), Hit(f"{pid} two", full=False)]

    monkeypatch.setattr(OnlineReader, "databases", staticmethod(lambda: {"federal_register": "Federal Register",
                                                                         "ecfr": "eCFR"}))
    monkeypatch.setattr(sources, "search", fake_search)
    r = reader()
    text = asyncio.run(r.search("notice period"))
    assert "[W1] federal_register one (Federal Register, Rule, 2026-01-05)\nAbout notice." in text
    assert "[W2] federal_register two" in text and "only its snippet can be read" in text
    assert "eCFR could not be searched" in text and "do not cite them" in text
    again = asyncio.run(r.search("x", "federal_register"))
    assert again.startswith("[W3]")                           # numbering continues
    assert asyncio.run(r.search("x", "nope")).startswith("ERROR: unknown database")


def test_reading_saves_first_and_hands_back_citable_passages(monkeypatch):
    r = reader()
    r.hits = [("federal_register", {"ref": "2026-1", "title": "Notice rule"})]

    class Fed:
        async def fetch(self, ref, hit):
            return sources.FetchedDoc("Notice rule", "Thirty days' notice is required.",
                                      "https://www.federalregister.gov/d/2026-1",
                                      metadata={"provider": "federal_register"})

    saved = []

    async def fake_save(self, doc):
        saved.append(doc)
        return 77, 3, False

    async def fake_passages(self, doc_id, version, query):
        return [Passage(chunk_id=5, document_id=doc_id, title="Notice rule", content="Thirty days.",
                        page_start=None, page_end=None, similarity=0.8, corpus=self.library,
                        version=version)]

    monkeypatch.setattr(sources, "get_provider", lambda pid: Fed())
    monkeypatch.setattr(OnlineReader, "_save", fake_save)
    monkeypatch.setattr(OnlineReader, "_passages", fake_passages)
    out = asyncio.run(r.read("W1", "notice period"))
    assert saved[0]["source_uri"] == "https://www.federalregister.gov/d/2026-1"
    assert "saved it to the library “Fetched: Test” (version 3)" in out
    assert "[1] Notice rule (library “Fetched: Test” v3)\nThirty days." in out
    assert r.ledger.get(1).document_id == 77                  # citable like any passage
    assert r.fetched == [{"source": "W1", "title": "Notice rule",
                          "url": "https://www.federalregister.gov/d/2026-1",
                          "provider": "federal_register", "document_id": 77,
                          "corpus": "Fetched: Test", "version": 3}]
    assert asyncio.run(r.read("W9")).startswith("ERROR: there is no search result W9")
    assert asyncio.run(r.read("file:///etc/passwd")).startswith("ERROR: give a search result")
    r.max_reads = 1
    assert "the most it may" in asyncio.run(r.read("W1"))


def test_a_private_address_is_refused():
    out = asyncio.run(reader().read("http://127.0.0.1:5433/"))
    assert out.startswith("ERROR: could not read")


def test_libraries_named_for_testing_start_hidden():
    assert default_hidden("test") and default_hidden("test-regs") and default_hidden("Contract TEST set")
    assert not default_hidden("case_law")
    assert library_hidden("test", False) is False             # someone chose to show it
    assert library_hidden("case_law", True) is True


def test_pickers_fold_hidden_libraries_away_unless_chosen():
    from ailawlab.web.app import templates

    macro = templates.env.get_template("_library_picker.html").module.library_picker
    libs = [{"name": "case_law", "documents": 3, "hidden": False},
            {"name": "test", "documents": 1, "hidden": True},
            {"name": "test-regs", "documents": 1, "hidden": True}]
    html = str(macro("p", libs, ["test-regs"]))
    folded = html.split("<details", 1)
    assert "case_law" in folded[0] and "test-regs" in folded[0]     # chosen stays in view
    assert 'value="test"' in folded[1] and "Hidden libraries (1)" in folded[1]
    assert "<details" not in str(macro("p", libs[:1], []))
