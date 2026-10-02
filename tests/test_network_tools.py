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


def test_search_result_markers_become_the_passages_read_from_them():
    from ailawlab.network_tools import resolve_result_markers

    read = {2: [5, 6], 4: [9]}
    text, unread = resolve_result_markers(
        "The best-interests standard governs [W2]. Abandonment is analyzed too [W2, W4]. "
        "A third case agrees [W7]. Ordinary [3] stays.", lambda n: read.get(n, []))
    assert text == ("The best-interests standard governs [5]. Abandonment is analyzed too [5, 9]. "
                    "A third case agrees [W7]. Ordinary [3] stays.")
    assert unread == [7]                                     # cited, never read: unsupported


def test_several_results_are_read_in_one_call(monkeypatch):
    calls = []

    async def fake_one(self, ref, look_for=""):
        calls.append(ref)
        return f"read {ref}"

    monkeypatch.setattr(OnlineReader, "_read_one", fake_one)
    r = reader()
    assert asyncio.run(r.read("W1, W3 W4")) == "read W1\n\nread W3\n\nread W4"
    assert calls == ["W1", "W3", "W4"]
    out = asyncio.run(r.read(", ".join(f"W{i}" for i in range(1, 8))))
    assert out.count("read W") == OnlineReader.PER_CALL and "Only the first 5" in out
    calls.clear()
    asyncio.run(r.read("https://example.org/a, b"))           # one address, not split
    assert calls == ["https://example.org/a, b"]


def test_the_agent_is_asked_for_several_sources_and_never_a_w_number():
    from ailawlab.tools import online_tools

    [_, read_tool] = online_tools(reader())
    assert "W1, W3, W4" in read_tool.description and "never cite a W number" in read_tool.description


def test_the_number_of_sources_is_chosen_and_bounded():
    from ailawlab.graphs.agentic_workflow import _sources_goal
    from ailawlab.network_tools import sources_wanted

    assert sources_wanted({}) == 5 and sources_wanted({"network_sources": "12"}) == 12
    assert sources_wanted({"network_sources": 0}) == 5           # unset reads as the default
    assert sources_wanted({"network_sources": 99}) == 20 and sources_wanted({"network_sources": "x"}) == 5
    r = OnlineReader(None, Libraries(None, []), SourceLedger(), Notes(), "run-1", "lib", wanted=12)
    assert r.wanted == 12 and r.max_reads == 14                  # room for two unreadable ones
    assert reader().max_reads == 8                                # no number set: the old budget
    assert "read the 12 most relevant results" in _sources_goal(12)
    assert "single most relevant result" in _sources_goal(1)


def test_the_experiment_page_shows_the_number_of_sources():
    from ailawlab.web.views import experiment_view

    v = experiment_view({"mode": "agentic_workflow", "name": "Study",
                         "config": {"allow_network": True, "network_sources": 7}}, [])
    assert {"label": "Sources to read online", "value": "7",
            "note": "the most relevant results, read before answering"} in v["facts"]


def test_an_answer_with_too_few_sources_is_sent_back_for_more():
    from langgraph.graph import END

    from ailawlab.graphs.agentic_workflow import more_sources_note, should_continue

    r = reader(wanted=8)
    r.hits = [("ecfr", {}), ("ecfr", {}), ("ecfr", {}), ("ecfr", {})]
    r.read_hits = {1: [1], 2: [2]}
    r.fetched = [{}, {}]
    note = more_sources_note(r)
    assert "read 2 of the 8 sources" in note and "W3, W4" in note and "search again" in note
    held = {"iterations": 3, "max_iterations": 12, "scratchpad": [
        {"role": "assistant", "content": "early answer"}, {"role": "user", "content": note}]}
    assert should_continue(held) == "reason"
    assert should_continue({**held, "scratchpad": [{"role": "assistant", "content": "final"}]}) == END
    assert should_continue({**held, "iterations": 12}) == END                # never past the limit


def test_the_last_step_has_no_tools_and_always_answers():
    import asyncio
    from types import SimpleNamespace

    from ailawlab.graphs.agentic_workflow import FINAL_STEP, reason_node
    from ailawlab.grounding import SourceLedger
    from ailawlab.rag import Libraries
    from ailawlab.router import LLMResult

    offered: list = []
    notes: list[str] = []

    class Router:
        async def chat(self, messages, tools=None, **kw):
            offered.append(tools)
            calls = [{"function": {"name": "search_libraries", "arguments": {"query": "q"}}}]
            return LLMResult(text="" if tools else "The answer.", thinking=None, host="h",
                             queue_wait_ms=0, eval_ms=1, prompt_tokens=120_000 if tools else 10,
                             output_tokens=1, tool_calls=calls if tools else [])

    async def note(message, **kw):
        notes.append(message)
        return 0

    async def nothing(*a, **k):
        return 0

    tracer = SimpleNamespace(note=note, error=nothing, llm_call=nothing, record_citations=nothing)
    registry = SimpleNamespace(schemas=lambda: [{"name": "search_libraries"}])
    ctx = SimpleNamespace(router=Router(), tracer=tracer, libraries=Libraries(None, []),
                          opt=lambda k, d=None: {"registry": registry, "ledger": SourceLedger()}.get(k, d))
    config = {"configurable": {"ctx": ctx}}
    state = {"task": "t", "scratchpad": [], "iterations": 0, "max_iterations": 400}

    # An ordinary step offers tools; a prompt past the share of the context asks it to wrap up.
    out = asyncio.run(reason_node(state, config))
    assert offered[-1] and not out["done"] and out["wrap_up"]
    # The next step offers none and is told to answer, so the run ends with an answer.
    out = asyncio.run(reason_node({**state, **out, "scratchpad": []}, config))
    assert offered[-1] is None and out["done"] and out["answer"] == "The answer."
    assert any("context is nearly full" in n for n in notes)
    # So does the step the limit falls on.
    out = asyncio.run(reason_node({**state, "iterations": 399}, config))
    assert offered[-1] is None and out["answer"] == "The answer." and "last step" in FINAL_STEP


def test_given_pages_are_clean_addresses_each_once_and_capped():
    from ailawlab.network_tools import given_pages

    text = "https://a.gov/x\n  https://a.gov/x \nnot a link\nftp://b.org/f\nhttp://c.org/\n"
    assert given_pages({"web_pages": text}) == ["https://a.gov/x", "http://c.org/"]
    assert given_pages({"web_pages": [f"https://a.gov/{i}" for i in range(30)]})[-1] == "https://a.gov/9"
    assert given_pages({}) == [] and given_pages({"web_pages": None}) == []


def test_pages_given_are_read_first_and_never_count_against_the_agent():
    import asyncio

    r = reader(wanted=2)
    r.online = False

    async def fake_read(source, look_for="", budget=True):
        if "broken" in source:
            return f"ERROR: could not read {source}"
        r.fetched.append({"source": source})
        return f"Read “{source}”"

    r._read_one = fake_read
    text = asyncio.run(r.read_given(["https://a.gov/x", "https://broken.example/", "https://a.gov/y"], "q"))
    assert "ERROR: could not read https://broken.example/" in text and text.count("Read “") == 2
    assert r.given_read == 2 and r.agent_reads == 0
    assert all(f.get("given") for f in r.fetched)
