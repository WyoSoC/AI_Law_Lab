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


def test_outside_tools_are_offered_as_the_run_allows():
    def names(r):
        return set(default_registry(Libraries(None, []), SourceLedger(), reader=r).names())

    assert names(None) == {"calculate"}
    assert names(reader(databases=True, web=False)) >= {"search_databases", "read", "search_libraries"}
    assert "search_web" not in names(reader(databases=True, web=False))
    assert names(reader(databases=False, web=True)) >= {"search_web", "read"}
    assert "search_databases" not in names(reader(databases=False, web=True))
    # Reading is held to the permissions too, whichever way a source is named.
    db_only = reader(databases=True, web=False)
    assert "not allowed" in asyncio.run(db_only.read("https://example.org/page"))
    assert "not allowed" in asyncio.run(db_only.search_web("anything"))
    web_only = reader(databases=False, web=True)
    web_only.hits = [("federal_register", {"ref": "2026-1", "title": "Rule"})]
    assert "not allowed" in asyncio.run(web_only.read("W1"))
    assert "not allowed" in asyncio.run(web_only.search("anything"))


def test_permissions_read_the_two_boxes_and_the_old_single_one():
    from ailawlab.network_tools import permissions

    assert permissions({}) == (False, False)
    assert permissions({"use_databases": True, "use_web": True}) == (True, True)
    assert permissions({"allow_network": True}) == (True, False)        # before the boxes split
    assert permissions({"allow_network": True, "use_databases": False}) == (False, False)


def test_web_addresses_are_found_in_a_brief_and_cleaned():
    from ailawlab.network_tools import clean_urls, urls_in

    brief = ("Compare https://www.dol.gov/fact-sheets/21. and (see https://a.org/x?y=1), "
             "plus https://www.dol.gov/fact-sheets/21 again.")
    assert urls_in(brief) == ["https://www.dol.gov/fact-sheets/21", "https://a.org/x?y=1"]
    assert clean_urls("https://a.gov/x\n https://a.gov/x \nftp://b.org\nhttp://c.org/") == \
        ["https://a.gov/x", "http://c.org/"]


def test_web_search_results_are_numbered_leads_with_addresses(monkeypatch):
    from ailawlab import network_tools

    async def fake_brave(query, count=None):
        return [{"title": "Fact Sheet #21", "url": "https://www.dol.gov/fs21", "snippet": "Records.",
                 "age": "2025", "site": "U.S. Department of Labor"}]

    monkeypatch.setattr(network_tools, "brave_search", fake_brave)
    r = reader(databases=False, web=True)
    out = asyncio.run(r.search_web("FLSA records"))
    assert out.startswith("[W1] Fact Sheet #21 (U.S. Department of Labor, 2025)\nhttps://www.dol.gov/fs21\nRecords.")
    assert r.searches == [{"kind": "web", "query": "FLSA records", "where": "the open web",
                           "results": [{"n": 1, "title": "Fact Sheet #21", "url": "https://www.dol.gov/fs21"}]}]
    assert network_tools.strip_tags("<strong>FLSA</strong> &amp; records") == "FLSA & records"


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
    # Read again (by another agent, say), the same document is not fetched or saved twice.
    saved.clear()
    again = asyncio.run(r.read("W1", "cure period"))
    assert not saved and "was already read in this run" in again and len(r.fetched) == 1


def test_a_private_address_is_refused():
    out = asyncio.run(reader(web=True).read("http://127.0.0.1:5433/"))
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
    out = asyncio.run(r.read(", ".join(f"W{i}" for i in range(1, OnlineReader.PER_CALL + 3))))
    assert out.count("read W") == OnlineReader.PER_CALL and f"Only the first {OnlineReader.PER_CALL}" in out
    calls.clear()
    asyncio.run(r.read("https://example.org/a, b"))           # one address, not split
    assert calls == ["https://example.org/a, b"]


def test_the_agent_is_asked_for_several_sources_and_never_a_w_number():
    from ailawlab.tools import online_tools

    [_, read_tool] = online_tools(reader())
    assert "W1, W3, W4" in read_tool.description and "never cite a W number" in read_tool.description


def test_the_experiment_page_shows_what_a_study_may_use():
    from ailawlab.web.views import experiment_view

    v = experiment_view({"mode": "agentic_workflow", "name": "Study",
                         "config": {"use_databases": True, "use_web": False, "time_limit_minutes": 45}}, [])
    facts = {f["label"]: f["value"] for f in v["facts"]}
    assert facts == {"Libraries": "the planner chooses", "Legal databases": "allowed",
                     "Open web": "not allowed", "Time limit": "45 minutes",
                     "Saves what it reads to": "“Fetched: Study”"}
    assert v["launch_defaults"]["choose_libraries"] and v["launch_defaults"]["own_library"] == "Fetched: Study"
    old = experiment_view({"mode": "agentic_workflow", "name": "Old",
                           "config": {"libraries": ["case_law"], "allow_network": True}}, [])
    facts = {f["label"]: f["value"] for f in old["facts"]}
    assert facts["Libraries"] == "“case_law”" and facts["Legal databases"] == "allowed"


def test_older_tool_results_shrink_but_keep_their_passage_numbers():
    from ailawlab.graphs.agentic_workflow import compact

    long = ("Read “A” and saved it.\n\n[7] A v. B (library “x” v2), p. 3\n" + "passage text " * 30
            + "\n\n[8] A v. B (library “x” v2), p. 4\nmore text")
    convo = [{"role": "user", "content": "brief"},
             {"role": "assistant", "content": "note one [7]", "tool_calls": [{}]},
             {"role": "tool", "content": long},
             {"role": "assistant", "content": "note two", "tool_calls": [{}]},
             {"role": "tool", "content": "No results." + " " * 400},
             {"role": "tool", "content": "short"}]
    out = compact(convo, keep=1)
    assert out[2]["content"] == ("Read “A” and saved it. … (shortened to save room; your notes say what it "
                                 "showed; its passages, still citable by number:\n"
                                 "[7] A v. B (library “x” v2), p. 3\n[8] A v. B (library “x” v2), p. 4)")
    assert out[4]["content"].endswith("your notes say what it showed)")
    assert out[1]["content"] == "note one [7]" and out[5]["content"] == "short"
    assert compact(convo, keep=3)[2]["content"] == long and convo[2]["content"] == long   # not changed in place


def test_a_report_citing_nothing_goes_back_once_with_the_passages_seen():
    from types import SimpleNamespace

    from ailawlab.graphs.agentic_workflow import investigate
    from ailawlab.router import LLMResult

    replies = iter([
        LLMResult(text="", thinking=None, host="h", queue_wait_ms=0, eval_ms=1, prompt_tokens=10,
                  output_tokens=1, tool_calls=[{"function": {"name": "search_libraries", "arguments": {"query": "q"}}}]),
        LLMResult(text="Let me search for circuit cases next.", thinking=None, host="h",
                  queue_wait_ms=0, eval_ms=1, prompt_tokens=10, output_tokens=1),
        LLMResult(text="The test has two parts [3].", thinking=None, host="h",
                  queue_wait_ms=0, eval_ms=1, prompt_tokens=10, output_tokens=1)])
    told: list[str] = []

    class Router:
        async def chat(self, messages, **kw):
            told.append(messages[-1]["content"])
            return next(replies)

    async def nothing(*a, **k):
        return 0

    async def search(name, args):
        return "[3] Smith v. Jones (library “cases” v1)\nTwo parts.", 1

    registry = SimpleNamespace(schemas=lambda: [{}], names=lambda: ["search_libraries"], call=search)
    opts = {"registry": registry, "should_stop": lambda: "", "reader": None}
    ctx = SimpleNamespace(router=Router(), libraries=Libraries(None, []), opt=lambda k, d=None: opts.get(k, d),
                          tracer=SimpleNamespace(note=nothing, error=nothing, llm_call=nothing, tool_call=nothing))
    sub = {"id": "Q1", "question": "q", "look_in": []}
    found = asyncio.run(investigate(ctx, {"plan": {"question": "s", "sub_questions": [sub]}}, sub))
    assert "If you are still researching, continue" in told[-1] and "[3] Smith v. Jones" in told[-1]
    assert found["text"] == "The test has two parts [3]." and found["steps"] == 3


def test_a_research_agent_reports_findings_on_its_last_step_and_when_stopped():
    from types import SimpleNamespace

    from ailawlab.graphs.agentic_workflow import FINAL_STEP, investigate
    from ailawlab.router import LLMResult

    offered: list = []
    stop = {"why": ""}

    class Router:
        async def chat(self, messages, tools=None, **kw):
            offered.append((tools, messages[-1]["content"]))
            calls = [{"function": {"name": "calculate", "arguments": {"expression": "1+1"}}}]
            return LLMResult(text="" if tools else "Findings [1].", thinking=None, host="h",
                             queue_wait_ms=0, eval_ms=1, prompt_tokens=100, output_tokens=1,
                             tool_calls=calls if tools else [])

    async def nothing(*a, **k):
        return 0

    calls_made = []

    class Registry:
        def schemas(self):
            return [{"name": "calculate"}]

        def names(self):
            return ["calculate"]

        async def call(self, name, args):
            calls_made.append(name)
            if len(calls_made) == 3:
                stop["why"] = "time limit"
            return "2", 1

    tracer = SimpleNamespace(note=nothing, error=nothing, llm_call=nothing, tool_call=nothing)
    opts = {"registry": Registry(), "should_stop": lambda: stop["why"], "reader": None}
    ctx = SimpleNamespace(router=Router(), tracer=tracer, libraries=Libraries(None, []),
                          opt=lambda k, d=None: opts.get(k, d))
    sub = {"id": "Q1", "question": "What is 1+1?", "look_in": []}
    found = asyncio.run(investigate(ctx, {"plan": {"question": "Sums", "sub_questions": [sub]}}, sub))
    # Three steps with tools; then the clock ran out, so the next step had none and reported.
    assert [t is not None for t, _ in offered] == [True, True, True, False]
    assert offered[-1][1] == FINAL_STEP
    assert found == {"id": "Q1", "question": "What is 1+1?", "text": "Findings [1].",
                     "steps": 4, "ended": "time limit"}


def test_an_empty_last_report_is_asked_for_again_without_thinking():
    from types import SimpleNamespace

    from ailawlab.graphs.agentic_workflow import investigate
    from ailawlab.router import LLMResult

    thought: list = []

    class Router:
        async def chat(self, messages, tools=None, think=None, **kw):
            thought.append(think)
            return LLMResult(text="" if think else "Findings [2].", thinking=None, host="h",
                             queue_wait_ms=0, eval_ms=1, prompt_tokens=10, output_tokens=1)

    async def nothing(*a, **k):
        return 0

    registry = SimpleNamespace(schemas=list, names=list)
    opts = {"registry": registry, "should_stop": lambda: "stopped", "reader": None}
    ctx = SimpleNamespace(router=Router(), tracer=SimpleNamespace(note=nothing, error=nothing, llm_call=nothing),
                          libraries=Libraries(None, []), opt=lambda k, d=None: opts.get(k, d))
    sub = {"id": "Q2", "question": "q", "look_in": []}
    found = asyncio.run(investigate(ctx, {"plan": {"question": "s", "sub_questions": [sub]}}, sub))
    assert thought == [True, False] and found["text"] == "Findings [2]." and found["ended"] == "stopped"


def test_the_clock_says_when_and_why_research_must_end(monkeypatch):
    from ailawlab.graphs import agentic_workflow

    now = {"t": 1000.0}
    monkeypatch.setattr(agentic_workflow.time, "monotonic", lambda: now["t"])
    pressed = {"stop": False}
    check = agentic_workflow.deadline_check(1000.0, 10, lambda: pressed["stop"])
    assert check() == ""
    now["t"] += 10 * 60
    assert check() == "time limit"
    pressed["stop"] = True
    assert check() == "stopped"


def test_pages_named_in_the_brief_are_read_first_and_marked():
    r = reader(web=True)

    async def fake_read(source, look_for=""):
        if "broken" in source:
            return f"ERROR: could not read {source}"
        r.fetched.append({"source": source})
        return f"Read “{source}”"

    r._read_one = fake_read
    text = asyncio.run(r.read_given(["https://a.gov/x", "https://broken.example/", "https://a.gov/y"], "q"))
    assert "ERROR: could not read https://broken.example/" in text and text.count("Read “") == 2
    assert all(f.get("given") for f in r.fetched) and len(r.fetched) == 2
