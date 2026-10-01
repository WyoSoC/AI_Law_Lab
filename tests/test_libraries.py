"""Several libraries per experiment, and citations traced back through them. Pure: the one
search test stands in for the database."""
from __future__ import annotations

import asyncio

import pytest

from ailawlab import rag
from ailawlab.grounding import SourceLedger, cited_numbers, in_list, link_citations
from ailawlab.rag import (
    Libraries,
    LibraryPin,
    Passage,
    format_passages,
    library_names,
    run_library_names,
    wanted_versions,
)


def passage(chunk_id: int, corpus: str = "cases", version: int | None = 3, content: str = "",
            document_id: int | None = None, title: str = "Roe v. Doe", similarity: float = 0.8) -> Passage:
    return Passage(chunk_id=chunk_id, document_id=document_id or chunk_id * 10, title=title,
                   content=content or f"text {chunk_id}", page_start=4, page_end=4,
                   similarity=similarity, corpus=corpus, version=version)


# ---------------------------------------------------------------- which libraries


def test_library_names_reads_new_and_old_configs():
    assert library_names({"libraries": ["cases", " regs  x ", "cases", "", 7]}) == ["cases", "regs x"]
    assert library_names({"libraries": "cases"}) == ["cases"]
    assert library_names({"libraries": []}) == []                       # deliberately none
    assert library_names({"corpus": "water"}) == ["water"]               # before several
    assert library_names({}, "document_analysis") == ["default"]
    assert library_names({}, "roleplay") == [] and library_names({"corpus": ""}, "roleplay") == []
    # `libraries` wins over a leftover `corpus`.
    assert library_names({"libraries": ["a"], "corpus": "b"}) == ["a"]


def test_a_run_choice_replaces_the_experiments_libraries():
    config = {"libraries": ["cases", "regs"]}
    assert run_library_names("agentic_workflow", config, {}) == ["cases", "regs"]
    assert run_library_names("agentic_workflow", config, {"libraries": ["regs"]}) == ["regs"]
    assert run_library_names("roleplay", config, {"libraries": []}) == []
    # An old-style launch naming one library is not overridden by the experiment's list.
    assert run_library_names("roleplay", config, {"corpus": "water"}) == ["water"]


def test_wanted_versions():
    names = ["cases", "regs"]
    assert wanted_versions(names, {"library_versions": {"cases": "4", "other": 1, "regs": ""}}) == {"cases": 4}
    assert wanted_versions(["cases"], {"corpus_version": 2}) == {"cases": 2}     # older links
    assert wanted_versions(names, {"corpus_version": 2}) == {}                  # ambiguous: ignored
    with pytest.raises(ValueError, match="not a number"):
        wanted_versions(names, {"library_versions": {"cases": "latest"}})


# ---------------------------------------------------------------- searching several


def test_libraries_search_all_together_once_each(monkeypatch):
    calls = []

    async def fake_search(router, query, scope, ids, top_k):
        calls.append((sorted(ids), top_k))
        rows = [passage(1, "cases", None, "same text", similarity=0.9),
                passage(2, "regs", None, "same text", similarity=0.85),   # a copy: dropped
                passage(3, "regs", None, "other", similarity=0.7)]
        return [p for p in rows if p.document_id in ids]

    monkeypatch.setattr(rag, "_search", fake_search)
    libs = Libraries(None, [LibraryPin("cases", 3, [10]), LibraryPin("regs", 2, [20, 30]),
                            LibraryPin("empty", None, [])])
    assert libs.searchable() == ["cases", "regs"] and bool(libs)
    assert libs.describe() == "“cases” v3, “regs” v2, “empty” (empty)"

    found = asyncio.run(libs.search("q", top_k=2))
    assert [(p.chunk_id, p.corpus, p.version) for p in found] == [(1, "cases", 3), (3, "regs", 2)]
    assert calls[-1] == ([10, 20, 30], 4)            # over-fetched, to make up for copies
    only = asyncio.run(libs.search("q", top_k=2, library="regs"))
    assert [p.chunk_id for p in only] == [2, 3] and calls[-1] == ([20, 30], 2)
    assert asyncio.run(Libraries(None, [LibraryPin("empty", None, [])]).search("q")) == []


def test_prompts_name_each_passages_library():
    text = format_passages([passage(1), passage(2, "regs", None)], numbers=[4, 7])
    assert text.startswith("[4] Roe v. Doe, p. 4 (library “cases” v3)\ntext 1")
    assert "[7] Roe v. Doe, p. 4 (library “regs”)\ntext 2" in text
    plain = format_passages([passage(1, corpus="")])
    assert plain == "[1] Roe v. Doe, p. 4\ntext 1"


# ---------------------------------------------------------------- tracing citations


def test_cited_numbers():
    assert cited_numbers("See [2], [1, 3] and [2][4]; not [2024] or [x].") == [2, 1, 3, 4]
    assert cited_numbers("Per [S2] and [S1, S3]; [2] is not a role-play marker.", "S") == [2, 1, 3]


def test_citations_name_document_library_and_version():
    passages = [passage(1), passage(2, "regs", 2)]
    cites = link_citations("Held [2]; also [5].", in_list(passages), context="Answer")
    assert cites[0] == {"quoted_text": "[2]", "source_label": "Roe v. Doe, p. 4", "chunk_id": 2,
                        "document_id": 20, "corpus": "regs", "corpus_version": 2,
                        "similarity": 0.8, "verdict": "grounded", "context": "Answer"}
    assert cites[1]["verdict"] == "unsupported" and cites[1]["quoted_text"] == "[5]"
    turn = link_citations("Under [S1].", in_list(passages), prefix="S", context="Turn 3 · Al")
    assert turn[0]["quoted_text"] == "[S1]" and turn[0]["corpus"] == "cases"


def test_the_ledger_keeps_one_number_per_passage_across_searches():
    ledger = SourceLedger()
    assert ledger.number([passage(5), passage(6)]) == [1, 2]
    assert ledger.number([passage(7), passage(5)]) == [3, 1]          # 5 keeps its number
    assert ledger.get(3).chunk_id == 7 and ledger.get(4) is None and ledger.get(0) is None
    srcs = ledger.sources(cited=[1, 3])
    assert [(s["marker"], s["cited"], s["corpus"], s["version"]) for s in srcs] == [
        ("1", True, "cases", 3), ("2", False, "cases", 3), ("3", True, "cases", 3)]


def test_the_search_tool_numbers_across_calls_and_can_target_a_library(monkeypatch):
    from ailawlab.tools import default_registry

    class FakeLibraries(Libraries):
        async def search(self, query, top_k=None, library=None):
            if library == "regs":
                return [passage(9, "regs", 2)]
            return [passage(1), passage(9, "regs", 2)]

    libs = FakeLibraries(None, [LibraryPin("cases", 3, [10]), LibraryPin("regs", 2, [90])])
    ledger = SourceLedger()
    registry = default_registry(libs, ledger)
    [schema] = [t for t in registry.schemas() if t["function"]["name"] == "search_libraries"]
    assert schema["function"]["parameters"]["properties"]["library"]["enum"] == ["cases", "regs"]
    first, _ = asyncio.run(registry.call("search_libraries", {"query": "q"}))
    assert first.startswith("[1] Roe v. Doe, p. 4 (library “cases” v3)") and "[2] " in first
    second, _ = asyncio.run(registry.call("search_libraries", {"query": "q", "library": "regs"}))
    assert second.startswith("[2] ")                                     # same passage, same number
    bad, _ = asyncio.run(registry.call("search_libraries", {"query": "q", "library": "nope"}))
    assert bad.startswith("ERROR: there is no library")
    # No library with documents: no search tool at all, rather than one that finds nothing.
    assert default_registry(Libraries(None, []), SourceLedger()).names() == ["calculate"]


# ---------------------------------------------------------------- the run page


def test_references_gather_citations_by_document():
    from ailawlab.web.views import cited_references, references_text

    rows = [
        {"quoted_text": "[S1]", "context": "Turn 1 · Al", "verdict": "grounded", "doc_id": 10,
         "library": "cases", "library_version": 3, "source_label": "Roe v. Doe, p. 4"},
        {"quoted_text": "[S2]", "context": "Turn 2 · Bea", "verdict": "grounded", "doc_id": 20,
         "library": "regs", "library_version": 2, "source_label": "40 CFR 1"},
        {"quoted_text": "[S3]", "context": "Turn 3 · Al", "verdict": "grounded", "doc_id": 10,
         "library": "cases", "library_version": 3, "source_label": "Roe v. Doe, p. 9"},
        {"quoted_text": "[S7]", "context": "Turn 3 · Al", "verdict": "unsupported", "doc_id": None},
    ]
    docs = {10: {"id": 10, "title": "Roe v. Doe", "source_uri": "https://example.com/roe",
                 "metadata": {"link": "https://example.com/roe", "site": "Example"}}}
    refs = cited_references(rows, docs)
    first, second = refs["references"]
    assert (first["n"], first["text"], first["url"], first["library_text"]) == (
        1, "Roe v. Doe. Example.", "https://example.com/roe", "“cases” v3")
    assert [u["marker"] for u in first["uses"]] == ["[S1]", "[S3]"] and first["available"]
    assert second["text"] == "40 CFR 1" and not second["available"]      # document since deleted
    assert refs["unsupported"] == [{"marker": "[S7]", "context": "Turn 3 · Al"}]
    assert refs["by_library"] == [{"library": "“cases” v3", "documents": 1},
                                  {"library": "“regs” v2", "documents": 1}]
    text = references_text({"experiment_name": "E", "id": "abcdef1234"}, refs)
    assert "1. Roe v. Doe. Example. https://example.com/roe\n   Searched in library “cases” v3. " \
           "Cited: [S1] Turn 1 · Al; [S3] Turn 3 · Al." in text
    assert "[S7] Turn 3 · Al" in text


def test_an_answer_links_its_markers_to_documents():
    from ailawlab.web.views import run_view

    run = {"mode": "agentic_workflow", "status": "succeeded",
           "libraries": [{"name": "cases", "version": 3}, {"name": "regs", "version": None}],
           "result": {"answer": "Yes [2].", "sources": [passage(1).source("1"),
                                                        passage(2, "regs", 2).source("2", True)]}}
    v = run_view(run, "/lab")
    assert ('<a class="cite" href="/lab/sources/documents/20" title="Roe v. Doe, p. 4 — “regs” v2">[2]</a>'
            in v["summary_html"])
    assert v["rerun_versions"] == {"cases": 3} and len(v["sources"]) == 2


def test_the_pdf_report_lists_references():
    from ailawlab.web.report_pdf import run_report
    from ailawlab.web.views import run_view

    run = {"id": "b4fbcbf5-e770-4ac2-849a-556ffc50bcfa", "experiment_name": "Research",
           "mode": "document_analysis", "status": "succeeded",
           "libraries": [{"name": "cases", "version": 3}, {"name": "regs", "version": 2}],
           "result": {"answer": "Held [1]."}}
    refs = {"references": [{"n": 1, "text": "Roe v. Doe.", "url": "", "library_text": "“cases” v3",
                            "uses": [{"marker": "[1]", "context": "Answer"}]}],
            "unsupported": [{"marker": "[4]", "context": "Answer"}]}
    without = run_report(run, run_view(run))
    with_refs = run_report(run, {**run_view(run), "refs": refs})
    assert with_refs.startswith(b"%PDF") and len(with_refs) > len(without)
