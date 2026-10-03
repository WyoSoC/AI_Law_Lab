"""Role-play case files: each agent's own libraries, disclosed by citing them. Pure: the
search test stands in for the database and the cluster."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ailawlab.agent_spec import parse_markdown, to_markdown
from ailawlab.graphs.roleplay_policy import (
    disclose,
    format_exhibits,
    private_briefs,
)
from ailawlab.rag import Libraries, LibraryPin, Passage, all_run_library_names, case_files

DANA = {"id": "dana", "name": "Dana", "libraries": ["provider-file"]}
SAM = {"id": "sam", "name": "Sam", "libraries": ["client-file", "provider-file "]}


def test_case_files_read_and_write_as_an_agent_file_section():
    [agent] = parse_markdown("# Dana Reyes\n\n## Role\nCounsel\n\n## Case files\n- provider-file\n"
                             "- emails 2024\n").agents
    assert agent["libraries"] == ["provider-file", "emails 2024"]
    assert "## Case files\n- provider-file\n- emails 2024" in to_markdown([agent])
    assert parse_markdown(to_markdown([agent])).agents[0]["libraries"] == agent["libraries"]
    assert case_files(SAM) == ["client-file", "provider-file"] and case_files({}) == []


def test_a_run_pins_shared_libraries_and_every_case_file():
    config = {"libraries": ["cases"], "agents": [DANA, SAM]}
    assert all_run_library_names("roleplay", config, {}) == ["cases", "provider-file", "client-file"]
    # A run that brings its own cast brings its own case files.
    assert all_run_library_names("roleplay", config, {"agents": [DANA]}) == ["cases", "provider-file"]
    assert all_run_library_names("agentic_workflow", config, {}) == ["cases"]


def p(chunk_id: int, corpus: str) -> Passage:
    return Passage(chunk_id=chunk_id, document_id=chunk_id * 10, title=f"Doc {chunk_id}",
                   content=f"text {chunk_id}", page_start=None, page_end=None, similarity=0.7,
                   corpus=corpus, version=1)


def _fake_libraries(searched: list[list[str]]):
    class Fake(Libraries):
        async def search(self, query, top_k=None, library=None):
            searched.append(self.names)
            return [p(1 if n == "provider-file" else 2 if n == "client-file" else 3, n)
                    for n in self.names][:top_k]

    pins = [LibraryPin(n, 1, [1]) for n in ("cases", "provider-file", "client-file")]
    libs = Fake(None, pins)
    libs.subset = lambda names: Fake(None, [x for x in pins if x.name in names])
    return libs


async def _note(*a, **k):
    return 0


def _ctx(libs):
    return SimpleNamespace(libraries=libs, tracer=SimpleNamespace(retrieval=_note, note=_note))


def test_an_agent_can_search_only_its_own_case_file_and_the_shared_sources():
    from ailawlab.graphs.roleplay import TurnSources

    searched: list[list[str]] = []
    ctx = _ctx(_fake_libraries(searched))
    sam = {"id": "sam", "name": "Sam", "libraries": ["client-file"]}
    turn = TurnSources(ctx, {"libraries": ["cases"], "exhibits": []}, sam)
    assert turn.places() == ["own", "shared"]
    assert [t["function"]["name"] for t in turn.tools()] == ["search_case_file", "search_legal_sources"]
    own = asyncio.run(turn.search("the lease", "own"))
    shared = asyncio.run(turn.search("the statute", "shared"))
    assert [h["marker"] for h in own + shared] == ["S1", "S2"]
    assert [(x.corpus, private) for x, private in turn.found] == [("client-file", True), ("cases", False)]
    assert "provider-file" not in {n for names in searched for n in names}
    # The same passage found again keeps its number.
    assert [h["marker"] for h in asyncio.run(turn.search("again", "own"))] == ["S1"]
    assert len(turn.found) == 2 and turn.searches == 3
    text = turn.result_text(own, "own")
    assert text.startswith("[S1] Doc 2 (your case file “client-file” v1)\ntext 2")
    # A passage already on the record comes back as its exhibit, not as a new source.
    turn = TurnSources(ctx, {"libraries": ["cases"], "exhibits": [{"chunk_id": 2, "marker": "E1"}]}, sam)
    hits = asyncio.run(turn.search("lease", "own"))
    assert [h["marker"] for h in hits] == ["E1"] and turn.found == []
    assert "already on the record" in turn.result_text(hits, "own")
    # No case file and no shared library: no tools, and nothing is searched.
    searched.clear()
    bare = TurnSources(ctx, {"libraries": []}, {"id": "x"})
    assert bare.places() == [] and bare.tools() == []
    assert asyncio.run(bare.search("anything", "shared")) == [] and searched == []


def test_the_research_note_says_where_an_agent_may_look():
    from ailawlab.graphs.roleplay import TurnSources, research_note

    ctx = _ctx(_fake_libraries([]))
    both = research_note(TurnSources(ctx, {"libraries": ["cases"]}, {"id": "s", "libraries": ["client-file"]}))
    assert "search_case_file" in both and "search_legal_sources" in both and "discloses it" in both
    shared = research_note(TurnSources(ctx, {"libraries": ["cases"]}, {"id": "s"}))
    assert "search_case_file" not in shared and "discloses" not in shared
    assert research_note(TurnSources(ctx, {"libraries": []}, {"id": "s"})) == ""


def test_older_exhibits_are_listed_by_label_only():
    exhibits = [{"marker": f"E{i}", "label": f"L{i}", "content": f"body{i}", "name": "S", "turn": i}
                for i in range(1, 12)]
    text = format_exhibits(exhibits)
    assert "body1\n" not in text and "[E1] L1 (disclosed by S in turn 1)\n\n[E2]" in text and "body11" in text


def test_citing_a_passage_discloses_it_once():
    sources = [{"marker": "S1", "label": "Memo", "chunk_id": 7, "document_id": 70, "corpus": "provider-file",
                "version": 2, "private": True, "content": "Own fact."},
               {"marker": "S2", "label": "Statute", "chunk_id": 8, "document_id": 80, "corpus": "cases",
                "version": 1, "private": False, "content": "Law."}]
    after, added = disclose([], sources, [2, 1, 9], DANA, 4)        # [S9] names nothing
    assert added == ["E1", "E2"]
    assert [(e["marker"], e["label"], e["from_marker"], e["disclosed_by"], e["turn"], e["private"])
            for e in after] == [("E1", "Statute", "S2", "dana", 4, False),
                                ("E2", "Memo", "S1", "dana", 4, True)]
    again, added = disclose(after, sources, [1], SAM, 5)            # already on the record
    assert added == [] and again == after


def test_the_assessor_knows_who_held_which_case_files():
    assert "own case files: provider-file" in private_briefs([DANA])


def test_run_and_experiment_pages_show_case_files_and_exhibits():
    from ailawlab.web.views import experiment_view, run_view

    exp = experiment_view({"mode": "roleplay", "config": {"libraries": ["cases"], "agents": [DANA, SAM]}}, [])
    assert {"label": "Case files", "value": "2 libraries", "note": "private to Dana, Sam until cited"} in exp["facts"]
    assert exp["case_files"] == [{"name": "provider-file", "holders": ["Dana", "Sam"]},
                                 {"name": "client-file", "holders": ["Sam"]}]
    assert exp["agents"][0]["case_files"] == ["provider-file"]

    exhibit = {"marker": "E1", "label": "Memo", "document_id": 70, "corpus": "provider-file", "version": 2,
               "disclosed_by": "dana", "name": "Dana", "turn": 1, "private": True, "from_marker": "S1"}
    run = {"mode": "roleplay", "config_snapshot": {"libraries": ["cases"], "agents": [DANA, SAM]},
           "libraries": [{"name": "cases", "version": 1}, {"name": "provider-file", "version": 2}],
           "result": {"exhibits": [exhibit], "transcript": [
               {"turn": 1, "agent_id": "dana", "name": "Dana", "content": "See [S1].", "disclosed": ["E1"],
                "sources": [{**exhibit, "marker": "S1", "cited": True}]},
               {"turn": 2, "agent_id": "sam", "name": "Sam", "content": "[E1] proves nothing."}]}}
    v = run_view(run, "/lab")
    assert v["exhibits"][0]["color"] == 1 and v["speakers"][0]["disclosed"] == 1
    assert 'href="/lab/sources/documents/70"' in v["transcript"][1]["html"]      # Sam's [E1]
    assert v["rerun_shared"] == ["cases"] and v["rerun_versions"] == {"cases": 1, "provider-file": 2}
