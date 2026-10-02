"""End-to-end graph tests. These execute real LangGraph runs against the Sparks and
Postgres -- the point is to catch wiring failures that mocks would hide."""
from __future__ import annotations

import pytest

from ailawlab import experiments
from ailawlab.db import close_pool, fetch_all, fetch_one
from ailawlab.rag import Corpus
from ailawlab.router import get_router
from ailawlab.tracing import run_metrics

CONTRACT = """
MASTER SERVICES AGREEMENT

1. TERM. This Agreement commences on the Effective Date and continues for
twenty-four (24) months, renewing automatically for successive twelve (12) month
terms unless either party gives written notice of non-renewal at least ninety (90)
days before the end of the then-current term.

2. TERMINATION FOR CONVENIENCE. Client may terminate this Agreement for
convenience upon sixty (60) days written notice. Provider shall have no reciprocal
right of termination for convenience.

3. INDEMNIFICATION. Provider shall indemnify, defend, and hold harmless Client
from any and all claims arising out of Provider's performance, without limitation
as to amount. Client shall have no indemnification obligation to Provider.

4. LIMITATION OF LIABILITY. Provider's aggregate liability shall not exceed the
fees paid in the twelve (12) months preceding the claim; provided, however, that
this limitation shall not apply to Provider's indemnification obligations under
Section 3.

5. GOVERNING LAW. This Agreement is governed by the laws of the State of Wyoming.
"""

AUTHORITY = """
Wyoming Contract Law — Unconscionability and Limitation of Liability

Wyoming courts enforce limitation-of-liability provisions in commercial contracts
between sophisticated parties absent unconscionability. See Roberts v. Klinkosh,
where the court upheld a liability cap in an arm's-length commercial agreement.

However, an indemnification obligation that is uncapped while the corresponding
limitation of liability is capped creates asymmetric risk allocation. Wyoming
follows the general rule that such asymmetry is enforceable between commercial
parties but is scrutinized where one party lacked meaningful choice.

Termination-for-convenience clauses granting the right to only one party are
generally enforceable in Wyoming commercial agreements.
"""


REGULATION = """
Model Commercial Services Regulation — Notice of Termination

A party that terminates a commercial services agreement for convenience must give the
other party written notice stating the effective date of termination. Notice periods
shorter than thirty (30) days are disfavoured where the agreement runs longer than one year.
"""


@pytest.fixture(scope="module", autouse=True)
async def seed_corpus():
    router = await get_router()
    await Corpus(router, name="test").add_document("Wyoming Contract Law Notes", AUTHORITY,
                                                   doc_type="authority")
    # A second library, so runs exercise searching several at once.
    await Corpus(router, name="test-regs").add_document("Model Termination Regulation", REGULATION,
                                                        doc_type="authority")
    yield
    # The experiments these tests create go to the trash, so they do not crowd the dashboard;
    # their runs are kept there for inspection.
    for row in await fetch_all("SELECT id FROM experiments WHERE deleted_at IS NULL AND name LIKE %s",
                               ("%(test)",)):
        await experiments.trash_experiment(str(row["id"]))
    await close_pool()


async def test_corpus_search_returns_scored_passages():
    router = await get_router()
    corpus = Corpus(router, name="test")
    hits = await corpus.search("uncapped indemnification with a liability cap")
    assert hits, "expected at least one passage"
    assert 0.0 <= hits[0].similarity <= 1.0
    assert hits[0].cite_label()


async def test_document_analysis_run_produces_answer_and_trace():
    exp = await experiments.create_experiment(
        "Contract risk review (test)", "document_analysis",
        config={"libraries": ["test", "test-regs"]},
    )
    run = await experiments.create_run(str(exp["id"]), inputs={
        "document_title": "Master Services Agreement",
        "document_text": CONTRACT,
        "question": "What are the principal risks to the Provider in this agreement?",
    })
    run_id = str(run["id"])
    result = await experiments.execute_run(run_id)

    assert result["answer"].strip(), "no answer produced"
    assert result["plan"], "planner produced no sub-questions"
    assert result["findings"], "no findings recorded"

    row = await fetch_one("SELECT status FROM runs WHERE id=%s", (run_id,))
    assert row["status"] == "succeeded"

    events = await fetch_all(
        "SELECT event_type, COUNT(*) AS n FROM run_events WHERE run_id=%s GROUP BY event_type",
        (run_id,),
    )
    kinds = {e["event_type"]: e["n"] for e in events}
    assert kinds.get("llm_call", 0) >= 3, f"expected plan+analyze+synthesize calls, got {kinds}"
    assert kinds.get("retrieval", 0) >= 1

    # Both libraries were pinned, and every source and grounded citation names its library.
    pinned = await fetch_all("SELECT corpus, version FROM run_libraries WHERE run_id=%s "
                             "ORDER BY position", (run_id,))
    assert [p["corpus"] for p in pinned] == ["test", "test-regs"]
    assert all(p["version"] for p in pinned)
    assert all(src["corpus"] in ("test", "test-regs") and src["version"] for src in result["sources"])
    cites = await fetch_all("SELECT * FROM citations WHERE run_id=%s AND verdict='grounded'", (run_id,))
    assert all(c["corpus"] and c["document_id"] and c["corpus_version"] for c in cites)

    metrics = await run_metrics(run_id)
    assert metrics["performance"]["output_tokens"] > 0
    assert metrics["host_distribution"], "no host attribution recorded"


async def test_agentic_workflow_uses_tools():
    exp = await experiments.create_experiment(
        "Research agent (test)", "agentic_workflow",
        config={"libraries": ["test", "test-regs"], "max_iterations": 4},
    )
    run = await experiments.create_run(str(exp["id"]), inputs={
        "task": "Search the corpus and tell me whether Wyoming enforces "
                "termination-for-convenience clauses that favour only one party.",
    })
    run_id = str(run["id"])
    result = await experiments.execute_run(run_id)

    assert result["answer"].strip()
    tool_events = await fetch_all(
        "SELECT payload FROM run_events WHERE run_id=%s AND event_type='tool_call'", (run_id,)
    )
    assert tool_events, "agent never called a tool"
    assert any(e["payload"]["tool"] == "search_libraries" for e in tool_events)
    # Sources are numbered across the run's searches; each names its library and version.
    assert result["sources"], "searches returned nothing to cite"
    assert [s["marker"] for s in result["sources"]] == [str(n) for n in range(1, len(result["sources"]) + 1)]
    assert all(s["corpus"] in ("test", "test-regs") and s["version"] for s in result["sources"])


async def test_roleplay_agents_keep_separate_memory():
    exp = await experiments.create_experiment(
        "Negotiation (test)", "roleplay",
        config={"corpus": "test", "max_turns": 4},
    )
    run = await experiments.create_run(str(exp["id"]), inputs={
        "scenario": "Provider and Client negotiate the indemnification cap in a "
                    "master services agreement governed by Wyoming law.",
        "agents": [
            {"id": "provider", "name": "Dana Reyes", "role": "counsel for the Provider",
             "goal": "Cap indemnification at 12 months of fees."},
            {"id": "client", "name": "Sam Okafor", "role": "counsel for the Client",
             "goal": "Keep Provider indemnification uncapped."},
        ],
    })
    run_id = str(run["id"])
    result = await experiments.execute_run(run_id)

    assert result["turns"] >= 2
    assert result["outcome"].strip()
    speakers = {t["agent_id"] for t in result["transcript"]}
    assert len(speakers) >= 2, f"only one agent ever spoke: {speakers}"

    # The separation invariant: each agent's memory is scoped to its own agent_id.
    rows = await fetch_all(
        "SELECT agent_id, COUNT(*) AS n FROM memory_messages WHERE run_id=%s GROUP BY agent_id",
        (run_id,),
    )
    by_agent = {r["agent_id"]: r["n"] for r in rows}
    assert len(by_agent) >= 2, f"agents shared a memory scope: {by_agent}"


async def test_roleplay_case_files_stay_private_until_cited():
    exp = await experiments.create_experiment(
        "Case files (test)", "roleplay",
        config={"libraries": [], "max_turns": 4, "word_limit": 200, "agents": [
            {"id": "provider", "name": "Dana Reyes", "role": "counsel for the Provider",
             "goal": "Keep the Provider's right to terminate for convenience.",
             "libraries": ["test"]},
            {"id": "client", "name": "Sam Okafor", "role": "counsel for the Client",
             "goal": "Require at least thirty days' notice of any termination.",
             "libraries": ["test-regs"]},
        ]},
    )
    run = await experiments.create_run(str(exp["id"]), inputs={
        "scenario": "Provider and Client negotiate the termination clause of a Wyoming services "
                    "agreement. Each side should support its position with its own sources."})
    run_id = str(run["id"])
    result = await experiments.execute_run(run_id)

    pinned = await fetch_all("SELECT corpus FROM run_libraries WHERE run_id=%s ORDER BY position", (run_id,))
    assert [p["corpus"] for p in pinned] == ["test", "test-regs"]
    # Each agent searched only its own case file.
    hits = await fetch_all("SELECT agent_id, payload FROM run_events WHERE run_id=%s "
                           "AND event_type='retrieval'", (run_id,))
    assert hits, "no agent consulted its case file"
    own = {"provider": "test", "client": "test-regs"}
    for h in hits:
        assert {x["library"] for x in h["payload"]["hits"]} <= {own[h["agent_id"]]}
    for t in result["transcript"]:
        for src in t.get("sources", []):
            assert src["private"] and src["corpus"] == own[t["agent_id"]]
    # Whatever was disclosed names who disclosed it and comes from that person's own file.
    for e in result["exhibits"]:
        assert e["corpus"] == own[e["disclosed_by"]] and e["private"]


async def test_agent_reads_online_and_cites_a_saved_copy():
    exp = await experiments.create_experiment(
        "Online research (test)", "agentic_workflow",
        config={"libraries": [], "max_iterations": 6, "allow_network": True,
                "fetch_library": "test-fetched"},
    )
    run = await experiments.create_run(str(exp["id"]), inputs={
        "task": "Use search_online to find the eCFR section on how a federal agency must "
                "post a notice of proposed rulemaking (5 CFR or 1 CFR), read the best result "
                "with read_online, and answer in two sentences, citing the passage you read."})
    run_id = str(run["id"])
    result = await experiments.execute_run(run_id)

    tools = [e["payload"]["tool"] for e in await fetch_all(
        "SELECT payload FROM run_events WHERE run_id=%s AND event_type='tool_call' ORDER BY seq", (run_id,))]
    assert "search_online" in tools, f"agent never searched online: {tools}"
    if "read_online" in tools and result["fetched"]:
        f = result["fetched"][0]
        doc = await fetch_one("SELECT corpus, metadata FROM documents WHERE id=%s", (f["document_id"],))
        # A document read before (by this test's earlier runs) is reused, not saved twice.
        assert doc["corpus"] == "test-fetched" and doc["metadata"]["fetched_by_run"]
        pinned = await fetch_one("SELECT version FROM run_libraries WHERE run_id=%s AND corpus=%s",
                                 (run_id, "test-fetched"))
        assert pinned and pinned["version"] == f["version"]
        assert all(s["corpus"] == "test-fetched" for s in result["sources"])


async def test_library_queries_run_against_the_database():
    """Every library query, run for real: a placeholder miscount fails here, not on a page."""
    from ailawlab import rag

    await rag.corpus_experiments("test")
    assert any(c["corpus"] == "test" for c in await rag.list_corpora())
    await rag.list_versions("test")
    await rag.check_idle("no such library")


async def test_removing_several_documents_and_reverting():
    from ailawlab import rag

    name = "test-revert"
    await rag.delete_corpus(name)
    store = Corpus(await get_router(), name=name)
    ids = [await store.add_document(f"Doc {c}", f"Text of document {c}, for the revert test.")
           for c in "ABC"]
    v1 = await rag.record_version(name)
    assert sorted(v1["document_ids"]) == sorted(ids)

    gone = await rag.remove_documents(name, ids[:2])                 # two at once ...
    assert {g["id"] for g in gone} == set(ids[:2])
    v2 = await rag.latest_version(name)
    assert v2["version"] == v1["version"] + 1 and list(v2["document_ids"]) == [ids[2]]  # ... one version

    plan = await rag.revert_plan(name, v1["version"])
    assert {d["id"] for d in plan["put_back"]} == set(ids[:2]) and plan["take_out"] == []
    v3 = await rag.revert_to(name, v1["version"])
    assert sorted(v3["document_ids"]) == sorted(ids) and v3["change"].startswith(f"Reverted to version {v1['version']}.")
    assert {d["id"] for d in await rag.corpus_documents(name)} == set(ids)

    # A revert can be reverted, and the earlier versions are untouched.
    v4 = await rag.revert_to(name, v2["version"])
    assert list(v4["document_ids"]) == [ids[2]]
    assert sorted((await rag.get_version(name, v1["version"]))["document_ids"]) == sorted(ids)
    await rag.delete_corpus(name)


async def test_document_analysis_can_ask_the_libraries_directly():
    exp = await experiments.create_experiment(
        "Ask the libraries (test)", "document_analysis",
        config={"libraries": ["test", "test-regs"]},
    )
    # A question with neither a document nor a library cannot run, and says why.
    empty = await experiments.create_experiment("No sources (test)", "document_analysis",
                                                config={"libraries": []})
    with pytest.raises(ValueError, match="choose at least one library"):
        await experiments.create_run(str(empty["id"]), inputs={"question": "Anything?"})
    await experiments.trash_experiment(str(empty["id"]))

    run = await experiments.create_run(str(exp["id"]), inputs={
        "question": "How much notice must a party give to terminate a services agreement for "
                    "convenience, and are one-sided termination rights enforceable in Wyoming?"})
    run_id = str(run["id"])
    result = await experiments.execute_run(run_id)
    await experiments.trash_experiment(str(exp["id"]))

    assert result["plan"] and result["findings"] == []          # no document to analyze
    assert result["answer"].strip() and result["sources"], "nothing was retrieved or answered"
    first = result["answer"].lstrip()[:200].lower()
    assert not any(w in first for w in ("memorandum", "to:", "from:", "[your name]")), first
    assert all(s["corpus"] in ("test", "test-regs") for s in result["sources"])
    cites = await fetch_all("SELECT verdict, corpus FROM citations WHERE run_id=%s", (run_id,))
    assert cites and all(c["corpus"] for c in cites if c["verdict"] == "grounded")


async def test_renaming_a_library_everywhere():
    from ailawlab import rag
    from ailawlab.db import execute

    old, new = "test-rename-old", "test-rename-new"
    for n in (old, new):
        await rag.delete_corpus(n)
    doc = await Corpus(await get_router(), name=old).add_document("Doc R", "Text for the rename test.")
    v = await rag.record_version(old)
    exp = await experiments.create_experiment("Rename (test)", "roleplay", config={
        "libraries": [old], "agents": [{"id": "a", "name": "A", "libraries": [old]}]})
    await rag.set_hidden(old, True)

    with pytest.raises(ValueError, match="already a library"):
        await rag.rename_library(old, "test")                         # taken
    await execute("INSERT INTO crawls (corpus, start_url, host, max_pages) VALUES (%s, 'https://e.org/', 'e.org', 1)", (old,))
    with pytest.raises(rag.CorpusBusy, match="crawl"):
        await rag.rename_library(old, new)                            # a crawl is adding to it
    await execute("UPDATE crawls SET status='finished' WHERE corpus=%s", (old,))

    counts = await rag.rename_library(old, new)
    assert counts["documents"] == 1 and counts["corpus_versions"] == 1 and counts["experiments"] == 1
    assert (await rag.get_document(doc))["corpus"] == new
    assert (await rag.get_version(new, v["version"]))["document_ids"] == v["document_ids"]
    assert await rag.get_version(old, v["version"]) is None
    config = (await experiments.get_experiment(str(exp["id"])))["config"]
    assert config["libraries"] == [new] and config["agents"][0]["libraries"] == [new]
    assert await rag.is_hidden(new)                                    # its setting moved with it
    assert (await rag.current_names())[old] == new
    await experiments.trash_experiment(str(exp["id"]))
    await execute("DELETE FROM crawls WHERE corpus=%s", (new,))
    await rag.delete_corpus(new)
