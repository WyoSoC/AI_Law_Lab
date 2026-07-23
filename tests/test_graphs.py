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


@pytest.fixture(scope="module", autouse=True)
async def seed_corpus():
    router = await get_router()
    corpus = Corpus(router, name="test")
    await corpus.add_document("Wyoming Contract Law Notes", AUTHORITY, doc_type="authority")
    yield
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
        config={"corpus": "test"},
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

    metrics = await run_metrics(run_id)
    assert metrics["performance"]["output_tokens"] > 0
    assert metrics["host_distribution"], "no host attribution recorded"


async def test_agentic_workflow_uses_tools():
    exp = await experiments.create_experiment(
        "Research agent (test)", "agentic_workflow",
        config={"corpus": "test", "max_iterations": 4},
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
    assert any(e["payload"]["tool"] == "search_corpus" for e in tool_events)


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
