"""Document analysis pipeline: answer a question from a document, from libraries, or both.

plan -> analyze (fan-out over sub-questions) -> ground (RAG) -> synthesize

With a document, each sub-question is answered from the document and the findings are
grounded in the run's libraries. With no document, the question is asked of the libraries
directly: each sub-question is a search, and the answer is written from the passages found.
Either way the answer cites the numbered passages, and every citation is traced to its
document, library and version.

The fan-out is deliberate: sub-questions are independent, so they run concurrently and
the router's admission control decides how many actually execute at once. This is where
the cluster's parallelism is worth spending.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from ..config import settings
from ..grounding import cited_numbers, in_list, link_citations
from ..rag import Passage, format_passages
from .answer_style import ANSWER_STYLE, strip_letter_format
from .state import DocState, ctx_from

log = logging.getLogger(__name__)

PLAN_PROMPT = """You are planning how to examine a document to answer a question about it.

Document title: {title}
Question: {question}

Break this into 2-5 specific sub-questions that together answer the question.
Return ONLY a JSON array of strings, no prose, no markdown fence.

Example: ["What are the termination provisions?", "Which party bears indemnity risk?"]"""

PLAN_LIBRARY_PROMPT = """You are planning how to answer a research question from a library of sources.

Question: {question}

Break this into 2-5 specific sub-questions, each one something a search of the sources could
answer, that together answer the question. Return ONLY a JSON array of strings, no prose, no
markdown fence."""

ANALYZE_PROMPT = """Answer the sub-question using ONLY the document excerpt below. If the excerpt does not answer it, say exactly: NOT ADDRESSED IN DOCUMENT.

Sub-question: {subq}

--- DOCUMENT EXCERPT ---
{excerpt}
--- END EXCERPT ---

Answer concisely and quote the operative language verbatim where relevant."""

SYNTHESIZE_PROMPT = """Answer this question about a document.

Question: {question}

What the document says, by sub-question:
{findings}

Passages from the libraries that may support or qualify it, numbered, each labelled with its
source and the library it came from:
{passages}

Answer the question from what the document says, citing the library passages as [1], [2]
where they bear on a point. Do not invent citations. If the passages do not support a point,
say so plainly rather than reaching.

""" + ANSWER_STYLE

SYNTHESIZE_LIBRARY_PROMPT = """Answer this question from the sources below.

Question: {question}

Passages retrieved from the libraries, numbered, each labelled with its source and the
library it came from:
{passages}

Answer using only these passages, and cite each point as [1], [2] matching the numbers above.
Do not invent citations or draw on outside knowledge. If the passages do not answer part of
the question, say so plainly.

""" + ANSWER_STYLE


def _has_document(state: DocState) -> bool:
    return bool((state.get("document_text") or "").strip())


def _extract_json_array(text: str) -> list[str]:
    """Pull a JSON array out of a model reply that may be fenced or prefaced."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        return [str(x) for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        return []


async def plan_node(state: DocState, config: RunnableConfig) -> dict:
    ctx = ctx_from(config)
    question = state.get("question", "Summarize this document and flag legal risks.")
    prompt = (PLAN_PROMPT.format(title=state.get("document_title", "(untitled)"), question=question)
              if _has_document(state) else PLAN_LIBRARY_PROMPT.format(question=question))
    res = await ctx.router.chat(
        [{"role": "user", "content": prompt}],
        think=False,          # planning is structural, not the reasoning we want traced
        temperature=0.1,
        options={"num_predict": 500},
    )
    await ctx.tracer.llm_call(res, node="plan", prompt_preview=question)

    plan = _extract_json_array(res.text)
    if not plan:
        # A failed plan shouldn't abort the run; fall back to the question itself.
        await ctx.tracer.note("planner returned no parseable plan; using question as-is",
                              node="plan")
        plan = [question]
    return {"plan": plan, "question": question}


async def analyze_node(state: DocState, config: RunnableConfig) -> dict:
    """Run every sub-question against the document concurrently (nothing to do without one)."""
    if not _has_document(state):
        return {"findings": []}
    ctx = ctx_from(config)
    text = state.get("document_text", "")
    # gemma4 has a 128K context, so a whole document usually fits without chunking.
    excerpt = text[:400_000]

    async def one(subq: str) -> dict:
        res = await ctx.router.chat(
            [{"role": "user", "content": ANALYZE_PROMPT.format(subq=subq, excerpt=excerpt)}],
            think=True,       # this is the substantive reasoning worth capturing
            num_ctx=131072,
            temperature=0.2,
            options={"num_predict": 1200},
        )
        await ctx.tracer.llm_call(res, node="analyze", prompt_preview=subq)
        return {
            "sub_question": subq,
            "answer": res.text.strip(),
            "thinking": res.thinking,
            "addressed": "NOT ADDRESSED IN DOCUMENT" not in res.text.upper(),
            "truncated": res.truncated,
        }

    findings = await asyncio.gather(*(one(q) for q in state.get("plan", [])))
    return {"findings": list(findings)}


async def ground_node(state: DocState, config: RunnableConfig) -> dict:
    """Retrieve passages from the run's libraries.

    With a document: one search for the question and the sub-questions the document
    addressed. Without: a search per sub-question and one for the whole question, run
    concurrently, merged so each passage appears once, best match first.
    """
    ctx = ctx_from(config)
    if not ctx.libraries:
        return {"passages": []}
    if _has_document(state):
        addressed = [f for f in state.get("findings", []) if f.get("addressed")]
        if not addressed:
            return {"passages": []}
        query = state.get("question", "") + " " + " ".join(f["sub_question"] for f in addressed)
        passages = await ctx.libraries.search(query)
        await ctx.tracer.retrieval(query, passages, node="ground")
        return {"passages": [p.__dict__ for p in passages]}

    queries = [state.get("question", ""), *state.get("plan", [])]

    async def one(q: str) -> list[Passage]:
        found = await ctx.libraries.search(q, top_k=4)
        await ctx.tracer.retrieval(q, found, node="ground")
        return found

    merged: dict[int, Passage] = {}
    for found in await asyncio.gather(*(one(q) for q in queries if q.strip())):
        for p in found:
            if p.chunk_id not in merged or p.similarity > merged[p.chunk_id].similarity:
                merged[p.chunk_id] = p
    best = sorted(merged.values(), key=lambda p: p.similarity, reverse=True)
    return {"passages": [p.__dict__ for p in best[:settings.rag_top_k * 2]]}


async def synthesize_node(state: DocState, config: RunnableConfig) -> dict:
    ctx = ctx_from(config)
    passages = [Passage(**p) for p in state.get("passages", [])]
    findings_text = "\n\n".join(
        f"### {f['sub_question']}\n{f['answer']}" for f in state.get("findings", [])
    )
    passages_text = format_passages(passages) or "(nothing was retrieved from the libraries)"
    prompt = (SYNTHESIZE_PROMPT.format(question=state.get("question", ""),
                                       findings=findings_text or "(none)", passages=passages_text)
              if _has_document(state) else
              SYNTHESIZE_LIBRARY_PROMPT.format(question=state.get("question", ""),
                                               passages=passages_text))
    res = await ctx.router.chat(
        [{"role": "user", "content": prompt}],
        think=True,
        num_ctx=131072,
        temperature=0.3,
        options={"num_predict": 2000},
    )
    event_id = await ctx.tracer.llm_call(res, node="synthesize")

    # A marker past the end of the passage list is a fabricated citation; it is recorded
    # as unsupported rather than dropped, since that signal is the point of the evaluation.
    answer, stripped = strip_letter_format(res.text)
    if stripped:
        await ctx.tracer.note("removed a memo-style header or sign-off from the answer "
                              "(the reply as written is in the trace above)", node="synthesize")
    citations = link_citations(answer, in_list(passages), context="Answer")
    await ctx.tracer.record_citations(event_id, citations)
    cited = set(cited_numbers(answer))
    return {"answer": answer, "citations": citations,
            "sources": [p.source(str(n), n in cited) for n, p in enumerate(passages, start=1)]}


def build_document_graph():
    g = StateGraph(DocState)
    g.add_node("plan", plan_node)
    g.add_node("analyze", analyze_node)
    g.add_node("ground", ground_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "analyze")
    g.add_edge("analyze", "ground")
    g.add_edge("ground", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()
