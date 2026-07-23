"""Document analysis pipeline.

plan -> analyze (fan-out over sub-questions) -> ground (RAG) -> synthesize -> cite_check

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

from ..rag import format_passages
from .state import DocState, ctx_from

log = logging.getLogger(__name__)

PLAN_PROMPT = """You are a legal analyst planning how to examine a document.

Document title: {title}
Analyst's question: {question}

Break this into 2-5 specific sub-questions that together answer the analyst's question.
Return ONLY a JSON array of strings, no prose, no markdown fence.

Example: ["What are the termination provisions?", "Which party bears indemnity risk?"]"""

ANALYZE_PROMPT = """You are analyzing a legal document. Answer the sub-question using ONLY the document excerpt below. If the excerpt does not answer it, say exactly: NOT ADDRESSED IN DOCUMENT.

Sub-question: {subq}

--- DOCUMENT EXCERPT ---
{excerpt}
--- END EXCERPT ---

Answer concisely and quote the operative language verbatim where relevant."""

SYNTHESIZE_PROMPT = """You are a legal analyst writing findings for a colleague.

Original question: {question}

Findings from document analysis:
{findings}

Supporting authority retrieved from the corpus:
{passages}

Write a clear, well-organized answer. Cite supporting authority as [1], [2] matching the
numbered passages above. Do not invent citations. If the authority does not support a
point, say so plainly rather than reaching."""


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
    res = await ctx.router.chat(
        [{"role": "user", "content": PLAN_PROMPT.format(
            title=state.get("document_title", "(untitled)"), question=question)}],
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
    """Run every sub-question against the document concurrently."""
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
    """Retrieve supporting authority for the findings actually addressed."""
    ctx = ctx_from(config)
    addressed = [f for f in state.get("findings", []) if f.get("addressed")]
    if not addressed:
        return {"passages": []}

    query = state.get("question", "") + " " + " ".join(f["sub_question"] for f in addressed)
    passages = await ctx.corpus.search(query)
    await ctx.tracer.retrieval(query, passages, node="ground")
    return {"passages": [p.__dict__ for p in passages]}


async def synthesize_node(state: DocState, config: RunnableConfig) -> dict:
    ctx = ctx_from(config)
    from ..rag import Passage

    passages = [Passage(**p) for p in state.get("passages", [])]
    findings_text = "\n\n".join(
        f"### {f['sub_question']}\n{f['answer']}" for f in state.get("findings", [])
    )
    res = await ctx.router.chat(
        [{"role": "user", "content": SYNTHESIZE_PROMPT.format(
            question=state.get("question", ""),
            findings=findings_text or "(none)",
            passages=format_passages(passages) or "(no supporting authority retrieved)")}],
        think=True,
        num_ctx=131072,
        temperature=0.3,
        options={"num_predict": 2000},
    )
    event_id = await ctx.tracer.llm_call(res, node="synthesize")

    citations = _link_citations(res.text, passages)
    await ctx.tracer.record_citations(event_id, citations)
    return {"answer": res.text.strip(), "citations": citations}


def _link_citations(answer: str, passages: list) -> list[dict]:
    """Map [n] markers in the answer back to the passages they refer to.

    A marker pointing past the end of the passage list is a fabricated citation and is
    recorded as unsupported rather than dropped -- that signal is the point of the
    evaluation tier.
    """
    citations = []
    for marker in sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)}):
        if 1 <= marker <= len(passages):
            p = passages[marker - 1]
            citations.append({
                "quoted_text": f"[{marker}]",
                "source_label": p.cite_label(),
                "chunk_id": p.chunk_id,
                "similarity": p.similarity,
                "verdict": "grounded",
            })
        else:
            citations.append({
                "quoted_text": f"[{marker}]",
                "source_label": None,
                "chunk_id": None,
                "similarity": None,
                "verdict": "unsupported",
            })
    return citations


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
