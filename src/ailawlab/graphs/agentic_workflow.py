"""Agentic workflow: a planned study, researched in parallel, written up by a lead agent.

    plan (reviewed by the researcher before the run; research.py)
      -> research: one agent per sub-question, several at once, each a tool-using loop
      -> write-up: a lead agent answers the research question from their findings

There is no budget on sources: each research agent searches and reads until it judges the
evidence for its sub-question strong, or new searches stop turning up anything. What bounds
a run is its time limit and the researcher's Stop button; either ends research with what has
been found, and the write-up still runs. An agent's context is kept small enough to go on
reading: its own notes stay whole, and tool results older than the last few shrink to a
line (the passages keep their numbers, so what it noted remains citable). Past the context
share, or on its last step, an agent gets no tools and reports its findings.

Every passage is numbered once for the whole run (grounding.SourceLedger), whichever agent
found it, so a [n] in the answer names one passage, document, library and version.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from ..config import settings
from ..grounding import cited_numbers, link_citations
from ..network_tools import current_agent, resolve_result_markers
from .answer_style import ANSWER_STYLE, strip_letter_format
from .state import AgenticState, ctx_from

log = logging.getLogger(__name__)

# The document given with a run goes to each research agent, up to this many characters
# (about 15k tokens), leaving room in its context for the research.
DOCUMENT_CHARS = 60_000

RESEARCH_PROMPT = """You are a legal researcher working on one part of a larger study. Research your sub-question thoroughly from real sources, using the tools; do not answer from memory.

{places}

How to work:
- Search, then read the most promising results. Follow leads: a case cites another, a regulation names its statute, a scholar names a source.
- Prefer primary and authoritative sources; note where sources disagree.
- There is no limit on how much you search or read. Keep going until the evidence for your sub-question is strong, and stop when new searches stop turning up relevant material.
- After each search or read, write a short note in your reply: what you learned, with the passage numbers [n] that support it. Older tool results are shortened to save room, but your notes are kept whole, so note everything you will need.
- Cite a passage as [n] with the number a search or read gave it. Never cite a search result's W number; read it first.

When you are done, reply without calling a tool: your findings for this sub-question, each claim with its [n] citations, then what you could not establish."""

FINAL_STEP = ("Research time for this sub-question is over: no more tools. Report your findings "
              "now from what you have found, each claim with its [n] citations, then what you "
              "could not establish.")

WRITE_UP_PROMPT = """You are the lead researcher. Your research agents have finished; write the study's answer from their findings.

RESEARCH BRIEF
{brief}

RESEARCH QUESTION
{question}

APPROACH
{approach}

FINDINGS, BY SUB-QUESTION
{findings}

PASSAGES CITED IN THE FINDINGS (cite them by these numbers)
{passages}

Write the answer to the research question:
- Start with a direct answer in a few sentences.
- Then a section per sub-question, each under a short Markdown heading in your own words ("### ..."), synthesizing rather than repeating the findings, and comparing where the brief asks for a comparison.
- Cite each claim with the passage numbers [n] above; cite only those numbers, and only for what the passage supports.
- End with a section "### What could not be established" listing the gaps honestly.

""" + ANSWER_STYLE


def _places(ctx, sub: dict[str, Any], registry) -> str:
    """Where this agent may look, and where the plan suggests it start."""
    tools = set(registry.names())
    lines = []
    names = ctx.libraries.searchable()
    reader = ctx.opt("reader")
    if "search_libraries" in tools:
        libs = names + ([reader.library] if reader and reader.library not in names else [])
        lines.append("Libraries of curated sources (search_libraries): "
                     + "; ".join(f"“{n}”" for n in libs) + ".")
    if "search_databases" in tools:
        lines.append("Public legal databases (search_databases): case law, the Federal Register, "
                     "the eCFR, govinfo, SEC EDGAR.")
    if "search_web" in tools:
        lines.append("The open web (search_web), and any public page by address (read).")
    if not lines:
        lines.append("You have no sources to search: say what the answer would need.")
    if sub.get("look_in"):
        lines.append("The plan suggests starting with: " + ", ".join(sub["look_in"]) + ".")
    return "\n".join(lines)


def _brief_message(state: AgenticState, sub: dict[str, Any]) -> str:
    plan = state.get("plan") or {}
    parts = [f"THE STUDY\n{plan.get('question') or state.get('brief', '')}"]
    if plan.get("approach"):
        parts.append(f"APPROACH\n{plan['approach']}")
    others = [s for s in plan.get("sub_questions") or [] if s["id"] != sub["id"]]
    if others:
        parts.append("OTHER SUB-QUESTIONS (other researchers have these; do not research them)\n"
                     + "\n".join(f"- {s['question']}" for s in others))
    parts.append(f"YOUR SUB-QUESTION ({sub['id']})\n{sub['question']}")
    if plan.get("enough_when"):
        parts.append(f"ENOUGH WHEN\n{plan['enough_when']}")
    if (pages := (state.get("given_pages") or "").strip()):
        parts.append("PAGES NAMED IN THE BRIEF, ALREADY READ (cite their passages by number)\n" + pages)
    text = (state.get("document_text") or "").strip()
    if text:
        cut = len(text) > DOCUMENT_CHARS
        parts.append(f"A DOCUMENT GIVEN WITH THIS RUN: “{state.get('document_title') or 'Untitled'}”"
                     + (f" (its first {DOCUMENT_CHARS:,} characters)" if cut else "")
                     + f"\n--- DOCUMENT ---\n{text[:DOCUMENT_CHARS]}\n--- END DOCUMENT ---")
    return "\n\n".join(parts)


# A passage's heading in a tool result: "[12] Title (library “x” v3), p. 4".
_PASSAGE_HEAD = re.compile(r"^\[(\d+)\] (.+)$", re.MULTILINE)


def passages_seen(text: str) -> dict[int, str]:
    """The passages a tool result handed out, by number, with their headings. Pure."""
    return {int(m.group(1)): m.group(2).strip()[:200] for m in _PASSAGE_HEAD.finditer(text or "")}


def compact(messages: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """The agent's conversation with all but the last `keep` tool results shortened to their
    first line and the headings of the passages they gave (so their numbers stay citable).
    The agent's own notes are kept whole. Pure."""
    tool_at = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    shorten = set(tool_at[:-keep] if keep else tool_at)
    out = []
    for i, m in enumerate(messages):
        if i in shorten and len(m.get("content") or "") > 300:
            first = (m["content"].strip().splitlines() or [""])[0][:240]
            heads = [f"[{n}] {h}" for n, h in passages_seen(m["content"]).items()]
            m = {**m, "content": f"{first} … (shortened to save room; your notes say what it showed"
                                 + ("; its passages, still citable by number:\n" + "\n".join(heads)
                                    if heads else ")")
                                 + (")" if heads else "")}
        out.append(m)
    return out


def _tool_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = call.get("function", {})
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return fn.get("name", ""), args if isinstance(args, dict) else {}


async def investigate(ctx, state: AgenticState, sub: dict[str, Any]) -> dict[str, Any]:
    """One research agent's loop over one sub-question. Returns its findings."""
    registry = ctx.opt("registry")
    tracer = ctx.tracer
    agent = sub["id"]
    current_agent.set(agent)        # this task's own copy: reads are traced to this agent
    system = RESEARCH_PROMPT.format(places=_places(ctx, sub, registry))
    convo: list[dict[str, Any]] = [{"role": "user", "content": _brief_message(state, sub)}]
    steps, wrap_up, ended, nudged = 0, False, "finished", False
    write_now = False                   # the next reply is writing, not research: no thinking
    seen: dict[int, str] = {}           # passages this agent was handed, by number
    asked_to_cite = False
    earlier = ""                        # findings it gave before being asked to cite them
    await tracer.note(f"{agent} started: {sub['question']}", agent_id=agent, node="research")
    while True:
        steps += 1
        stop = ctx.opt("should_stop")()
        last = bool(stop) or wrap_up or steps >= settings.agent_max_steps
        if last:
            ended = stop or ("context full" if wrap_up else "step limit")
        messages = [{"role": "system", "content": system},
                    *compact(convo, settings.agent_keep_results)]
        if last:
            messages.append({"role": "user", "content": FINAL_STEP})
        res = await ctx.router.chat(
            messages, model=settings.agent_model,
            tools=None if last else registry.schemas(), think=not write_now,
            num_ctx=settings.agent_context_tokens, temperature=0.2,
            options={"num_predict": 8000 if last else 4000}, timeout=900)
        await tracer.llm_call(res, node="research", agent_id=agent, prompt_preview=sub["question"])
        write_now = False

        if res.tool_calls and not last:
            convo.append({"role": "assistant", "content": res.text or "", "tool_calls": res.tool_calls})
            for call in res.tool_calls:
                name, args = _tool_args(call)
                output, elapsed_ms = await registry.call(name, args)
                await tracer.tool_call(name, args, output, node="act", agent_id=agent,
                                       eval_ms=elapsed_ms)
                convo.append({"role": "tool", "content": output, "tool_name": name})
                seen.update(passages_seen(output))
            wrap_up = res.prompt_tokens > settings.agent_context_tokens * settings.agent_wrap_up_share
            continue

        text, _ = strip_letter_format(res.text)
        if not text.strip() and not nudged:
            # Thinking used the step without a reply: ask once more for the findings, without
            # thinking, so the reply is the findings themselves.
            nudged = True
            write_now = True
            if not last:
                convo.append({"role": "user", "content": "Report your findings now, with [n] citations."})
            continue
        if text.strip() and seen and not cited_numbers(text) and not asked_to_cite:
            # A reply that cites nothing is either a plan said aloud ("let me search ...") or
            # findings without their sources. Either way it goes back once, with the passages
            # this agent has seen, so it continues or reports with citations.
            asked_to_cite = True
            write_now = True
            earlier = text
            convo.append({"role": "assistant", "content": text})
            convo.append({"role": "user", "content": cite_note(seen, still_researching=not last)})
            await tracer.note(f"{agent}: its report cited no passages; asked again", agent_id=agent,
                              node="research")
            continue
        if not text.strip() and earlier:
            text = earlier
            await tracer.note(f"{agent}: kept its earlier findings (the rewrite came back empty)",
                              agent_id=agent, node="research")
        if not text.strip():
            text = "(no findings were reported)"
            await tracer.error(f"{agent} reported no findings", agent_id=agent, node="research")
        await tracer.note(f"{agent} finished after {steps} step{'' if steps == 1 else 's'}"
                          + ("" if ended == "finished" else f" ({ended})"), agent_id=agent,
                          node="research")
        return {"id": agent, "question": sub["question"], "text": text.strip(),
                "steps": steps, "ended": ended}


def cite_note(seen: dict[int, str], still_researching: bool) -> str:
    """What an agent whose report cites no passages is told. Pure."""
    index = "\n".join(f"[{n}] {h}" for n, h in list(seen.items())[:80])
    return (("If you are still researching, continue with the tools. If you are done, give "
             if still_researching else "Give ")
            + "your findings again, citing the passage numbers [n] that support each claim. "
            "Passages you have been given:\n" + index)


async def research_node(state: AgenticState, config: RunnableConfig) -> dict:
    """Every sub-question researched by its own agent, several at once."""
    ctx = ctx_from(config)
    subs = (state.get("plan") or {}).get("sub_questions") or []
    gate = asyncio.Semaphore(max(1, settings.research_parallel))

    async def one(sub: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            if (stop := ctx.opt("should_stop")()):
                return {"id": sub["id"], "question": sub["question"], "steps": 0, "ended": stop,
                        "text": "(not researched: " + stop + " before this sub-question started)"}
            try:
                return await investigate(ctx, state, sub)
            except Exception as e:
                log.exception("research agent %s failed", sub["id"])
                await ctx.tracer.error(f"{sub['id']} failed: {type(e).__name__}: {e}",
                                       agent_id=sub["id"], node="research")
                return {"id": sub["id"], "question": sub["question"], "steps": 0,
                        "ended": "failed", "text": f"(research failed: {type(e).__name__})"}

    findings = await asyncio.gather(*(one(s) for s in subs))
    reader = ctx.opt("reader")
    if reader:
        # A W number cited in a finding becomes the passages read from it.
        for f in findings:
            f["text"], _ = resolve_result_markers(f["text"], reader.cite_numbers)
    stopped = next((f["ended"] for f in findings if f["ended"] in ("time limit", "stopped")), "")
    return {"findings": list(findings), "steps": sum(f["steps"] for f in findings),
            "stopped": stopped}


def _cited_passages(ledger, findings: list[dict[str, Any]], limit_chars: int = 60_000) -> str:
    numbers = sorted({n for f in findings for n in cited_numbers(f["text"])})
    out, used = [], 0
    for n in numbers:
        p = ledger.get(n)
        if p is None:
            continue
        excerpt = " ".join(p.content.split())[:700]
        line = f"[{n}] {p.cite_label()}: {excerpt}"
        if used + len(line) > limit_chars:
            out.append(f"(and {len(numbers) - len(out)} more passages, not shown for room)")
            break
        out.append(line)
        used += len(line)
    return "\n\n".join(out) or "(no passages were cited)"


async def write_up_node(state: AgenticState, config: RunnableConfig) -> dict:
    """The lead agent answers the research question from the findings."""
    ctx = ctx_from(config)
    ledger = ctx.opt("ledger")
    reader = ctx.opt("reader")
    plan = state.get("plan") or {}
    findings = state.get("findings") or []
    await ctx.tracer.note("writing the answer from the findings", node="write_up", agent_id="lead")
    prompt = WRITE_UP_PROMPT.format(
        brief=state.get("brief", ""), question=plan.get("question", ""),
        approach=plan.get("approach", "") or "(none given)",
        findings="\n\n".join(f"{f['id']}. {f['question']}\n{f['text']}" for f in findings),
        passages=_cited_passages(ledger, findings))
    res = await ctx.router.chat(
        [{"role": "user", "content": prompt}], model=settings.agent_model, think=True,
        num_ctx=settings.agent_context_tokens, temperature=0.2,
        options={"num_predict": 8000}, timeout=1200)
    event_id = await ctx.tracer.llm_call(res, node="write_up", agent_id="lead",
                                         prompt_preview=plan.get("question", ""))
    answer, stripped = strip_letter_format(res.text)
    if stripped:
        await ctx.tracer.note("removed a memo-style header or sign-off from the answer",
                              node="write_up", agent_id="lead")
    if not answer.strip():
        await ctx.tracer.error("the write-up produced no answer", node="write_up", agent_id="lead")
        answer = "(no answer was written; the findings by sub-question are below)"
    unread: list[int] = []
    if reader:
        answer, unread = resolve_result_markers(answer, reader.cite_numbers)
    citations = link_citations(answer, ledger.get, context="Answer") + [
        {"quoted_text": f"[W{n}]", "source_label": None, "chunk_id": None, "document_id": None,
         "corpus": None, "corpus_version": None, "similarity": None, "verdict": "unsupported",
         "context": "Answer (a search result that was never read)"} for n in dict.fromkeys(unread)]
    for f in findings:
        citations += link_citations(f["text"], ledger.get, context=f"Findings {f['id']}")
    await ctx.tracer.record_citations(event_id, citations)
    cited = cited_numbers(answer) + [n for f in findings for n in cited_numbers(f["text"])]
    return {"answer": answer, "citations": citations, "sources": ledger.sources(cited),
            "fetched": list(reader.fetched) if reader else [],
            "searches": list(reader.searches) if reader else []}


def build_agentic_graph():
    g = StateGraph(AgenticState)
    g.add_node("research", research_node)
    g.add_node("write_up", write_up_node)
    g.set_entry_point("research")
    g.add_edge("research", "write_up")
    g.add_edge("write_up", END)
    return g.compile()


def deadline_check(started: float, minutes: int, stop_requested) -> Any:
    """A function telling an agent whether research must end now, and why. `stop_requested`
    says whether the researcher pressed Stop."""
    deadline = started + minutes * 60

    def should_stop() -> str:
        if stop_requested():
            return "stopped"
        if time.monotonic() >= deadline:
            return "time limit"
        return ""

    return should_stop
