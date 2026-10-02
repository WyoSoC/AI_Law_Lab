"""Agentic workflow: a tool-using ReAct loop over gemma4's native function calling.

reason -> (tool calls?) -> act -> reason -> ... -> finish

Uses Ollama's `tools` parameter rather than prompt-scraped JSON. gemma4 reports the
`tools` capability, so tool calls arrive as structured `message.tool_calls` and the
usual parse-failure class of bug disappears.
"""
from __future__ import annotations

import json
import logging

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from ..config import settings
from ..grounding import cited_numbers, link_citations
from ..network_tools import resolve_result_markers
from .answer_style import ANSWER_STYLE, strip_letter_format
from .state import AgenticState, ctx_from

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant answering a question from legal sources. You have tools available and should use them to ground your work in real sources rather than recalling from memory.

{sources}

Guidelines:
- Search the libraries before asserting what a document or authority says.
- Quote operative language rather than paraphrasing when precision matters.
- Cite a passage as [n] with the number the search gave it; cite only passages a search returned.
- If the tools cannot establish something, say so explicitly. Do not fill the gap with plausible-sounding invention.
- When you have enough to answer, give the final answer directly with citations.

""" + ANSWER_STYLE


def _sources_goal(n: int) -> str:
    """What the agent is asked to read online before answering."""
    if n == 1:
        return "After searching, read the single most relevant result before answering."
    return (f"Draw on {n} independent sources: after searching, read the {n} most relevant "
            "results before answering (read_online takes several at once, e.g. W1, W3, W4; search "
            "again if the first search does not give enough), and read fewer only if fewer are "
            "relevant.")


def _sources_line(names: list[str]) -> str:
    if not names:
        return ("No library of legal sources is available for this task, so you cannot search "
                "for authority: say where the answer would need one.")
    if len(names) == 1:
        return f"You can search one library of legal sources: “{names[0]}”."
    return ("You can search these libraries of legal sources, all at once or one at a time: "
            + "; ".join(f"“{n}”" for n in names) + ".")


# A document given with the task goes into the first message whole, up to this many
# characters (about 50k tokens), leaving the rest of the 128K window for the work.
DOCUMENT_CHARS = 200_000

FINAL_STEP = ("This is your last step: no more tools. Write your answer now from what you have "
              "found, citing passages by number as before. Say plainly what you could not "
              "establish.")


def _task_message(state: AgenticState) -> str:
    """The task, with the document the run was given, if any."""
    task = state["task"]
    if pages := (state.get("given_pages") or "").strip():
        task += ("\n\nWeb pages given with this task, already read and saved (cite their "
                 f"passages by number):\n\n{pages}")
    text = (state.get("document_text") or "").strip()
    if not text:
        return task
    cut = len(text) > DOCUMENT_CHARS
    return (f"{task}\n\nA document was provided with this task: "
            f"“{state.get('document_title') or 'Untitled'}”"
            + (f" (only its first {DOCUMENT_CHARS:,} characters are shown)" if cut else "")
            + ". Quote it where it matters; cite library passages by number.\n\n"
            f"--- DOCUMENT ---\n{text[:DOCUMENT_CHARS]}\n--- END DOCUMENT ---")


async def reason_node(state: AgenticState, config: RunnableConfig) -> dict:
    """One model turn: either request tools or produce the final answer."""
    ctx = ctx_from(config)
    registry = ctx.opt("registry")

    reader = ctx.opt("reader")
    names = ctx.libraries.searchable()
    sources = (_sources_line(names) if names or not reader
               else "You have no library of legal sources yet.")
    if reader:
        sources += (" You may also search public legal databases online (search_online) and "
                    "read results or a public web page (read_online). What you read is saved "
                    f"to the library “{reader.library}” and comes back as numbered passages "
                    "you cite like any other. Search results themselves are not sources: never "
                    "cite a [W] number. " + _sources_goal(reader.wanted))
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(sources=sources)},
                {"role": "user", "content": _task_message(state)}]
    messages.extend(state.get("scratchpad", []))

    max_iter = state.get("max_iterations", settings.agent_max_steps)
    # The last step gets no tools, so the run always ends with an answer.
    last_step = state.get("iterations", 0) + 1 >= max_iter or bool(state.get("wrap_up"))
    if last_step:
        messages.append({"role": "user", "content": FINAL_STEP})
        await ctx.tracer.note(
            "last step: " + ("the agent's context is nearly full" if state.get("wrap_up")
                             else f"the {max_iter}-step limit is reached")
            + "; it answers now from what it has found", node="reason")

    res = await ctx.router.chat(
        messages,
        tools=None if last_step else registry.schemas(),
        think=True,
        num_ctx=settings.agent_context_tokens,
        temperature=0.2,
        options={"num_predict": 1500},
    )
    event_id = await ctx.tracer.llm_call(res, node="reason", prompt_preview=state["task"])

    iterations = state.get("iterations", 0) + 1

    if res.tool_calls and not last_step:
        # Record the assistant's tool request verbatim so the model sees its own
        # call alongside the result on the next turn.
        return {
            "scratchpad": [{"role": "assistant", "content": res.text or "",
                            "tool_calls": res.tool_calls}],
            "iterations": iterations,
            "wrap_up": res.prompt_tokens > settings.agent_context_tokens * settings.agent_wrap_up_share,
            "done": False,
        }

    answer, stripped = strip_letter_format(res.text)
    if stripped:
        await ctx.tracer.note("removed a memo-style header or sign-off from the answer "
                              "(the reply as written is in the trace above)", node="reason")
    # An answer before the agent has read the sources it was asked for is held back, and it is
    # told what it has not read yet. At most twice, and never on the last step, so it cannot
    # loop; it may still answer with fewer if it says the rest are not relevant.
    nudges = state.get("nudges", 0)
    if (answer and reader and reader.agent_reads < reader.wanted and nudges < 2
            and not last_step and iterations < max_iter - 1 and reader.agent_reads < reader.max_reads):
        note = more_sources_note(reader)
        await ctx.tracer.note(f"answer held back: {reader.agent_reads} of {reader.wanted} sources "
                              "read; asked to read more", node="reason")
        return {
            "scratchpad": [{"role": "assistant", "content": answer},
                           {"role": "user", "content": note}],
            "iterations": iterations,
            "nudges": nudges + 1,
            "done": False,
        }

    if not answer and res.truncated:
        # Reasoning consumed the whole budget. Don't silently return "".
        await ctx.tracer.error("reason node truncated before producing an answer", node="reason")
        answer = "(no answer produced: token budget exhausted during reasoning)"
    elif not answer and last_step:
        await ctx.tracer.error("the last step produced no answer", node="reason")
        answer = "(no answer produced on the last step)"

    out = {
        "scratchpad": [{"role": "assistant", "content": answer}],
        "answer": answer,
        "iterations": iterations,
        "done": True if answer else iterations >= max_iter or last_step,
    }
    if answer:
        ledger = ctx.opt("ledger")
        unread_cited: list[int] = []
        if reader:
            # A model may cite a search result ([W2]) instead of the passages it read from it;
            # those become the read document's passage numbers, and a result it never read
            # is recorded as unsupported.
            resolved, unread_cited = resolve_result_markers(answer, reader.cite_numbers)
            if resolved != answer:
                await ctx.tracer.note("search-result markers in the answer ([W…]) were turned into "
                                      "the passage numbers of the documents read from them", node="reason")
                answer = resolved
                out["answer"] = answer
                out["scratchpad"] = [{"role": "assistant", "content": answer}]
        # [n] markers resolve through the run-wide numbering the search tool handed out.
        citations = link_citations(answer, ledger.get, context="Answer") + [
            {"quoted_text": f"[W{n}]", "source_label": None, "chunk_id": None, "document_id": None,
             "corpus": None, "corpus_version": None, "similarity": None, "verdict": "unsupported",
             "context": "Answer (a search result that was never read)"} for n in dict.fromkeys(unread_cited)]
        await ctx.tracer.record_citations(event_id, citations)
        out.update(citations=citations, sources=ledger.sources(cited_numbers(answer)),
                   fetched=list(ctx.opt("reader").fetched) if ctx.opt("reader") else [])
    return out


async def act_node(state: AgenticState, config: RunnableConfig) -> dict:
    """Execute the tool calls the model just requested."""
    ctx = ctx_from(config)
    registry = ctx.opt("registry")

    last = state["scratchpad"][-1]
    results, messages = [], []
    for call in last.get("tool_calls", []):
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        output, elapsed_ms = await registry.call(name, args)
        await ctx.tracer.tool_call(name, args, output, node="act", eval_ms=elapsed_ms)
        results.append({"tool": name, "args": args, "output": output, "ms": elapsed_ms})
        messages.append({"role": "tool", "content": output, "tool_name": name})

    return {"scratchpad": messages, "tool_results": results}


def should_continue(state: AgenticState) -> str:
    if state.get("done"):
        return END
    if state.get("iterations", 0) >= state.get("max_iterations", settings.agent_max_steps):
        return END
    last = state["scratchpad"][-1] if state.get("scratchpad") else {}
    if last.get("tool_calls"):
        return "act"
    # A held-back answer ends with a message asking for more sources: think again.
    return "reason" if last.get("role") == "user" else END


def more_sources_note(reader) -> str:
    """What an agent that answered too soon is told: how far it is from the sources asked for,
    and which search results it has not read."""
    unread = [f"W{n}" for n in range(1, len(reader.hits) + 1) if n not in reader.read_hits]
    have = reader.agent_reads
    ask = (f"You have read {have} of the {reader.wanted} sources this task asks for. Before "
           "answering, read more of the relevant ones")
    if unread:
        ask += f": search results not read yet are {', '.join(unread[:12])}"
    ask += (". If those are not relevant, search again (search_online, perhaps another database "
            "or wording). If no more relevant sources exist, answer now and say so.")
    return ask


def build_agentic_graph():
    g = StateGraph(AgenticState)
    g.add_node("reason", reason_node)
    g.add_node("act", act_node)

    g.set_entry_point("reason")
    g.add_conditional_edges("reason", should_continue, {"act": "act", "reason": "reason", END: END})
    g.add_edge("act", "reason")
    return g.compile()
