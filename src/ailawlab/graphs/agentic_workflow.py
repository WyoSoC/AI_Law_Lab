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

from .state import AgenticState, ctx_from

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a legal research agent. You have tools available and should use them to ground your work in real sources rather than recalling from memory.

Guidelines:
- Search the corpus before asserting what a document or authority says.
- Quote operative language rather than paraphrasing when precision matters.
- If the tools cannot establish something, say so explicitly. Do not fill the gap with plausible-sounding invention.
- When you have enough to answer, give the final answer directly with citations."""


async def reason_node(state: AgenticState, config: RunnableConfig) -> dict:
    """One model turn: either request tools or produce the final answer."""
    ctx = ctx_from(config)
    registry = ctx.opt("registry")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["task"]}]
    messages.extend(state.get("scratchpad", []))

    res = await ctx.router.chat(
        messages,
        tools=registry.schemas(),
        think=True,
        num_ctx=131072,
        temperature=0.2,
        options={"num_predict": 1500},
    )
    await ctx.tracer.llm_call(res, node="reason", prompt_preview=state["task"])

    iterations = state.get("iterations", 0) + 1
    max_iter = state.get("max_iterations", 8)

    if res.tool_calls:
        # Record the assistant's tool request verbatim so the model sees its own
        # call alongside the result on the next turn.
        return {
            "scratchpad": [{"role": "assistant", "content": res.text or "",
                            "tool_calls": res.tool_calls}],
            "iterations": iterations,
            "done": False,
        }

    answer = res.text.strip()
    if not answer and res.truncated:
        # Reasoning consumed the whole budget. Don't silently return "".
        await ctx.tracer.error("reason node truncated before producing an answer", node="reason")
        answer = "(no answer produced: token budget exhausted during reasoning)"

    return {
        "scratchpad": [{"role": "assistant", "content": answer}],
        "answer": answer,
        "iterations": iterations,
        "done": True if answer else iterations >= max_iter,
    }


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
    if state.get("iterations", 0) >= state.get("max_iterations", 8):
        return END
    last = state["scratchpad"][-1] if state.get("scratchpad") else {}
    return "act" if last.get("tool_calls") else END


def build_agentic_graph():
    g = StateGraph(AgenticState)
    g.add_node("reason", reason_node)
    g.add_node("act", act_node)

    g.set_entry_point("reason")
    g.add_conditional_edges("reason", should_continue, {"act": "act", END: END})
    g.add_edge("act", "reason")
    return g.compile()
