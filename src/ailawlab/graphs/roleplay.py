"""Role-play simulation: several agents with distinct roles, goals, and private memory.

moderator -> speak -> moderator -> ... -> assess

Each agent gets its own Memory scoped to (run_id, agent_id), so opposing counsel do not
share a recollection of the negotiation. That separation is the whole point of the mode:
a shared buffer would leak one side's private reasoning into the other's context and
quietly invalidate the experiment.
"""
from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from ..memory import Memory, MemoryScope
from .state import RoleplayState, ctx_from

log = logging.getLogger(__name__)

AGENT_PROMPT = """You are {name}, {role}.

Scenario: {scenario}

Your objective: {goal}

You are participating in a live exchange. Stay in character. Speak only as {name}, in the
first person, and keep each turn under 200 words. Do not narrate the other party's
actions or speak on their behalf. If you reach agreement or conclude the matter, say so
explicitly."""

MODERATOR_PROMPT = """You are moderating a legal role-play exercise.

Scenario: {scenario}
Participants: {roster}

Transcript so far:
{transcript}

Who should speak next? Reply with ONLY the participant's id from this list: {ids}
If the exchange has reached a natural conclusion or agreement, reply with only: DONE"""

ASSESS_PROMPT = """You are evaluating a completed legal role-play exercise.

Scenario: {scenario}

Transcript:
{transcript}

Assess: What was the outcome? Did the parties reach agreement, and on what terms? Which
positions shifted, and what caused the shift? Note any point where a participant asserted
a legal proposition that appeared unsupported. Be concrete and cite turn numbers."""


def _format_transcript(transcript: list[dict], limit: int | None = None) -> str:
    rows = transcript[-limit:] if limit else transcript
    return "\n\n".join(f"[turn {t['turn']}] {t['name']}: {t['content']}" for t in rows) or "(nothing yet)"


async def moderator_node(state: RoleplayState, config: RunnableConfig) -> dict:
    """Pick the next speaker, or end the scene."""
    ctx = ctx_from(config)
    agents = state["agents"]
    turn = state.get("turn", 0)

    if turn >= state.get("max_turns", 12):
        return {"done": True, "next_speaker": ""}

    # First turn: no transcript to reason about, so just start with the first agent.
    if turn == 0:
        return {"next_speaker": agents[0]["id"], "done": False}

    ids = [a["id"] for a in agents]
    res = await ctx.router.chat(
        [{"role": "user", "content": MODERATOR_PROMPT.format(
            scenario=state["scenario"],
            roster=", ".join(f"{a['id']} ({a['name']}, {a['role']})" for a in agents),
            transcript=_format_transcript(state.get("transcript", []), limit=8),
            ids=", ".join(ids))}],
        think=False,          # turn-taking is bookkeeping, not reasoning worth tracing
        temperature=0.0,
        options={"num_predict": 24},
    )
    await ctx.tracer.llm_call(res, node="moderator", agent_id="moderator")

    choice = res.text.strip().upper()
    if "DONE" in choice:
        return {"done": True, "next_speaker": ""}

    # Match the reply against known ids; fall back to round-robin if it's unusable.
    for aid in ids:
        if aid.upper() in choice:
            return {"next_speaker": aid, "done": False}

    fallback = ids[turn % len(ids)]
    await ctx.tracer.note(f"moderator reply {res.text.strip()!r} unparseable; "
                          f"falling back to round-robin -> {fallback}", node="moderator")
    return {"next_speaker": fallback, "done": False}


async def speak_node(state: RoleplayState, config: RunnableConfig) -> dict:
    """The selected agent takes a turn, using its own private memory."""
    ctx = ctx_from(config)
    agent = next(a for a in state["agents"] if a["id"] == state["next_speaker"])
    turn = state.get("turn", 0) + 1

    memory = Memory(MemoryScope(run_id=ctx.run_id, agent_id=agent["id"]), ctx.router)

    system = agent.get("system_prompt") or AGENT_PROMPT.format(
        name=agent["name"], role=agent["role"],
        scenario=state["scenario"], goal=agent.get("goal", "Represent your side effectively."),
    )
    # What this agent can see: the public transcript, plus its own recollection.
    recent = _format_transcript(state.get("transcript", []), limit=6)
    prompt = (f"Transcript so far:\n{recent}\n\nIt is your turn. Respond as {agent['name']}."
              if state.get("transcript") else
              f"You open the exchange. Respond as {agent['name']}.")

    messages = await memory.build_messages(prompt, system_prompt=system)
    res = await ctx.router.chat(
        messages, think=True, num_ctx=32768, temperature=0.7,
        options={"num_predict": 700},
    )
    await ctx.tracer.llm_call(res, node="speak", agent_id=agent["id"], prompt_preview=prompt)

    content = res.text.strip() or "(no response produced)"
    await memory.add_message("user", prompt)
    await memory.add_message("assistant", content)
    if await memory.maybe_compact():
        await ctx.tracer.event("memory_write", agent_id=agent["id"], node="speak",
                               payload={"action": "compacted short-term into long-term"})

    return {
        "transcript": [{"turn": turn, "agent_id": agent["id"], "name": agent["name"],
                        "role": agent["role"], "content": content,
                        "thinking": res.thinking, "host": res.host}],
        "turn": turn,
    }


async def assess_node(state: RoleplayState, config: RunnableConfig) -> dict:
    ctx = ctx_from(config)
    res = await ctx.router.chat(
        [{"role": "user", "content": ASSESS_PROMPT.format(
            scenario=state["scenario"],
            transcript=_format_transcript(state.get("transcript", [])))}],
        think=True, num_ctx=131072, temperature=0.2,
        options={"num_predict": 1500},
    )
    await ctx.tracer.llm_call(res, node="assess", agent_id="evaluator")
    return {"outcome": res.text.strip()}


def route_after_moderator(state: RoleplayState) -> str:
    return "assess" if state.get("done") else "speak"


def build_roleplay_graph():
    g = StateGraph(RoleplayState)
    g.add_node("moderator", moderator_node)
    g.add_node("speak", speak_node)
    g.add_node("assess", assess_node)

    g.set_entry_point("moderator")
    g.add_conditional_edges("moderator", route_after_moderator,
                            {"speak": "speak", "assess": "assess"})
    g.add_edge("speak", "moderator")
    g.add_edge("assess", END)
    return g.compile()
