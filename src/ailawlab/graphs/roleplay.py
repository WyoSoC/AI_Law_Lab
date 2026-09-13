"""Role-play simulation: several agents with distinct roles, goals, and private memory.

moderator -> speak -> moderator -> ... -> assess

Each agent gets its own Memory scoped to (run_id, agent_id), so opposing counsel do not
share a recollection of the negotiation. That separation is the whole point of the mode:
a shared buffer would leak one side's private reasoning into the other's context and
quietly invalidate the experiment. The same holds for each agent's private negotiation
notes, bottom line, and confidential facts: only that agent (and, afterwards, the
assessor) ever sees them. The moderator sees public roles and objectives only.

The model supplies judgement; the rules it is held to live in roleplay_policy.py, along
with the research each rule comes from.
"""
from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from ..config import settings
from ..memory import Memory, MemoryScope, estimate_tokens
from .roleplay_policy import (
    MODERATOR_ID,
    character_reminder,
    compose_agent_prompt,
    context_window,
    format_entries,
    format_ledger,
    intervention_due,
    intervention_text,
    last_speaker,
    ledger_schema,
    may_conclude,
    memory_budget,
    moderator_schema,
    pick_speaker,
    private_briefs,
    since_last_turn,
    speaking_counts,
    speech_budget,
    transcript_segments,
)
from .state import RoleplayState, ctx_from

log = logging.getLogger(__name__)

MODERATOR_PROMPT = """You are the neutral moderator of a legal role-play.

Scenario: {scenario}

Participants (id, name, role, public objective):
{roster}

Turns spoken so far: {counts}

Earlier turns (shortened):
{earlier}

The most recent turn, in full:
{last}

Answer four questions about the most recent turn and the exchange:
- question_for: did the most recent speaker put a direct question to, or demand an answer
  from, one specific other participant? Give that participant's id -- the person being
  asked, never the person asking -- or "nobody".
- stalled: over the last several turns, are the parties restating the same positions and
  arguments without any new proposal, concession, or question?
- concluded: have the parties clearly reached agreement on terms, or has someone clearly
  and finally walked away?
- next_speaker: whose turn should come next for the exchange to be productive and fair?"""

LEDGER_REQUEST = """Before you speak, update your private notes on this negotiation. Keep
each entry short and concrete (names, numbers, terms). Record honestly what has been
conceded on each side, including by you, and list every argument you have already made so
that you can avoid repeating it. These notes are for you alone."""

SUMMARY_PROMPT = """Summarize turns {first} to {last} of a legal role-play for an evaluator.
For each participant, record their positions, offers, concessions, questions asked, and any
agreement or walk-away, citing turn numbers. Be factual. 300 words at most.

{entries}"""

ASSESS_PROMPT = """You are evaluating a completed legal role-play exercise.

Scenario: {scenario}

Private information the participants held (they could not see each other's):
{briefs}

{transcript}

Assess the exercise concretely, citing turn numbers:
1. Outcome: did the parties reach agreement, and on what exact terms? If not, why not?
2. Movement: which positions shifted, what caused each shift, and where did the exchange
   repeat itself instead of moving?
3. Discipline: did anyone accept terms past their bottom line, or reveal confidential
   information? Did revealing it help or hurt them?
4. Law: note any legal proposition asserted that appears unsupported.
5. Realism: were the participants' concessions and resistance plausible for real counsel
   in this situation, or did anyone give in or dig in implausibly?"""

# Above this, a transcript is summarized in sections before assessment. gemma4's window is
# 128K tokens; 100 turns of 1000 words is ~140K, so full transcripts cannot simply be sent.
ASSESS_DIRECT_TOKENS = 60_000


def _json(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


async def moderator_node(state: RoleplayState, config: RunnableConfig) -> dict:
    """Pick the next speaker, intervene on a stall, or end the scene."""
    ctx = ctx_from(config)
    agents = state["agents"]
    transcript = state.get("transcript", [])
    turn = state.get("turn", 0)

    if turn >= state.get("max_turns", settings.default_max_turns):
        return {"done": True, "next_speaker": "", "directive": ""}
    if turn == 0:
        return {"next_speaker": agents[0]["id"], "done": False, "directive": ""}

    ids = [a["id"] for a in agents]
    names = {a["id"]: a.get("name", a["id"]) for a in agents}
    counts = speaking_counts(agents, transcript)
    recent = transcript[-8:]
    res = await ctx.router.chat(
        [{"role": "user", "content": MODERATOR_PROMPT.format(
            scenario=state["scenario"],
            roster="\n".join(f"- {a['id']} ({names[a['id']]}, {a.get('role') or 'participant'})"
                             f": {a.get('goal') or 'no stated objective'}" for a in agents),
            counts=", ".join(f"{names[i]} {n}" for i, n in counts.items()),
            earlier=format_entries(recent[:-1], max_words=120),
            last=format_entries(recent[-1:]))}],
        think=False,          # turn-taking is bookkeeping, not reasoning worth tracing
        temperature=0.0,
        num_ctx=context_window(estimate_tokens(format_entries(recent)) + 2000),
        options={"num_predict": 300},
        format=moderator_schema(ids),
    )
    await ctx.tracer.llm_call(res, node="moderator", agent_id=MODERATOR_ID)
    decision = _json(res.text)
    if not decision:
        await ctx.tracer.note(f"moderator reply was not valid JSON: {res.text[:200]!r}",
                              node="moderator")

    if decision.get("concluded"):
        if may_conclude(transcript, agents):
            return {"done": True, "next_speaker": "", "directive": ""}
        await ctx.tracer.note("moderator judged the scene concluded, but not everyone has "
                              "spoken twice yet; continuing", node="moderator")

    previous = last_speaker(transcript)
    next_id, rule = pick_speaker(agents, transcript, decision.get("question_for"),
                                 decision.get("next_speaker"))
    update: dict = {"next_speaker": next_id, "done": False}
    directive: list[str] = []
    if rule == "answering a direct question" and previous:
        directive.append(f"{names[previous]} put a direct question to you in the last turn. "
                         "Answer it directly before anything else.")

    if intervention_due(bool(decision.get("stalled")), turn, state.get("last_intervention", 0),
                        len(agents)):
        count = sum(1 for t in transcript if t.get("agent_id") == MODERATOR_ID)
        technique, text = intervention_text(count, names[next_id])
        update["transcript"] = [{"turn": turn, "agent_id": MODERATOR_ID, "name": "Moderator",
                                 "role": f"intervention: {technique}", "content": text}]
        update["last_intervention"] = turn
        directive.append(f"The moderator just said to you: {text}")

    update["directive"] = "\n".join(directive)
    await ctx.tracer.note(f"turn {turn + 1}: {names[next_id]} speaks ({rule})"
                          + (f"; {update['transcript'][0]['role']}"
                             if "transcript" in update else ""),
                          node="moderator")
    return update


async def _update_ledger(ctx, agent: dict, system: str, incoming: str, previous: dict | None,
                         num_ctx: int) -> dict | None:
    """The agent's private negotiation notes, carried forward turn to turn."""
    res = await ctx.router.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"{incoming}\n\n{format_ledger(previous)}\n\n{LEDGER_REQUEST}"}],
        think=False, temperature=0.3, num_ctx=num_ctx,
        options={"num_predict": 900}, format=ledger_schema(),
    )
    await ctx.tracer.llm_call(res, node="ledger", agent_id=agent["id"])
    ledger = _json(res.text)
    if not ledger:
        await ctx.tracer.note("private notes update was not valid JSON; keeping the previous "
                              "notes", node="ledger", agent_id=agent["id"])
        return previous
    return ledger


async def speak_node(state: RoleplayState, config: RunnableConfig) -> dict:
    """The selected agent updates its private notes, then takes its turn."""
    ctx = ctx_from(config)
    agents = state["agents"]
    agent = next((a for a in agents if a["id"] == state["next_speaker"]), None)
    if agent is None:
        raise ValueError(f"moderator selected unknown agent {state['next_speaker']!r}")
    name = agent.get("name", agent["id"])
    turn = state.get("turn", 0) + 1
    word_limit = int(state.get("word_limit", settings.default_word_limit))
    think = True
    transcript = state.get("transcript", [])

    system = compose_agent_prompt(agent, agents, state["scenario"], word_limit)
    new = since_last_turn(transcript, agent["id"])
    has_spoken = len(new) < len(transcript)
    if not transcript:
        incoming = "You open the exchange."
    else:
        label = "Since your last turn" if has_spoken else "What has been said so far"
        incoming = f"{label}:\n{format_entries(new)}"

    num_ctx = context_window(estimate_tokens(system) + estimate_tokens(incoming) + 20_000
                             + speech_budget(word_limit, think) + 1500)
    ledger = await _update_ledger(ctx, agent, system, incoming,
                                  state.get("ledgers", {}).get(agent["id"]), num_ctx)

    memory = Memory(MemoryScope(run_id=ctx.run_id, agent_id=agent["id"]), ctx.router,
                    token_budget=memory_budget(word_limit, len(agents), num_ctx, think))
    prompt = "\n\n".join(p for p in (
        incoming,
        format_ledger(ledger),
        f"Note from the moderator: {state['directive']}" if state.get("directive") else "",
        character_reminder(agent, word_limit),
        f"It is your turn. Respond as {name}.",
    ) if p)
    messages = await memory.build_messages(prompt, system_prompt=system)

    res = await ctx.router.chat(messages, think=think, num_ctx=num_ctx, temperature=0.7,
                                options={"num_predict": speech_budget(word_limit, think)})
    await ctx.tracer.llm_call(res, node="speak", agent_id=agent["id"], prompt_preview=prompt)
    if res.truncated and not res.text.strip():
        await ctx.tracer.note("reasoning used the whole token budget before any reply; "
                              "retrying this turn without reasoning", node="speak",
                              agent_id=agent["id"])
        res = await ctx.router.chat(messages, think=False, num_ctx=num_ctx, temperature=0.7,
                                    options={"num_predict": speech_budget(word_limit, False)})
        await ctx.tracer.llm_call(res, node="speak", agent_id=agent["id"])

    content = res.text.strip() or "(no response produced)"
    words = len(content.split())
    if words > word_limit * 1.15:
        await ctx.tracer.note(f"{name} used {words} words against a limit of {word_limit}",
                              node="speak", agent_id=agent["id"])

    # Memory keeps what the agent heard and what it said, once each. The notes, reminder
    # and moderator note are rebuilt every turn, so storing them would only duplicate.
    await memory.add_message("user", incoming)
    await memory.add_message("assistant", content)
    if await memory.maybe_compact():
        await ctx.tracer.event("memory_write", agent_id=agent["id"], node="speak",
                               payload={"action": "compacted short-term into long-term"})

    return {
        "transcript": [{"turn": turn, "agent_id": agent["id"], "name": name,
                        "role": agent.get("role", ""), "content": content,
                        "private_notes": ledger, "thinking": res.thinking, "host": res.host}],
        "turn": turn,
        "ledgers": {**state.get("ledgers", {}), agent["id"]: ledger},
        "directive": "",
    }


async def _summarize_segment(ctx, segment: list[dict]) -> str:
    entries = format_entries(segment)
    res = await ctx.router.chat(
        [{"role": "user", "content": SUMMARY_PROMPT.format(
            first=segment[0]["turn"], last=segment[-1]["turn"], entries=entries)}],
        think=False, temperature=0.1, num_ctx=context_window(estimate_tokens(entries) + 1500),
        options={"num_predict": 700},
    )
    await ctx.tracer.llm_call(res, node="assess", agent_id="evaluator")
    return f"Turns {segment[0]['turn']}-{segment[-1]['turn']}:\n{res.text.strip()}"


async def assess_node(state: RoleplayState, config: RunnableConfig) -> dict:
    ctx = ctx_from(config)
    transcript = state.get("transcript", [])
    total = sum(estimate_tokens(t["content"]) + 20 for t in transcript)

    if total <= ASSESS_DIRECT_TOKENS:
        body = f"Transcript:\n{format_entries(transcript)}"
    else:
        # Sections are summarized concurrently: the router spreads them across both Sparks.
        head, tail = transcript[:-6], transcript[-6:]
        summaries = await asyncio.gather(*(_summarize_segment(ctx, seg)
                                           for seg in transcript_segments(head, 24_000)))
        await ctx.tracer.note(f"transcript of ~{total} tokens summarized in {len(summaries)} "
                              "sections before assessment", node="assess")
        body = ("Summaries of the earlier turns:\n\n" + "\n\n".join(summaries)
                + f"\n\nThe final turns, verbatim:\n{format_entries(tail)}")

    prompt = ASSESS_PROMPT.format(scenario=state["scenario"],
                                  briefs=private_briefs(state["agents"]), transcript=body)
    res = await ctx.router.chat(
        [{"role": "user", "content": prompt}],
        think=True, num_ctx=context_window(estimate_tokens(prompt) + 6000), temperature=0.2,
        options={"num_predict": 3000},
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
