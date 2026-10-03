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

from .. import seats
from ..agent_spec import played_by_person
from ..config import settings
from ..grounding import in_list, link_citations
from ..memory import Memory, MemoryScope, estimate_tokens
from ..models import model_info, thinks
from ..rag import Passage, case_files
from ..tools import text_tool_calls, tool_args
from .roleplay_policy import (
    MODERATOR_ID,
    SOURCE_WORDS,
    character_reminder,
    cited_sources,
    compose_agent_prompt,
    context_window,
    disclose,
    format_entries,
    format_exhibits,
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
    truncate_words,
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
{exhibits}
Assess the exercise concretely, citing turn numbers:
1. Outcome: did the parties reach agreement, and on what exact terms? If not, why not?
2. Movement: which positions shifted, what caused each shift, and where did the exchange
   repeat itself instead of moving?
3. Discipline: did anyone accept terms past their bottom line, or reveal confidential
   information? Did revealing it help or hurt them?
4. Law: note any legal proposition asserted that appears unsupported. Where parties drew on
   their own case files, say which disclosures moved the exchange and whether any exhibit
   was misdescribed by the side that cited it.
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
    if ctx.opt("stop_requested", lambda: False)():
        await ctx.tracer.note("stopped by request: the exchange ends here and is assessed",
                              node="moderator")
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
        seats.publish(ctx.run_id, entries=update["transcript"])

    update["directive"] = "\n".join(directive)
    await ctx.tracer.note(f"turn {turn + 1}: {names[next_id]} speaks ({rule})"
                          + (f"; {update['transcript'][0]['role']}"
                             if "transcript" in update else ""),
                          node="moderator")
    return update


async def _update_ledger(ctx, agent: dict, system: str, incoming: str, previous: dict | None,
                         num_ctx: int, model: str) -> dict | None:
    """The agent's private negotiation notes, carried forward turn to turn."""
    res = await ctx.router.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"{incoming}\n\n{format_ledger(previous)}\n\n{LEDGER_REQUEST}"}],
        model=model, think=False, temperature=0.3, num_ctx=num_ctx,
        options={"num_predict": 900}, format=ledger_schema(),
    )
    await ctx.tracer.llm_call(res, node="ledger", agent_id=agent["id"])
    ledger = _json(res.text)
    if not ledger:
        await ctx.tracer.note("private notes update was not valid JSON; keeping the previous "
                              "notes", node="ledger", agent_id=agent["id"])
        return previous
    return ledger


# ---------------------------------------------------------------- finding sources
#
# A speaker searches for what it needs instead of being handed passages: an AI agent through
# tools, a person through the search box on their page. Both go through TurnSources, so a
# passage is numbered [S1], [S2] in the order that speaker found it during the turn, and
# citing one from a case file discloses it exactly as before.

SEARCHES_PER_TURN = 4
HITS_PER_SEARCH = 4

TOOLS = {
    "own": ("search_case_file",
            ("Search your own case file: evidence and documents only you hold. No one else "
             "has seen these passages. Citing one in your turn discloses it to everyone as an "
             "exhibit.")),
    "shared": ("search_legal_sources",
               "Search the shared legal sources that every participant can consult."),
}


class TurnSources:
    """The passages one speaker found during one turn, and the searches that found them."""

    def __init__(self, ctx, state: RoleplayState, agent: dict):
        self.ctx, self.agent = ctx, agent
        own = [n for n in case_files(agent) if ctx.libraries.subset([n])]
        shared = [n for n in state.get("libraries") or [] if n not in own]
        self.libraries = {"own": own, "shared": [n for n in shared if ctx.libraries.subset([n])]}
        self.exhibits = {e["chunk_id"]: e["marker"] for e in state.get("exhibits") or []}
        self.found: list[tuple[Passage, bool]] = []
        self.searches = 0

    def places(self) -> list[str]:
        return [k for k in ("own", "shared") if self.libraries[k]]

    def tools(self) -> list[dict]:
        return [{"type": "function", "function": {
                    "name": TOOLS[k][0], "description": TOOLS[k][1],
                    "parameters": {"type": "object", "required": ["query"], "properties": {
                        "query": {"type": "string",
                                  "description": "What you are looking for, in plain words."}}}}}
                for k in self.places()]

    def tool_place(self, name: str) -> str | None:
        return next((k for k in self.places() if TOOLS[k][0] == name), None)

    async def search(self, query: str, where: str) -> list[dict]:
        """Search one place; returns the hits as listed, each with its [S] number (new or
        already given) or its exhibit marker."""
        query = " ".join(str(query or "").split())[:500]
        if where not in self.places() or not query:
            return []
        self.searches += 1
        try:
            hits = await self.ctx.libraries.subset(self.libraries[where]).search(
                query, top_k=HITS_PER_SEARCH)
        except Exception as e:  # noqa: BLE001 - a failed search is the speaker's to work around
            await self.ctx.tracer.note(f"legal source search failed ({type(e).__name__})",
                                       node="speak", agent_id=self.agent["id"])
            return []
        await self.ctx.tracer.retrieval(query, hits, node="speak", agent_id=self.agent["id"])
        known = {p.chunk_id: i for i, (p, _) in enumerate(self.found, start=1)}
        out = []
        for p in hits:
            if p.chunk_id in self.exhibits:
                out.append({"marker": self.exhibits[p.chunk_id], "passage": p, "new": False})
                continue
            if p.chunk_id not in known:
                self.found.append((p, where == "own"))
                known[p.chunk_id] = len(self.found)
            out.append({"marker": f"S{known[p.chunk_id]}", "passage": p,
                        "new": known[p.chunk_id] == len(self.found)})
        return out

    @staticmethod
    def label(p: Passage, private: bool) -> str:
        where = (f" (your case file {p.library_label()})" if private
                 else f" (library {p.library_label()})" if p.corpus else "")
        return p.cite_label() + where

    def result_text(self, hits: list[dict], where: str) -> str:
        if not hits:
            return "Nothing relevant was found there. Try other words, or speak without it."
        rows = []
        for h in hits:
            p = h["passage"]
            if h["marker"].startswith("E"):
                rows.append(f"[{h['marker']}] {p.cite_label()} is already on the record as an exhibit.")
            else:
                rows.append(f"[{h['marker']}] {self.label(p, where == 'own')}\n"
                            f"{truncate_words(p.content, SOURCE_WORDS)}")
        return "\n\n".join(rows)

    def listing(self) -> list[dict]:
        """The passages found so far, for a person's page."""
        return [{"marker": f"S{i}", "label": self.label(p, private), "private": private,
                 "content": truncate_words(p.content, SOURCE_WORDS)}
                for i, (p, private) in enumerate(self.found, start=1)]


def research_note(sources: TurnSources) -> str:
    """How an AI speaker may look things up this turn, or "" when it has nowhere to look.

    Worded to invite a search. Measured 2026-10-03 on an opening turn: a note that said to
    search "when you need evidence ..., not by habit" got gemma4 to search 0 times in 6;
    this one, 6 in 6 (qwen3.6: 2 in 4, then 4 in 4)."""
    places = sources.places()
    if not places:
        return ""
    holds, tools = [], []
    if "own" in places:
        holds.append("your case file holds evidence no one else has seen")
        tools.append("search_case_file")
    if "shared" in places:
        holds.append("the shared legal sources hold the law every participant can consult")
        tools.append("search_legal_sources")
    own = ("Citing a passage from your own case file discloses it: it goes on the record as an "
           "exhibit everyone can read and answer. Disclose one when it strengthens your "
           "position; keep back one that would hurt it. " if "own" in places else "")
    return (f"{' and '.join(holds).capitalize()}. Before you speak, look up what you need with "
            f"{' or '.join(tools)} (up to {SEARCHES_PER_TURN} searches this turn): a position "
            "backed by the record is stronger than one asserted. Passages come back numbered "
            f"[S1], [S2]; cite one by its number when you rely on it. {own}Cite only what a "
            "passage actually says, and do not invent other authority.")


# ---------------------------------------------------------------- taking a turn

async def speak_node(state: RoleplayState, config: RunnableConfig) -> dict:
    """The selected participant takes a turn: an AI agent updates its private notes, looks
    up what it needs and speaks; a person is asked, and the run waits for their reply."""
    ctx = ctx_from(config)
    agents = state["agents"]
    agent = next((a for a in agents if a["id"] == state["next_speaker"]), None)
    if agent is None:
        raise ValueError(f"moderator selected unknown agent {state['next_speaker']!r}")
    seats.publish(ctx.run_id, speaking=agent["id"])
    if played_by_person(agent):
        return await _person_turn(ctx, state, agent)
    return await _agent_turn(ctx, state, agent)


async def _agent_turn(ctx, state: RoleplayState, agent: dict) -> dict:
    agents = state["agents"]
    name = agent.get("name", agent["id"])
    turn = state.get("turn", 0) + 1
    word_limit = int(state.get("word_limit", settings.default_word_limit))
    model = agent.get("model") or settings.chat_model
    think = thinks(await model_info(model))
    transcript = state.get("transcript", [])

    system = compose_agent_prompt(agent, agents, state["scenario"], word_limit)
    new = since_last_turn(transcript, agent["id"])
    has_spoken = len(new) < len(transcript)
    if not transcript:
        incoming = "You open the exchange."
    else:
        label = "Since your last turn" if has_spoken else "What has been said so far"
        incoming = f"{label}:\n{format_entries(new)}"

    sources = TurnSources(ctx, state, agent)
    exhibits = list(state.get("exhibits") or [])
    on_record = format_exhibits(exhibits) if exhibits else ""
    # Room for every search's passages, on top of what the prompt holds before any search.
    found_room = SEARCHES_PER_TURN * HITS_PER_SEARCH * 320 if sources.places() else 0
    num_ctx = context_window(estimate_tokens(system) + estimate_tokens(incoming) + 20_000
                             + estimate_tokens(on_record) + found_room
                             + speech_budget(word_limit, think) + 1500)
    ledger = await _update_ledger(ctx, agent, system, incoming,
                                  state.get("ledgers", {}).get(agent["id"]), num_ctx, model)

    memory = Memory(MemoryScope(run_id=ctx.run_id, agent_id=agent["id"]), ctx.router,
                    token_budget=memory_budget(word_limit, len(agents), num_ctx, think))
    prompt = "\n\n".join(p for p in (
        incoming,
        format_ledger(ledger),
        on_record,
        research_note(sources),
        f"Note from the moderator: {state['directive']}" if state.get("directive") else "",
        character_reminder(agent, word_limit),
        f"It is your turn. Respond as {name}.",
    ) if p)
    messages = await memory.build_messages(prompt, system_prompt=system)

    async def say(convo: list[dict], tools: list[dict] | None, thinking: bool):
        return await ctx.router.chat(convo, model=model, tools=tools, think=thinking,
                                     num_ctx=num_ctx, temperature=0.7,
                                     options={"num_predict": speech_budget(word_limit, thinking)})

    # The agent may search a few times, then speaks. Its last call has no tools, so the
    # turn always ends in words.
    convo = list(messages)
    while True:
        tools = sources.tools() if sources.searches < SEARCHES_PER_TURN else []
        res = await say(convo, tools or None, think)
        event_id = await ctx.tracer.llm_call(res, node="speak", agent_id=agent["id"],
                                             prompt_preview=prompt if len(convo) == len(messages) else "")
        calls = res.tool_calls or (text_tool_calls(res.text, [t["function"]["name"] for t in tools])
                                   if tools else [])
        if not (tools and calls):
            break
        convo.append({"role": "assistant", "content": "" if not res.tool_calls else res.text or "",
                      "tool_calls": calls})
        for call in calls:
            tool, args = tool_args(call)
            where = sources.tool_place(tool)
            if where is None:
                output = f"There is no tool called {tool!r}."
            elif sources.searches >= SEARCHES_PER_TURN:
                output = "No more searches this turn. Take your turn now."
            else:
                output = sources.result_text(await sources.search(args.get("query", ""), where), where)
            await ctx.tracer.tool_call(tool, args, output, node="speak", agent_id=agent["id"])
            convo.append({"role": "tool", "content": output, "tool_name": tool})
    if text_tool_calls(res.text, [TOOLS[k][0] for k in TOOLS]):
        # Out of searches but still asking for one, as text: ask once more for the turn itself.
        convo.append({"role": "user", "content": "You cannot search any more this turn. Take "
                      f"your turn now, in your own words, as {name}."})
        res = await say(convo, None, think)
        event_id = await ctx.tracer.llm_call(res, node="speak", agent_id=agent["id"])
    if res.truncated and not res.text.strip():
        await ctx.tracer.note("reasoning used the whole token budget before any reply; "
                              "retrying this turn without reasoning", node="speak",
                              agent_id=agent["id"])
        res = await say(convo, None, False)
        event_id = await ctx.tracer.llm_call(res, node="speak", agent_id=agent["id"])

    content = res.text.strip() or "(no response produced)"
    # Memory keeps what the agent heard and what it said, once each. The notes, reminder,
    # moderator note and passages are rebuilt every turn, so storing them would only duplicate.
    await memory.add_message("user", incoming)
    await memory.add_message("assistant", content)
    if await memory.maybe_compact():
        await ctx.tracer.event("memory_write", agent_id=agent["id"], node="speak",
                               payload={"action": "compacted short-term into long-term"})
    return await _record_turn(ctx, state, agent, turn, content, sources.found, event_id,
                              {"private_notes": ledger, "thinking": res.thinking,
                               "host": res.host, "model": model})


async def _person_turn(ctx, state: RoleplayState, agent: dict) -> dict:
    """Ask the person in this seat for their turn and wait for it."""
    name = agent.get("name", agent["id"])
    turn = state.get("turn", 0) + 1
    live = seats.get(ctx.run_id)
    seat = live.seats[agent["id"]]
    sources = TurnSources(ctx, state, agent)
    await ctx.tracer.note(f"waiting for {name} ({seat.player_name}) to take turn {turn}",
                          node="speak", agent_id=agent["id"])
    content = await seats.await_reply(ctx.run_id, agent["id"], turn, state.get("directive", ""),
                                      sources)
    if content is None:
        if live.stopped:
            return {"directive": ""}
        # The turn is spent, so a person who has gone away cannot hold the run forever.
        text = (f"{name} did not reply within {live.reply_minutes} minute"
                f"{'' if live.reply_minutes == 1 else 's'}, so the exchange moves on.")
        await ctx.tracer.note(text, node="speak", agent_id=agent["id"])
        entry = {"turn": turn - 1, "agent_id": MODERATOR_ID, "name": "Moderator",
                 "role": "no reply", "content": text}
        seats.publish(ctx.run_id, entries=[entry])
        return {"transcript": [entry], "turn": turn, "directive": ""}
    event_id = await ctx.tracer.event("note", node="speak", agent_id=agent["id"], payload={
        "message": f"{name}'s turn, typed by {seat.player_name}", "response": content,
        "player": seat.player_name})
    return await _record_turn(ctx, state, agent, turn, content, sources.found, event_id,
                              {"private_notes": None, "thinking": None, "host": "",
                               "played_by": seat.player_name})


async def _record_turn(ctx, state: RoleplayState, agent: dict, turn: int, content: str,
                       found: list[tuple[Passage, bool]], event_id: int | None,
                       extra: dict) -> dict:
    """Put a turn on the record: the entry, the passages it found and cited, and what its
    citations disclosed."""
    name = agent.get("name", agent["id"])
    word_limit = int(state.get("word_limit", settings.default_word_limit))
    words = len(content.split())
    if words > word_limit * 1.15:
        await ctx.tracer.note(f"{name} used {words} words against a limit of {word_limit}",
                              node="speak", agent_id=agent["id"])
    exhibits = list(state.get("exhibits") or [])
    passages = [p for p, _ in found]
    entry = {"turn": turn, "agent_id": agent["id"], "name": name,
             "role": agent.get("role", ""), "content": content, **extra}
    cited = cited_sources(content, len(passages))
    records = [{**p.source(f"S{i}", i in cited), "private": private,
                "holder": agent["id"] if private else None, "content": p.content}
               for i, (p, private) in enumerate(found, start=1)]
    # What this turn cited goes on the record as exhibits the others can see from now on.
    after, disclosed = disclose(exhibits, records, cited, agent, turn)
    if records:
        entry["sources"] = [{k: v for k, v in r.items() if k != "content"} for r in records]
    if disclosed:
        entry["disclosed"] = disclosed
    if found or exhibits or state.get("libraries") or case_files(agent):
        # [S2] names this turn's second passage only, [E1] the first exhibit on the record
        # before this turn; a marker naming neither is recorded as unsupported.
        where = f"Turn {turn} · {name}"
        await ctx.tracer.record_citations(event_id, [
            *link_citations(content, in_list(passages), prefix="S", context=where),
            *link_citations(content, _exhibit_lookup(exhibits), prefix="E", context=where)])
    if disclosed:
        await ctx.tracer.note(f"{name} disclosed {', '.join(disclosed)}", node="speak",
                              agent_id=agent["id"])
    # One note per turn spoken, which is also how a run's progress is counted.
    by = extra.get("played_by") or extra.get("model") or ""
    await ctx.tracer.event("note", node="speak", agent_id=agent["id"], payload={
        "message": f"turn {turn}: {name} spoke {words} words" + (f" ({by})" if by else ""),
        "spoken": turn})
    seats.publish(ctx.run_id, entries=[entry], exhibits=after, speaking="")
    ledgers = state.get("ledgers", {})
    if extra.get("private_notes") is not None:
        ledgers = {**ledgers, agent["id"]: extra["private_notes"]}
    return {
        "transcript": [entry],
        "turn": turn,
        "ledgers": ledgers,
        "exhibits": after,
        "directive": "",
    }


def _exhibit_lookup(exhibits: list[dict]):
    """[E<n>] -> the exhibit as a Passage, for link_citations."""
    def lookup(n: int) -> Passage | None:
        if not 1 <= n <= len(exhibits):
            return None
        e = exhibits[n - 1]
        return Passage(chunk_id=e["chunk_id"], document_id=e["document_id"], title=e["label"],
                       content=e.get("content", ""), page_start=None, page_end=None,
                       similarity=e.get("similarity") or 0.0, corpus=e.get("corpus") or "",
                       version=e.get("version"))
    return lookup


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
    seats.publish(ctx.run_id, speaking="", phase="assessing")
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

    exhibits = state.get("exhibits") or []
    prompt = ASSESS_PROMPT.format(scenario=state["scenario"],
                                  briefs=private_briefs(state["agents"]), transcript=body,
                                  exhibits=f"\n{format_exhibits(exhibits)}\n" if exhibits else "")
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
