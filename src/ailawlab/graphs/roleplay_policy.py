"""Role-play decisions that need no model call, kept pure so they can be tested offline.

The engine in roleplay.py asks the model for judgement (who was asked a question, whether
talks have stalled, what an agent will say) and leaves the rules here. Each rule traces
to a finding about how multi-agent LLM exchanges actually go wrong:

* Private negotiation notes. LLM negotiators without a plan re-propose the same deal each
  round (Abdelnabi et al. 2023, arXiv:2309.17234), and in a study of simulated
  negotiations the one intervention that moved runs off near-universal impasse was a
  structured ledger of concessions made, concessions received, current state, opponent
  assessment, and open issues. More tokens, free-form reflection, and reasoning mode did
  not (arXiv:2604.11840). LEDGER_FIELDS follows that ledger, plus the arguments already
  made, which is what the patent-dispute run on this platform kept repeating.
* A question gets an answer. Letting the current speaker select the next one ("adjacency
  pairs") cut dialogue breakdowns, while free self-selection let single agents monopolize
  the floor (arXiv:2412.04937). pick_speaker applies that rule first, then balances turns.
* No early endings. LLM agents drift toward premature consensus (arXiv:2509.23055), so a
  scene may not conclude until everyone has spoken at least twice.
* Mediator interventions. When positions stall, the moderator uses what mediators use to
  break an impasse: reframing around interests, narrowing to one issue, and reality
  testing the cost of no deal.
* A character reminder right before each turn. Instruction-following measurably drifts
  within 8-12 turns of a dialog (arXiv:2402.10962); restating the core of the character
  at the end of the prompt, where attention is strongest, counters that cheaply.
"""
from __future__ import annotations

from ..memory import estimate_tokens

MODERATOR_ID = "moderator"

LEDGER_FIELDS: tuple[tuple[str, str], ...] = (
    ("concessions_made", "Concessions you have made so far"),
    ("concessions_received", "Concessions the others have made so far"),
    ("where_things_stand", "Where things stand on each issue"),
    ("read_on_the_others", "What the others really want, and how far you think they will move"),
    ("open_issues", "Issues still open"),
    ("arguments_already_made", "Arguments you have already made (do not repeat them)"),
    ("next_move", "Your plan for this turn"),
)

INTERVENTIONS: tuple[tuple[str, str], ...] = (
    ("reframe",
     ("The discussion has stalled on positions. {name}, before restating anything, say what "
      "you think the other side needs underneath its demand, and propose one option that "
      "could serve both sides.")),
    ("narrow",
     ("The discussion has stalled. {name}, pick the single open issue closest to agreement "
      "and make a concrete, specific proposal on that issue alone.")),
    ("reality test",
     ("The discussion has stalled. {name}, consider plainly what happens to your side if "
      "there is no agreement at all. Then either make a conditional offer (if you do X, we "
      "will do Y) or say clearly that you are prepared to walk away.")),
)


# ---------------------------------------------------------------- agent prompts


def _others_line(agent: dict, agents: list[dict]) -> str:
    others = [f"{a.get('name', a['id'])} ({a.get('role') or 'participant'})"
              for a in agents if a["id"] != agent["id"]]
    return "; ".join(others) or "the other participants"


def compose_agent_prompt(agent: dict, agents: list[dict], scenario: str, word_limit: int) -> str:
    """An agent's system prompt from its character fields.

    A hand-written system_prompt wins outright: the structured fields are the UI's way of
    building a prompt for people who do not want to write one, not a constraint on those
    who do.
    """
    if agent.get("system_prompt"):
        return agent["system_prompt"]

    def section(label: str, value: str | list | None) -> str:
        if isinstance(value, list):
            value = "\n".join(f"- {v}" for v in value)
        return f"\n{label}:\n{value}\n" if value else ""

    private = section("Your bottom line", agent.get("bottom_line")) + section(
        "Confidential information only you know", agent.get("confidential"))
    if private:
        private = ("\nPRIVATE. The other participants cannot see anything in this part.\n"
                   + private)

    return f"""You are {agent.get('name', agent['id'])}, {agent.get('role') or 'a party to this matter'}.
{section('Background', agent.get('backstory'))}{section('Demeanor', agent.get('demeanor'))}{section('Behavioral tendencies', agent.get('tendencies'))}{section('What you care about most', agent.get('priorities'))}{section('Other notes', agent.get('notes'))}
Scenario: {scenario}

Your objective: {agent.get('goal') or 'Represent your side effectively.'}
{private}
You are in a live exchange with: {_others_line(agent, agents)}.

How to take your turn:
- Stay in character. Let your background shape your temperament, your words, and how
  readily you concede.
- Speak only as yourself, in the first person. Never write lines for anyone else or
  describe what they do.
- Move the exchange forward every turn: answer anything put to you, then add a new
  argument, a concrete proposal, a conditional concession, or a pointed question. Do not
  restate an argument you have already made.
- Weigh every offer against your interests and your bottom line. Do not accept terms past
  your bottom line; walking away is a legitimate result.
- Keep confidential information to yourself unless revealing a specific fact clearly helps.
- Keep your turn under {word_limit} words.
- If you reach agreement, state the agreed terms explicitly. If you walk away, say so."""


def character_reminder(agent: dict, word_limit: int) -> str:
    """The core of the character, restated at the end of every turn's prompt."""
    name = agent.get("name", agent["id"])
    if agent.get("system_prompt"):
        return f"Reminder: stay in character as {name}. Keep your turn under {word_limit} words."
    parts = [f"Reminder: you are {name}, {agent.get('role') or 'a party to this matter'}."]
    if agent.get("goal"):
        parts.append(f"Your objective: {agent['goal']}")
    if agent.get("bottom_line"):
        parts.append(f"Your private bottom line: {agent['bottom_line']}")
    if agent.get("demeanor"):
        parts.append(f"Your demeanor: {agent['demeanor']}")
    parts.append(f"Say something new, and keep your turn under {word_limit} words.")
    return "\n".join(parts)


# ---------------------------------------------------------------- transcript views


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit]) + " ..."


def format_entries(entries: list[dict], max_words: int | None = None) -> str:
    rows = []
    for t in entries:
        content = truncate_words(t["content"], max_words) if max_words else t["content"]
        if t.get("agent_id") == MODERATOR_ID:
            rows.append(f"[moderator, after turn {t['turn']}] {content}")
        else:
            rows.append(f"[turn {t['turn']}] {t['name']}: {content}")
    return "\n\n".join(rows) or "(nothing yet)"


def since_last_turn(transcript: list[dict], agent_id: str) -> list[dict]:
    """What an agent has not yet responded to: everything after its own last turn.

    Storing only this in an agent's memory (not a fresh window of recent turns each time)
    keeps the memory linear. The old approach stored the last six turns on every turn, so
    each statement landed in memory up to six times.
    """
    last = max((i for i, t in enumerate(transcript) if t.get("agent_id") == agent_id),
               default=-1)
    return transcript[last + 1:]


def speaking_counts(agents: list[dict], transcript: list[dict]) -> dict[str, int]:
    counts = {a["id"]: 0 for a in agents}
    for t in transcript:
        if t.get("agent_id") in counts:
            counts[t["agent_id"]] += 1
    return counts


def last_speaker(transcript: list[dict]) -> str | None:
    return next((t["agent_id"] for t in reversed(transcript)
                 if t.get("agent_id") != MODERATOR_ID), None)


# ---------------------------------------------------------------- moderation


def moderator_schema(ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "question_for": {"type": "string", "enum": [*ids, "nobody"]},
            "stalled": {"type": "boolean"},
            "concluded": {"type": "boolean"},
            "next_speaker": {"type": "string", "enum": ids},
            "reason": {"type": "string"},
        },
        "required": ["question_for", "stalled", "concluded", "next_speaker", "reason"],
    }


def pick_speaker(agents: list[dict], transcript: list[dict], question_for: str | None,
                 suggested: str | None) -> tuple[str, str]:
    """Choose the next speaker. Returns (agent id, which rule decided it).

    In order: someone asked a direct question answers it; otherwise the moderator's
    suggestion stands unless it would let one agent speak twice running or pull more than
    two turns ahead of the quietest participant, in which case the quietest speaks.
    """
    ids = [a["id"] for a in agents]
    previous = last_speaker(transcript)
    counts = speaking_counts(agents, transcript)
    others = [i for i in ids if i != previous] or ids
    quietest = min(others, key=lambda i: (counts[i], _roster_distance(ids, previous, i)))

    if question_for in ids and question_for != previous:
        return question_for, "answering a direct question"
    if suggested not in others:
        return quietest, "round-robin (moderator's pick was unusable)"
    if counts[suggested] - min(counts[i] for i in others) >= 2:
        return quietest, "balancing speaking time"
    return suggested, "moderator's choice"


def _roster_distance(ids: list[str], previous: str | None, candidate: str) -> int:
    """Seats after the previous speaker, so ties go round the table in order."""
    if previous not in ids:
        return ids.index(candidate)
    return (ids.index(candidate) - ids.index(previous)) % len(ids)


def may_conclude(transcript: list[dict], agents: list[dict]) -> bool:
    """No scene ends before every participant has spoken at least twice."""
    return min(speaking_counts(agents, transcript).values(), default=0) >= 2


def intervention_due(stalled: bool, turn: int, last_intervention: int, n_agents: int) -> bool:
    """Intervene on a stall, but give the parties room to respond before doing it again."""
    return stalled and turn - last_intervention >= max(4, 2 * n_agents)


def intervention_text(count: int, name: str) -> tuple[str, str]:
    """The (technique, public statement) for the moderator's `count`-th intervention."""
    technique, text = INTERVENTIONS[count % len(INTERVENTIONS)]
    return technique, text.format(name=name)


# ---------------------------------------------------------------- notes, sizing, assessment


def ledger_schema() -> dict:
    keys = [k for k, _ in LEDGER_FIELDS]
    return {"type": "object", "properties": {k: {"type": "string"} for k in keys},
            "required": keys}


def format_ledger(ledger: dict | None) -> str:
    if not ledger:
        return "Your private notes: none yet. This is your first turn."
    lines = [f"- {label}: {ledger.get(key) or '(none)'}" for key, label in LEDGER_FIELDS]
    return "Your private notes (only you can see these):\n" + "\n".join(lines)


def speech_budget(word_limit: int, think: bool) -> int:
    """num_predict for one turn. Measured: ~870 words cost ~1.4k tokens with thinking on."""
    return int(word_limit * 1.6) + (2048 if think else 256)


def context_window(tokens_needed: int) -> int:
    """Smallest standard window that fits, capped at gemma4's 128K."""
    for size in (32768, 65536):
        if tokens_needed <= size * 0.9:
            return size
    return 131072


def memory_budget(word_limit: int, n_agents: int, num_ctx: int, think: bool) -> int:
    """Short-term memory tokens left once the fixed parts of a turn's prompt are reserved."""
    incoming = int(word_limit * 1.4) * max(1, n_agents - 1) + 400
    reserved = speech_budget(word_limit, think) + 3500 + incoming
    return max(4000, min(20000, num_ctx - reserved))


def transcript_segments(transcript: list[dict], max_tokens: int) -> list[list[dict]]:
    """Consecutive runs of entries, each small enough to summarize in one call."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for t in transcript:
        cost = estimate_tokens(t["content"]) + 20
        if current and size + cost > max_tokens:
            segments.append(current)
            current, size = [], 0
        current.append(t)
        size += cost
    if current:
        segments.append(current)
    return segments


def private_briefs(agents: list[dict]) -> str:
    """Bottom lines and confidential facts, for the assessor only."""
    rows = []
    for a in agents:
        bits = [f"bottom line: {a['bottom_line']}" if a.get("bottom_line") else "",
                f"confidential: {a['confidential']}" if a.get("confidential") else ""]
        if any(bits):
            rows.append(f"- {a.get('name', a['id'])}: " + "; ".join(b for b in bits if b))
    return "\n".join(rows) or "(no participant had a bottom line or confidential information)"


# Measured 2026-09-14 on a live three-agent run at 1000 words a turn: turns took 26 s early
# and 30 s by turn 20 (reply ~17 s, private notes ~4 s, moderator ~2.5 s, plus memory and
# database work), growing as each agent's prompt lengthens. 38 s a turn leaves room for
# that growth. An earlier probe with several requests sharing a Spark measured ~70 s, which
# overstated a normal run by more than half.
ESTIMATE = {"per_turn_s": 14.0, "per_1000_words_s": 24.0, "assess_s": 60.0}


def estimate_run_seconds(max_turns: int, word_limit: int) -> float:
    """Wall time for a run that uses every turn -- an upper bound people can plan around."""
    per_turn = ESTIMATE["per_turn_s"] + ESTIMATE["per_1000_words_s"] * word_limit / 1000
    return max_turns * per_turn + ESTIMATE["assess_s"]
