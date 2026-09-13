"""AI help with a role-play cast: draft agents from a scenario, and review a cast against it.

Both write for the same readers as agent_spec.py -- lawyers, not programmers. A draft comes
back in the standard Markdown layout and goes through the same forgiving reader as an
uploaded file, so anything the model gets wrong about the format surfaces as the same
plain-language warnings a person would see. Nothing is saved: a draft lands in the builder
for a person to edit before the experiment is created.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable

from .agent_spec import SECTIONS, ParseResult, normalize_agent, parse_markdown
from .router import LLMRouter

DRAFT_PROMPT = """You are helping a law professor set up a realistic legal role-play for
research and teaching.

Scenario: {scenario}
{notes}
Write {count} participants for this scenario. Make them feel like real people in this kind
of matter:
- Give each a distinct role and an objective that pulls against at least one other
  participant's, but leave room for trade-offs rather than a pure zero-sum fight.
- Priorities are the interests behind each position (why they want it), not a restatement
  of the objective.
- Give every negotiating party a concrete private bottom line (a number, a term, or a
  condition) and what they would do instead of agreeing. Set the bottom lines so that a
  deal is possible but has to be worked for.
- Give each at least one confidential fact that would change the negotiation if revealed.
- Tendencies are specific, observable habits, one per line.
- Stay consistent with the scenario's jurisdiction and area of law. Do not invent case
  citations.

Write only the participants, in exactly this layout, with no introduction or commentary.
Repeat the whole block for each participant:

# Full name

{headings}"""

REVIEW_PROMPT = """You are reviewing the cast of a legal role-play before it runs. The
person who built it is a lawyer, not a programmer. Point out only real problems, in plain
language they can act on.

Scenario: {scenario}

Cast:
{cast}

Check:
- Does each participant's role and objective fit this scenario? (For example, an
  objective about indemnification caps does not fit a patent infringement dispute.)
- Is anyone essential to this scenario missing, or is anyone redundant?
- Can the bottom lines both hold at once? If no agreement is possible, say so -- that may
  be intended, but the builder should know.
- Is any confidential information irrelevant to the scenario, or already public in
  another participant's description?
- Are any two participants so alike that they will sound the same?

Use "warning" for something likely to spoil the exercise and "suggestion" for an
improvement. Use the participant's id, or "cast" for a problem with the cast as a whole.
Return an empty list if there is nothing worth raising."""


def _headings() -> str:
    return "\n\n".join(f"## {s.heading}\n({s.hint})" for s in SECTIONS if s.in_template)


def _strip_fences(text: str) -> str:
    """Models sometimes wrap Markdown in a code fence even when told not to."""
    m = re.fullmatch(r"\s*```[a-zA-Z]*\n(.*?)\n```\s*", text, re.DOTALL)
    return m.group(1) if m else text


async def draft_cast(router: LLMRouter, scenario: str, count: int, notes: str = "",
                     taken: Iterable[str] = ()) -> ParseResult:
    """Ask gemma4 for `count` agents in the standard layout, and read them back."""
    res = await router.chat(
        [{"role": "user", "content": DRAFT_PROMPT.format(
            scenario=scenario.strip(), count=count, headings=_headings(),
            notes=f"\nWhat the professor wants from this cast: {notes.strip()}\n" if notes.strip()
            else "")}],
        think=True, temperature=0.8, num_ctx=16384,
        options={"num_predict": 2048 + 900 * count},
    )
    result = parse_markdown(_strip_fences(res.text), source="AI draft", taken=taken)
    if res.truncated:
        result.warnings.append("AI draft: the draft was cut off before it finished, so the "
                               "last agent may be incomplete. Check it, or draft again.")
    if result.agents and len(result.agents) != count:
        result.warnings.append(f"AI draft: asked for {count} agents but got "
                               f"{len(result.agents)}.")
    return result


def _cast_for_review(agents: list[dict]) -> str:
    blocks = []
    for a in (normalize_agent(x) for x in agents):
        lines = [f"[{a.get('id')}] {a.get('name', '')}"]
        for s in SECTIONS:
            if s.key in ("id", "system_prompt") or not a.get(s.key):
                continue
            value = a[s.key]
            lines.append(f"  {s.heading}: " + ("; ".join(value) if isinstance(value, list) else value))
        if a.get("system_prompt"):
            lines.append(f"  Full prompt: {a['system_prompt']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def review_cast(router: LLMRouter, agents: list[dict], scenario: str) -> list[dict]:
    """Model-judged problems with a cast, in the same shape as agent_spec.check_cast."""
    ids = [a.get("id") for a in agents if a.get("id")]
    schema = {
        "type": "object",
        "properties": {"issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": [*ids, "cast"]},
                "level": {"type": "string", "enum": ["warning", "suggestion"]},
                "message": {"type": "string"},
            },
            "required": ["agent", "level", "message"],
        }}},
        "required": ["issues"],
    }
    # think=False: structured output with a schema was verified on the cluster with thinking
    # off (2026-09-13); it has not been checked with thinking on.
    res = await router.chat(
        [{"role": "user", "content": REVIEW_PROMPT.format(
            scenario=scenario.strip(), cast=_cast_for_review(agents))}],
        think=False, temperature=0.2, num_ctx=16384, options={"num_predict": 1500},
        format=schema,
    )
    try:
        issues = json.loads(res.text).get("issues", [])
    except (json.JSONDecodeError, AttributeError):
        return [{"level": "warning", "agent": None,
                 "message": "The AI review did not return a usable answer. Try again."}]
    return [{"level": i.get("level", "suggestion"),
             "agent": None if i.get("agent") == "cast" else i.get("agent"),
             "message": i.get("message", "").strip()}
            for i in issues if isinstance(i, dict) and i.get("message")]
