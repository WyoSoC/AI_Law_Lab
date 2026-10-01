"""Shared state and runtime context for all LangGraph experiment modes."""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig

from ..rag import Libraries
from ..router import LLMRouter
from ..tracing import Tracer


@dataclass
class RunContext:
    """Non-serializable services a graph node needs.

    Passed through LangGraph's `config.configurable` rather than in the state dict,
    because state is checkpointed and these are live connections.
    """
    run_id: str
    router: LLMRouter
    tracer: Tracer
    libraries: Libraries        # what the run retrieves from, each library pinned to a version
    config: dict[str, Any]

    def opt(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


def ctx_from(config: RunnableConfig) -> RunContext:
    return config["configurable"]["ctx"]


class DocState(TypedDict, total=False):
    """Document analysis pipeline state."""
    document_id: int | None
    document_title: str
    document_text: str
    question: str
    plan: list[str]
    findings: Annotated[list[dict], operator.add]
    passages: list[dict]
    answer: str
    citations: list[dict]
    sources: list[dict]         # each passage given to the synthesis, by marker; see Passage.source
    error: str


class AgenticState(TypedDict, total=False):
    """Agentic workflow (tool-using ReAct loop) state."""
    task: str
    scratchpad: Annotated[list[dict], operator.add]
    tool_results: Annotated[list[dict], operator.add]
    iterations: int
    max_iterations: int
    answer: str
    citations: list[dict]
    sources: list[dict]         # every passage any search returned, by its run-wide number
    done: bool
    error: str


class RoleplayState(TypedDict, total=False):
    """Multi-agent role-play simulation state."""
    scenario: str
    # Each agent: {id, name, role, goal, backstory, demeanor, tendencies, priorities,
    # bottom_line, confidential, notes} or a raw {system_prompt} that overrides them.
    # See agent_spec.py for the file format these come from.
    agents: list[dict]
    transcript: Annotated[list[dict], operator.add]
    turn: int
    max_turns: int
    word_limit: int             # per-turn word cap handed to each speaker
    next_speaker: str
    directive: str              # moderator's instruction to the next speaker, if any
    last_intervention: int      # turn of the moderator's last impasse intervention
    ledgers: dict[str, dict]    # agent_id -> that agent's private negotiation notes
    libraries: list[str]        # shared libraries every agent may search; empty for none
    # Passages disclosed on the record, in order: [E1], [E2], ... Each was cited by an agent
    # from its own case file or a shared library, and every agent may see and cite it after.
    exhibits: list[dict]
    outcome: str
    done: bool
    error: str
