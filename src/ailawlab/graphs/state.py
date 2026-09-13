"""Shared state and runtime context for all LangGraph experiment modes."""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig

from ..rag import Corpus
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
    corpus: Corpus
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
    error: str


class AgenticState(TypedDict, total=False):
    """Agentic workflow (tool-using ReAct loop) state."""
    task: str
    scratchpad: Annotated[list[dict], operator.add]
    tool_results: Annotated[list[dict], operator.add]
    iterations: int
    max_iterations: int
    answer: str
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
    outcome: str
    done: bool
    error: str
