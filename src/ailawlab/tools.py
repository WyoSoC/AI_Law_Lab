"""External tool tier: callable actions exposed to agents via gemma4's native tool API.

Design constraints for a legal research setting:
  * Every tool result is traced, so a claim derived from a tool call can be audited.
  * Tools are declared in the schema Ollama expects and dispatched by name from a
    registry, so an experiment can be configured with a subset of tools without code
    changes.
  * Network-touching tools are opt-in per experiment, not on by default -- an
    experiment about model reasoning shouldn't silently reach the open internet.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .rag import Corpus, format_passages

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Awaitable[str]]
    requires_network: bool = False

    def schema(self) -> dict[str, Any]:
        """Ollama / OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None, allow_network: bool = False):
        self._tools: dict[str, Tool] = {}
        self.allow_network = allow_network
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def available(self) -> list[Tool]:
        return [t for t in self._tools.values() if self.allow_network or not t.requires_network]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.available()]

    def names(self) -> list[str]:
        return [t.name for t in self.available()]

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, int]:
        """Dispatch a tool call. Returns (result_text, elapsed_ms).

        Errors are returned as text rather than raised: a failed tool call is
        information the agent should see and recover from, not a crash of the run.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: no such tool '{name}'. Available: {', '.join(self.names())}", 0
        if tool.requires_network and not self.allow_network:
            return f"ERROR: tool '{name}' needs network access, disabled for this experiment", 0

        t0 = time.perf_counter()
        try:
            result = await tool.fn(**args)
        except TypeError as e:
            result = f"ERROR: bad arguments for '{name}': {e}"
        except Exception as e:
            log.exception("tool %s failed", name)
            result = f"ERROR: tool '{name}' failed: {e}"
        return str(result), int((time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------- built-in tools


def corpus_search_tool(corpus: Corpus) -> Tool:
    async def search_corpus(query: str, top_k: int = 5) -> str:
        passages = await corpus.search(query, top_k=int(top_k))
        if not passages:
            return "No matching passages in the corpus."
        return format_passages(passages)

    return Tool(
        name="search_corpus",
        description=(
            "Search the ingested legal corpus (case law, statutes, contracts, filings) "
            "for passages relevant to a query. Returns numbered passages with source "
            "labels you can cite as [1], [2], and so on."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "top_k": {"type": "integer", "description": "How many passages (default 5)."},
            },
            "required": ["query"],
        },
        fn=search_corpus,
    )


def calculator_tool() -> Tool:
    async def calculate(expression: str) -> str:
        # Deliberately not eval(): damages/interest arithmetic does not justify
        # handing an agent arbitrary code execution.
        import ast
        import operator as op

        ops = {
            ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
            ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod,
        }

        def ev(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in ops:
                return ops[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](ev(node.operand))
            raise ValueError(f"unsupported expression element: {ast.dump(node)}")

        return str(ev(ast.parse(expression, mode="eval").body))

    return Tool(
        name="calculate",
        description="Evaluate an arithmetic expression (damages, interest, deadlines in days).",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        fn=calculate,
    )


def default_registry(corpus: Corpus, allow_network: bool = False) -> ToolRegistry:
    return ToolRegistry([corpus_search_tool(corpus), calculator_tool()],
                        allow_network=allow_network)
