"""External tool tier: callable actions exposed to agents via gemma4's native tool API.

Design constraints for a legal research setting:
  * Every tool result is traced, so a claim derived from a tool call can be audited.
  * Tools are declared in the schema Ollama expects and dispatched by name from a
    registry, so an experiment can be configured with a subset of tools without code
    changes.
  * Network-touching tools are opt-in per experiment, not on by default -- an
    experiment about model reasoning shouldn't silently reach the open internet. The two
    installed (network_tools.py) save whatever they read into a library, so a citation to
    something read online is as traceable as one to a curated library.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .grounding import SourceLedger
from .network_tools import OnlineReader
from .rag import Libraries, format_passages

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


def library_search_tool(libraries: Libraries, ledger: SourceLedger,
                        extra: list[str] | None = None) -> Tool:
    """Search the run's libraries. Passages are numbered across the whole run through
    `ledger`, so the [n] the agent cites in its answer names one passage, whichever search
    returned it. With several libraries the agent may confine a search to one of them.
    `extra` names a library that may only fill during the run (where documents read online
    are saved); it is searchable once it holds something."""
    names = [*libraries.searchable(), *[n for n in extra or [] if n not in libraries.searchable()]]

    async def search_libraries(query: str, top_k: int = 5, library: str = "") -> str:
        library = (library or "").strip()
        if library and library not in names:
            return (f"ERROR: there is no library named “{library}”. "
                    f"Libraries: {', '.join(names)}")
        passages = await libraries.search(query, top_k=max(1, min(int(top_k), 10)),
                                          library=library or None)
        if not passages:
            return "No matching passages" + (f" in “{library}”." if library else ".")
        return format_passages(passages, ledger.number(passages))

    params: dict[str, Any] = {
        "query": {"type": "string", "description": "What to search for."},
        "top_k": {"type": "integer", "description": "How many passages (default 5, at most 10)."},
    }
    description = (
        "Search the legal sources available for this task (case law, statutes, regulations, "
        "contracts, filings) for passages relevant to a query. Returns numbered passages, "
        "each with its source and the library it came from. A passage keeps its number in "
        "every search, so cite it as [n] with that number.")
    if len(names) > 1:
        params["library"] = {"type": "string", "enum": names,
                             "description": "Search only this library. Leave it out to search all of them."}
        description += " Libraries: " + "; ".join(f"“{n}”" for n in names) + "."
    return Tool(
        name="search_libraries",
        description=description,
        parameters={"type": "object", "properties": params, "required": ["query"]},
        fn=search_libraries,
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


def online_tools(reader: OnlineReader) -> list[Tool]:
    """Search public legal databases and read what is found, saving it to a library. Both
    need the network, so a registry offers them only when the experiment allows it."""
    dbs = reader.databases()

    async def search_online(query: str, database: str = "") -> str:
        return await reader.search(query, database)

    async def read_online(source: str, look_for: str = "") -> str:
        return await reader.read(source, look_for)

    return [
        Tool(name="search_online",
             description=("Search public legal databases online for documents not in the "
                          "libraries: " + "; ".join(f"{k} ({v})" for k, v in dbs.items())
                          + ". Returns numbered results [W1], [W2] with snippets. Results are "
                          "leads, not sources: read one with read_online before relying on it."),
             parameters={"type": "object", "properties": {
                 "query": {"type": "string", "description": "What to search for."},
                 "database": {"type": "string", "enum": list(dbs),
                              "description": "Search only this database. Leave it out to search all."}},
                 "required": ["query"]},
             fn=search_online, requires_network=True),
        Tool(name="read_online",
             description=(f"Read a search result (by its number, e.g. W2) or a public web "
                          f"address. The document is saved to the library “{reader.library}” "
                          "and the passages most relevant to `look_for` come back numbered, to "
                          "cite as [n] like any library passage. At most "
                          f"{reader.max_reads} reads per task."),
             parameters={"type": "object", "properties": {
                 "source": {"type": "string",
                            "description": "A result number such as W2, or an https:// address."},
                 "look_for": {"type": "string",
                              "description": "What to find in it; picks which passages come back."}},
                 "required": ["source"]},
             fn=read_online, requires_network=True),
    ]


def default_registry(libraries: Libraries, ledger: SourceLedger,
                     allow_network: bool = False, reader: OnlineReader | None = None) -> ToolRegistry:
    """Library search (when the run has a library with anything in it, or may fill one by
    reading online), arithmetic, and with `reader` the network tools."""
    extra = [reader.library] if reader else []
    tools = [library_search_tool(libraries, ledger, extra)] if libraries or reader else []
    if reader:
        tools += online_tools(reader)
    return ToolRegistry([*tools, calculator_tool()], allow_network=allow_network)
