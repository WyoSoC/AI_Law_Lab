"""Citation markers in model output, traced back to the passages they name.

Every mode hands a model numbered passages and asks it to cite them: document analysis as
[1], [2] over one retrieval, an agent as [1], [2] over every search it makes in a run, and a
role-play speaker as [S1], [S2] over the passages shown for that turn. This module turns the
markers in what the model wrote into citation records that name the document, the library
it sits in and the version the run searched. A marker that names no passage the model was
given is recorded as unsupported rather than dropped: catching fabrication is the point.
Pure: nothing here touches the database.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .rag import Passage


def cited_numbers(text: str, prefix: str = "") -> list[int]:
    """The numbers of the [n] (or [<prefix>n]) markers in `text`, in order, each once.

    A run of markers written together, [1, 3] or [1][2], counts each number. Up to three
    digits, so a year in brackets is not taken for a citation.
    """
    seen: list[int] = []
    p = re.escape(prefix)
    pattern = rf"\[{p}(\d{{1,3}}(?:\s*,\s*{p}\d{{1,3}})*)\]"
    for group in re.findall(pattern, text):
        for part in group.split(","):
            n = int(part.strip().removeprefix(prefix))
            if n not in seen:
                seen.append(n)
    return seen


def link_citations(text: str, lookup: Callable[[int], Passage | None], *,
                   prefix: str = "", context: str = "") -> list[dict[str, Any]]:
    """One citation record per distinct marker in `text`.

    `lookup` maps a marker's number to the passage it names, or None if the model was given
    no such passage. `context` says where the marker appeared ("Answer", "Turn 7 · Dana").
    """
    out = []
    for n in sorted(cited_numbers(text, prefix)):
        p = lookup(n)
        if p is None:
            out.append({"quoted_text": f"[{prefix}{n}]", "source_label": None, "chunk_id": None,
                        "document_id": None, "corpus": None, "corpus_version": None,
                        "similarity": None, "verdict": "unsupported", "context": context})
        else:
            out.append({"quoted_text": f"[{prefix}{n}]", "source_label": p.cite_label(),
                        "chunk_id": p.chunk_id, "document_id": p.document_id,
                        "corpus": p.corpus or None, "corpus_version": p.version,
                        "similarity": p.similarity, "verdict": "grounded", "context": context})
    return out


def in_list(passages: list[Passage]) -> Callable[[int], Passage | None]:
    """Lookup for passages numbered 1, 2, 3 in the order given."""
    return lambda n: passages[n - 1] if 1 <= n <= len(passages) else None


class SourceLedger:
    """Run-wide numbers for retrieved passages.

    An agent searches several times in one run. If each search numbered its results from 1,
    a [2] in the final answer could mean any search's second passage. Here a passage gets a
    number the first time any search returns it and keeps it, so a marker names exactly one
    passage however many searches the run made.
    """

    def __init__(self) -> None:
        self._numbers: dict[int, int] = {}        # chunk id -> number
        self.passages: list[Passage] = []          # passage n is self.passages[n - 1]

    def number(self, passages: list[Passage]) -> list[int]:
        """Each passage's number, giving new ones to passages not seen before."""
        out = []
        for p in passages:
            if p.chunk_id not in self._numbers:
                self.passages.append(p)
                self._numbers[p.chunk_id] = len(self.passages)
            out.append(self._numbers[p.chunk_id])
        return out

    def get(self, n: int) -> Passage | None:
        return self.passages[n - 1] if 1 <= n <= len(self.passages) else None

    def sources(self, cited: list[int] | None = None) -> list[dict[str, Any]]:
        """Every passage handed out, numbered, with whether the given markers cite it."""
        cited_set = set(cited or [])
        return [p.source(str(n), n in cited_set) for n, p in enumerate(self.passages, start=1)]
