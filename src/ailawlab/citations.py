"""Citation lists for libraries: one reference per document, from what its source recorded.

Each provider stores different facts about a document (a court opinion's filing date, a
regulation's CFR citation, a Federal Register document number, a web page's site), so each
gets its own form. The aim is a reference list a reader can follow back to every source an
experiment could have drawn on, not a particular citation manual: Bluebook form needs
reporter citations and pin cites that the sources do not supply. Pure.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any


def _date(value: Any) -> str:
    """'1994-11-18' or a timestamp as 'November 18, 1994'; anything else unchanged."""
    if isinstance(value, datetime | date):
        d = value
    else:
        text = str(value or "").strip()
        try:
            d = datetime.fromisoformat(text[:10])
        except ValueError:
            return text
    return f"{d:%B} {d.day}, {d.year}"


def _s(*parts: str, sep: str = ", ") -> str:
    """Non-empty parts joined into one sentence ending in a period, or ""."""
    text = sep.join(p.strip().rstrip(".,;") for p in parts if p and p.strip())
    return f"{text}." if text else ""


def _paren(*parts: str) -> str:
    inner = ", ".join(p for p in parts if p)
    return f" ({inner})" if inner else ""


def citation(doc: dict[str, Any]) -> dict[str, Any]:
    """One document's reference: {id, text, url}. `text` is plain; `url` may be "".

    Two sentences: what the document is, then where it came from and when it was read.
    """
    m = doc.get("metadata") or {}
    title = " ".join(str(doc.get("title") or "Untitled").split())
    url = doc.get("source_uri") or m.get("link") or ""
    url = url if str(url).startswith(("http://", "https://")) else ""
    provider = m.get("provider")
    retrieved = m.get("retrieved_at") or doc.get("created_at")
    read = f"retrieved {_date(retrieved)}" if retrieved else ""

    if provider == "courtlistener":
        what = _s(title + _paren(str(m.get("court") or "").strip(), _date(m.get("date_filed"))))
        where = _s("CourtListener", f"opinion no. {m['opinion_id']}" if m.get("opinion_id") else "",
                   "search snippet only" if m.get("snippet_only") else "", read)
    elif provider == "ecfr":
        heading = re.sub(r"^§+\s*[\w.-]+\s*", "", title).strip() if m.get("citation") else title
        what = _s(" — ".join(p for p in (m.get("citation") or "", heading) if p))
        where = _s("eCFR", f"current as of {_date(m['as_of'])}" if m.get("as_of") else "", read)
    elif provider == "federal_register":
        what = _s(title)
        where = _s(f"Federal Register document {m.get('document_number', '')}".strip()
                   + _paren(str(m.get("type") or ""),
                            f"published {_date(m['publication_date'])}" if m.get("publication_date") else ""),
                   "abstract only" if m.get("abstract_only") else "", read)
    elif provider == "govinfo":
        what = _s(title)
        where = _s("govinfo", str(m.get("collection") or ""),
                   f"issued {_date(m['date_issued'])}" if m.get("date_issued") else "",
                   f"package {m['package_id']}" if m.get("package_id") else "", read)
    elif provider == "edgar":
        what = _s(title)
        where = _s("SEC EDGAR", f"accession no. {m['accession']}" if m.get("accession") else "", read)
    elif m.get("link"):
        what = _s(title)
        where = _s(str(m.get("site") or "") or "Web page",
                   f"published {_date(m['published'])}" if m.get("published") else "",
                   "PDF" if doc.get("doc_type") == "PDF" else "", read)
    elif doc.get("source_uri") and not url:
        what = _s(title)
        where = _s(f"Uploaded file {doc['source_uri']}", read)
    else:
        what, where = _s(title), _s(read[:1].upper() + read[1:])
    return {"id": doc.get("id"), "text": f"{what} {where}".strip(), "url": url}


def citation_list(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """References for a set of documents, in title order."""
    return sorted((citation(d) for d in docs), key=lambda c: c["text"].casefold())


def as_text(entries: list[dict[str, Any]], heading: str = "") -> str:
    """A numbered plain-text list, for copying into a paper or a memo."""
    lines = [heading, ""] if heading else []
    for i, c in enumerate(entries, start=1):
        lines.append(f"{i}. {c['text']}" + (f" {c['url']}" if c["url"] else ""))
    return "\n".join(lines) + "\n"
