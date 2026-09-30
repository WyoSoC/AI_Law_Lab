"""What the experiment page shows: plain-language views of an experiment's stored settings.

Settings are stored as JSON because every run snapshots them, but the people reading the
page before launching a run are lawyers, not programmers. This module turns a config into
labelled facts, a readable cast, a source card and one-line run summaries. It has no
FastAPI or database dependency, so all of it can be tested offline.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from ..agent_spec import SECTIONS, check_cast, normalize_agent
from ..config import settings
from ..graphs.roleplay_policy import estimate_run_seconds

MODE_LABELS = {
    "document_analysis": "Document analysis",
    "agentic_workflow": "Agentic workflow",
    "roleplay": "Role-play",
}

MODE_BLURBS = {
    "document_analysis": (
        "Each run analyzes one document. The question is split into sub-questions, each is "
        "answered from the document, and the answer is grounded in the corpus with numbered "
        "citations that are checked afterwards."),
    "agentic_workflow": (
        "Each run gives an agent one task. It works in a think-then-act loop, calling tools "
        "such as corpus search, until it can answer."),
    "roleplay": (
        "Each run plays out the scenario between the cast. A moderator decides who speaks, "
        "steps in when talks stall, and ends the scene; an evaluator then assesses the "
        "outcome."),
}

_HEADINGS = {s.key: s.heading for s in SECTIONS}
_PUBLIC_KEYS = ("backstory", "demeanor", "tendencies", "priorities", "notes")
_PRIVATE_KEYS = ("bottom_line", "confidential")


def duration_text(seconds: float) -> str:
    """A rough duration a person can plan around: "about 1 hour 50 minutes"."""
    minutes = round(seconds / 60)
    if minutes < 2:
        return "about a minute"
    if minutes < 60:
        return f"about {minutes} minutes"
    hours, rest = divmod(minutes, 60)
    rest = round(rest / 10) * 10
    if rest == 60:
        hours, rest = hours + 1, 0
    text = f"about {hours} hour{'s' if hours != 1 else ''}"
    return f"{text} {rest} minutes" if rest else text


def elapsed_text(seconds: float) -> str:
    """A measured duration, shorter and exact: "45 s", "12 min", "1 h 52 min"."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


def _date(value: Any) -> str:
    """"2026-09-12T21:45:00+00:00" or a datetime -> "September 12, 2026"."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    if not isinstance(value, datetime):
        return ""
    return f"{value:%B} {value.day}, {value.year}"


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_url(value: Any) -> str:
    """Only http(s) links are rendered: the stored config can be hand-edited as raw JSON."""
    return value if isinstance(value, str) and re.match(r"^https?://", value, re.IGNORECASE) else ""


def snippet(text: str, width: int = 170) -> str:
    """The start of a model-written answer as one plain line, without Markdown."""
    lines = [line for line in (text or "").splitlines() if not line.lstrip().startswith("#")]
    plain = re.sub(r"[*_`]+", "", " ".join(lines))
    plain = " ".join(plain.split())
    if len(plain) <= width:
        return plain
    cut = plain[:width].rsplit(" ", 1)[0]
    return f"{cut}…"


def _fact(label: str, value: str, note: str = "") -> dict[str, str]:
    return {"label": label, "value": value, "note": note}


def agent_view(raw: dict) -> dict[str, Any]:
    """One cast member: name, role and objective up front, the rest grouped for a card."""
    a = normalize_agent(raw)

    def item(key: str) -> dict[str, Any]:
        # "entries", not "items": in a template, d.items would be the dict's method.
        value = a[key]
        return ({"label": _HEADINGS[key], "entries": value} if isinstance(value, list)
                else {"label": _HEADINGS[key], "text": value})

    return {
        "name": a.get("name") or a.get("id") or "Unnamed participant",
        "role": a.get("role", ""),
        "goal": a.get("goal", ""),
        "details": [item(k) for k in _PUBLIC_KEYS if a.get(k)],
        "private": [item(k) for k in _PRIVATE_KEYS if a.get(k)],
        "prompt": a.get("system_prompt", ""),
    }


def source_view(source: dict) -> dict[str, str]:
    words = source.get("words")
    parts = [
        str(source.get("site") or ""),
        f"published {_date(source['published'])}" if source.get("published") else "",
        f"read {_date(source['retrieved_at'])}" if source.get("retrieved_at") else "",
        (f"{words:,} words" + (" (shortened)" if source.get("truncated") else ""))
        if isinstance(words, int) else "",
    ]
    return {"title": str(source.get("title") or "Source"), "url": safe_url(source.get("url")),
            "meta": " · ".join(p for p in parts if p)}


def run_summary(run: dict, mode: str, turns_so_far: int | None = None) -> str:
    """One line saying where a run is, or what it found."""
    status = run.get("status")
    result = run.get("result") or {}
    if status == "failed":
        error = (run.get("error") or "unknown error").strip().splitlines()[0]
        return f"Failed: {snippet(error, 140)}"
    if status in ("pending", "running"):
        if mode == "roleplay" and turns_so_far is not None:
            limit = _int((run.get("inputs") or {}).get("max_turns")
                         or (run.get("config_snapshot") or {}).get("max_turns"),
                         settings.default_max_turns)
            return f"In progress: {turns_so_far} of up to {limit} turns so far"
        return "Starting" if status == "pending" else "In progress"
    if mode == "roleplay":
        text = snippet(result.get("outcome", ""))
        return f"{result.get('turns', 0)} turns · {text}" if text else f"{result.get('turns', 0)} turns"
    return snippet(result.get("answer", "")) or "Finished, with no answer text"


def run_rows(runs: list[dict], mode: str, progress: dict[str, int] | None = None,
             now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    progress = progress or {}
    rows = []
    for r in runs:
        rid = str(r["id"])
        active = r.get("status") in ("pending", "running")
        started, finished = r.get("started_at"), r.get("finished_at")
        if started and finished:
            took = elapsed_text((finished - started).total_seconds())
        elif started and active:
            took = f"{elapsed_text((now - started).total_seconds())} so far"
        else:
            took = "—"
        rows.append({
            "id": rid, "short": rid[:8], "status": r.get("status"), "active": active,
            "started": f"{started:%b} {started.day}, {started:%H:%M}" if started else "—",
            "took": took, "summary": run_summary(r, mode, progress.get(rid)),
        })
    return rows


def experiment_view(exp: dict, runs: list[dict], progress: dict[str, int] | None = None,
                    corpus_documents: int | None = None,
                    now: datetime | None = None) -> dict[str, Any]:
    """Everything the experiment page renders, derived from the stored experiment."""
    config = exp.get("config") or {}
    mode = exp.get("mode", "")
    view: dict[str, Any] = {
        "mode_label": MODE_LABELS.get(mode, mode.replace("_", " ").capitalize()),
        "blurb": MODE_BLURBS.get(mode, ""),
        "created": _date(exp.get("created_at")),
        "runs": run_rows(runs, mode, progress, now),
    }
    view["active_runs"] = [r for r in view["runs"] if r["active"]]

    if mode == "roleplay":
        agents = [a for a in config.get("agents") or [] if isinstance(a, dict)]
        turns = _int(config.get("max_turns"), settings.default_max_turns)
        words = _int(config.get("word_limit"), settings.default_word_limit)
        people = len(agents)
        source = config.get("source")
        corpus = str(config.get("corpus") or "")
        facts = [
            _fact("Cast", f"{people} {'person' if people == 1 else 'people'}"),
            _fact("Max turns", str(turns), "the moderator may end sooner"),
            _fact("Words per turn", f"{words:,}"),
            _fact("Longest run", duration_text(estimate_run_seconds(turns, words)),
                  "if every turn is used"),
        ]
        if corpus:
            facts.append(_fact("Legal sources", corpus, "passages the cast can cite"))
        view.update(
            facts=facts,
            corpus=corpus,
            scenario=str(config.get("scenario") or ""),
            agents=[agent_view(a) for a in agents],
            cast_errors=[i["message"] for i in check_cast(agents) if i["level"] == "error"],
            source=source_view(source) if isinstance(source, dict) else None,
            launch_defaults={"max_turns": turns, "word_limit": words, "corpus": corpus},
        )
        return view

    corpus = str(config.get("corpus") or "default")
    documents = ("" if corpus_documents is None else
                 f"{corpus_documents} document{'' if corpus_documents == 1 else 's'} in it")
    facts = [_fact("Corpus", corpus, documents)]
    if mode == "agentic_workflow":
        facts += [
            _fact("Max tool steps", str(_int(config.get("max_iterations"), 8)),
                  "think-then-act cycles before it must answer"),
            _fact("Network tools", "allowed" if config.get("allow_network") else "not allowed",
                  "none are installed yet, so the agent uses corpus search either way"),
        ]
    view.update(facts=facts, launch_defaults={})
    return view


# ---------------------------------------------------------------- run page and report
#
# A finished run is read like a document: the result summary first, then (for a role-play)
# the transcript with each person in their own colour, then the numbers and the trace. The
# same view feeds the PDF export, so the page and the file say the same thing.

# How many distinct speaker colours the stylesheet and the PDF define.
PALETTE_SIZE = 8

_TURN_REF = re.compile(r"\b(Turns?)\s+(\d{1,3})\b")


def link_turns(html: str) -> str:
    """Make an assessor's "Turn 17" / "Turns 3-7" references jump to that turn."""
    return _TURN_REF.sub(lambda m: f'<a class="turn-ref" href="#turn-{m.group(2)}">{m.group(1)} {m.group(2)}</a>', html)


def _speakers(transcript: list[dict], agents: list[dict]) -> dict[str, dict[str, Any]]:
    """Each speaker's name, role, colour slot and turn count, in cast order."""
    order = [a.get("id") for a in agents if isinstance(a, dict) and a.get("id")]
    for t in transcript:
        if t.get("agent_id") and t["agent_id"] != "moderator" and t["agent_id"] not in order:
            order.append(t["agent_id"])
    info: dict[str, dict[str, Any]] = {}
    by_id = {a.get("id"): a for a in agents if isinstance(a, dict)}
    for i, aid in enumerate(order):
        a = by_id.get(aid, {})
        info[aid] = {"id": aid, "name": a.get("name") or aid, "role": a.get("role", ""),
                     "color": i % PALETTE_SIZE + 1, "turns": 0, "words": 0, "cited": 0}
    for t in transcript:
        s = info.get(t.get("agent_id"))
        if s:
            s["name"] = t.get("name") or s["name"]
            s["role"] = t.get("role") or s["role"]
            s["turns"] += 1
            s["words"] += len((t.get("content") or "").split())
            s["cited"] += sum(1 for src in t.get("sources") or [] if src.get("cited"))
    return info


def _link_sources(html: str, sources: list[dict], prefix: str) -> str:
    """Turn a reply's [S2] markers into links to the document the passage came from."""
    by_marker = {s.get("marker"): s for s in sources}

    def link(m: re.Match) -> str:
        src = by_marker.get(m.group(1))
        if not src or not src.get("document_id"):
            return m.group(0)
        title = str(src.get("label", "")).replace('"', "&quot;")
        return (f'<a class="cite" href="{prefix}/sources/documents/{int(src["document_id"])}" '
                f'title="{title}">[{m.group(1)}]</a>')

    return re.sub(r'<span class="cite">\[(S\d+)\]</span>', link, html)


def run_view(run: dict, prefix: str = "") -> dict[str, Any]:
    """The run page's readable parts: summary, transcript, cast and sources. `prefix` is the
    site's mount path, for links to source documents."""
    from .markdown import to_html

    result = run.get("result") or {}
    config = {**(run.get("config_snapshot") or {}), **(run.get("inputs") or {})}
    mode = run.get("mode", "")
    started, finished = run.get("started_at"), run.get("finished_at")
    view: dict[str, Any] = {
        "mode_label": MODE_LABELS.get(mode, mode),
        "started": f"{started:%B} {started.day}, {started:%Y at %H:%M} UTC" if started else "",
        "took": elapsed_text((finished - started).total_seconds()) if started and finished else "",
        "summary_title": "Assessment" if mode == "roleplay" else "Answer",
        "summary_md": result.get("outcome") or result.get("answer") or "",
    }
    view["summary_html"] = link_turns(to_html(view["summary_md"])) if view["summary_md"] else ""

    if mode == "roleplay":
        transcript = [t for t in result.get("transcript") or [] if isinstance(t, dict)]
        agents = [a for a in config.get("agents") or [] if isinstance(a, dict)]
        speakers = _speakers(transcript, agents)
        entries = []
        for t in transcript:
            moderator = t.get("agent_id") == "moderator"
            s = speakers.get(t.get("agent_id"), {})
            entries.append({
                "turn": t.get("turn"), "moderator": moderator,
                "name": "Moderator" if moderator else s.get("name", t.get("name", "")),
                "role": t.get("role", ""), "color": s.get("color", 0),
                "agent_id": t.get("agent_id"),
                "content": t.get("content", ""),
                "html": _link_sources(to_html(t.get("content", "")), t.get("sources") or [], prefix),
                "words": len((t.get("content") or "").split()),
                "private_notes": t.get("private_notes") or {}, "thinking": t.get("thinking") or "",
                "sources": t.get("sources") or [], "host": t.get("host", ""),
            })
        view.update(
            scenario=str(config.get("scenario") or ""),
            corpus=str(config.get("corpus") or ""),
            speakers=list(speakers.values()),
            transcript=entries,
            turns=sum(1 for e in entries if not e["moderator"]),
            interventions=sum(1 for e in entries if e["moderator"]),
            cast=[{**agent_view(a), "id": a.get("id")} for a in agents],
        )
    return view
