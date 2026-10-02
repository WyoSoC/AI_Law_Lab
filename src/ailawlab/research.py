"""Research planning for the agentic workflow.

An agentic experiment's description is its research brief. Before a run, a planner reads the
brief and decides how to research it: the question restated, the sub-questions that answer
it, which libraries to search (from a profile of each and a trial search of each against the
brief), whether the legal databases or the open web are needed, the web addresses the brief
names, and what would count as enough. The researcher reviews and may edit the plan; the run
then carries it out (graphs/agentic_workflow.py) and stores it with its result.

Plans are drafted in the background (a draft takes from a few seconds to a minute or two
while the model loads), so the launch form asks for one and then polls for it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from .config import settings
from .db import fetch_all
from .network_tools import clean_urls, permissions, urls_in, web_search_ready
from .rag import Libraries, LibraryPin, library_names, list_corpora
from .router import LLMRouter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- run settings


def time_limit_minutes(config: dict[str, Any]) -> int:
    """The run's time limit in minutes: the run's or experiment's, within bounds. Pure."""
    try:
        n = int(config.get("time_limit_minutes") or settings.research_time_limit_min)
    except (TypeError, ValueError):
        n = settings.research_time_limit_min
    return max(5, min(n, settings.research_time_limit_max))


def research_settings(config: dict[str, Any], inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a run of an agentic experiment may use, the run's choices over the experiment's:
    whether the planner chooses among all libraries or only named ones, the two outside
    permissions, the time limit. Pure."""
    merged = {**config, **(inputs or {})}
    databases, web = permissions(merged)
    choose = merged.get("choose_libraries")
    named = library_names(merged, "agentic_workflow") if (
        "libraries" in merged or "corpus" in merged) else []
    return {
        # Experiments from before the planner named their libraries; those are kept to.
        "choose_libraries": bool(choose) if choose is not None else not named,
        "libraries": named,
        "use_databases": databases,
        "use_web": web,
        "time_limit_minutes": time_limit_minutes(merged),
    }


def brief_of(exp: dict[str, Any], inputs: dict[str, Any] | None = None) -> str:
    """The research brief: the experiment's description (a run from before briefs could still
    carry a task, which is used then). Pure."""
    task = str((inputs or {}).get("task") or "").strip()
    return task or str(exp.get("description") or "").strip()


# ---------------------------------------------------------------- library profiles


async def candidate_libraries(choose: bool, named: list[str]) -> list[str]:
    """The libraries a plan may pick from: every library in view, or the ones named."""
    if not choose:
        return named
    return [c["corpus"] for c in await list_corpora() if c["documents"] and not c["hidden"]]


async def library_profiles(router: LLMRouter, names: list[str], brief: str) -> list[dict[str, Any]]:
    """What the planner knows about each library: its size, kinds of document, a sample of
    titles, and how well its closest passages match the brief (a trial search)."""
    if not names:
        return []
    rows = await fetch_all(
        "SELECT corpus, id, title, doc_type, metadata->>'provider' AS provider FROM documents "
        "WHERE corpus = ANY(%s) AND removed_at IS NULL ORDER BY corpus, id", (names,))
    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["corpus"], []).append(r)
    profiles = []
    for name in names:
        docs = by_name.get(name, [])
        if not docs:
            continue
        kinds: dict[str, int] = {}
        for d in docs:
            kind = d["provider"] or d["doc_type"] or "document"
            kinds[kind] = kinds.get(kind, 0) + 1
        # Searched as the library is now, without recording a version: planning is not a run.
        probe = Libraries(router, [LibraryPin(name, 0, [d["id"] for d in docs])])
        try:
            found = await probe.search(brief[:2000], top_k=3)
        except Exception as e:  # noqa: BLE001 - a failed trial search leaves the profile without it
            log.warning("trial search of %s failed: %s", name, e)
            found = []
        profiles.append({
            "name": name,
            "documents": len(docs),
            "kinds": kinds,
            "titles": [d["title"] for d in docs[:25]],
            "match": round(max((p.similarity for p in found), default=0.0), 2),
            "closest": list(dict.fromkeys(p.title for p in found)),
        })
    return profiles


def _profile_text(p: dict[str, Any]) -> str:
    kinds = ", ".join(f"{n} {k}" for k, n in p["kinds"].items())
    more = p["documents"] - len(p["titles"])
    return (f"- “{p['name']}”: {p['documents']} documents ({kinds}). "
            f"Trial search against the brief: best match {p['match']:.2f}"
            + (f", closest: {'; '.join(p['closest'])}" if p["closest"] else ", nothing close")
            + ". Titles: " + "; ".join(p["titles"]) + (f"; and {more} more" if more > 0 else "") + ".")


# ---------------------------------------------------------------- the plan

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "approach": {"type": "string"},
        "sub_questions": {"type": "array", "items": {
            "type": "object",
            "properties": {"question": {"type": "string"},
                           "look_in": {"type": "array", "items": {"type": "string"}}},
            "required": ["question", "look_in"]}},
        "libraries": {"type": "array", "items": {"type": "string"}},
        "library_reasons": {"type": "string"},
        "enough_when": {"type": "string"},
    },
    "required": ["question", "approach", "sub_questions", "libraries", "enough_when"],
}

PLANNER_PROMPT = """You plan legal research. Read the research brief and decide how to research it well.

RESEARCH BRIEF
{brief}
{document}
LIBRARIES YOU MAY SEARCH (curated collections; a match near 0.5 or above means the library holds closely related material)
{libraries}

OUTSIDE SOURCES
{outside}

Write the plan:
- question: the research question the brief asks, stated precisely in one or two sentences.
- approach: two to four sentences on how to research it: what kinds of authority matter (cases, statutes, regulations, scholarship, religious or philosophical texts, ...), which jurisdictions and period, and what to compare.
- sub_questions: two to {max_sub} questions that together answer it, each answerable by research, none overlapping. For each, look_in lists where to look: library names from the list above{look_in_extra}.
- libraries: the libraries worth searching, by exact name. Leave out a library that cannot help; choose none if none can.
- library_reasons: one sentence on why those libraries, and why any others were left out.
- enough_when: what evidence would be enough to answer well, so the research knows when to stop.

Answer only with JSON."""


def _outside_text(databases: bool, web: bool) -> str:
    lines = []
    lines.append("- databases: public legal databases (CourtListener case law, the Federal Register, the eCFR, "
                 "govinfo, SEC EDGAR) " + ("ARE allowed" if databases else "are NOT allowed"))
    lines.append("- web: open-web search and reading public pages "
                 + ("ARE allowed" if web else "are NOT allowed"))
    return "\n".join(lines)


def normalize_plan(raw: dict[str, Any], candidates: list[str], databases: bool, web: bool,
                   brief: str) -> dict[str, Any]:
    """A plan as the run will use it: names checked against what is allowed, sub-questions
    numbered and capped, the brief's web addresses attached. Also applied to a plan the
    researcher edited. Pure."""
    allowed_places = set(candidates) | ({"databases"} if databases else set()) | ({"web"} if web else set())
    libraries = [n for n in dict.fromkeys(raw.get("libraries") or []) if n in candidates]
    subs = []
    for item in raw.get("sub_questions") or []:
        if isinstance(item, str):
            item = {"question": item, "look_in": []}
        q = " ".join(str(item.get("question") or "").split())
        if not q:
            continue
        look = [x for x in dict.fromkeys(item.get("look_in") or []) if x in allowed_places]
        subs.append({"id": f"Q{len(subs) + 1}", "question": q, "look_in": look})
        if len(subs) >= settings.research_max_sub_questions:
            break
    # A library a sub-question looks in is searched, even if the list above left it out.
    for s in subs:
        libraries += [x for x in s["look_in"] if x in candidates and x not in libraries]
    urls = clean_urls(raw["urls"]) if "urls" in raw else urls_in(brief)
    return {
        "question": " ".join(str(raw.get("question") or "").split()) or brief[:500],
        "approach": str(raw.get("approach") or "").strip(),
        "sub_questions": subs or [{"id": "Q1", "question": brief[:500], "look_in": []}],
        "libraries": libraries,
        "library_reasons": str(raw.get("library_reasons") or "").strip(),
        "enough_when": str(raw.get("enough_when") or "").strip(),
        "urls": urls,
        "use_databases": databases,
        "use_web": web,
    }


async def draft_plan(router: LLMRouter, brief: str, settings_: dict[str, Any],
                     document_title: str = "", document_text: str = "") -> dict[str, Any]:
    """Draft a plan for a brief: profile the candidate libraries, then ask the planner."""
    if not brief.strip():
        raise ValueError("This experiment has no research brief: its description is empty.")
    web = settings_["use_web"] and web_search_ready()
    databases = settings_["use_databases"]
    candidates = await candidate_libraries(settings_["choose_libraries"], settings_["libraries"])
    profiles = await library_profiles(router, candidates, brief)
    document = ""
    if document_text.strip():
        document = (f"\nA DOCUMENT IS GIVEN WITH THIS RUN: “{document_title or 'Untitled'}”. "
                    f"Its opening:\n{document_text.strip()[:3000]}\n")
    extra = ", or “databases” or “web” where those are allowed" if (databases or web) else ""
    prompt = PLANNER_PROMPT.format(
        brief=brief.strip(), document=document,
        libraries="\n".join(_profile_text(p) for p in profiles) or "(no libraries hold anything)",
        outside=_outside_text(databases, web), max_sub=settings.research_max_sub_questions,
        look_in_extra=extra)
    t0 = time.monotonic()
    res = await router.chat([{"role": "user", "content": prompt}], model=settings.agent_model,
                            think=True, format=PLAN_SCHEMA, num_ctx=32768, temperature=0.2,
                            options={"num_predict": 6000}, timeout=600)
    try:
        raw = json.loads(res.text)
    except json.JSONDecodeError as e:
        raise ValueError("the planner did not return a readable plan; draft it again") from e
    plan = normalize_plan(raw if isinstance(raw, dict) else {}, [p["name"] for p in profiles],
                          databases, web, brief)
    plan["profiles"] = profiles
    plan["candidates"] = [p["name"] for p in profiles]
    plan["model"] = settings.agent_model
    plan["drafted_in_s"] = round(time.monotonic() - t0, 1)
    plan["web_unavailable"] = settings_["use_web"] and not web_search_ready()
    return plan


# ---------------------------------------------------------------- drafting in the background

_jobs: dict[str, dict[str, Any]] = {}
_JOB_TTL_S = 3600


def start_draft(router: LLMRouter, brief: str, settings_: dict[str, Any],
                document_title: str = "", document_text: str = "") -> str:
    """Start drafting a plan; returns a job id to poll with draft_status."""
    now = time.time()
    for key in [k for k, j in _jobs.items() if now - j["started"] > _JOB_TTL_S]:
        _jobs.pop(key, None)
    job_id = uuid.uuid4().hex
    job: dict[str, Any] = {"status": "drafting", "started": now, "plan": None, "error": None}
    _jobs[job_id] = job

    async def run() -> None:
        try:
            job["plan"] = await draft_plan(router, brief, settings_, document_title, document_text)
            job["status"] = "ready"
        except Exception as e:  # noqa: BLE001 - shown to the researcher, who can try again
            log.warning("plan draft failed: %s", e)
            job["status"], job["error"] = "failed", str(e) or type(e).__name__

    job["task"] = asyncio.create_task(run())
    return job_id


def draft_status(job_id: str) -> dict[str, Any] | None:
    job = _jobs.get(job_id)
    if job is None:
        return None
    return {"status": job["status"], "plan": job["plan"], "error": job["error"],
            "elapsed_s": round(time.time() - job["started"])}


