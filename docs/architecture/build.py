"""Build the system architecture note -- a full-page diagram plus a component reference.

The diagram is drawn, not embedded: every box, rule and arrow is vector output from the
layout below, so the note stays crisp at any zoom and has no image dependency. Keep the
boxes in step with the code they name -- each one cites the module it describes, and the
"Where each box lives" table at the end is the index tying the two together.

The generators are not dependencies of the app, so run this from a throwaway environment:

    uv venv /tmp/docenv && uv pip install --python /tmp/docenv/bin/python reportlab
    /tmp/docenv/bin/python docs/architecture/build.py

Output goes to src/ailawlab/web/static/docs/, where the web portal links to it.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "ailawlab" / "web" / "static" / "docs"
STEM = "architecture"

TITLE = "AI Law Lab: System Architecture"
SUBTITLE = "How a request becomes a traced, reproducible experiment run"
AUTHOR = "Jian Gong"
AFFILIATION = "AI Law Lab, University of Wyoming"
DATELINE = "Architecture note, September 2026. Describes the software at commit f689797."
FOOTER = "AI Law Lab · Architecture note"

# Palette taken from the portal's own CSS custom properties (web/static/style.css), so the
# note and the site it is linked from read as one thing: blue = application, purple =
# inference, green = durable state, grey = anything outside the university's hardware.
INK = "#23221f"
MUTED = "#6d6a63"
LINE = "#d9d4c9"
KINDS = {
    "plain": ("#ffffff", "#c9c3b6", INK),
    "app":   ("#e6f1fb", "#9dc2e4", "#14497e"),
    "llm":   ("#eeedfe", "#b3aeea", "#3f3894"),
    "data":  ("#e1f5ee", "#8ecdb8", "#0b5844"),
    "ext":   ("#f3f1ec", "#c2bcaf", "#5a564e"),
}


def t(text: str) -> str:
    """Collapse a triple-quoted paragraph to one line."""
    return " ".join(text.split())


# --------------------------------------------------------------------------- diagram

# One entry per horizontal band, top to bottom. A band is as tall as its fullest box needs
# to be, with `h` as a floor, so editing the text below never silently clips it; if the page
# still runs short, every band and gap is scaled alike. `w` is a relative width in the band.
BANDS = [
    {"label": "People", "h": 22, "boxes": [
        {"kind": "plain", "title": "Faculty, students, and external research partners",
         "body": "A browser is the whole client. Nothing below leaves University of Wyoming "
                 "hardware except where marked."},
    ]},
    {"label": "Front door", "h": 46, "boxes": [
        {"kind": "app", "title": "Apache httpd · datahive.uwyo.edu", "w": 1.0,
         "body": "TLS with the campus certificate, HSTS, CSP frame-ancestors. Shares the Open "
                 "OnDemand vhost. ProxyPass /ai_law_lab/ passes the path through unstripped; "
                 "flushpackets=on keeps the trace stream unbuffered; timeout 900 s covers a "
                 "model call that is still thinking."},
        {"kind": "app", "title": "uvicorn · 127.0.0.1:8088", "w": 1.0,
         "body": "ai-law-lab.service under systemd, run with root_path=/ai_law_lab and "
                 "--proxy-headers. Bound to loopback: the reverse proxy is the only way in."},
    ]},
    {"label": "Web app", "h": 60, "boxes": [
        {"kind": "app", "title": "Pages · web/app.py, views.py",
         "body": "Server-rendered Jinja2 for the dashboard, builder, experiment, run and "
                 "source pages. views.py renders stored JSON config as plain language: a "
                 "lawyer reads settings, not a config file."},
        {"kind": "app", "title": "Experiment and cast API",
         "body": "Markdown agent and experiment files (agent_spec.py), AI-drafted casts "
                 "with real names substituted (cast_assistant.py), and reading a link, PDF "
                 "or pasted text (source_material.py)."},
        {"kind": "app", "title": "Live run trace · SSE",
         "body": "/api/runs/{id}/stream pushes new run_events rows until the run ends. A "
                 "poll, not LISTEN/NOTIFY: events arrive seconds apart, and no viewer holds "
                 "a connection of its own."},
        {"kind": "app", "title": "Legal sources · sources.py",
         "body": "Search and ingest across five providers, plus direct PDF upload. A click "
                 "ingests at most ten documents: each one holds embedding slots on the "
                 "cluster while it is chunked."},
    ]},
    {"label": "Experiments", "h": 38, "boxes": [
        {"kind": "app", "title": "Experiment manager · experiments.py",
         "body": "Defines an experiment, then at launch copies its configuration into the run "
                 "(runs.config_snapshot) and executes it in a background task, returning the "
                 "run id at once. Editing an experiment later never rewrites the conditions a "
                 "past result was produced under. A cast that cannot run fails before the "
                 "first model call, not halfway through a scene."},
    ]},
    {"label": "Graphs", "h": 72, "boxes": [
        {"kind": "llm", "title": "document_analysis",
         "body": "plan → analyze → ground → synthesize.  Sub-questions are answered "
                 "concurrently against the document, grounded in the corpus, then synthesized "
                 "with numbered citations. A citation marker pointing past the retrieved "
                 "passages is recorded as unsupported, not dropped."},
        {"kind": "llm", "title": "agentic_workflow",
         "body": "reason ⇄ act.  A ReAct loop over gemma4's native function calling: tool "
                 "calls arrive as structured fields on the message, not scraped out of text, "
                 "and the loop returns to reason for as long as the model keeps asking for "
                 "tools."},
        {"kind": "llm", "title": "roleplay",
         "body": "moderator → speak → moderator, then assess.  Three model calls a turn: "
                 "the speaker's private ledger, the reply, and the moderator's verdict on who "
                 "goes next and whether the scene is over. The rules and the studies behind "
                 "them are in graphs/roleplay_policy.py."},
    ]},
    {"label": "Services", "h": 54, "boxes": [
        {"kind": "llm", "title": "Corpus · rag.py",
         "body": "Chunks of 2400 characters overlapping by 300, embedded and retrieved "
                 "top-6 above cosine 0.35. Page anchors let a citation point where a human "
                 "can check it."},
        {"kind": "llm", "title": "Tools · tools.py",
         "body": "The registry the agentic loop is given: corpus search and safe "
                 "arithmetic. Network tools are gated behind allow_network, and none are "
                 "installed."},
        {"kind": "llm", "title": "Memory · memory.py",
         "body": "Verbatim recent turns in a token budget; overflow is summarized, "
                 "embedded and recalled by similarity. Scoped to (run_id, agent_id): one "
                 "party's reasoning cannot reach another's."},
        {"kind": "llm", "title": "Tracer · tracing.py",
         "body": "One append-only row per model call, tool call, retrieval and node "
                 "transition, with queue wait and model time in separate columns and the "
                 "reasoning trace in its own."},
    ]},
    {"label": "Router", "h": 42, "boxes": [
        {"kind": "llm", "title": "LLM router · router.py — one bounded queue for the cluster",
         "body": "Capacity is 16 slots per healthy host, 32 across the cluster: measured "
                 "throughput stops improving there while latency keeps climbing, so admitting "
                 "more would buy nothing and destroy the latency signal in the results. "
                 "Waiting here is the backpressure, and that wait is reported apart from model "
                 "time. Admitted work goes to the least-loaded healthy host; hosts are probed "
                 "every 30 s and parked after three consecutive failures."},
    ]},
    {"label": "Sparks", "h": 36, "boxes": [
        {"kind": "llm", "title": "test-spark3 · Ollama",
         "body": "NVIDIA DGX Spark (GB10, 20 cores, 121 GB unified memory). gemma4:latest for "
                 "chat, nomic-embed-text for embeddings. Reachable by Tailscale name only."},
        {"kind": "llm", "title": "test-spark4 · Ollama",
         "body": "The second unit, identical. The lab-subnet addresses 10.99.252.31/.32 are "
                 "not routable from the orchestrator and appear in no configuration."},
    ]},
    {"label": "State", "h": 50, "boxes": [
        {"kind": "data", "w": 2.3,
         "title": "PostgreSQL 17 + pgvector · ailawlab-db container on 127.0.0.1:5433",
         "body": "experiments · runs, each holding the config it ran under · run_events, the "
                 "append-only trace · citations with a grounded / unsupported / unchecked "
                 "verdict · documents and chunks, 768-dimension vectors indexed with HNSW · "
                 "memory_messages and memory_long_term, scoped per agent. An async pool, not "
                 "one shared connection: about 32 agents can be in flight at once."},
        {"kind": "ext", "w": 1.0, "dashed": True,
         "title": "Online legal sources · outbound HTTPS",
         "body": "CourtListener, the Federal Register, the eCFR, govinfo and SEC EDGAR, "
                 "plus any page or PDF a cast is drafted from. Queries for public material "
                 "only — never an uploaded document."},
    ]},
]

# Arrows down the middle of the stack: (upper band index, lower band index). The chain is
# the path a launch actually takes; it deliberately stops at the hosts, because nothing
# flows from a Spark into Postgres -- that edge is the dashed spine on the left instead.
FLOW = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

GUTTER, LEFT_CHANNEL, RIGHT_CHANNEL = 44.0, 14.0, 16.0
BAND_GAP, BOX_GAP = 6.8, 10.0
TITLE_SIZE, BODY_SIZE, BODY_LEAD = 7.6, 6.1, 7.3


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    from reportlab.pdfbase import pdfmetrics

    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and pdfmetrics.stringWidth(trial, font, size) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _arrowhead(c, x: float, y: float, dx: float, dy: float, size: float = 3.4) -> None:
    """Filled triangle at (x, y) pointing along (dx, dy)."""
    import math

    angle = math.atan2(dy, dx)
    left = angle + math.radians(148)
    right = angle - math.radians(148)
    p = c.beginPath()
    p.moveTo(x, y)
    p.lineTo(x + size * math.cos(left), y + size * math.sin(left))
    p.lineTo(x + size * math.cos(right), y + size * math.sin(right))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def _polyline(c, points: list[tuple[float, float]], color, *, dashed: bool = False) -> None:
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(0.8)
    c.setLineJoin(1)
    if dashed:
        c.setDash(2.4, 2.4)
    path = c.beginPath()
    path.moveTo(*points[0])
    for pt in points[1:]:
        path.lineTo(*pt)
    c.drawPath(path, stroke=1, fill=0)
    c.setDash()
    (x0, y0), (x1, y1) = points[-2], points[-1]
    _arrowhead(c, x1, y1, x1 - x0, y1 - y0)
    c.restoreState()


PAD = 5.5


def _box_height(spec: dict, w: float) -> float:
    """How tall this box must be for its title and body to fit at `w` points wide."""
    inner = w - 2 * PAD
    title = len(_wrap(spec["title"], "Sans-Bold", TITLE_SIZE, inner))
    lines = len(_wrap(spec.get("body", ""), "Sans", BODY_SIZE, inner))
    return PAD + title * (TITLE_SIZE + 1.6) + (1.6 + lines * BODY_LEAD if lines else 0) + PAD - 1


def _measure(main_w: float) -> dict[str, list]:
    """Box widths and band heights for the whole stack, worked out before anything is drawn."""
    widths, heights = [], []
    for band in BANDS:
        weights = [b.get("w", 1.0) for b in band["boxes"]]
        unit = (main_w - BOX_GAP * (len(band["boxes"]) - 1)) / sum(weights)
        row = [unit * weight for weight in weights]
        widths.append(row)
        heights.append(max([band["h"]] + [_box_height(spec, w)
                                          for spec, w in zip(band["boxes"], row)]))
    return {"widths": widths, "heights": heights}


def _channel_label(c, colors, x: float, y0: float, y1: float, text: str) -> None:
    """Name a dashed edge, set along its vertical run in the margin beside the stack."""
    c.saveState()
    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Sans", 6.0)
    c.translate(x, (y0 + y1) / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, text)
    c.restoreState()


def _box(c, x: float, y: float, w: float, h: float, spec: dict) -> None:
    """One labelled box. `y` is its bottom edge."""
    from reportlab.lib import colors

    fill, edge, ink = (colors.HexColor(v) for v in KINDS[spec["kind"]])
    c.saveState()
    c.setFillColor(fill)
    c.setStrokeColor(edge)
    c.setLineWidth(0.8)
    if spec.get("dashed"):
        c.setDash(2.6, 2.4)
    c.roundRect(x, y, w, h, 3.2, stroke=1, fill=1)
    c.setDash()

    pad = PAD
    inner = w - 2 * pad
    ty = y + h - pad - TITLE_SIZE + 1.4
    c.setFillColor(ink)
    c.setFont("Sans-Bold", TITLE_SIZE)
    for line in _wrap(spec["title"], "Sans-Bold", TITLE_SIZE, inner):
        c.drawString(x + pad, ty, line)
        ty -= TITLE_SIZE + 1.6

    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Sans", BODY_SIZE)
    by = ty - 1.6
    for line in _wrap(spec.get("body", ""), "Sans", BODY_SIZE, inner):
        if by < y + 3:                      # never spill out of the box
            break
        c.drawString(x + pad, by, line)
        by -= BODY_LEAD
    c.restoreState()


def draw_diagram(c, width: float, height: float) -> None:
    """Draw the whole architecture at the canvas origin, within width x height points.

    Kept a plain function rather than a Flowable subclass so this module imports without
    reportlab installed; build_pdf() wraps it in a flowable when it actually renders.
    """
    from reportlab.lib import colors

    left = GUTTER + LEFT_CHANNEL
    main_w = width - left - RIGHT_CHANNEL
    spine_x = left - LEFT_CHANNEL / 2
    channel_x = left + main_w + RIGHT_CHANNEL / 2

    plan = _measure(main_w)
    natural = sum(plan["heights"]) + BAND_GAP * (len(BANDS) - 1)
    scale = min(1.0, height / natural)
    gap = BAND_GAP * scale

    # Place every band top-down, remembering each box's rectangle for the arrows.
    placed: list[dict] = []
    y = height
    for band, widths, h in zip(BANDS, plan["widths"], plan["heights"]):
        h *= scale
        y -= h
        x = left
        rects = []
        for spec, w in zip(band["boxes"], widths):
            _box(c, x, y, w, h, spec)
            rects.append((x, y, w, h))
            x += w + BOX_GAP
        placed.append({"label": band["label"], "y": y, "h": h, "rects": rects})
        y -= gap

    # Rotated tier labels in the left gutter.
    c.saveState()
    c.setFillColor(colors.HexColor(MUTED))
    c.setFont("Sans-Bold", 6.4)
    for band in placed:
        c.saveState()
        c.translate(GUTTER - 13, band["y"] + band["h"] / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, band["label"].upper())
        c.restoreState()
    c.restoreState()

    arrow = colors.HexColor("#8e887c")

    # The execution path: a short arrow from the middle of one band to the next.
    for upper, lower in FLOW:
        top, bottom = placed[upper], placed[lower]
        x_mid = left + main_w / 2
        _polyline(c, [(x_mid, top["y"] - 1), (x_mid, bottom["y"] + bottom["h"] + 1)], arrow)

    # The router fans out to each host individually -- that split is the whole point of
    # the tier, so it gets two arrows rather than one down the middle.
    router = placed[6]
    for hx, hy, hw, hh in placed[7]["rects"]:
        _polyline(c, [(hx + hw / 2, router["y"] - 1), (hx + hw / 2, hy + hh + 1)], arrow)

    # Left spine: everything above persists through the shared services into Postgres.
    services, state = placed[5], placed[8]
    px, py, _pw, ph = state["rects"][0]
    _polyline(c, [
        (services["rects"][0][0], services["y"] + services["h"] / 2),
        (spine_x, services["y"] + services["h"] / 2),
        (spine_x, py + ph + gap / 2),
        (px + 30, py + ph + gap / 2),
        (px + 30, py + ph + 1),
    ], arrow, dashed=True)
    _channel_label(c, colors, spine_x - 1.5, py + ph, services["y"], "trace \u00b7 corpus \u00b7 memory")

    # Right channel: the one edge that leaves campus.
    sx, sy, sw, sh = placed[2]["rects"][3]
    ex, ey, ew, eh = state["rects"][1]
    _polyline(c, [
        (sx + sw + 1, sy + sh / 2),
        (channel_x, sy + sh / 2),
        (channel_x, ey + eh + gap / 2),
        (ex + ew - 30, ey + eh + gap / 2),
        (ex + ew - 30, ey + eh + 1),
    ], arrow, dashed=True)
    _channel_label(c, colors, channel_x + 6.4, ey + eh, sy, "leaves campus")


# --------------------------------------------------------------------------- note text

SECTIONS: list[tuple[str, object]] = [
    ("h1", "Reading the diagram"),
    ("body", t("""
        Each band is a tier, and the stack reads top to bottom in the order a request passes
        through it. Solid arrows are the path a launched run takes. Dashed arrows are the two
        edges that are not part of that path: what every tier persists, and the one direction
        in which traffic leaves the university. Colour carries the same meaning as on the
        portal itself -- blue for the application, purple for anything that consumes model
        capacity, green for durable state, grey for hardware and services outside the lab.
    """)),
    ("bullets", [
        t("""**Two hops, one process.** Apache terminates TLS and proxies to a uvicorn worker
             bound to loopback. There is exactly one application process; concurrency inside it
             is asyncio, not workers, which is why a single bounded queue can speak for the
             whole cluster."""),
        t("""**The orchestration band is three graphs, not three services.** They share the
             router, corpus, memory and tracer beneath them, and differ only in their nodes."""),
        t("""**Everything narrows at the router.** Every model call in every mode, including
             embeddings during ingestion, passes through one admission point. That is what makes
             a timing number from one run comparable with a timing number from another."""),
    ]),

    ("h1", "1 · Front door"),
    ("body", t("""
        The lab is served at `https://datahive.uwyo.edu/ai_law_lab/` from the same Apache vhost
        as the Open OnDemand portal. Three details in that configuration are load-bearing. The
        proxy passes the mount path through **unstripped**, because the application runs with
        `root_path=/ai_law_lab` and strips the prefix itself; proxying to a bare root would 404
        its `/static` mount. `flushpackets=on` keeps the live trace stream from being buffered
        into uselessness. And `timeout 900` covers a model call that emits nothing while it is
        thinking -- with the default the browser would lose a role-play turn mid-generation.
    """)),
    ("body", t("""
        `X-Forwarded-Proto` is set on the location and uvicorn runs with `--proxy-headers`:
        without both, the application would build pages from an http-scheme request and emit
        `http://` absolute URLs on an HSTS-pinned site. The unit file deliberately has no
        `EnvironmentFile` -- `.env` holds JSON whose inner quotes systemd's parser strips --
        so pydantic-settings reads it directly, relative to the working directory.
    """)),

    ("h1", "2 · Web application"),
    ("body", t("""
        The interface is server-rendered Jinja2 with small amounts of JavaScript for the cast
        builder, not a single-page application. `views.py` exists so that what a page shows is
        a plain-language rendering of stored settings rather than the stored settings: an
        experiment page says how long a role-play is likely to take and shows each cast member
        as a card, because the people configuring experiments here are lawyers.
    """)),
    ("body", t("""
        A run's trace reaches the browser over server-sent events. The endpoint polls
        `run_events` once a second rather than using `LISTEN`/`NOTIFY`: events arrive seconds
        apart, so a one-second poll is well inside budget and avoids holding a dedicated
        database connection open for every viewer.
    """)),

    ("h1", "3 · Experiment manager"),
    ("body", t("""
        `experiments.py` is the seam between the web tier and the graphs. Launching copies the
        experiment's configuration into `runs.config_snapshot` and only then starts work, so
        editing an experiment afterwards cannot rewrite the conditions a past result was
        produced under. Execution happens in a background task and the run id comes back
        immediately, which is what lets a browser watch an hour-long scene without holding a
        request open. A failure is recorded as a status with its trace intact, not raised: a
        failed experiment is a result.
    """)),
    ("body", t("""
        Before a role-play starts, the cast is checked. Without that, a duplicate agent id or
        a moderator pick matching no agent surfaces mid-run as a bare `StopIteration`, long
        after the cause was knowable. Lesser problems are written to the trace instead.
    """)),

    ("h1", "4 · Orchestration"),
    ("body", t("""
        Each mode is a LangGraph state graph over a typed state dictionary. Live services --
        the router, tracer and corpus -- travel in `config.configurable` rather than in state,
        because state is checkpointed and connections are not serializable.
    """)),
    ("table", {
        "header": ["Mode", "Nodes", "What ends it"],
        "rows": [
            ["document_analysis", "plan → analyze → ground → synthesize",
             "The pipeline is linear and runs once."],
            ["agentic_workflow", "reason ⇄ act",
             "The model stops emitting tool calls, or the iteration ceiling is reached."],
            ["roleplay", "moderator → speak → moderator … → assess",
             ("The moderator closes the scene, or max_turns is hit. Never before everyone "
              "has spoken twice.")],
        ],
        "mono_first": True,
        "caption": t("""
            The recursion limit handed to a role-play is three times its turn ceiling plus
            twenty, since each turn costs a moderator hop and a speak hop.
        """),
    }),

    ("h1", "5 · Shared services"),
    ("bullets", [
        t("""**Corpus** (`rag.py`) ingests a document once -- identity is a SHA-256 of its text
             -- chunks it at 2400 characters with 300 of overlap, embeds in batches so one large
             document cannot hold a cluster slot indefinitely, and retrieves the top six chunks
             above cosine 0.35 with page anchors attached."""),
        t("""**Tools** (`tools.py`) is the registry the agentic loop is given: corpus search and
             a safe arithmetic evaluator. `allow_network` gates network-touching tools, and none
             are installed, so that switch currently changes nothing."""),
        t("""**Memory** (`memory.py`) keeps a verbatim buffer inside a token budget and folds
             the overflow into model-written summaries that are embedded and recalled by
             similarity. Its scope is `(run_id, agent_id)`: in a negotiation, a shared buffer
             would leak one side's private reasoning into the other's context and quietly
             invalidate the exercise."""),
        t("""**Tracer** (`tracing.py`) writes one sequenced row per model call, tool call,
             retrieval and node transition. Queue wait and model time go in separate columns
             because the benchmark showed them diverging threefold under load; reported
             together, a run that was merely waiting looks like a run that was slow."""),
    ]),

    ("h1", "6 · Inference and the queue"),
    ("body", t("""
        Two DGX Spark units serve `gemma4:latest` through Ollama, reachable by Tailscale name
        only. The router in front of them is the piece of this architecture most worth
        explaining, because the obvious design is wrong. Benchmarking both hosts on 2026-07-23
        found that fifty concurrent requests all complete -- nothing errors -- while aggregate
        throughput stops improving at about sixteen:
    """)),
    ("table", {
        "header": ["Concurrency", "Aggregate tok/s", "p50 latency"],
        "rows": [["1", "18", "0.66 s"], ["16", "160", "1.87 s"],
                 ["32", "131", "3.29 s"], ["50", "160", "5.22 s"]],
        "mono_first": True,
        "caption": t("""
            Past sixteen, the extra requests are queueing inside Ollama, not running.
            Round-robin dispatch would push fifty requests at a host and then report the
            resulting latency as though it were model speed.
        """),
    }),
    ("body", t("""
        So the router keeps a single global semaphore sized to the healthy hosts' combined
        capacity -- sixteen slots each, thirty-two in total -- and queues everything else
        itself. Waiting there *is* the backpressure, and the time spent is what gets recorded
        as `queue_wait_ms`. Admitted work goes to the least-loaded healthy host, which
        self-balances better than round-robin when one agent is summarizing a two-hundred-page
        contract and another is emitting a one-line reply. Hosts are probed every thirty
        seconds, parked after three consecutive failures, and the global capacity is rebuilt
        when the healthy count changes.
    """)),
    ("body", t("""
        One model property shapes the tiers above: gemma4 reasons before answering by default.
        That reasoning is exactly what a research setting wants, and it is captured to its own
        column -- but it costs roughly an order of magnitude more output tokens on short
        answers. Nodes doing mechanical work (turn-taking, summarizing, planning) therefore
        pass `think=False`; nodes doing substantive legal reasoning pass `think=True`.
    """)),

    ("h1", "7 · State"),
    ("body", t("""
        One PostgreSQL 17 database with pgvector, in a container on loopback, holds everything
        a run produces. It is reached through an async pool rather than a single connection:
        the design this memory tier was ported from held one SQLite connection, which
        serializes hard under the roughly thirty-two concurrent agents this cluster sustains.
        Vector columns are 768-dimensional and indexed with HNSW, chosen over IVFFlat because
        the corpus grows incrementally and HNSW needs no retraining.
    """)),
    ("table", {
        "header": ["Tables", "What they hold"],
        "rows": [
            ["experiments, runs", ("Definitions, and per-run snapshots of the "
                                   "configuration each result was produced under.")],
            ["run_events, citations", ("The append-only trace, and asserted citations with "
                                       "a grounded / unsupported / unchecked verdict.")],
            ["documents, chunks", "The corpus: text, page anchors, and embeddings."],
            ["memory_messages,\nmemory_long_term", ("Short-term buffers and summarized "
                                                    "long-term memory, per (run, agent), "
                                                    "with provenance back to a trace event "
                                                    "or chunk.")],
        ],
        "mono_first": True,
        "caption": t("""
            Everything is addressable by `run_id`, and every foreign key cascades from the run,
            so a result can be audited or removed as a unit.
        """),
    }),

    ("h1", "8 · What leaves campus"),
    ("body", t("""
        Inference is local, so documents and queries stay on university hardware. The single
        exception is drawn dashed on the right of the diagram: outbound requests to public
        legal sources -- CourtListener, the Federal Register, the eCFR, govinfo and SEC EDGAR
        -- and to a page or PDF a cast is being drafted from. These carry a search term or a
        URL, never an uploaded document.
    """)),
    ("body", t("""
        That outbound path is guarded, because this server can reach the Sparks, Postgres and
        the university network. Only `http`/`https` links whose host resolves exclusively to
        public addresses are fetched, and every redirect is re-checked the same way. A source
        is capped at 12,000 words before it reaches a prompt. Real people and private
        organizations in a source are renamed before a cast is drafted from it, and the same
        substitution is applied to the model's output: a draft invents bottom lines and
        confidential facts, and attaching those to a real company is the failure worth
        engineering against.
    """)),

    ("h1", "Where each box lives"),
    ("table", {
        "header": ["Box on the diagram", "Source"],
        "rows": [
            ["Pages, SSE, HTTP API", "src/ailawlab/web/app.py, web/views.py, web/templates/"],
            ["Experiment and cast API", "agent_spec.py, cast_assistant.py, source_material.py"],
            ["Legal sources", "sources.py"],
            ["Experiment manager", "experiments.py"],
            ["Orchestration", ("graphs/document_analysis.py, graphs/agentic_workflow.py,\n"
                               "graphs/roleplay.py, graphs/roleplay_policy.py, "
                               "graphs/state.py")],
            ["Corpus, tools, memory, tracer", "rag.py, tools.py, memory.py, tracing.py"],
            ["LLM router", "router.py"],
            ["State", "db.py, db/schema.sql, docker-compose.yml"],
            ["Front door", ("/etc/apache2/sites-available/ood-portal.conf,\n"
                            "/etc/systemd/system/ai-law-lab.service, .env")],
        ],
        "mono_first": False,
        "caption": t("""
            Settings quoted throughout this note are defaults from `config.py`; every one is
            overridable by an `AILAWLAB_`-prefixed environment variable.
        """),
    }),

    ("h1", "Rebuilding this note"),
    ("code", {
        "lines": [
            "uv venv /tmp/docenv",
            "uv pip install --python /tmp/docenv/bin/python reportlab",
            "/tmp/docenv/bin/python docs/architecture/build.py",
        ],
        "caption": t("""
            The diagram is laid out in `docs/architecture/build.py`; the boxes are data at the
            top of that file. Re-run it after a change to the tiers, and re-run
            `scripts/bench_sparks.py` after any hardware or Ollama change, since the numbers in
            section 6 are what the router's slot count is set from.
        """),
    }),
]


def _pdf_markup(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font name="Mono">\1</font>', text)
    return re.sub(r"(https?://[^\s<]+)", r'<link href="\1" color="#185fa5">\1</link>', text)


def build_pdf(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate,
        Flowable,
        Frame,
        KeepTogether,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        Preformatted,
        Spacer,
        Table,
        TableStyle,
    )

    fonts = Path("/usr/share/fonts/truetype/dejavu")
    faces = {"Serif": "DejaVuSerif.ttf", "Serif-Bold": "DejaVuSerif-Bold.ttf",
             "Serif-Italic": "DejaVuSerif-Italic.ttf",
             "Serif-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
             "Sans": "DejaVuSans.ttf", "Sans-Bold": "DejaVuSans-Bold.ttf",
             "Mono": "DejaVuSansMono.ttf"}
    for name, file in faces.items():
        target = fonts / file
        if not target.exists():                      # italics ship in fonts-dejavu-extra
            target = fonts / ("DejaVuSerif-Bold.ttf" if "Bold" in name else "DejaVuSerif.ttf")
        pdfmetrics.registerFont(TTFont(name, str(target)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                                  italic="Serif-Italic", boldItalic="Serif-BoldItalic")

    ink, muted, rule = colors.HexColor(INK), colors.HexColor(MUTED), colors.HexColor(LINE)
    body = ParagraphStyle("body", fontName="Serif", fontSize=9.6, leading=13.4,
                          alignment=TA_JUSTIFY, textColor=ink, spaceAfter=6)
    styles = {
        "h1": ParagraphStyle("h1", parent=body, fontName="Serif-Bold", fontSize=11.2,
                             leading=14, spaceBefore=12, spaceAfter=5, alignment=TA_LEFT),
        "caption": ParagraphStyle("caption", parent=body, fontSize=8.4, leading=11,
                                  alignment=TA_LEFT, textColor=muted, spaceBefore=4),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.3, leading=10.6,
                               alignment=TA_LEFT, spaceAfter=0),
        "cell_mono": ParagraphStyle("cell_mono", parent=body, fontName="Mono", fontSize=7.6,
                                    leading=10.2, alignment=TA_LEFT, spaceAfter=0),
        "cell_head": ParagraphStyle("cell_head", parent=body, fontName="Serif-Bold",
                                    fontSize=8.3, leading=10.6, alignment=TA_LEFT, spaceAfter=0),
        "mono": ParagraphStyle("mono", fontName="Mono", fontSize=7.9, leading=11, textColor=ink),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2,
                                 spaceAfter=5),
    }

    def title_block(canvas, doc) -> None:
        """Header for the diagram page, drawn outside the frame so the art gets the rest."""
        w, h = landscape(letter)
        canvas.saveState()
        canvas.setFillColor(ink)
        canvas.setFont("Serif-Bold", 15.5)
        canvas.drawCentredString(w / 2, h - 45, TITLE)
        canvas.setFont("Serif-Italic", 10.2)
        canvas.setFillColor(muted)
        canvas.drawCentredString(w / 2, h - 59, SUBTITLE)
        canvas.setFont("Serif", 8.4)
        canvas.drawCentredString(w / 2, h - 71, f"{AUTHOR}  ·  {AFFILIATION}  ·  {DATELINE}")
        canvas.setStrokeColor(rule)
        canvas.setLineWidth(0.6)
        canvas.line(44, h - 80, w - 44, h - 80)
        canvas.restoreState()
        footer(canvas, doc, landscape(letter))

    def footer(canvas, doc, size=letter) -> None:
        canvas.saveState()
        canvas.setFont("Serif", 7.8)
        canvas.setFillColor(muted)
        canvas.drawString(44, 26, FOOTER)
        canvas.drawRightString(size[0] - 44, 26, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(str(path), pagesize=landscape(letter), title=TITLE,
                          author=f"{AUTHOR}, {AFFILIATION}", subject=SUBTITLE,
                          keywords="architecture, LangGraph, local inference, legal AI")
    lw, lh = landscape(letter)
    doc.addPageTemplates([
        PageTemplate(id="diagram", pagesize=landscape(letter), onPage=title_block,
                     frames=[Frame(44, 38, lw - 88, lh - 126, id="art",
                                   leftPadding=0, rightPadding=0,
                                   topPadding=0, bottomPadding=0)]),
        PageTemplate(id="notes", pagesize=letter, onPage=footer,
                     frames=[Frame(66, 52, letter[0] - 132, letter[1] - 118, id="col",
                                   leftPadding=0, rightPadding=0,
                                   topPadding=0, bottomPadding=0)]),
    ])

    class _DiagramFlowable(Flowable):
        def __init__(self, width: float, height: float):
            Flowable.__init__(self)
            self.width, self.height = width, height

        def wrap(self, avail_w: float, avail_h: float) -> tuple[float, float]:
            self.width = avail_w
            self.height = min(self.height, avail_h)
            return self.width, self.height

        def draw(self) -> None:
            draw_diagram(self.canv, self.width, self.height)

    story: list = [_DiagramFlowable(lw - 88, lh - 126),
                   NextPageTemplate("notes"), PageBreak()]
    for kind, value in SECTIONS:
        if kind in ("h1", "caption"):
            story.append(Paragraph(_pdf_markup(value), styles[kind]))
        elif kind == "body":
            story.append(Paragraph(_pdf_markup(value), body))
        elif kind == "bullets":
            story += [Paragraph(_pdf_markup(b), styles["bullet"], bulletText="•")
                      for b in value]
        elif kind == "table":
            rows = [[Paragraph(_pdf_markup(h), styles["cell_head"]) for h in value["header"]]]
            for row in value["rows"]:
                rows.append([
                    Paragraph(_pdf_markup(cell).replace("\n", "<br/>"),
                              styles["cell_mono" if j == 0 and value.get("mono_first")
                                     else "cell"])
                    for j, cell in enumerate(row)])
            widths = ([1.35, 2.05, 2.4] if len(value["header"]) == 3 else [1.6, 2.9])
            total = letter[0] - 132
            table = Table(rows, colWidths=[w / sum(widths) * total for w in widths],
                          repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, rule),
                ("LINEBELOW", (0, 1), (-1, -2), 0.3, colors.HexColor("#efece4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
            ]))
            story.append(KeepTogether([table, Paragraph(_pdf_markup(value["caption"]),
                                                        styles["caption"])]))
            story.append(Spacer(1, 4))
        elif kind == "code":
            box = Table([[Preformatted("\n".join(value["lines"]), styles["mono"])]],
                        colWidths=[letter[0] - 132], hAlign="LEFT")
            box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f5f0")),
                ("BOX", (0, 0), (-1, -1), 0.5, rule),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(KeepTogether([box, Paragraph(_pdf_markup(value["caption"]),
                                                      styles["caption"])]))

    doc.build(story)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{STEM}.pdf"
    build_pdf(target)
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
