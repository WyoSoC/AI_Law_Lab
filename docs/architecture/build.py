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
DATELINE = "Architecture note, October 2026. Describes the software at commit 74d2d40."
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
#
# The diagram follows the layering of the one in the project README: people, the web
# interface, the experiment manager, the three experiment modes, the services they share,
# the router and the cluster, and the record every run leaves. Each box says what the tier
# is for in a line or two; the detail is in the sections that follow it, not on the art.

ROWS = [
    {"id": "people", "label": "People", "h": 30, "boxes": [
        {"id": "users", "kind": "plain", "title": "Faculty, students and research partners",
         "lines": ["A web browser is the whole client."]},
    ]},
    {"id": "interface", "label": "Interface", "h": 44, "boxes": [
        {"id": "apache", "kind": "app", "w": 0.8, "title": "Apache · datahive.uwyo.edu",
         "lines": ["TLS, reverse proxy to /ai_law_lab"]},
        {"id": "portal", "kind": "app", "w": 1.5, "title": "Web portal · FastAPI",
         "lines": ["Experiment builder · legal sources and libraries ·",
                   "live run pages · references and PDF reports"]},
        {"id": "keycloak", "kind": "app", "w": 0.9, "title": "Keycloak · sign-in",
         "lines": ["UW sign-on, Google, Microsoft;", "approval and roles"]},
    ]},
    {"id": "manager", "label": "Experiments", "h": 34, "boxes": [
        {"id": "manager", "kind": "app", "title": "Experiment manager",
         "lines": ["Snapshots the configuration · pins every library to a version · "
                   "runs in the background and returns at once"]},
    ]},
    {"id": "modes", "label": "Modes", "h": 50, "boxes": [
        {"id": "doc", "kind": "llm", "title": "Document analysis",
         "lines": ["plan → analyze → ground → synthesize",
                   "numbered citations, checked afterwards"]},
        {"id": "agent", "kind": "llm", "title": "Agentic workflow",
         "lines": ["reason ⇄ act", "searches libraries, cites what it found"]},
        {"id": "roleplay", "kind": "llm", "title": "Role-play simulation",
         "lines": ["moderator → speak → … → assess",
                   "private memory, case files, exhibits"]},
    ]},
    {"id": "services", "label": "Services", "h": 50, "boxes": [
        {"id": "tools", "kind": "llm", "w": 0.85, "title": "External tools",
         "lines": ["library search · calculator", "network tools opt-in"]},
        {"id": "memory", "kind": "llm", "w": 0.85, "title": "Memory",
         "lines": ["private to each agent", "recent turns · summaries"]},
        {"id": "rag", "kind": "llm", "w": 1.15, "title": "Legal RAG · versioned libraries",
         "lines": ["several libraries searched together,", "page anchors, pinned versions"]},
        {"id": "online", "kind": "ext", "w": 1.05, "dashed": True, "title": "Online legal databases",
         "lines": ["CourtListener · Federal Register · eCFR", "govinfo · SEC EDGAR · web pages"]},
    ]},
    {"id": "router", "label": "Router", "h": 34, "boxes": [
        {"id": "router", "kind": "llm", "title": "LLM router · one bounded queue",
         "lines": ["16 slots per host · least-loaded dispatch · queue wait timed apart from "
                   "model time"]},
    ]},
    {"id": "cluster", "label": "Cluster", "h": 32, "boxes": [
        {"id": "spark3", "kind": "hw", "title": "test-spark3 · NVIDIA DGX Spark",
         "lines": ["Ollama · gemma4 (chat) · nomic-embed-text (embeddings)"]},
        {"id": "spark4", "kind": "hw", "title": "test-spark4 · NVIDIA DGX Spark",
         "lines": ["Ollama · gemma4 (chat) · nomic-embed-text (embeddings)"]},
    ]},
    {"id": "record", "label": "Record", "h": 46, "boxes": [
        {"id": "postgres", "kind": "data", "w": 1.0, "title": "PostgreSQL + pgvector",
         "lines": ["experiments and runs · libraries, versions", "and embeddings · memory · accounts"]},
        {"id": "logging", "kind": "data", "w": 1.45, "title": "Logging and evaluation engine",
         "lines": ["a trace of every call · each citation traced to its document,",
                   "library and version · references, exhibits, metrics"]},
    ]},
]

GUTTER = 50.0                  # row labels
LEFT_CHANNEL = 26.0            # model calls run down the left
RIGHT_CHANNEL = 30.0           # the trace runs down the right
BOX_GAP = 12.0
TITLE_SIZE, BODY_SIZE, BODY_LEAD = 8.0, 6.7, 8.2
PAD = 6.0

KINDS["hw"] = ("#f3f1ec", "#b8b2a4", "#3f3c36")


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


def _label(c, x: float, y: float, text: str, *, size: float = 6.2, angle: float = 0,
           anchor: str = "middle", color: str = MUTED, back: str | None = "#ffffff") -> None:
    """A small edge label, with a white backing so it reads across a line."""
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics

    w = pdfmetrics.stringWidth(text, "Sans", size)
    c.saveState()
    c.translate(x, y)
    c.rotate(angle)
    x0 = {"middle": -w / 2, "start": 0, "end": -w}[anchor]
    if back:
        c.setFillColor(colors.HexColor(back))
        c.rect(x0 - 1.5, -size * 0.3, w + 3, size * 1.15, stroke=0, fill=1)
    c.setFillColor(colors.HexColor(color))
    c.setFont("Sans", size)
    c.drawString(x0, 0, text)
    c.restoreState()


def _box(c, rect: tuple[float, float, float, float], spec: dict) -> None:
    from reportlab.lib import colors

    x, y, w, h = rect
    fill, stroke, ink = KINDS[spec["kind"]]
    c.saveState()
    c.setFillColor(colors.HexColor(fill))
    c.setStrokeColor(colors.HexColor(stroke))
    c.setLineWidth(0.8)
    if spec.get("dashed"):
        c.setDash(3, 2)
    c.roundRect(x, y, w, h, 4, stroke=1, fill=1)
    c.setDash()
    lines = [ln for raw in spec.get("lines", []) for ln in _wrap(raw, "Sans", BODY_SIZE, w - 2 * PAD)]
    block = TITLE_SIZE + 2.5 + BODY_LEAD * len(lines)
    if block > h - 4:
        raise ValueError(f"box {spec['id']!r} needs {block:.0f} pt, row allows {h - 4:.0f}")
    top = y + h / 2 + block / 2 - TITLE_SIZE
    c.setFillColor(colors.HexColor(ink))
    c.setFont("Sans-Bold", TITLE_SIZE)
    c.drawCentredString(x + w / 2, top, spec["title"])
    c.setFillColor(colors.HexColor(INK))
    c.setFont("Sans", BODY_SIZE)
    for i, line in enumerate(lines):
        c.drawCentredString(x + w / 2, top - 2.5 - BODY_LEAD * (i + 1), line)
    c.restoreState()


def draw_diagram(c, width: float, height: float) -> None:
    """Draw the whole architecture at the canvas origin, within width x height points.

    Kept a plain function rather than a Flowable subclass so this module imports without
    reportlab installed; build_pdf() wraps it in a flowable when it actually renders.
    """
    from reportlab.lib import colors

    left = GUTTER + LEFT_CHANNEL
    main_w = width - left - RIGHT_CHANNEL
    gap = (height - sum(r["h"] for r in ROWS)) / (len(ROWS) - 1)

    # Place the rows top-down; remember every box's rectangle (x, y, w, h) by id.
    R: dict[str, tuple[float, float, float, float]] = {}
    row_y: dict[str, tuple[float, float]] = {}
    y = height
    for row in ROWS:
        y -= row["h"]
        row_y[row["id"]] = (y, y + row["h"])
        weights = [b.get("w", 1.0) for b in row["boxes"]]
        free = main_w - BOX_GAP * (len(weights) - 1)
        x = left
        for box, wt in zip(row["boxes"], weights):
            bw = free * wt / sum(weights)
            R[box["id"]] = (x, y, bw, row["h"])
            x += bw + BOX_GAP
        y -= gap

    # Row labels and faint separators in the left gutter.
    rule = colors.HexColor(LINE)
    for row in ROWS:
        y0, y1 = row_y[row["id"]]
        c.saveState()
        c.setFillColor(colors.HexColor(MUTED))
        c.setFont("Sans-Bold", 6.4)
        c.translate(GUTTER - 30, (y0 + y1) / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, row["label"].upper())
        c.restoreState()

    for row in ROWS:
        for box in row["boxes"]:
            _box(c, R[box["id"]], box)

    def cx(i: str) -> float:
        return R[i][0] + R[i][2] / 2

    def top(i: str) -> float:
        return R[i][1] + R[i][3]

    def bottom(i: str) -> float:
        return R[i][1]

    def mid(i: str) -> float:
        return R[i][1] + R[i][3] / 2

    solid, soft = colors.HexColor("#6d6a63"), colors.HexColor("#9a958a")

    # People reach the portal through Apache; Keycloak vouches for who they are.
    _polyline(c, [(cx("apache"), bottom("users")), (cx("apache"), top("apache"))], solid)
    _polyline(c, [(R["apache"][0] + R["apache"][2], mid("apache")), (R["portal"][0], mid("portal"))], solid)
    _polyline(c, [(R["keycloak"][0], mid("keycloak")),
                  (R["portal"][0] + R["portal"][2], mid("portal"))], solid)

    # The portal hands experiments to the manager, which launches one of the three modes.
    _polyline(c, [(cx("portal"), bottom("portal")), (cx("portal"), top("manager"))], solid)
    for m in ("doc", "agent", "roleplay"):
        _polyline(c, [(cx(m), bottom("manager")), (cx(m), top(m))], solid)

    # Modes use the shared services: a bus beneath them, with a drop to each service.
    bus_y = (bottom("doc") + top("tools")) / 2
    c.saveState()
    c.setStrokeColor(solid)
    c.setLineWidth(0.8)
    for m in ("doc", "agent", "roleplay"):
        c.line(cx(m), bottom(m), cx(m), bus_y)
    bus_left = GUTTER + LEFT_CHANNEL / 2
    c.line(bus_left, bus_y, max(cx("rag"), cx("roleplay")), bus_y)
    c.restoreState()
    for s in ("tools", "memory", "rag"):
        _polyline(c, [(cx(s), bus_y), (cx(s), top(s))], solid)
    _label(c, (cx("agent") + cx("roleplay")) / 2, bus_y - 2.2, "retrieve · call tools · remember")

    # Every mode's own model calls go straight to the router, down the left channel.
    _polyline(c, [(bus_left, bus_y), (bus_left, mid("router")), (R["router"][0], mid("router"))], solid)
    _label(c, bus_left - 3, (bus_y + mid("router")) / 2, "model calls", angle=90, back="#ffffff")

    # Retrieval and memory need embeddings and summaries: they queue at the router too.
    for s in ("memory", "rag"):
        _polyline(c, [(cx(s), bottom(s)), (cx(s), top("router"))], solid)
    _label(c, (cx("memory") + cx("rag")) / 2, (bottom("rag") + top("router")) / 2 - 2,
           "embed · summarize")

    # Libraries are filled from the online databases: the one outbound path, dashed.
    _polyline(c, [(R["online"][0], mid("online")), (R["rag"][0] + R["rag"][2], mid("rag"))],
              soft, dashed=True)
    _label(c, cx("online"), bottom("online") - 7.5, "ingested on request · outbound HTTPS",
           size=5.8, back=None)

    # The router admits work to the least-loaded Spark.
    for h in ("spark3", "spark4"):
        _polyline(c, [(cx(h), bottom("router")), (cx(h), top(h))], solid)

    # Everything above the cluster is traced: a dashed spine down the right to the record.
    spine_x = left + main_w + RIGHT_CHANNEL / 2
    for src in ("roleplay", "router"):
        c.saveState()
        c.setStrokeColor(soft)
        c.setLineWidth(0.8)
        c.setDash(2.4, 2.4)
        c.line(R[src][0] + R[src][2], mid(src), spine_x, mid(src))
        c.restoreState()
    _polyline(c, [(spine_x, mid("roleplay")), (spine_x, mid("logging")),
                  (R["logging"][0] + R["logging"][2], mid("logging"))], soft, dashed=True)
    _label(c, spine_x + 3, (mid("roleplay") + mid("logging")) / 2, "every call traced",
           angle=90, back="#ffffff")

    # The record is kept in Postgres, with the libraries, embeddings and memory.
    _polyline(c, [(R["logging"][0], mid("logging")), (R["postgres"][0] + R["postgres"][2], mid("postgres"))],
              solid)


SECTIONS: list[tuple[str, object]] = [
    ("h1", "Reading the diagram"),
    ("body", t("""
        Rows are tiers, read top to bottom in the order a launched run passes through them.
        Solid arrows are that path. The two dashed lines are not part of it: on the right,
        the trace that everything above the cluster writes as it works; on the services row,
        the one outbound path, by which libraries are filled from public legal databases.
        Colour carries the meaning it has on the portal itself -- blue for the application,
        purple for anything that consumes model capacity, grey for hardware and outside
        services, green for what is kept.
    """)),
    ("bullets", [
        t("""**Two hops, one process.** Apache terminates TLS and proxies to a uvicorn worker
             bound to loopback. There is exactly one application process; concurrency inside it
             is asyncio, not workers, which is why a single bounded queue can speak for the
             whole cluster."""),
        t("""**Three modes, one set of services.** Document analysis, the agentic workflow and
             the role-play are three graphs over the same retrieval, tools, memory and tracer.
             They differ in their nodes, not in what they stand on."""),
        t("""**Everything narrows at the router.** Every model call in every mode, and every
             embedding and summary the services need, passes through one admission point.
             That is what makes a timing number from one run comparable with another's."""),
        t("""**Every run leaves a record that can be followed back.** The configuration it ran
             under, the version of every library it searched, every call it made, and for each
             citation the document, library and version it points to."""),
    ]),

    ("h1", "1 · Interface"),
    ("body", t("""
        The lab is served at `https://datahive.uwyo.edu/ai_law_lab/` from the same Apache vhost
        as the Open OnDemand portal. Three details in that configuration are load-bearing. The
        proxy passes the mount path through **unstripped**, because the application runs with
        `root_path=/ai_law_lab` and strips the prefix itself; proxying to a bare root would 404
        its `/static` mount. `flushpackets=on` keeps the live trace stream from being buffered
        into uselessness. And `timeout 900` covers a model call that emits nothing while it is
        thinking -- with the default the browser would lose a role-play turn mid-generation.
        `X-Forwarded-Proto` is set on the location and uvicorn runs with `--proxy-headers`, so
        the application never emits `http://` URLs on an HSTS-pinned site.
    """)),
    ("body", t("""
        Sign-in is a self-hosted Keycloak, served at `/sso/` (`/auth` belongs to another
        application on the same host). People sign in with UW sign-on, Google, Microsoft or a
        Keycloak account of their own. The portal runs OpenID Connect's authorization-code flow
        with PKCE itself (`web/auth.py`), with one unusual detail: this server cannot reach its
        own public address, so browsers are sent to Keycloak's public URL while the code
        exchange and signing keys go over loopback. Keycloak proves who someone is; the portal
        decides what they may do (`accounts.py`). A verified `uwyo.edu` address is approved as a
        researcher on first sign-in; anyone else waits for an administrator. A viewer reads and
        exports, a researcher also builds experiments, runs them and curates libraries, and an
        administrator also manages people and deletes things for good. Each request re-reads
        the person from the database, so disabling someone takes effect on their next click.
    """)),

    ("h1", "2 · Web portal"),
    ("body", t("""
        The interface is server-rendered Jinja2 with small amounts of JavaScript, not a
        single-page application. `views.py` turns stored settings into plain language, because
        the people configuring experiments here are lawyers: an experiment page says how long a
        role-play is likely to take and shows each person in the cast as a card, with their
        case files.
    """)),
    ("body", t("""
        **Legal Sources** is where libraries are built: search the online databases and ingest
        what is wanted, upload PDFs and text files, or add web pages by address and check them
        for changes later. Each library page lists its versions, what changed in each, the runs
        that searched each one, and a citation list for any version. **Run pages** show the
        answer or the role-play transcript with every citation marker linked to its document;
        the passages the model was given and which it cited; a References section listing the
        cited documents with the library version each came from and every place it was cited;
        for a role-play, the exhibits each side disclosed; and the metrics and the full trace.
        A run exports as a PDF report, and "run again with these versions" repeats it against
        the libraries exactly as it searched them. While a run is in progress its trace reaches
        the browser over server-sent events, polled from `run_events` once a second.
    """)),

    ("h1", "3 · Experiment manager"),
    ("body", t("""
        `experiments.py` is the seam between the portal and the graphs. Launching copies the
        experiment's configuration into `runs.config_snapshot` before any work starts, so
        editing an experiment afterwards cannot rewrite the conditions a past result was
        produced under. Libraries are versioned the same way: each library a run will search
        -- the experiment's own, or those chosen at launch, and in a role-play every agent's
        case files -- is pinned to a recorded version and written to `run_libraries`, so the
        result can be traced to, and re-run against, exactly the documents it searched.
    """)),
    ("body", t("""
        Execution happens in a background task and the run id comes back immediately, which is
        what lets a browser watch an hour-long scene without holding a request open. A failure
        is recorded as a status with its trace intact, not raised: a failed experiment is a
        result. A role-play's cast is checked before the first model call, so a duplicate agent
        id fails at once rather than halfway through a scene.
    """)),

    ("h1", "4 · Experiment modes"),
    ("body", t("""
        Each mode is a LangGraph state graph over a typed state dictionary. Live services --
        the router, tracer and the run's pinned libraries -- travel in `config.configurable`
        rather than in state, because state is checkpointed and connections are not
        serializable.
    """)),
    ("table", {
        "header": ["Mode", "Nodes", "Sources and citations"],
        "rows": [
            ["document_analysis", "plan → analyze → ground → synthesize",
             ("Sub-questions are answered from the document concurrently; the findings are "
              "grounded in one search of all the run's libraries, and the answer cites the "
              "passages as [1], [2].")],
            ["agentic_workflow", "reason ⇄ act",
             ("The agent searches the libraries as a tool, all at once or one at a time. A "
              "passage keeps one number across all its searches, so a [3] in its answer names "
              "one passage. Ends when the model stops asking for tools.")],
            ["roleplay", "moderator → speak → moderator … → assess",
             ("Each turn the speaker sees passages from its own case files and the shared "
              "libraries as [S1], [S2]. Citing one discloses it as an exhibit, [E1], that "
              "everyone may cite afterwards. Ends when the moderator closes the scene or the "
              "turn limit is reached, never before everyone has spoken twice.")],
        ],
        "mono_first": True,
        "caption": t("""
            The role-play's rules -- who speaks next, when an impasse warrants intervention,
            when a scene is over -- and the studies behind them are in
            `graphs/roleplay_policy.py` and the companion note on role-play moderation.
        """),
    }),

    ("h1", "5 · Shared services"),
    ("bullets", [
        t("""**Legal RAG** (`rag.py`). A library is a named set of documents, filled from
             uploads, the online databases (`sources.py`) or web pages (`web_links.py`). A
             document is chunked at 2400 characters with 300 of overlap, on paragraph
             boundaries, embedded, and kept with page anchors so a citation points where a
             person can check it. Documents are never edited in place: removing one sets
             `removed_at`, and re-reading a changed page adds a new row, so every recorded
             version of a library can still be searched exactly as it was. A run's libraries
             are searched together -- the top six passages above cosine 0.35, ranked against
             each other whichever library they come from, with text held in two libraries
             returned once -- and each passage is labelled with its library in the prompt."""),
        t("""**Citations** (`grounding.py`, `citations.py`). Every citation marker a model
             writes becomes a record holding the passage, document, library and version it
             names, and where it appeared ("Answer", "Turn 7 · Dana Reyes"). A marker that
             names no passage the model was given is recorded as unsupported rather than
             dropped -- catching fabrication is the point. Each library also produces a citation
             list, one reference per document from what its source recorded, which is the form
             a run's References take."""),
        t("""**Tools** (`tools.py`) is the registry the agentic loop is given: library search
             and a safe arithmetic evaluator. `allow_network` gates network-touching tools, and
             none are installed, so that switch currently changes nothing."""),
        t("""**Memory** (`memory.py`) keeps a verbatim buffer inside a token budget and folds
             the overflow into model-written summaries that are embedded and recalled by
             similarity. Its scope is `(run_id, agent_id)`: in a negotiation, a shared buffer
             would leak one side's private reasoning into the other's context. Case files are
             private the same way: an agent's own libraries are searched only on its turns,
             and another agent sees a passage from one only after it is cited."""),
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

    ("h1", "7 · Record"),
    ("body", t("""
        One PostgreSQL 17 database with pgvector, in a container on loopback, holds everything
        a run produces and everything it searched. It is reached through an async pool rather
        than a single connection, since about thirty-two agents can be in flight at once.
        Vectors are 768-dimensional and indexed with HNSW, chosen over IVFFlat because
        libraries grow incrementally and HNSW needs no retraining.
    """)),
    ("table", {
        "header": ["Tables", "What they hold"],
        "rows": [
            ["experiments, runs", ("Definitions, and per-run snapshots of the "
                                   "configuration each result was produced under.")],
            ["run_libraries", "Each library a run searched, and the version it searched."],
            ["run_events, citations", ("The append-only trace, and every citation marker with "
                                       "its passage, document, library, version and a "
                                       "grounded / unsupported verdict.")],
            ["documents, chunks,\ncorpus_versions", ("The libraries: text, page anchors and "
                                                     "embeddings, and the exact set of "
                                                     "documents each version held.")],
            ["memory_messages,\nmemory_long_term", ("Short-term buffers and summarized "
                                                    "long-term memory, per (run, agent).")],
            ["users, audit_log", ("Who may use the lab and in what role, and a log of "
                                  "approvals, role changes and permanent deletions.")],
        ],
        "mono_first": True,
        "caption": t("""
            Everything a run produces is addressable by `run_id` and cascades from the run, so
            a result can be audited or removed as a unit. Libraries are kept apart from runs:
            deleting a run never touches what it searched.
        """),
    }),

    ("h1", "8 · Outbound traffic"),
    ("body", t("""
        Model inference runs on the two Sparks; no prompt or document is sent to a commercial
        model API. The server makes outbound requests in one case, drawn dashed on the
        services row: fetching public material -- searches and documents from CourtListener,
        the Federal Register, the eCFR, govinfo and SEC EDGAR, a web page added to a library,
        or a page or PDF a cast is drafted from. These carry a search term or an address,
        never an uploaded document.
    """)),
    ("body", t("""
        That path is guarded, because this server can reach the Sparks, Postgres and the
        university network. Only `http`/`https` links whose host resolves exclusively to public
        addresses are fetched, and every redirect is re-checked the same way. Real people and
        private organizations in a source are renamed before a cast is drafted from it, and the
        same substitution is applied to the model's output: a draft invents bottom lines and
        confidential facts, and attaching those to a real company is the failure worth
        engineering against.
    """)),

    ("h1", "Where each box lives"),
    ("table", {
        "header": ["Box on the diagram", "Source"],
        "rows": [
            ["Apache", "/etc/apache2/sites-available/ood-portal.conf"],
            ["Web portal", ("web/app.py, web/views.py, web/templates/, web/report_pdf.py,\n"
                            "agent_spec.py, cast_assistant.py, source_material.py")],
            ["Keycloak sign-in", "web/auth.py, accounts.py, scripts/keycloak_setup.py, docs/keycloak.md"],
            ["Experiment manager", "experiments.py"],
            ["Modes", ("graphs/document_analysis.py, graphs/agentic_workflow.py,\n"
                       "graphs/roleplay.py, graphs/roleplay_policy.py, graphs/state.py")],
            ["Legal RAG", "rag.py, sources.py, web_links.py"],
            ["Tools, memory", "tools.py, memory.py"],
            ["LLM router", "router.py"],
            ["Logging and evaluation", "tracing.py, grounding.py, citations.py"],
            ["PostgreSQL", "db.py, db/schema.sql, docker-compose.yml"],
        ],
        "mono_first": False,
        "caption": t("""
            Paths under `src/ailawlab/` unless absolute. Settings quoted throughout this note are
            defaults from `config.py`; every one is overridable by an `AILAWLAB_`-prefixed
            environment variable. The service runs as `ai-law-lab.service` under systemd.
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
            The diagram is laid out in `docs/architecture/build.py`; its rows and boxes are data
            at the top of that file. Re-run it after a change to the tiers, and re-run
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
