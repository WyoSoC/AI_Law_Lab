"""Build the network access note -- what the lab reaches outside, when, and by whose leave.

Like the architecture note, this is drawn and typeset here, not exported from a word
processor: the one figure (how a research agent's request to read a page is decided) is
vector output from the layout below. Keep the numbers in step with the code they quote --
each section names the module and setting it describes, and "Where it lives" at the end is
the index tying the two together.

    .venv/bin/python docs/network-access/build.py

(reportlab is already in the app's environment, for the run reports.) Output goes to
src/ailawlab/web/static/docs/, where the web portal links to it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "ailawlab" / "web" / "static" / "docs"
STEM = "network-access"

TITLE = "AI Law Lab: Outside Data Access and Agent Web Reading"
SUBTITLE = "Which services the lab calls, when its agents go online, and how a site's wishes are honoured"
AUTHOR = "Jian Gong"
AFFILIATION = "AI Law Lab, University of Wyoming"
FOOTER = "AI Law Lab · Data access note"

# The portal's palette (web/static/style.css), as in the architecture note.
INK = "#23221f"
MUTED = "#6d6a63"
LINE = "#d9d4c9"
LINK = "#185fa5"
KINDS = {
    "step":  ("#e6f1fb", "#9dc2e4", "#14497e"),     # the lab doing something
    "check": ("#f3f1ec", "#c2bcaf", "#3f3c36"),     # a question asked of the site
    "keep":  ("#e1f5ee", "#8ecdb8", "#0b5844"),     # saved, citable
    "stop":  ("#fbeaea", "#e3a5a5", "#8a2b2b"),     # not read
}


def t(text: str) -> str:
    """Collapse a triple-quoted paragraph to one line."""
    return " ".join(text.split())


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --------------------------------------------------------------------------- the figure
#
# A research agent asks to read an address (or a search result's address). Each check on the
# left either lets it through to the next or turns it away to the right, where the refusal is
# told to the agent and kept on the run page.

FLOW = [
    ("step", "An agent asks to read a page", "an address, or a search result [W3]"),
    ("check", "Is it a public web address?", "http(s), resolving only to public IPs"),
    ("check", "Does robots.txt let the lab in?", "rules for AILawLab-agent, else for *"),
    ("check", "Does robots.txt let AI agents in?", "GPTBot, ClaudeBot, ChatGPT-User, …"),
    ("check", "Are its mining rights left open?", "no reservation in /.well-known/tdmrep.json"),
    ("step", "Fetch the page once", "as AILawLab-agent/1.0, ≤ 8 MB, 45 s"),
    ("check", "Is the page free of opt-outs?", "no noai or tdm-reservation in its headers or <meta>"),
    ("keep", "Save it to the run's library, then cite it", "a new library version; numbered passages"),
]
REFUSALS = {
    1: "Refused as unsafe; nothing is fetched",
    2: "Not read: the site closes it to the lab",
    3: "Not read: the site closes it to AI agents",
    4: "Not read: mining rights reserved",
    6: "Fetched, but not kept or shown to the agent",
}


def draw_flow(c, width: float, height: float) -> None:
    from reportlab.lib import colors

    n = len(FLOW)
    box_w, box_h = width * 0.52, 30.0
    gap = (height - n * box_h) / (n - 1)
    x = 0.0
    side_x, side_w = x + box_w + 40, width - box_w - 40

    def box(bx, by, bw, bh, kind, title, sub, bold=True):
        fill, stroke, ink = KINDS[kind]
        c.setFillColor(colors.HexColor(fill))
        c.setStrokeColor(colors.HexColor(stroke))
        c.setLineWidth(0.8)
        c.roundRect(bx, by, bw, bh, 4, stroke=1, fill=1)
        c.setFillColor(colors.HexColor(ink))
        c.setFont("Sans-Bold" if bold else "Sans", 8.0)
        c.drawCentredString(bx + bw / 2, by + bh - 12, title)
        if sub:
            c.setFillColor(colors.HexColor(INK))
            c.setFont("Sans", 6.8)
            c.drawCentredString(bx + bw / 2, by + 7, sub)

    def arrow(x0, y0, x1, y1, label=""):
        import math
        c.setStrokeColor(colors.HexColor(MUTED))
        c.setFillColor(colors.HexColor(MUTED))
        c.setLineWidth(0.8)
        c.line(x0, y0, x1, y1)
        a = math.atan2(y1 - y0, x1 - x0)
        p = c.beginPath()
        p.moveTo(x1, y1)
        for d in (148, -148):
            p.lineTo(x1 + 3.4 * math.cos(a + math.radians(d)), y1 + 3.4 * math.sin(a + math.radians(d)))
        p.close()
        c.drawPath(p, stroke=0, fill=1)
        if label:
            c.setFont("Sans", 6.4)
            c.drawString((x0 + x1) / 2 + (3 if x0 == x1 else -6), (y0 + y1) / 2 + (0 if x0 == x1 else 3), label)

    for i, (kind, title, sub) in enumerate(FLOW):
        y = height - box_h - i * (box_h + gap)
        box(x, y, box_w, box_h, kind, title, sub)
        if i < n - 1:
            arrow(x + box_w / 2, y, x + box_w / 2, y - gap, "yes" if kind == "check" else "")
        if i in REFUSALS:
            arrow(x + box_w, y + box_h / 2, side_x, y + box_h / 2, "no")
            box(side_x, y + 4, side_w, box_h - 8, "stop", REFUSALS[i], "", bold=False)
    c.setFont("Sans", 6.6)
    c.setFillColor(colors.HexColor(MUTED))
    c.drawString(side_x, -2, "Every refusal is told to the agent and listed on the run page.")


# --------------------------------------------------------------------------- the text

SECTIONS: list[tuple[str, object]] = [
    ("h1", "Summary"),
    ("body", t("""
        The lab's language models run on its own hardware, and its experiments search its own
        versioned libraries. It reaches beyond the university in three situations only, each
        allowed by a person: a researcher searching the public legal databases or adding web
        pages to a library on the **Legal Sources** page; a researcher drafting a role-play cast
        from a news story or opinion given by its address; and a research run (the
        **agentic workflow**) whose researcher ticked *legal databases* or *the open web*. Role-play
        and document-analysis runs never go online.
    """)),
    ("body", t("""
        What leaves the server is a search term or an address, never a document a researcher
        uploaded. What comes back is saved into a library before anything cites it, so every
        citation names a fixed copy: a document, a library and a version, with the address it
        came from and when it was read. And since October 2026 a page is read on the lab's
        own initiative -- by a research agent, or by a crawl -- only if the site has not said
        it does not want AI systems reading it (section 5).
    """)),

    ("h1", "1 · Everything the server connects to"),
    ("table", {
        "header": ["Destination", "What is sent", "When"],
        "widths": [1.5, 2.2, 2.3],
        "rows": [
            ["The two DGX Sparks (Tailscale, inside the university)",
             "Prompts and text to embed", "Every model call; never leaves university hardware"],
            ["CourtListener, Federal Register, eCFR, govinfo, SEC EDGAR",
             "A search term, or the identifier of a result",
             "A person searches or adds on Legal Sources; an agent searches or reads (databases allowed)"],
            ["Brave Search API", "A search query written by an agent",
             "An agent searches the open web (open web allowed)"],
            ["Any public website", "A request for one address",
             "An agent reads a page; a page named in the brief; a crawl; a person adds or re-checks a link, or drafts a cast from one"],
            ["UW Shibboleth, InCommon (mdq.incommon.org), Google, Microsoft",
             "Sign-in exchanges (Keycloak)", "Someone signs in with that provider"],
        ],
        "caption": t("""
            No prompt, document or answer is sent to a commercial model API. The search terms an
            agent writes do leave the university (to the database or to Brave), and they are
            drawn from the research brief: a brief that must stay confidential should be run
            with both outside permissions off.
        """),
    }),

    ("h1", "2 · The legal databases"),
    ("body", t("""
        Five public repositories are reached through the interfaces their publishers provide
        for programs to use (`sources.py`). Each is a staging step, not a live oracle: a search
        returns candidates, and nothing is relied on until it has been fetched, chunked,
        embedded and saved into a library, with its canonical address and retrieval time in
        the document's metadata.
    """)),
    ("table", {
        "header": ["Database", "Interface and credential", "What it holds; notes"],
        "widths": [1.25, 2.35, 2.4],
        "rows": [
            ["CourtListener", "REST API v4, courtlistener.com/api/rest/v4; API token "
             "(`AILAWLAB_COURTLISTENER_TOKEN`, set on the server)",
             "Opinions from federal and state courts (Free Law Project). Without a token only "
             "the search snippet can be saved, and it is labelled as a snippet."],
            ["Federal Register", "API v1, federalregister.gov/api/v1; no key",
             "Proposed and final rules, notices, presidential documents since 1994; full text "
             "where published, else the abstract, flagged as such."],
            ["eCFR", "ecfr.gov/api (search and versioner); no key",
             "The Code of Federal Regulations as in force, read as of each title's latest issue date."],
            ["govinfo (GPO)", "api.govinfo.gov; api.data.gov key "
             "(`AILAWLAB_GOVINFO_API_KEY`, set on the server)",
             "US Code, Public Laws, bills, reports; HTML or text rendition, else the PDF."],
            ["SEC EDGAR", "efts.sec.gov full-text search; www.sec.gov/Archives; no key",
             "Company filings since 2001. Requests name the lab and a contact address, as the "
             "SEC's fair-access policy asks."],
        ],
        "caption": t("""
            Credentials live in the server's `.env`, never in the code or the browser. A database
            whose credential is missing reports itself unavailable rather than failing a page.
        """),
    }),
    ("body", t("""
        **Pacing.** A person's search goes straight through. Research agents call far more
        often, several at a time, so each database's requests from agents are spaced at least
        one second apart across every run on the server; a database that answers 429 ("too
        many requests") is left alone for three minutes, and agents are told to search
        elsewhere meanwhile (`network_tools.paced`). One agent search asks each database for
        four results.
    """)),
    ("body", t("""
        **Open-web search.** Brave's Search API (`AILAWLAB_BRAVE_API_KEY`, set on the server)
        answers an agent's web searches with eight results each. Calls are spaced 1.1 seconds
        apart, the API's per-second limit, and one 429 is retried after two seconds. Results are
        leads, numbered [W1], [W2]...: an agent cannot cite a search result, only what it read.
    """)),
    ("body", t("""
        **Why robots.txt does not govern these.** robots.txt addresses crawlers wandering a
        website. These interfaces exist so that programs can query them, and their use is
        governed by their own terms, keys and rate limits, which the lab follows. The documents
        they return are largely government works or public filings; that a document is public
        does not settle every use of it, and researchers remain responsible for theirs.
    """)),

    ("h1", "3 · When a research run goes online"),
    ("body", t("""
        An agentic-workflow run has two permissions, each the researcher's to grant on the plan
        before the run starts: *search the legal databases* and *search the open web* (which
        includes reading any public page). With both off, the run is entirely local. With
        either on, network access happens at these points and no others:
    """)),
    ("bullets", [
        t("""**Planning: none.** The planner drafts sub-questions from the brief on the Sparks.
           It notes any web addresses written in the brief, but does not open them."""),
        t("""**Before research: pages named in the brief.** If the open web is allowed, each
           address in the brief is read once, before any agent starts, so every agent has it.
           These reads pass the same checks as an agent's (section 5)."""),
        t("""**During research: the agents' own tools.** Each research agent (up to four at once)
           decides for itself when to call `search_databases`, `search_web` and `read`, as its
           sub-question needs. Reading is not budgeted; the run's time limit (two hours by
           default) and the researcher's Stop button end it. A document is read once per run,
           however many agents ask for it."""),
        t("""**Write-up: none.** The lead agent writes the answer from the findings and the
           passages already read."""),
    ]),
    ("body", t("""
        Everything read is saved into a library -- by default "Fetched: <experiment name>" --
        which records a new version, and the run is pinned to it, so later library searches by
        any agent cover the new document too. The run page then lists every search with the
        results it gave, every document read (with the library version it was saved to), and
        every page that was not read because its site does not allow it.
    """)),

    ("h1", "4 · What a person adds"),
    ("body", t("""
        On the **Legal Sources** page a person can search the databases and add chosen results;
        add web pages by address (up to 20 at a time) and later *check them again*, which re-reads
        each page and rebuilds its passages only if the text changed (`web_links.py`); or crawl a
        page's links into a library (`crawler.py`). In the cast builder a person can draft a
        role-play from a page given by address (`source_material.py`). Adding a link or drafting
        from one is that person reading the page, as they would in a browser, so it is not put
        through the AI checks below; a crawl, which reads many pages automatically, is.
    """)),

    ("h1", "5 · Does the site let AI agents read it?"),
    ("body", t("""
        Before a research agent's read and before every page of a crawl, the lab asks what the
        site has published about automated and AI use, and obeys it (`site_rules.py`). Figure 1
        shows the order for an agent.
    """)),
    ("figure", None),
    ("caption", t("""
        Figure 1. How a research agent's request to read a page is decided. A crawl asks the
        same questions under its own name, AILawLab-crawler, and records a skipped page as
        "not open to AI use".
    """)),
    ("bullets", [
        t("""**robots.txt, for the lab** (RFC 9309). Read first. Agents are known to it as
           `AILawLab-agent`, crawls as `AILawLab-crawler`; a site may write rules for either, or
           for all crawlers (`*`). The most specific matching rule decides, and the `*` and `$`
           wildcards are understood -- Python's built-in reader ignores them, which would have
           let `Disallow: /*.pdf$` through. A missing robots.txt sets no rules; one that
           cannot be read because the site is failing means nothing is read."""),
        t("""**robots.txt, for AI agents.** Many sites now turn named AI agents away while
           admitting search engines. If the site's robots.txt closes this page to any of
           ChatGPT-User, Claude-User, Perplexity-User, MistralAI-User, GPTBot, OAI-SearchBot,
           ClaudeBot, Claude-SearchBot, anthropic-ai, PerplexityBot, CCBot, Google-Extended,
           Applebot-Extended, Bytespider, meta-externalagent or cohere-ai, the lab treats that as
           a refusal of AI systems in general, which it is one of. Only rules written for those
           agents by name count."""),
        t("""**Text-and-data-mining reservations** (W3C TDM Reservation Protocol). A site can
           reserve its TDM rights in `/.well-known/tdmrep.json`, by location; on the page in a
           `tdm-reservation: 1` header; or in a `<meta name="tdm-reservation">` tag. A
           reservation means not read; its licensing address, if given, is passed on."""),
        t("""**"noai" on the page.** A `noai` directive in an `X-Robots-Tag` header (for every
           agent, or for the lab's name) or in a robots `<meta>` tag in the page's head.
           These can only be seen once the page arrives, so such a page is fetched but neither
           kept nor shown to the agent."""),
    ]),
    ("body", t("""
        A refused agent is told plainly why and asked to find another source; the trace records
        it; and the run page lists each refused address with its reason. A site's rules are
        read at most once an hour per site, across all runs (RFC 9309 allows a day), and a
        failing site is asked again next time rather than remembered. If a page redirects to
        another site, that site's rules are checked too.
    """)),

    ("h1", "6 · Other safeguards on every fetch"),
    ("bullets", [
        t("""**Public addresses only.** This server can reach the Sparks, its database and the
           university network, so a link is fetched only if it is `http`/`https` and its host
           resolves exclusively to public addresses; every redirect (at most five) is checked
           again (`source_material.check_public_url`)."""),
        t("""**Bounded.** A response is cut off at 8 MB and 45 seconds; a page kept in a library
           keeps up to 100,000 words, a cast source 12,000."""),
        t("""**Named.** Requests say who is asking: `AILawLab-agent/1.0` for agents,
           `AILawLab-crawler/1.0` for crawls, each with the lab's address and a contact, so a
           site owner can see who is reading and how to reach us."""),
        t("""**Polite crawling.** A crawl stays on one website and goes one level deep, fetches
           one page at a time at least two seconds apart (longer if robots.txt asks, up to 30),
           takes at most 100 pages, stops on a 429 or 503 or three failures in a row, runs one
           at a time per site, and stores the licence or copyright notice it finds with every
           document."""),
        t("""**Recorded.** Searches with their results, documents read with their library
           versions, pages refused with the reason, and every crawl with the rules it ran under
           and the outcome for each link, are kept in the database."""),
    ]),

    ("h1", "7 · The lab's own interface"),
    ("body", t("""
        The browser talks to the lab through JSON endpoints under `/ai_law_lab/api/`. They are
        the portal's plumbing, not a public API: every one needs a signed-in, approved account,
        and changes need the researcher or admin role (`web/auth.py`). The ones that reach
        outside are `/api/sources/search` and `/api/sources/ingest` (databases),
        `/api/sources/links` and the two `refresh` endpoints (web links), `/api/crawls` and
        `/api/crawls/preview` (crawls), and `/api/source/read` (a cast source). Runs are
        launched from the experiment page and followed live through `/api/runs/{id}/stream`.
    """)),

    ("h1", "8 · Limits and open questions"),
    ("bullets", [
        t("""The AI-agent list is maintained by hand. New agents appear; the list is in
           `site_rules.AI_AGENTS` and should be reviewed each term."""),
        t("""Machine-readable AI preferences are still being standardized (the IETF AI
           Preferences working group). The lab follows robots.txt, `noai` and TDMRep, which sites
           use today, and should adopt the IETF vocabulary once it settles."""),
        t("""A site's permission to read is not a licence to reuse. The licence notice a crawl
           records, and the address and retrieval time kept with every document, are there so a
           researcher can check before relying on, quoting or publishing what was collected."""),
        t("""Pages a person adds by hand are not put through the AI checks (section 4). If the
           lab's practice should be stricter, the same check can be applied there with a line of
           code."""),
    ]),

    ("h1", "Where it lives"),
    ("table", {
        "header": ["Topic", "Source"],
        "widths": [1.6, 2.9],
        "mono_second": True,
        "rows": [
            ["Legal databases", "sources.py"],
            ["Agents' searching and reading; pacing", "network_tools.py, tools.py"],
            ["Run permissions and pages named in the brief", "research.py, experiments.py"],
            ["Site permission checks", "site_rules.py"],
            ["Fetching, address guard, limits", "source_material.py"],
            ["Web links in a library", "web_links.py"],
            ["Crawls", "crawler.py"],
            ["Settings quoted here", "config.py (each overridable by an AILAWLAB_ variable)"],
            ["Run page: read online, refused", "web/views.py, web/templates/run.html"],
        ],
        "caption": t("""
            Paths under `src/ailawlab/`. Tests: `tests/test_site_rules.py`,
            `tests/test_network_tools.py`, `tests/test_crawler.py`. Rebuild this note with
            `.venv/bin/python docs/network-access/build.py`.
        """),
    }),
]


# --------------------------------------------------------------------------- typesetting

def _pdf_markup(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Code first, set aside, so a "*" inside it is not read as emphasis.
    code: list[str] = []

    def keep(m: re.Match) -> str:
        code.append(f'<font name="Mono" size="8.4">{m.group(1)}</font>')
        return f"\x00{len(code) - 1}\x00"

    text = re.sub(r"`(.+?)`", keep, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: code[int(m.group(1))], text)


def build_pdf(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Flowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    fonts = Path("/usr/share/fonts/truetype/dejavu")
    faces = {"Serif": "DejaVuSerif.ttf", "Serif-Bold": "DejaVuSerif-Bold.ttf",
             "Serif-Italic": "DejaVuSerif-Italic.ttf", "Serif-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
             "Sans": "DejaVuSans.ttf", "Sans-Bold": "DejaVuSans-Bold.ttf", "Mono": "DejaVuSansMono.ttf"}
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
        "title": ParagraphStyle("title", parent=body, fontName="Serif-Bold", fontSize=15.5,
                                leading=19, alignment=TA_LEFT, spaceAfter=3),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontName="Serif-Italic", fontSize=10.2,
                                   leading=13, textColor=muted, alignment=TA_LEFT, spaceAfter=3),
        "byline": ParagraphStyle("byline", parent=body, fontSize=8.4, leading=11, textColor=muted,
                                 alignment=TA_LEFT, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=body, fontName="Serif-Bold", fontSize=11.2,
                             leading=14, spaceBefore=12, spaceAfter=5, alignment=TA_LEFT,
                             keepWithNext=1),
        "caption": ParagraphStyle("caption", parent=body, fontSize=8.4, leading=11,
                                  alignment=TA_LEFT, textColor=muted, spaceBefore=4),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.3, leading=10.6,
                               alignment=TA_LEFT, spaceAfter=0),
        "cell_mono": ParagraphStyle("cell_mono", parent=body, fontName="Mono", fontSize=7.6,
                                    leading=10.2, alignment=TA_LEFT, spaceAfter=0),
        "cell_head": ParagraphStyle("cell_head", parent=body, fontName="Serif-Bold",
                                    fontSize=8.3, leading=10.6, alignment=TA_LEFT, spaceAfter=0),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=5),
    }

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Serif", 7.8)
        canvas.setFillColor(muted)
        canvas.drawString(66, 30, FOOTER)
        canvas.drawRightString(letter[0] - 66, 30, str(doc.page))
        canvas.restoreState()

    class Figure(Flowable):
        def __init__(self, width: float, height: float):
            Flowable.__init__(self)
            self.width, self.height = width, height

        def wrap(self, avail_w, avail_h):
            return self.width, self.height + 8

        def draw(self) -> None:
            self.canv.saveState()
            self.canv.translate(0, 8)
            draw_flow(self.canv, self.width, self.height)
            self.canv.restoreState()

    width = letter[0] - 132
    story: list = [
        Paragraph(_pdf_markup(TITLE), styles["title"]),
        Paragraph(_pdf_markup(SUBTITLE), styles["subtitle"]),
        Paragraph(f"{AUTHOR} · {AFFILIATION} · Data access note, October 2026. "
                  f"Describes the software at commit {commit()}.", styles["byline"]),
    ]
    for kind, value in SECTIONS:
        if kind == "h1":
            story.append(Paragraph(_pdf_markup(value), styles["h1"]))
        elif kind == "caption":
            story.append(Paragraph(_pdf_markup(value), styles["caption"]))
        elif kind == "body":
            story.append(Paragraph(_pdf_markup(value), body))
        elif kind == "bullets":
            story += [Paragraph(_pdf_markup(b), styles["bullet"], bulletText="•") for b in value]
        elif kind == "figure":
            story.append(Spacer(1, 4))
            story.append(Figure(width, 340))
        elif kind == "table":
            rows = [[Paragraph(_pdf_markup(h), styles["cell_head"]) for h in value["header"]]]
            for row in value["rows"]:
                rows.append([Paragraph(_pdf_markup(cell), styles["cell_mono" if j == 1 and value.get("mono_second")
                                                                 else "cell"])
                             for j, cell in enumerate(row)])
            widths = value["widths"]
            table = Table(rows, colWidths=[w / sum(widths) * width for w in widths],
                          repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, rule),
                ("LINEBELOW", (0, 1), (-1, -2), 0.3, colors.HexColor("#efece4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
            ]))
            story.append(table)
            story.append(Paragraph(_pdf_markup(value["caption"]), styles["caption"]))
            story.append(Spacer(1, 4))

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=66, rightMargin=66,
                            topMargin=60, bottomMargin=52, title=TITLE,
                            author=f"{AUTHOR}, {AFFILIATION}", subject=SUBTITLE,
                            keywords="robots.txt, TDMRep, noai, legal databases, AI agents, web access")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{STEM}.pdf"
    build_pdf(target)
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
