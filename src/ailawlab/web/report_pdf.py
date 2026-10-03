"""A run as a PDF: the result summary, the setup, and the full transcript in colour.

Built from views.run_view and the same Markdown blocks the page renders, so the file says
what the page says. Private notes and reasoning are left out unless asked for: a report
is the thing most likely to be shared, and those were never visible to the other parties.

Text is set in DejaVu Sans when the system has it (model output uses curly quotes, dashes,
section signs and the occasional symbol outside Latin-1), and falls back to Helvetica.
"""
from __future__ import annotations

import html
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .markdown import Block, parse

# Speaker colours, (text/rule, background), in the same order as the stylesheet's --p1..--p8.
PALETTE = [
    ("#1f5fa8", "#eaf1fa"), ("#a3500b", "#fbf0e6"), ("#0f7a5a", "#e6f5ef"), ("#6a3fb5", "#f1ecfb"),
    ("#b0284f", "#fbe9ee"), ("#0d7285", "#e3f3f6"), ("#6b6b12", "#f3f3e0"), ("#4a5563", "#eef0f2"),
]
INK, MUTED, LINE, ACCENT = "#23221f", "#6d6a63", "#d9d4c9", "#185fa5"
MODERATOR = ("#9a5b00", "#fbf3e4")

_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _fonts() -> tuple[str, str, str]:
    """Register DejaVu if present; return (regular, bold, mono) font names."""
    if "Report" in pdfmetrics.getRegisteredFontNames():
        return "Report", "Report-Bold", "Report-Mono"
    try:
        pdfmetrics.registerFont(TTFont("Report", _FONT_DIR / "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("Report-Bold", _FONT_DIR / "DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("Report-Mono", _FONT_DIR / "DejaVuSansMono.ttf"))
        # No oblique face is installed; italics fall back to the upright face.
        pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold",
                                      italic="Report", boldItalic="Report-Bold")
        return "Report", "Report-Bold", "Report-Mono"
    except Exception:  # noqa: BLE001 - any font problem: use the built-in faces
        return "Helvetica", "Helvetica-Bold", "Courier"


def _styles() -> dict[str, ParagraphStyle]:
    regular, bold, mono = _fonts()
    base = ParagraphStyle("body", fontName=regular, fontSize=9.6, leading=13.6, textColor=INK,
                          alignment=TA_LEFT, spaceAfter=5)
    return {
        "body": base,
        "title": ParagraphStyle("title", parent=base, fontName=bold, fontSize=18, leading=22, spaceAfter=4),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9, textColor=MUTED, spaceAfter=2),
        "h1": ParagraphStyle("h1", parent=base, fontName=bold, fontSize=13.5, leading=17,
                             spaceBefore=12, spaceAfter=6, textColor=ACCENT),
        "h2": ParagraphStyle("h2", parent=base, fontName=bold, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base, fontName=bold, fontSize=10.2, leading=14, spaceBefore=6, spaceAfter=3),
        "small": ParagraphStyle("small", parent=base, fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=2),
        "speaker": ParagraphStyle("speaker", parent=base, fontName=bold, fontSize=10.2, leading=13, spaceAfter=3),
        "quote": ParagraphStyle("quote", parent=base, leftIndent=12, textColor=MUTED),
        "code": ParagraphStyle("code", parent=base, fontName=mono, fontSize=8.2, leading=10.5),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.6, leading=11.5, spaceAfter=0),
        "mono": ParagraphStyle("mono", parent=base, fontName=mono),
    }


# ---------------------------------------------------------------- inline HTML -> reportlab markup


def _rl(markup: str, mono: str, link_color: str = ACCENT) -> str:
    """The Markdown reader's inline HTML in reportlab's paragraph markup."""
    s = markup.replace("&#x27;", "'")
    s = s.replace("<strong>", "<b>").replace("</strong>", "</b>")
    s = s.replace("<em>", "<i>").replace("</em>", "</i>")
    s = s.replace("<code>", f'<font face="{mono}">').replace("</code>", "</font>")
    s = s.replace("<br>", "<br/>")
    s = re.sub(r'<span class="cite">(.*?)</span>', rf'<font color="{link_color}">\1</font>', s)
    s = re.sub(r'<a class="turn-ref" href="(#[\w-]+)">(.*?)</a>', rf'<a href="\1" color="{link_color}">\2</a>', s)
    s = re.sub(r'<a href="([^"]+)"[^>]*>(.*?)</a>', rf'<a href="\1" color="{link_color}">\2</a>', s)
    return s


def _plain(text: str) -> str:
    """Plain text (names, notes, reasoning) safe for a reportlab paragraph."""
    return html.escape(text or "", quote=False).replace("\n", "<br/>")


def _blocks(blocks: list[Block], st: dict[str, ParagraphStyle], color: str = ACCENT) -> list:
    mono = st["code"].fontName
    out: list = []
    for b in blocks:
        t = b["type"]
        if t == "heading":
            style = st["h2"] if b["level"] <= 2 else st["h3"]
            out.append(Paragraph(_rl(b["html"], mono, color), style))
        elif t == "para":
            out.append(Paragraph(_rl(b["html"], mono, color), st["body"]))
        elif t == "list":
            items = []
            for it in b["items"]:
                inner: list = [Paragraph(_rl(it["html"], mono, color), st["body"])]
                if it["children"]:
                    inner.append(ListFlowable(
                        [ListItem(Paragraph(_rl(c, mono, color), st["body"])) for c in it["children"]],
                        bulletType="bullet", start="–", leftIndent=12, bulletFontSize=8))
                items.append(ListItem(inner))
            out.append(ListFlowable(items, bulletType="1" if b["ordered"] else "bullet",
                                    start=b["start"] if b["ordered"] else "•", leftIndent=14,
                                    bulletFontName=st["body"].fontName, bulletFontSize=8.5,
                                    bulletColor=color))
        elif t == "quote":
            for q in _blocks(b["blocks"], st, color):
                if isinstance(q, Paragraph):
                    q.style = st["quote"]
                out.append(q)
        elif t == "hr":
            out.append(HRFlowable(width="100%", thickness=0.5, color=LINE, spaceBefore=4, spaceAfter=6))
        elif t == "code":
            out.append(Paragraph(_plain(b["text"]), st["code"]))
        elif t == "table":
            data = [[Paragraph(f"<b>{_rl(c, mono, color)}</b>", st["cell"]) for c in b["header"]]]
            data += [[Paragraph(_rl(c, mono, color), st["cell"]) for c in r] for r in b["rows"]]
            table = Table(data, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("BACKGROUND", (0, 0), (-1, 0), "#f3f1ec"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            out.append(table)
            out.append(Spacer(1, 6))
    return out


def _boxed(flowables: list, rule: str, fill: str, width: float) -> Table:
    """Flowables in a tinted box with a coloured left rule, as a turn is on the page."""
    box = Table([[flowables]], colWidths=[width], splitInRow=1)
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, rule),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return box


# ---------------------------------------------------------------- the report


def run_report(run: dict[str, Any], view: dict[str, Any], include_private: bool = False) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    title = run.get("experiment_name") or "Experiment"
    short = str(run.get("id", ""))[:8]
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title=f"{title} (run {short})", author="AI Law Lab",
                            subject=f"{view.get('mode_label', '')} run report")
    width = doc.width
    story: list = []

    meta = [view.get("mode_label", ""), f"run {short}", run.get("status", "")]
    if view.get("started"):
        meta.append(f"started {view['started']}")
    if view.get("took"):
        meta.append(f"took {view['took']}")
    libraries = view.get("libraries") or []
    if libraries:
        meta.append(("library " if len(libraries) == 1 else "libraries ") + ", ".join(
            lib["name"] + (f" v{lib['version']}" if lib.get("version") else "") for lib in libraries))
    if view.get("turns") is not None and "turns" in view:
        meta.append(f"{view['turns']} turns")
    story += [Paragraph(_plain(title), st["title"]),
              Paragraph(_plain(" · ".join(m for m in meta if m)), st["sub"]),
              Paragraph(_plain(f"Report generated {datetime.now(UTC):%B %-d, %Y} by AI Law Lab, "
                               "University of Wyoming."
                               + (" Includes each speaker's private notes and reasoning."
                                  if include_private else "")), st["small"]),
              HRFlowable(width="100%", thickness=0.6, color=LINE, spaceBefore=6, spaceAfter=4)]

    if view.get("summary_md"):
        story.append(Paragraph(_plain(view.get("summary_title", "Summary")), st["h1"]))
        story += _blocks(parse(view["summary_md"]), st)

    if view.get("plan"):
        plan = view["plan"]
        story.append(CondPageBreak(2 * inch))
        story.append(Paragraph("Plan", st["h1"]))
        if view.get("stopped"):
            story.append(Paragraph(_plain("Research ended early ("
                                          + ("stopped by the researcher" if view["stopped"] == "stopped"
                                             else "time limit reached")
                                          + "); the answer was written from what had been found."), st["small"]))
        if view.get("brief"):
            story.append(Paragraph("<b>Brief:</b> " + _plain(view["brief"]), st["body"]))
        story.append(Paragraph("<b>Question:</b> " + _plain(plan.get("question", "")), st["body"]))
        if plan.get("approach"):
            story.append(Paragraph("<b>Approach:</b> " + _plain(plan["approach"]), st["body"]))
        for sub in plan.get("sub_questions") or []:
            story.append(Paragraph(f"<b>{_plain(sub.get('id', ''))}</b> " + _plain(sub.get("question", "")), st["body"]))
        story.append(Paragraph(_plain(
            "Libraries: " + (", ".join(plan.get("libraries") or []) or "none")
            + " · legal databases " + ("allowed" if plan.get("use_databases") else "not allowed")
            + " · open web " + ("allowed" if plan.get("use_web") else "not allowed")), st["small"]))
    if view.get("findings"):
        story.append(CondPageBreak(2 * inch))
        story.append(Paragraph("Findings by sub-question", st["h1"]))
        for f in view["findings"]:
            story.append(Paragraph(f"<b>{_plain(f.get('id', ''))}</b> " + _plain(f.get("question", ""))
                                   + f'<font color="{MUTED}" size="8">  {f.get("steps", 0)} steps'
                                   + ("" if f.get("ended") == "finished" else f" · {_plain(f.get('ended', ''))}")
                                   + "</font>", st["speaker"]))
            story += _blocks(parse(f.get("text") or ""), st)

    if view.get("transcript") is not None and "transcript" in view:
        story.append(CondPageBreak(2.5 * inch))
        story.append(Paragraph("Scenario and cast", st["h1"]))
        if view.get("scenario"):
            story.append(Paragraph(_plain(view["scenario"]), st["body"]))
        if libraries:
            story.append(Paragraph("<b>Legal sources:</b> " + _plain(", ".join(
                lib["name"] + (f" v{lib['version']}" if lib.get("version") else "") for lib in libraries)),
                st["body"]))
        colour_of = {s["id"]: PALETTE[(s["color"] - 1) % len(PALETTE)] for s in view.get("speakers", [])}
        rows = [[Paragraph("<b>Name</b>", st["cell"]), Paragraph("<b>Role</b>", st["cell"]),
                 Paragraph("<b>Objective</b>", st["cell"]), Paragraph("<b>Turns</b>", st["cell"])]]
        turns_by = {s["id"]: s["turns"] for s in view.get("speakers", [])}
        ids = [s["id"] for s in view.get("speakers", [])]
        cast = {a["id"]: a for a in view.get("cast", []) if a.get("id")}
        for aid in ids:
            a = cast.get(aid, {})
            name = next((s["name"] for s in view["speakers"] if s["id"] == aid), aid)
            rule = colour_of.get(aid, (INK, "#fff"))[0]
            rows.append([Paragraph(f'<font color="{rule}"><b>{_plain(name)}</b></font>', st["cell"]),
                         Paragraph(_plain(a.get("role", "")) + (
                             f'<br/><font color="{MUTED}" size="8">played by {_plain(a["played_by"])}</font>'
                             if a.get("played_by") else ""), st["cell"]),
                         Paragraph(_plain(a.get("goal", "")), st["cell"]),
                         Paragraph(str(turns_by.get(aid, 0)), st["cell"])])
        if len(rows) > 1:
            table = Table(rows, colWidths=[width * 0.2, width * 0.27, width * 0.43, width * 0.1],
                          repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, LINE),
                                       ("BACKGROUND", (0, 0), (-1, 0), "#f3f1ec"),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(table)

        story.append(CondPageBreak(2 * inch))
        story.append(Paragraph("Transcript", st["h1"]))
        for e in view["transcript"]:
            if e["moderator"]:
                rule, fill = MODERATOR
                head = Paragraph(f'<font color="{rule}"><b>Moderator</b></font>'
                                 f'<font color="{MUTED}" size="8">  {_plain(e["role"])} · after turn {e["turn"]}</font>',
                                 st["speaker"])
                story.append(KeepTogether([_boxed([head] + _blocks(parse(e["content"]), st, rule),
                                                  rule, fill, width), Spacer(1, 6)]))
                continue
            rule, fill = PALETTE[(e["color"] - 1) % len(PALETTE)] if e["color"] else (INK, "#f6f5f2")
            head = Paragraph(f'<a name="turn-{e["turn"]}"/><font color="{rule}"><b>{_plain(e["name"])}</b></font>'
                             f'<font color="{MUTED}" size="8">  {_plain(e["role"])} · turn {e["turn"]} · {e["words"]} words'
                             f'{" · " + _plain(e["by"]) if e.get("by") else ""}</font>',
                             st["speaker"])
            inner = [head] + _blocks(parse(e["content"]), st, rule)
            if e["sources"]:
                cited = [s for s in e["sources"] if s.get("cited")]
                lines = "<br/>".join(
                    f'[{_plain(s["marker"])}] {_plain(s["label"])}'
                    + (f' ({"own case file" if s.get("private") else "in"} {_plain(s["corpus"])}'
                       f'{" v" + str(s["version"]) if s.get("version") else ""})' if s.get("corpus") else "")
                    + (" — cited" if s.get("cited") else "")
                    for s in e["sources"])
                if e.get("disclosed"):
                    lines += "<br/><b>Disclosed:</b> " + ", ".join(_plain(m) for m in e["disclosed"])
                inner.append(Paragraph(f"<b>Legal sources given ({len(e['sources'])}, {len(cited)} cited)</b><br/>{lines}",
                                       st["small"]))
            if include_private and e["private_notes"]:
                notes = "<br/>".join(f"<b>{_plain(k.replace('_', ' ').capitalize())}:</b> {_plain(str(v))}"
                                     for k, v in e["private_notes"].items() if v)
                inner.append(Paragraph(f"<b>Private notes (only {_plain(e['name'])} saw these)</b><br/>{notes}",
                                       st["small"]))
            if include_private and e["thinking"]:
                inner.append(Paragraph(f"<b>Reasoning</b><br/>{_plain(e['thinking'])}", st["small"]))
            # A turn may run past a page; the box then splits rather than jumping pages.
            story.append(_boxed(inner, rule, fill, width))
            story.append(Spacer(1, 6))

    if view.get("exhibits"):
        story.append(CondPageBreak(1.5 * inch))
        story.append(Paragraph("Exhibits", st["h1"]))
        for e in view["exhibits"]:
            where = (f"{'case file' if e.get('private') else 'shared library'} {e['corpus']}"
                     + (f" v{e['version']}" if e.get("version") else "")) if e.get("corpus") else ""
            story.append(Paragraph(
                f"<b>[{_plain(e['marker'])}]</b> {_plain(e['label'])}"
                f'<br/><font color="{MUTED}" size="8">Disclosed by {_plain(e["name"])} in turn {e["turn"]}'
                + (f"; {_plain(where)}" if where else "") + ".</font>", st["body"]))

    refs = view.get("refs") or {}
    if refs.get("references") or refs.get("unsupported"):
        # Where each citation leads: the document's full reference, the library and version
        # it was found in, and every place the output cites it.
        story.append(CondPageBreak(1.5 * inch))
        story.append(Paragraph("References", st["h1"]))
        for ref in refs.get("references", []):
            uses = "; ".join(f"{u['marker']} {u['context']}".strip() for u in ref["uses"][:20])
            if len(ref["uses"]) > 20:
                uses += f"; and {len(ref['uses']) - 20} more"
            where = f"Searched in library {ref['library_text']}. " if ref.get("library_text") else ""
            story.append(Paragraph(
                f"{ref['n']}. {_plain(ref['text'])}" + (f" {_plain(ref['url'])}" if ref.get("url") else "")
                + f'<br/><font color="{MUTED}" size="8">{_plain(where)}Cited: {_plain(uses)}.</font>',
                st["body"]))
        if refs.get("unsupported"):
            story.append(Paragraph(
                "<b>Markers naming no passage the model was given (unsupported):</b> " + _plain(
                    "; ".join(f"{u['marker']} {u['context']}".strip() for u in refs["unsupported"])),
                st["small"]))

    def footer(canvas, doc_) -> None:
        canvas.saveState()
        canvas.setFont(st["small"].fontName, 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(doc_.leftMargin, 0.5 * inch, f"AI Law Lab · {title[:80]} · run {short}")
        canvas.drawRightString(doc_.leftMargin + doc_.width, 0.5 * inch, f"page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
