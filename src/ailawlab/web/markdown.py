"""A small, safe Markdown reader for what the models write: assessments, answers, turns.

gemma4 writes Markdown -- headings, bold, bullet and numbered lists, the odd table -- and
showing it as raw text buries the structure a reader needs. This covers what the models
actually produce, not the whole CommonMark spec, and is safe by construction: every
character of the source is HTML-escaped before any formatting is applied, so model output
can never inject markup. Links are kept only when they are http(s).

parse() returns blocks that both the web page (to_html) and the PDF export read, so the two
show the same structure. Inline formatting inside a block is a limited HTML subset:
<strong>, <em>, <code>, <a href>, and <span class="cite"> for role-play source markers
like [S2] and document-analysis markers like [3].
"""
from __future__ import annotations

import html
import re
from typing import Any

Block = dict[str, Any]

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_BULLET = re.compile(r"^(\s*)[-*+•]\s+(.*)$")
_NUMBER = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


# ---------------------------------------------------------------- inline

_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__")
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])|(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_CITE = re.compile(r"\[(S?\d{1,3})\]")


def inline(text: str) -> str:
    """One line or paragraph of Markdown as safe inline HTML."""
    out = html.escape(text, quote=True)
    # Code spans first, set aside so nothing inside them is formatted.
    codes: list[str] = []

    def keep_code(m: re.Match) -> str:
        codes.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(codes) - 1}\x00"

    out = _CODE.sub(keep_code, out)
    out = _LINK.sub(lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">{m.group(1)}</a>', out)
    out = _BOLD.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", out)
    out = _ITALIC.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", out)
    out = _CITE.sub(r'<span class="cite">[\1]</span>', out)
    return re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], out)


# ---------------------------------------------------------------- blocks


def _table_cells(line: str) -> list[str]:
    line = line.strip()
    line = line.removeprefix("|")
    line = line.removesuffix("|")
    return [c.strip() for c in line.split("|")]


def _list(lines: list[str], i: int) -> tuple[Block, int]:
    """A list starting at lines[i], with one level of nesting for indented items."""
    first = _NUMBER.match(lines[i]) or _BULLET.match(lines[i])
    ordered = bool(_NUMBER.match(lines[i]))
    base = len(first.group(1))
    start = int(first.group(2)) if ordered else 1
    items: list[dict[str, Any]] = []
    while i < len(lines):
        line = lines[i]
        m_num, m_bul = _NUMBER.match(line), _BULLET.match(line)
        m = m_num or m_bul
        if m and len(m.group(1)) <= base + 1:
            if bool(m_num) != ordered:
                break
            items.append({"text": m.group(3) if m_num else m.group(2), "children": []})
            i += 1
        elif m and items:                                   # indented: a nested item
            items[-1]["children"].append(m.group(3) if m_num else m.group(2))
            i += 1
        elif line.strip() and items and line.startswith((" ", "\t")):
            # A wrapped continuation of the previous item.
            target = items[-1]["children"] if items[-1]["children"] else None
            if target:
                target[-1] += " " + line.strip()
            else:
                items[-1]["text"] += " " + line.strip()
            i += 1
        else:
            break
    return ({"type": "list", "ordered": ordered, "start": start,
             "items": [{"html": inline(it["text"]),
                        "children": [inline(c) for c in it["children"]]} for it in items]}, i)


def parse(text: str) -> list[Block]:
    lines = (text or "").replace("\r\n", "\n").replace("\t", "    ").split("\n")
    blocks: list[Block] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append({"type": "para", "html": "<br>".join(inline(p.strip()) for p in para)})
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
        elif _FENCE.match(line):
            flush()
            fence = _FENCE.match(line).group(1)
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                body.append(lines[i])
                i += 1
            blocks.append({"type": "code", "text": "\n".join(body)})
            i += 1
        elif m := _HEADING.match(line):
            flush()
            blocks.append({"type": "heading", "level": len(m.group(1)), "html": inline(m.group(2))})
            i += 1
        elif _HR.match(line):
            flush()
            blocks.append({"type": "hr"})
            i += 1
        elif ("|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1])
              and "|" in lines[i + 1]):
            flush()
            header = [inline(c) for c in _table_cells(line)]
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells = [inline(c) for c in _table_cells(lines[i])]
                rows.append((cells + [""] * len(header))[:len(header)])
                i += 1
            blocks.append({"type": "table", "header": header, "rows": rows})
        elif _BULLET.match(line) or _NUMBER.match(line):
            flush()
            block, i = _list(lines, i)
            blocks.append(block)
        elif _QUOTE.match(line):
            flush()
            quoted: list[str] = []
            while i < len(lines) and (m := _QUOTE.match(lines[i])):
                quoted.append(m.group(1))
                i += 1
            blocks.append({"type": "quote", "blocks": parse("\n".join(quoted))})
        else:
            para.append(line)
            i += 1
    flush()
    return blocks


# ---------------------------------------------------------------- HTML


def blocks_html(blocks: list[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        t = b["type"]
        if t == "heading":
            # Model headings sit inside a page that already has h1/h2, so they start at h3.
            level = min(b["level"] + 2, 6)
            out.append(f"<h{level}>{b['html']}</h{level}>")
        elif t == "para":
            out.append(f"<p>{b['html']}</p>")
        elif t == "list":
            tag = "ol" if b["ordered"] else "ul"
            start = f' start="{b["start"]}"' if b["ordered"] and b["start"] != 1 else ""
            items = "".join(
                f"<li>{it['html']}"
                + (f"<ul>{''.join(f'<li>{c}</li>' for c in it['children'])}</ul>" if it["children"] else "")
                + "</li>" for it in b["items"])
            out.append(f"<{tag}{start}>{items}</{tag}>")
        elif t == "quote":
            out.append(f"<blockquote>{blocks_html(b['blocks'])}</blockquote>")
        elif t == "hr":
            out.append("<hr>")
        elif t == "code":
            out.append(f"<pre>{html.escape(b['text'])}</pre>")
        elif t == "table":
            head = "".join(f"<th>{c}</th>" for c in b["header"])
            rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in b["rows"])
            out.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                       f"<tbody>{rows}</tbody></table></div>")
    return "\n".join(out)


def to_html(text: str) -> str:
    """Markdown source as safe HTML, for use with Jinja's |safe."""
    return blocks_html(parse(text))
