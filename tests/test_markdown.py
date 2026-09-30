"""The Markdown reader used for run results. Pure."""
from __future__ import annotations

from ailawlab.web.markdown import inline, parse, to_html


def test_model_output_cannot_inject_markup():
    out = to_html('<script>alert(1)</script> **<img src=x onerror=alert(1)>** [x](javascript:alert(1))')
    assert "<script" not in out and "<img" not in out and "&lt;script&gt;" in out
    assert 'href="javascript' not in out           # only http(s) links become links


def test_inline_formatting():
    assert inline("**bold** and *it* and `a*b*c` and snake_case_name") == (
        "<strong>bold</strong> and <em>it</em> and <code>a*b*c</code> and snake_case_name")
    assert inline("see [S2] and [3]") == 'see <span class="cite">[S2]</span> and <span class="cite">[3]</span>'
    assert inline("[site](https://example.com/a)") == \
        '<a href="https://example.com/a" target="_blank" rel="noopener">site</a>'


def test_blocks_a_model_assessment_uses():
    text = """## Evaluation

### 1. Outcome

**Outcome:** No agreement.
Second line of the same paragraph.

*   **Stone:** wanted a bond
*   **Bell:** wanted acreage
    - nested detail

1. First
2. Second

| Party | Moved? |
|---|---|
| Stone | no |

> quoted text

---"""
    kinds = [b["type"] for b in parse(text)]
    assert kinds == ["heading", "heading", "para", "list", "list", "table", "quote", "hr"]
    blocks = parse(text)
    assert blocks[2]["html"].endswith("<br>Second line of the same paragraph.")
    assert blocks[3]["items"][1]["children"] == ["nested detail"]
    assert blocks[4]["ordered"] and len(blocks[4]["items"]) == 2
    assert blocks[5]["rows"] == [["Stone", "no"]]
    html = to_html(text)
    assert "<h4>Evaluation</h4>" in html and "<ol>" in html and "<blockquote>" in html


def test_plain_text_is_just_paragraphs():
    assert parse("One.\n\nTwo.") == [{"type": "para", "html": "One."}, {"type": "para", "html": "Two."}]
    assert parse("") == []
