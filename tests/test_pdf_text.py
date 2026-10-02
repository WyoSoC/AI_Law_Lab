"""Reading PDFs: text only, soup left out, Dhamma Palatino Pali restored. Pure."""
from __future__ import annotations

import io

from ailawlab import pdf_text, source_material
from ailawlab.pdf_text import page_is_text


def test_glyph_soup_is_told_from_text():
    assert page_is_text("Termination for convenience requires thirty days' written notice. " * 3)
    assert page_is_text("12")                                         # a page number alone
    # What pypdf made of a custom-encoded font, and what a font with no map gives.
    assert not page_is_text("theGproperGo˛˛˚sionGhereVG“xowG›tGt˛›tGtim˜G§˜nUGüp›n›nı›Gt˛˜GÑ›ky›n" * 2)
    assert not page_is_text("<\n\x15\x0b \x03 \x01\x06\x0c\x0c\r\x10\x06\x13\n\x16\x16\x03 \x13\x06\x1e*" * 4)
    assert not page_is_text("(cid:10)(cid:21)(cid:11) (cid:3)(cid:8)(cid:1)(cid:6)(cid:12)" * 4)


class _Page:
    """Stands in for a PDFium text page: chars with the font and box each was drawn in."""

    def __init__(self, chars):
        self.chars = chars


class _Raw:
    @staticmethod
    def FPDFText_CountChars(tp):
        return len(tp.chars)

    @staticmethod
    def FPDFText_GetUnicode(tp, k):
        return ord(tp.chars[k][0])

    @staticmethod
    def FPDFText_GetFontInfo(tp, k, buf, size, flags):
        name = tp.chars[k][1].encode("latin-1")
        buf.value = name
        return len(name)

    @staticmethod
    def FPDFText_GetCharBox(tp, k, left, right, bottom, top):
        box = tp.chars[k][2]
        for ref, v in zip((left, right, bottom, top), box):
            ref._obj.value = v
        return 1


def _page(segments):
    """[(text, font, ligature?)] -> chars; letters of a ligature share one box."""
    chars, x = [], 0.0
    for text, font, lig in segments:
        for ch in text:
            chars.append((ch, font, (x, x + 5, 0, 10)))
            if not lig:
                x += 6
        if lig:
            x += 6
    return _Page(chars)


def test_dhamma_palatino_pali_is_restored_only_in_those_fonts():
    page = _page([("Dıgha Nik›ya, satipa˛˛h›na, Saºgha, Buddhaª; ", "QEKEHW+DPalatino", False),
                  ("fi", "QEKEHW+DPalatinoBoldItalic", True),             # one glyph: Ā
                  ("nanda first ", "QEKEHW+DPalatinoBoldItalic", False),   # two glyphs: f, i
                  ("a real › stays", "GAHXDC+Palatino-Roman", False)])
    text = "".join(c[0] for c in page.chars)
    fixed, n = pdf_text._fix_pali(page, _Raw, text)
    assert fixed == "Dīgha Nikāya, satipaṭṭhāna, Saṅgha, Buddhaṃ; Ānanda first a real › stays"
    assert n == 8                                              # ı › ˛ ˛ › º ª, and Ā
    plain = _page([("No Pali here at all.", "Times-Roman", False)])
    assert pdf_text._fix_pali(plain, _Raw, "No Pali here at all.") == ("No Pali here at all.", 0)


def _pdf(pages):
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for lines in pages:
        for i, line in enumerate(lines):
            c.drawString(72, 720 - 16 * i, line)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_pdf_pages_are_kept_with_their_numbers():
    data = _pdf([["First page of the agreement."], ["Second page: termination."],
                 ["Third page: notice of thirty days."]])
    r = pdf_text.extract(data)
    assert r.pages == 3 and r.pages_kept == 3 and r.engine == "pdfium"
    doc = source_material.from_bytes(data, content_type="application/pdf", url="https://e.org/a.pdf")
    for offset, page in doc.page_map:                     # each anchor lands on its own page
        assert doc.text[offset:].startswith({1: "First", 2: "Second", 3: "Third"}[page])


def test_an_unreadable_pdf_says_why():
    assert "damaged" in pdf_text.unreadable_reason(pdf_text.PdfText(text="", engine="none"))
    assert "OCR" in pdf_text.unreadable_reason(pdf_text.PdfText(text="", pages=3, pages_unreadable=3))
    try:
        source_material.from_bytes(b"%PDF-1.5 not really", content_type="application/pdf")
    except source_material.SourceError as e:
        assert "damaged" in str(e) or "No text" in str(e)
