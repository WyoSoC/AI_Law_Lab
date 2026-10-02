"""Reading the text of a PDF: the text layer only, page by page, judged before it is kept.

PDFium (Google's PDF engine, the one in Chrome; pypdfium2, Apache-2.0/BSD) reads the text.
On 90 PDFs from one public library (2026-10-02) it produced readable text from 38, against
15 for pypdf, which this replaced: pypdf misreads embedded fonts with custom encodings and
returns glyph soup ("t˛˜GÑ›ky›nGw›s", every space a "G"). pdfminer.six read as many but took
twenty times as long; poppler and MuPDF read no more. Images are never read: this is text
extraction, not OCR.

Two things are done to what PDFium returns:

* Each page is checked, and a page whose "text" is not text -- control characters, no
  spaces, no letters, which is what a font with no character map yields -- is left out
  rather than stored as passages that would poison retrieval. A PDF with no readable page is
  reported as such (it would need OCR).
* Pali written in the "Dhamma Palatino" fonts (DPalatino*), which put accented letters in
  the slots of Mac punctuation, is turned back into Unicode: "Nik›ya" becomes "Nikāya". The
  map was read off the documents themselves (see PALI_MAP) and applies only to characters
  drawn in those fonts, so a real "›" elsewhere is left alone.
"""
from __future__ import annotations

import ctypes
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Dhamma Palatino slot -> letter, each confirmed against Pali words in the texts:
# nibb›na, S›vatthı, rÒpa, Buddhaª, Saºgha, sara˚a, satipa˛˛h›na, An›thapi˚˜ika, CÒ˘a,
# ≥h›nissaro, Upani˝ad, KauŸıtakı, ⁄vet›Ÿvatara, B¸had›ra˚yaka, ¿g-veda.
PALI_MAP = {"›": "ā", "ı": "ī", "Ò": "ū", "ª": "ṃ", "º": "ṅ", "˚": "ṇ", "˛": "ṭ", "˜": "ḍ",
            "˘": "ḷ", "≥": "Ṭ", "˝": "ṣ", "Ÿ": "ś", "⁄": "Ś", "¸": "ṛ", "¿": "Ṛ"}
PALI_FONTS = re.compile(r"^(?:[A-Z]{6}\+)?DPalatino", re.IGNORECASE)
# In these fonts the "fi" ligature slot holds "Ā"; PDFium reports that one glyph as "f" + "i"
# sharing a single box, which is how it is told apart from the two letters of "first".
_MARKERS = set(PALI_MAP) | {"f"}


def unreadable_reason(r: PdfText) -> str:
    """Why a PDF yielded no text, in words a person can act on."""
    if r.engine == "none":
        return "This PDF is damaged and cannot be opened."
    if r.pages_unreadable:
        return ("The text in this PDF is drawn with fonts that cannot be turned back into "
                "letters, so it could only be read by OCR (reading the pages as images).")
    return ("No text could be read from that PDF. It may be a scanned image, or damaged; if "
            "so, copy the text and paste it instead.")


@dataclass
class PdfText:
    text: str
    page_map: list[tuple[int, int]] = field(default_factory=list)   # (char offset, page number)
    pages: int = 0
    pages_kept: int = 0
    pages_unreadable: int = 0
    pali_fixed: int = 0
    engine: str = "pdfium"


def normalize_text(text: str) -> str:
    """Whitespace as every source is stored: single spaces, paragraphs one blank line apart.
    source_material applies the same to whole documents, so normalizing page by page first
    keeps page offsets true."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n[ \t]*\n\s*", "\n\n", text).strip()


def page_is_text(text: str) -> bool:
    """Whether a page's extracted text is text, not glyph soup. Pure.

    Soup is what a font without a character map gives: control characters, runs with no
    spaces, few letters. A page with very little text (a blank page, a page number) passes.
    """
    t = text.strip()
    if len(t) < 40:
        return True
    controls = sum(1 for ch in t if ord(ch) < 32 and ch not in "\n\r\t")
    if controls / len(t) > 0.02 or "(cid:" in t:
        return False
    tokens = t.split()
    if sum(len(w) for w in tokens) / max(len(tokens), 1) > 18:     # no word breaks
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters / len(t) > 0.35


def _fix_pali(textpage, raw, text: str) -> tuple[str, int]:
    """`text` with Dhamma Palatino letters turned into Unicode, judged char by char."""
    if not any(ch in _MARKERS for ch in text):
        return text, 0
    n = raw.FPDFText_CountChars(textpage)
    buf = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()

    def font(k: int) -> str:
        raw.FPDFText_GetFontInfo(textpage, k, buf, 128, ctypes.byref(flags))
        return buf.value.decode("latin-1", "replace")

    def box(k: int) -> tuple[float, ...]:
        vals = [ctypes.c_double() for _ in range(4)]
        raw.FPDFText_GetCharBox(textpage, k, *(ctypes.byref(v) for v in vals))
        return tuple(round(v.value, 2) for v in vals)

    out, fixed, k = [], 0, 0
    while k < n:
        ch = chr(raw.FPDFText_GetUnicode(textpage, k) or 0xFFFD)
        if ch in _MARKERS and PALI_FONTS.match(font(k)):
            if ch == "f":
                if k + 1 < n and chr(raw.FPDFText_GetUnicode(textpage, k + 1)) == "i" \
                        and box(k) == box(k + 1):
                    out.append("Ā")
                    fixed += 1
                    k += 2
                    continue
            else:
                out.append(PALI_MAP[ch])
                fixed += 1
                k += 1
                continue
        out.append(ch)
        k += 1
    return "".join(out), fixed


def extract(data: bytes) -> PdfText:
    """The readable text of a PDF, page by page. Raises ValueError if it cannot be opened."""
    import pypdfium2 as pdfium
    from pypdfium2 import raw

    try:
        doc = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        return _fallback(data, str(e))
    result = PdfText(text="", pages=len(doc))
    parts: list[str] = []
    offset = 0
    try:
        for i in range(len(doc)):
            try:
                page = doc[i]
                textpage = page.get_textpage()
            except pdfium.PdfiumError:
                result.pages_unreadable += 1
                continue
            text = textpage.get_text_range()
            text, fixed = _fix_pali(textpage, raw, text)
            textpage.close()
            page.close()
            text = normalize_text(text)
            if not text:
                continue
            if not page_is_text(text):
                result.pages_unreadable += 1
                continue
            result.pali_fixed += fixed
            result.page_map.append((offset, i + 1))
            parts.append(text)
            offset += len(text) + 2
            result.pages_kept += 1
    finally:
        doc.close()
    result.text = "\n\n".join(parts)
    return result


def _fallback(data: bytes, why: str) -> PdfText:
    """pypdf, for a file PDFium will not open (it is lenient with some broken files)."""
    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception as e:  # noqa: BLE001 - a PDF neither engine opens is reported, not raised
        log.warning("pdf could not be opened (%s; pypdf: %s)", why, e)
        return PdfText(text="", engine="none")
    result = PdfText(text="", pages=len(pages), engine="pypdf")
    parts, offset = [], 0
    for i, t in enumerate(pages):
        t = normalize_text(t)
        if t and page_is_text(t):
            result.page_map.append((offset, i + 1))
            parts.append(t)
            offset += len(t) + 2
            result.pages_kept += 1
        elif t:
            result.pages_unreadable += 1
    result.text = "\n\n".join(parts)
    return result
