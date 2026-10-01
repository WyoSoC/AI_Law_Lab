"""How a written answer should read, and a safety net for when the model ignores it.

The person reading an answer is a researcher who asked a question, not a colleague receiving
a memo. Telling the model it is "a legal analyst writing findings for a colleague" made it
dress answers as memoranda -- To / From / Date / Subject, an invented date, "[Your Name]" left
unfilled -- which says nothing and misstates who is involved. ANSWER_STYLE asks for the
answer itself; strip_letter_format removes a header or sign-off that slips through anyway,
so the stored answer starts with substance. The trace keeps the model's reply as written.
"""
from __future__ import annotations

import re

ANSWER_STYLE = """How to write the answer:
- Start with the answer itself. Use short headings or lists only where they help.
- Do not format it as a memo, letter or report: no To, From, Date, Re or Subject lines, no
  greeting, no sign-off or signature, and no placeholders such as [Your Name] or [Date].
- Do not describe yourself or the reader by a job title."""

# "To:", "**To:**", "**To**:", "- From :" ...
_HEADER = re.compile(
    r"^[\W_]*(to|from|date|re|subject|cc|bcc|attn|attention|prepared (by|for))\s*(\*\*|__)?\s*[:：]",
    re.IGNORECASE)
# A short title line naming the thing a memo: "Legal Analysis Findings Memorandum".
_TITLE = re.compile(r"^[\W_]*[^.!?\n]{0,80}\b(memo|memorandum)\b[^.!?\n]{0,40}$", re.IGNORECASE)
_RULE = re.compile(r"^\s*([-*_=])(\s*\1){2,}\s*$")
_SIGNOFF = re.compile(
    r"^\W*(sincerely|best regards|regards|kind regards|respectfully( submitted)?|"
    r"yours (truly|sincerely))\W*$", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^\W*\[(your |author'?s? |analyst'?s? )?(name|title|date|signature|"
                          r"position|organization|firm)\]\W*$", re.IGNORECASE)


def strip_letter_format(text: str) -> tuple[str, bool]:
    """`text` without a memo or letter header at the top or a sign-off at the end. Returns
    the cleaned text and whether anything was removed. Pure."""
    lines = (text or "").strip().splitlines()
    start = 0
    for i, line in enumerate(lines[:14]):
        if not line.strip() or _HEADER.match(line) or _TITLE.match(line) or _RULE.match(line) \
                or _PLACEHOLDER.match(line):
            start = i + 1
            continue
        break
    # Only something shaped like a memo header counts: two or more short header lines, or a
    # memo title and one. "Date: the statute took effect in 2019" alone is an answer.
    top = lines[:start]
    headers = sum(1 for x in top if _HEADER.match(x) and len(x) <= 160)
    titled = any(_TITLE.match(x) for x in top)
    if not (headers >= 2 or (titled and headers >= 1)):
        start = 0
    end = len(lines)
    for i in range(len(lines) - 1, max(start, len(lines) - 6) - 1, -1):
        line = lines[i]
        if not line.strip() or _SIGNOFF.match(line) or _PLACEHOLDER.match(line):
            if _SIGNOFF.match(line) or _PLACEHOLDER.match(line):
                end = i
            continue
        break
    cleaned = "\n".join(lines[start:end]).strip()
    return cleaned, cleaned != (text or "").strip()
