"""Answers read as answers: no memo dressing. Pure."""
from __future__ import annotations

from ailawlab.graphs.agentic_workflow import DOCUMENT_CHARS, _brief_message
from ailawlab.graphs.answer_style import ANSWER_STYLE, strip_letter_format

# What a run produced on 2026-10-01, header and all.
REPORTED = """**Legal Analysis Findings Memorandum**
**To:** Colleague
**From:** [Your Name], Legal Analyst
**Date:** October 26, 2023
**Subject:** Comparative Analysis: Conceptual Guardrails in Buddhist Texts (Analogy to AI Agent Constraints)

---

### 1. Overview
The texts describe restraint as a practice [1].

Sincerely,
[Your Name]"""


def test_a_memo_header_and_sign_off_are_removed():
    text, removed = strip_letter_format(REPORTED)
    assert removed and text == "### 1. Overview\nThe texts describe restraint as a practice [1]."
    plain = "# To: Colleague\n## From: Analyst\nThe answer [2]."
    assert strip_letter_format(plain) == ("The answer [2].", True)


def test_ordinary_answers_are_left_alone():
    for text in ("To be clear: the clause is void [2].",
                 "Date: the statute took effect in 2019 [1].\nMore follows.",
                 "Memorandum decisions are not precedential [3].",
                 "---\nThe libraries do not address this.",
                 "Regards to notice, the rule is thirty days [1]."):
        assert strip_letter_format(text) == (text.strip(), False), text


def test_the_prompts_ask_for_the_answer_itself():
    from ailawlab.graphs import document_analysis as da

    for prompt in (da.SYNTHESIZE_PROMPT, da.SYNTHESIZE_LIBRARY_PROMPT):
        assert ANSWER_STYLE in prompt and "colleague" not in prompt.lower()
        assert "legal analyst" not in prompt.lower()
    assert "[Your Name]" in ANSWER_STYLE                      # named, so the model avoids it


def test_a_research_agent_is_given_its_sub_question_the_study_and_the_document():
    plan = {"question": "How long is notice?", "approach": "Read the contract.",
            "sub_questions": [{"id": "Q1", "question": "Notice period?"},
                              {"id": "Q2", "question": "Cure period?"}],
            "enough_when": "The clause is quoted."}
    sub = plan["sub_questions"][0]
    msg = _brief_message({"plan": plan}, sub)
    assert "THE STUDY\nHow long is notice?" in msg and "YOUR SUB-QUESTION (Q1)\nNotice period?" in msg
    assert "- Cure period?" in msg and "Notice period?\n\nENOUGH WHEN" in msg     # the others, not its own
    msg = _brief_message({"plan": plan, "document_title": "MSA", "document_text": "Notice: 30 days."}, sub)
    assert "“MSA”" in msg and "Notice: 30 days." in msg
    long = _brief_message({"plan": plan, "document_text": "x" * (DOCUMENT_CHARS + 10)}, sub)
    assert f"its first {DOCUMENT_CHARS:,} characters" in long
