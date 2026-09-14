"""Agent specification files. Pure parsing and writing, so no cluster or database needed."""
from __future__ import annotations

from ailawlab.agent_spec import (
    assign_ids,
    check_cast,
    parse_files,
    parse_markdown,
    template,
    to_markdown,
)

DANA = """# Dana Reyes

## Role
Lead counsel for the Provider

## Objective
Cap the Provider's indemnification at 12 months of fees.

## Background
Fifteen years in technology transactions.

Treats unlimited liability as an existential threat.

## Demeanor
Calm, precise, relentlessly polite.

## Tendencies
- Anchors hard early,
  then concedes slowly
- Reframes every risk as a dollar figure

## Priorities
Protecting the balance sheet above any single clause.

## Bottom line
Will not accept indemnification above 24 months of fees.

## Confidential information
The Provider's insurer has already refused to cover uncapped indemnities.
"""


def test_reads_every_standard_section():
    r = parse_markdown(DANA, "dana.md")
    assert r.warnings == []
    [a] = r.agents
    assert a == {
        "id": "dana-reyes",
        "name": "Dana Reyes",
        "role": "Lead counsel for the Provider",
        "goal": "Cap the Provider's indemnification at 12 months of fees.",
        "backstory": "Fifteen years in technology transactions.\n\n"
                     "Treats unlimited liability as an existential threat.",
        "demeanor": "Calm, precise, relentlessly polite.",
        "tendencies": ["Anchors hard early, then concedes slowly",
                       "Reframes every risk as a dollar figure"],
        "priorities": "Protecting the balance sheet above any single clause.",
        "bottom_line": "Will not accept indemnification above 24 months of fees.",
        "confidential": "The Provider's insurer has already refused to cover uncapped indemnities.",
    }


def test_headings_are_forgiving_about_case_synonyms_and_style():
    text = """Notes to myself before the agents start -- not part of any agent.

#Sam Okafor
Role: General counsel for the Client

### GOAL:
Keep indemnification uncapped.

## Walk-away point (private)
Walk away before accepting any cap.

**Interests**
Avoiding another abandoned project.

## Behavioural tendencies
* Opens with the strongest position
1. Tests bluffs
• Concedes on price before risk
"""
    r = parse_markdown(text)
    assert r.warnings == []
    [a] = r.agents
    assert a["name"] == "Sam Okafor"
    assert a["role"] == "General counsel for the Client"
    assert a["goal"] == "Keep indemnification uncapped."
    assert a["bottom_line"] == "Walk away before accepting any cap."
    assert a["priorities"] == "Avoiding another abandoned project."
    assert a["tendencies"] == ["Opens with the strongest position", "Tests bluffs",
                               "Concedes on price before risk"]


def test_tendencies_without_dashes_are_read_one_per_line():
    r = parse_markdown("# Dana Reyes\n## Tendencies\nAnchors hard\nBluffs\n")
    assert r.agents[0]["tendencies"] == ["Anchors hard", "Bluffs"]


def test_unrecognised_sections_are_kept_and_reported():
    r = parse_markdown("# Dana Reyes\n\n## Leverage\nThe Client has no other vendor.\n", "dana.md")
    [a] = r.agents
    assert a["notes"] == "Leverage: The Client has no other vendor."
    [warning] = r.warnings
    assert warning.startswith("dana.md:") and '"Leverage"' in warning


def test_text_directly_under_the_name_is_kept_and_reported():
    r = parse_markdown("# Dana Reyes\nA veteran negotiator.\n\n## Role\nCounsel\n")
    [a] = r.agents
    assert a["notes"] == "A veteran negotiator."
    assert a["role"] == "Counsel"
    assert len(r.warnings) == 1


def test_a_cast_file_with_instructions_comments_and_windows_line_endings():
    text = ("﻿<!-- instructions\n# not an agent -->\nIntro text.\r\n\r\n"
            "# Dana Reyes\r\n## Role\r\nProvider counsel\r\n\r\n---\r\n\r\n"
            "# Sam Okafor\r\n## Role\r\nClient counsel\r\n")
    r = parse_markdown(text)
    assert r.warnings == []
    assert [(a["id"], a["role"]) for a in r.agents] == [
        ("dana-reyes", "Provider counsel"), ("sam-okafor", "Client counsel")]


def test_a_hash_followed_by_a_number_is_not_a_new_agent():
    r = parse_markdown("# Dana Reyes\n## Priorities\n#1 concern is cost\n")
    assert len(r.agents) == 1
    assert r.agents[0]["priorities"] == "#1 concern is cost"


def test_a_heading_with_no_name_is_named_and_reported():
    r = parse_markdown("# \n## Role\nCounsel\n")
    [a] = r.agents
    assert a["name"] == "Unnamed agent 1"
    assert len(r.warnings) == 1


def test_a_file_without_names_says_how_to_fix_it():
    r = parse_markdown("Role: counsel\nObjective: win\n", "cast.txt")
    assert r.agents == []
    [warning] = r.warnings
    assert warning.startswith("cast.txt:") and '"# Dana Reyes"' in warning


def test_download_then_upload_gives_back_the_same_agents():
    agents = [
        {"id": "provider", "name": "Dana Reyes", "role": "Lead counsel", "goal": "Cap it.",
         "backstory": "First paragraph.\n\n# Not a heading\n---\n**Role**",
         "tendencies": ["Anchors", "Bluffs"], "bottom_line": "24 months.",
         "confidential": "Insurer refused.", "notes": "Leverage: none."},
        {"id": "sam-okafor", "name": "Sam Okafor", "goal": "Uncapped."},
    ]
    r = parse_markdown(to_markdown(agents))
    assert r.warnings == []
    assert r.agents == agents


def test_ids_come_from_names_and_never_collide():
    agents = [{"name": "Dana Reyes"}, {"name": "Dana Reyes"}, {"name": "Émile Zola"}]
    warnings = assign_ids(agents, taken={"dana-reyes"})
    assert [a["id"] for a in agents] == ["dana-reyes-2", "dana-reyes-3", "emile-zola"]
    assert len(warnings) == 2


def test_several_files_upload_as_one_cast():
    r = parse_files([("a.md", "# Dana Reyes\n"), ("b.md", "# Dana Reyes\n\n# Sam Okafor\n")])
    assert [a["id"] for a in r.agents] == ["dana-reyes", "dana-reyes-2", "sam-okafor"]
    assert len(r.warnings) == 1


def test_the_blank_template_reads_cleanly():
    t = template()
    r = parse_markdown(t)
    assert r.warnings == []
    assert len(r.agents) == 1
    for heading in ("## Role", "## Objective", "## Tendencies", "## Bottom line",
                    "## Confidential information"):
        assert heading in t


def test_check_cast_needs_two_agents():
    issues = check_cast([{"id": "a", "name": "Dana", "goal": "Win"}])
    assert any(i["level"] == "error" and "at least two" in i["message"] for i in issues)


def test_check_cast_flags_duplicates_and_copied_objectives():
    issues = check_cast([
        {"id": "a", "name": "Dana", "role": "x", "goal": "Win", "bottom_line": "b"},
        {"id": "a", "name": "dana", "role": "y", "goal": "win ", "bottom_line": "b"},
    ])
    messages = [i["message"] for i in issues]
    assert any('share the id "a"' in m for m in messages)
    assert any("both named" in m for m in messages)
    assert any("same objective" in m for m in messages)


def test_check_cast_suggests_a_bottom_line_but_leaves_hand_written_prompts_alone():
    issues = check_cast([{"id": "a", "name": "A", "role": "r", "goal": "g"},
                         {"id": "b", "name": "B", "system_prompt": "You are B."}])
    assert [(i["level"], i["agent"]) for i in issues] == [("suggestion", "a")]


def test_bold_or_sub_heading_names_start_agents():
    r = parse_markdown("**Eleanor Vance**\n\n## Role\nCounsel\n\n***\n\n"
                       "### Marcus Thorne\n## Role\nDeveloper counsel\n")
    assert [(a["name"], a["role"]) for a in r.agents] == [
        ("Eleanor Vance", "Counsel"), ("Marcus Thorne", "Developer counsel")]
    assert r.warnings == []


def test_ordinary_bold_lines_and_empty_sections_are_not_taken_for_names():
    text = ("# Dana Reyes\n\n## Background\nLong career.\n\n**A key lesson learned**\n\n"
            "## Demeanor\nCalm.\n\n## Leverage\n\n## Role\nCounsel\n")
    r = parse_markdown(text)
    assert [a["name"] for a in r.agents] == ["Dana Reyes"]
    assert r.agents[0]["role"] == "Counsel"
    assert "A key lesson learned" in r.agents[0]["backstory"]


def test_small_typos_in_section_headings_are_forgiven():
    # "Demeanories" is what gemma4 wrote in a live draft.
    r = parse_markdown("# Dana Reyes\n## Demeanories\nCalm.\n## Backround\nLong career.\n"
                       "## Objetive\nWin.\n## Leverage\nNone.\n## Full prompting ideas\nX\n")
    [a] = r.agents
    assert (a["demeanor"], a["backstory"], a["goal"]) == ("Calm.", "Long career.", "Win.")
    assert "Leverage: None." in a["notes"] and "Full prompting ideas: X" in a["notes"]
    assert "system_prompt" not in a
