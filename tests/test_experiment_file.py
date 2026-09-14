"""Experiment files: a scenario, settings and cast in one Markdown file. Pure, offline."""
from __future__ import annotations

from ailawlab.agent_spec import (
    apply_limits,
    experiment_template,
    parse_files,
    parse_markdown,
    template,
    to_experiment_markdown,
    to_markdown,
)

SOURCE = {"title": "Wyoming's high court hears arguments", "kind": "web page",
          "url": "https://oilcity.news/story/", "site": "Oil City News",
          "published": "2026-09-12T21:45:00+00:00", "words": 952, "truncated": False,
          "retrieved_at": "2026-09-14T05:09:24+00:00", "sha256": "ab12" * 16}
AGENTS = [
    {"id": "evelyn-reed", "name": "Evelyn Reed", "role": "Mediator", "goal": "A durable deal.",
     "tendencies": ["Restates the objective", "Uses silence"]},
    {"id": "amelia-stone", "name": "Amelia Stone", "role": "Counsel for the State Board",
     "bottom_line": "No renewal without a remediation bond."},
]


def test_an_experiment_file_round_trips():
    scenario = "A mediation over gravel leases.\n\n# Not a heading\nThe county objects."
    text = to_experiment_markdown(scenario, {"max_turns": 60, "word_limit": 800}, AGENTS, SOURCE)
    r = parse_markdown(text, "casper.md")
    assert r.warnings == []
    assert r.is_experiment
    assert r.scenario == scenario
    assert r.settings == {"max_turns": 60, "word_limit": 800}
    assert r.source == SOURCE
    assert r.agents == AGENTS


def test_a_plain_agent_file_is_not_an_experiment():
    r = parse_markdown(to_markdown(AGENTS))
    assert not r.is_experiment and r.scenario == "" and r.settings == {} and r.source is None
    assert [a["name"] for a in r.agents] == ["Evelyn Reed", "Amelia Stone"]


def test_experiment_headings_and_settings_are_forgiving():
    text = """Notes before anything else.

# SCENARIO
The State Board denied the lease renewals.

---

# Role-play settings
- Maximum turns: 60
- Words per turn: 1,200

# Cast
The people below.

# Evelyn Reed
## Role
Mediator
"""
    r = parse_markdown(text)
    assert r.warnings == []
    assert r.is_experiment
    assert r.scenario == "The State Board denied the lease renewals."
    assert r.settings == {"max_turns": 60, "word_limit": 1200}
    assert [a["name"] for a in r.agents] == ["Evelyn Reed"]


def test_unreadable_settings_and_source_lines_are_reported():
    text = """# Settings
Max turns: plenty
Speed: fast
a line with no colon

# Source
Link: oilcity.news/story
Author: Someone

# Evelyn Reed
"""
    r = parse_markdown(text, "exp.md")
    assert r.settings == {}
    assert r.source is None
    assert len(r.warnings) == 5 and all(w.startswith("exp.md:") for w in r.warnings)
    assert any('"Max turns" needs a number' in w for w in r.warnings)
    assert any('"Speed" is not a setting' in w for w in r.warnings)
    assert any("not a web address" in w for w in r.warnings)


def test_settings_outside_the_allowed_range_are_brought_within_it():
    settings, warnings = apply_limits({"max_turns": 500, "word_limit": 10}, 100, 2000)
    assert settings == {"max_turns": 100, "word_limit": 40}
    assert warnings == ["Max turns must be between 2 and 100, so 500 was changed to 100.",
                        "Words per turn must be between 40 and 2,000, so 10 was changed to 40."]
    assert apply_limits({"max_turns": 24}, 100, 2000) == ({"max_turns": 24}, [])


def test_only_the_first_experiment_file_sets_the_scenario():
    first = to_experiment_markdown("First scenario.", {"max_turns": 30}, AGENTS[:1])
    second = to_experiment_markdown("Second scenario.", {"max_turns": 90}, AGENTS[1:])
    r = parse_files([("one.md", first), ("agents.md", "# Sam Okafor\n"), ("two.md", second)])
    assert r.is_experiment
    assert (r.scenario, r.settings) == ("First scenario.", {"max_turns": 30})
    assert [a["name"] for a in r.agents] == ["Evelyn Reed", "Sam Okafor", "Amelia Stone"]
    assert any(w.startswith("two.md:") and "already read" in w for w in r.warnings)


def test_the_templates_read_cleanly():
    r = parse_markdown(experiment_template(100, 1000, 100, 2000))
    assert r.warnings == []
    assert r.is_experiment and r.scenario == "" and r.settings == {"max_turns": 100, "word_limit": 1000}
    assert len(r.agents) == 1
    text = experiment_template()
    for heading in ("# Scenario", "# Settings", "Max turns: 100", "Words per turn: 1000", "## Role"):
        assert heading in text
    assert not parse_markdown(template()).is_experiment
