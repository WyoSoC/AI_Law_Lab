"""The experiment page's plain-language views. Pure: no database or cluster needed."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ailawlab.graphs.roleplay_policy import estimate_run_seconds
from ailawlab.web.views import (
    agent_view,
    duration_text,
    elapsed_text,
    experiment_view,
    run_summary,
    snippet,
    source_view,
)

NOW = datetime(2026, 9, 14, 6, 0, tzinfo=UTC)


def test_durations_read_like_a_person_would_say_them():
    assert duration_text(30) == "about a minute"
    assert duration_text(600) == "about 10 minutes"
    assert duration_text(3590) == "about 1 hour"
    assert duration_text(estimate_run_seconds(100, 1000)) == "about 1 hour"
    assert duration_text(estimate_run_seconds(20, 1000)) == "about 14 minutes"
    assert duration_text(7200) == "about 2 hours"
    assert (elapsed_text(45), elapsed_text(725), elapsed_text(6720)) == ("45 s", "12 min", "1 h 52 min")


def test_an_agent_card_keeps_private_sections_apart():
    card = agent_view({"id": "a", "name": "Amelia Stone", "role": "Counsel", "goal": "Win",
                       "tendencies": "Anchors hard\nBluffs", "bottom_line": "24 months",
                       "confidential": "Insurer refused", "demeanor": "Calm"})
    assert (card["name"], card["role"], card["goal"]) == ("Amelia Stone", "Counsel", "Win")
    assert card["details"] == [{"label": "Demeanor", "text": "Calm"},
                               {"label": "Tendencies", "entries": ["Anchors hard", "Bluffs"]}]
    assert [p["label"] for p in card["private"]] == ["Bottom line", "Confidential information"]


def test_source_links_must_be_web_links():
    card = source_view({"title": "Story", "url": "javascript:alert(1)", "site": "Oil City News",
                        "published": "2026-09-12T21:45:00+00:00", "words": 952,
                        "retrieved_at": "2026-09-14T05:09:24+00:00"})
    assert card["url"] == ""
    assert card["meta"] == ("Oil City News · published September 12, 2026 · "
                            "read September 14, 2026 · 952 words")
    assert source_view({"url": "https://oilcity.news/x"})["url"] == "https://oilcity.news/x"


def test_run_summaries_say_where_a_run_is_or_what_it_found():
    assert run_summary({"status": "failed", "error": "ValueError: the cast cannot run\nTrace"},
                       "roleplay") == "Failed: ValueError: the cast cannot run"
    running = {"status": "running", "inputs": {"scenario": "s"}, "config_snapshot": {"max_turns": 100}}
    assert run_summary(running, "roleplay", turns_so_far=23) == "In progress: 23 of up to 100 turns so far"
    done = {"status": "succeeded", "result": {"turns": 9, "outcome":
            "## Evaluation\n### 1. Outcome\n**Did the parties reach agreement? No.** They stalled."}}
    assert run_summary(done, "roleplay") == "9 turns · Did the parties reach agreement? No. They stalled."
    assert run_summary({"status": "succeeded", "result": {"answer": "x " * 200}},
                       "document_analysis").endswith("…")
    assert snippet("") == ""


def test_a_roleplay_experiment_view():
    exp = {"mode": "roleplay", "created_at": NOW - timedelta(hours=1), "config": {
        "max_turns": 100, "word_limit": 1000, "scenario": "A mediation.",
        "agents": [{"id": "a", "name": "A B"}],
        "source": {"title": "T", "url": "https://example.com/story"}}}
    runs = [{"id": "b4fbcbf5-e770-4ac2-849a-556ffc50bcfa", "status": "running",
             "started_at": NOW - timedelta(minutes=44), "finished_at": None, "result": None,
             "inputs": {}, "config_snapshot": {"max_turns": 100}}]
    view = experiment_view(exp, runs, progress={"b4fbcbf5-e770-4ac2-849a-556ffc50bcfa": 31}, now=NOW)
    assert view["mode_label"] == "Role-play" and view["created"] == "September 14, 2026"
    assert [f["value"] for f in view["facts"]] == ["1 person", "100", "1,000", "about 1 hour"]
    assert view["cast_errors"] == ["A role-play needs at least two agents."]
    assert view["source"]["url"] == "https://example.com/story"
    [row] = view["runs"]
    assert row["active"] and row["took"] == "44 min so far" and row["short"] == "b4fbcbf5"
    assert row["summary"] == "In progress: 31 of up to 100 turns so far"
    assert view["active_runs"] == [row]
    assert view["launch_defaults"] == {"max_turns": 100, "word_limit": 1000}


def test_document_and_agentic_experiment_views():
    doc = experiment_view({"mode": "document_analysis", "config": {"corpus": "test"}}, [],
                          corpus_documents=1, now=NOW)
    assert doc["facts"] == [{"label": "Corpus", "value": "test", "note": "1 document in it"}]
    agent = experiment_view({"mode": "agentic_workflow", "config": {"max_iterations": "4"}}, [], now=NOW)
    assert [f["value"] for f in agent["facts"]] == ["default", "4", "not allowed"]
