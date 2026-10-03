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
    assert view["launch_defaults"] == {"max_turns": 100, "word_limit": 1000, "libraries": [],
                                      "seating": [{"id": "a", "name": "A B", "model": "", "person": False}]}


def test_document_and_agentic_experiment_views():
    doc = experiment_view({"mode": "document_analysis", "config": {"corpus": "test"}}, [],
                          library_documents={"test": 1}, now=NOW)
    assert doc["facts"] == [{"label": "Library", "value": "“test”", "note": "1 document in it"}]
    # An agentic study from before the planner keeps to the libraries it named.
    both = experiment_view({"mode": "agentic_workflow", "name": "Treaties",
                            "config": {"libraries": ["cases", "regs"], "allow_network": True,
                                       "fetch_library": "  jurisprudence "}}, [], now=NOW)
    assert both["facts"][0] == {"label": "Libraries", "value": "“cases”, “regs”",
                                "note": "only these may be searched"}
    assert both["launch_defaults"] == {"choose_libraries": False, "libraries": ["cases", "regs"],
                                       "use_databases": True, "use_web": False,
                                       "time_limit_minutes": 120, "fetch_library": "jurisprudence",
                                       "own_library": "Fetched: Treaties"}
    new = experiment_view({"mode": "agentic_workflow", "name": "S", "config": {"choose_libraries": True}},
                          [], now=NOW)
    assert [f["value"] for f in new["facts"]] == ["the planner chooses", "not allowed", "not allowed",
                                                   "120 minutes"]
    doc_only = experiment_view({"mode": "document_analysis", "config": {}}, [], now=NOW)
    assert "allow_network" not in doc_only["launch_defaults"]
    none = experiment_view({"mode": "document_analysis", "config": {"libraries": []}}, [], now=NOW)
    assert none["facts"][0]["value"] == "None: answers without retrieved authority"


def test_a_run_view_gives_each_speaker_a_colour_and_renders_markdown():
    from ailawlab.web.views import link_turns, run_view

    run = {"mode": "roleplay", "status": "succeeded",
           "started_at": NOW - timedelta(minutes=12), "finished_at": NOW,
           "libraries": [{"name": "water", "version": 2, "documents": 5}],
           "config_snapshot": {"scenario": "A mediation.", "corpus": "water",
                               "agents": [{"id": "b", "name": "Bea"}, {"id": "a", "name": "Al"}]},
           "inputs": {},
           "result": {"outcome": "## Result\n\n**No deal** (Turn 2).", "transcript": [
               {"turn": 1, "agent_id": "a", "name": "Al", "content": "I *open*."},
               {"turn": 1, "agent_id": "moderator", "role": "reframing", "content": "Focus."},
               {"turn": 2, "agent_id": "b", "name": "Bea", "content": "Per [S1], no.",
                "sources": [{"marker": "S1", "label": "Case", "document_id": 1,
                             "similarity": 0.8, "cited": True}]}]}}
    v = run_view(run)
    assert [(s["name"], s["color"], s["turns"], s["cited"]) for s in v["speakers"]] == \
        [("Bea", 1, 1, 1), ("Al", 2, 1, 0)]                  # cast order, not speaking order
    assert v["turns"] == 2 and v["interventions"] == 1 and v["rerun_versions"] == {"water": 2}
    assert '<a class="turn-ref" href="#turn-2">Turn 2</a>' in v["summary_html"]
    assert "<strong>No deal</strong>" in v["summary_html"] and v["took"] == "12 min"
    assert v["transcript"][0]["html"] == "<p>I <em>open</em>.</p>" and v["transcript"][1]["moderator"]
    linked = run_view(run, "/lab")["transcript"][2]["html"]
    assert '<a class="cite" href="/lab/sources/documents/1" title="Case">[S1]</a>' in linked
    assert link_turns("Turns 3-7 and Turn 12") == (
        '<a class="turn-ref" href="#turn-3">Turns 3</a>-7 and <a class="turn-ref" href="#turn-12">Turn 12</a>')


def test_the_pdf_report_builds_with_and_without_private_notes():
    from ailawlab.web.report_pdf import run_report
    from ailawlab.web.views import run_view

    run = {"id": "b4fbcbf5-e770-4ac2-849a-556ffc50bcfa", "experiment_name": "Gravel", "mode": "roleplay",
           "status": "succeeded", "config_snapshot": {"agents": [{"id": "a", "name": "Al"}]},
           "result": {"outcome": "| a | b |\n|---|---|\n| 1 | 2 |\n\n- x\n  - y",
                      "transcript": [{"turn": 1, "agent_id": "a", "name": "Al", "content": "Hi — § 1 “q”.",
                                      "private_notes": {"where_we_stand": "far apart"},
                                      "thinking": "hmm"}]}}
    plain, full = run_report(run, run_view(run)), run_report(run, run_view(run), True)
    assert plain.startswith(b"%PDF") and len(full) > len(plain)


def test_activity_events_are_json_safe_and_keep_what_the_panel_reads():
    import json

    from ailawlab.web.views import activity_events

    rows = [{"seq": 4, "event_type": "llm_call", "node": "reason", "eval_ms": 3138, "queue_wait_ms": 0,
             "output_tokens": 120, "prompt_tokens": 1552, "thinking": "Read W1-W4.", "host": "spark3",
             "payload": {"tool_calls": [{"function": {"name": "read_online"}}]}, "created_at": NOW}]
    [e] = activity_events(rows)
    assert e["created_at"] == NOW.isoformat() and e["prompt_tokens"] == 1552 and "host" not in e
    assert json.loads(json.dumps(activity_events(rows)))[0]["thinking"] == "Read W1-W4."
