"""Role-play rules that need no model call: speaker choice, endings, sizing, prompts."""
from __future__ import annotations

from ailawlab.graphs.roleplay_policy import (
    MODERATOR_ID,
    character_reminder,
    compose_agent_prompt,
    context_window,
    format_entries,
    intervention_due,
    intervention_text,
    may_conclude,
    memory_budget,
    moderator_schema,
    pick_speaker,
    since_last_turn,
    speech_budget,
    transcript_segments,
)

AGENTS = [{"id": "a", "name": "Ann", "role": "buyer"},
          {"id": "b", "name": "Bo", "role": "seller"},
          {"id": "c", "name": "Cy", "role": "mediator"}]


def said(*speakers: str) -> list[dict]:
    return [{"turn": i + 1, "agent_id": s, "name": s, "content": f"turn {i + 1}"}
            for i, s in enumerate(speakers)]


def test_a_direct_question_is_answered_next():
    assert pick_speaker(AGENTS, said("a", "b"), question_for="c", suggested="a") == (
        "c", "answering a direct question")


def test_a_speaker_cannot_answer_their_own_question():
    who, _ = pick_speaker(AGENTS, said("a", "b"), question_for="b", suggested="b")
    assert who != "b"


def test_nobody_speaks_twice_running():
    who, rule = pick_speaker(AGENTS, said("a", "b"), question_for="nobody", suggested="b")
    assert who == "c" and rule.startswith("round-robin")


def test_one_agent_cannot_monopolize_the_floor():
    # a and b have traded four turns while c has not spoken; the moderator keeps picking a.
    who, rule = pick_speaker(AGENTS, said("a", "b", "a", "b"), question_for="nobody",
                             suggested="a")
    assert (who, rule) == ("c", "balancing speaking time")


def test_the_moderators_reasonable_choice_stands():
    assert pick_speaker(AGENTS, said("a", "b", "c"), None, "b") == ("b", "moderator's choice")


def test_the_moderator_schema_only_allows_real_ids():
    schema = moderator_schema(["a", "b"])
    assert schema["properties"]["next_speaker"]["enum"] == ["a", "b"]
    assert "nobody" in schema["properties"]["question_for"]["enum"]


def test_no_scene_ends_before_everyone_has_spoken_twice():
    two = AGENTS[:2]
    assert not may_conclude(said("a", "b", "a"), two)
    assert may_conclude(said("a", "b", "a", "b"), two)


def test_interventions_wait_for_a_stall_and_then_cool_down():
    assert not intervention_due(stalled=False, turn=10, last_intervention=0, n_agents=2)
    assert intervention_due(stalled=True, turn=10, last_intervention=0, n_agents=2)
    assert not intervention_due(stalled=True, turn=10, last_intervention=8, n_agents=2)
    techniques = [intervention_text(i, "Ann")[0] for i in range(4)]
    assert techniques == ["reframe", "narrow", "reality test", "reframe"]
    assert "Ann" in intervention_text(0, "Ann")[1]


def test_an_agent_sees_what_happened_since_its_last_turn():
    t = said("a", "b", "c", "a", "b")
    assert [x["agent_id"] for x in since_last_turn(t, "a")] == ["b"]
    assert [x["agent_id"] for x in since_last_turn(t, "c")] == ["a", "b"]
    assert len(since_last_turn(t, "new")) == 5


def test_moderator_statements_are_labelled_in_the_transcript():
    entries = [*said("a"), {"turn": 1, "agent_id": MODERATOR_ID, "name": "Moderator",
                            "content": "Stalled."}]
    assert "[moderator, after turn 1] Stalled." in format_entries(entries)


def test_private_sections_are_marked_and_hand_written_prompts_win():
    agent = {"id": "a", "name": "Ann", "role": "buyer", "goal": "Buy low",
             "bottom_line": "No more than $10", "tendencies": ["Anchors low"]}
    prompt = compose_agent_prompt(agent, AGENTS, "A sale.", 500)
    assert "PRIVATE" in prompt and "No more than $10" in prompt and "- Anchors low" in prompt
    assert "Bo (seller)" in prompt and "Ann (buyer)" not in prompt
    assert "under 500 words" in prompt
    assert compose_agent_prompt({"id": "x", "system_prompt": "Custom."}, AGENTS, "s", 5) == "Custom."
    assert "No more than $10" in character_reminder(agent, 500)


def test_budgets_grow_with_the_word_limit_and_fit_the_context():
    assert speech_budget(1000, think=True) > speech_budget(200, think=True) > 1000
    assert context_window(10_000) == 32768
    assert context_window(40_000) == 65536
    assert context_window(200_000) == 131072
    budget = memory_budget(1000, 2, 32768, think=True)
    assert 4000 <= budget <= 20000
    assert budget + speech_budget(1000, True) + 1400 < 32768


def test_long_transcripts_split_into_bounded_segments():
    t = [{"turn": i, "agent_id": "a", "name": "A", "content": "word " * 700} for i in range(10)]
    segments = transcript_segments(t, max_tokens=3000)
    assert sum(len(s) for s in segments) == 10
    assert len(segments) > 1 and all(len(s) >= 1 for s in segments)
