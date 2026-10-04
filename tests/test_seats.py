"""Roles played by a model of their own or by a person: the agent file fields, how a run
seats its cast, and a person's seat from joining to their turn. Pure: no database, no cluster."""
from __future__ import annotations

import asyncio

import pytest

from ailawlab import seats
from ailawlab.agent_spec import is_person, normalize_agent, parse_markdown, seat_cast, to_markdown


def test_played_by_and_model_read_and_write_as_agent_file_sections():
    [dana, sam] = parse_markdown("# Dana\n\n## Played by\nA person\n\n## Model\nqwen3.6:latest\n\n"
                                 "# Sam\n\n## Model\nhermes3:latest\n").agents
    # A person needs no model; the model line is dropped rather than left to confuse.
    assert dana["played_by"] == "person" and "model" not in dana
    assert sam["model"] == "hermes3:latest" and "played_by" not in sam
    text = to_markdown([dana, sam])
    assert "## Played by\nA person" in text and "## Model\nhermes3:latest" in text
    assert parse_markdown(text).agents == [dana, sam]
    assert is_person("a Human.") and is_person("Person") and not is_person("gemma4") and not is_person(None)
    assert "played_by" not in normalize_agent({"id": "x", "played_by": "AI"})


def test_a_run_can_change_who_plays_each_role():
    cast = [{"id": "dana", "name": "Dana", "model": "qwen3.6:latest"},
            {"id": "sam", "name": "Sam", "played_by": "person"},
            {"id": "ola", "name": "Ola"}]
    # As the experiment has it.
    assert [(a.get("model"), a.get("played_by")) for a in seat_cast(cast, None, None)] == [
        ("qwen3.6:latest", None), (None, "person"), (None, None)]
    # Swapped: Dana to the default, Sam to a model, Ola to a person.
    swapped = seat_cast(cast, {"dana": "", "sam": "gemma4:latest"}, {"ola": "u1"})
    assert [(a.get("model"), a.get("played_by")) for a in swapped] == [
        (None, None), ("gemma4:latest", None), (None, "person")]
    assert cast[0]["model"] == "qwen3.6:latest"        # the experiment's cast is untouched


def _state():
    return {"scenario": "A lease renewal.", "word_limit": 300,
            "agents": [{"id": "dana", "name": "Dana", "role": "Counsel", "bottom_line": "Walk at 5%",
                        "confidential": "The mine is closing."},
                       {"id": "sam", "name": "Sam", "role": "Landowner", "played_by": "person",
                        "libraries": ["sam-file"]}]}


def test_a_person_fills_in_their_profile_then_is_ready():
    async def go():
        live = seats.open_run("r1", _state(), {"sam": {"user": "u1", "name": "Pat"}}, 0)
        try:
            assert list(live.seats) == ["sam"] and live.phase == "joining"
            seats.set_profile(live, "sam", {"role": "  Ranch   owner ", "bottom_line": "No less than $40",
                                            "name": ""})
            seat = live.seats["sam"]
            assert seat.agent["role"] == "Ranch owner" and seat.agent["name"] == "Sam"
            assert [r["role"] for r in live.roster if r["id"] == "sam"] == ["Ranch owner"]
            waiting = asyncio.create_task(seats.wait_until_ready(live))
            await asyncio.sleep(0)
            assert not waiting.done()
            seats.set_ready(live, "sam")
            assert await asyncio.wait_for(waiting, 1) is True and live.phase == "running"
            with pytest.raises(ValueError):
                seats.set_profile(live, "sam", {"role": "changed"})
        finally:
            seats.close("r1")
    asyncio.run(go())


def test_a_person_sees_only_what_was_said_aloud():
    live = seats.open_run("r2", _state(), {"sam": {"user": "u1", "name": "Pat"}}, 0)
    try:
        seats.publish("r2", entries=[{"turn": 1, "agent_id": "dana", "name": "Dana", "role": "Counsel",
                                      "content": "We offer 3%.", "private_notes": {"plan": "go to 5%"},
                                      "thinking": "They will take 4%.", "sources": [{"marker": "S1"}],
                                      "model": "qwen3.6:latest"}],
                      exhibits=[{"marker": "E1", "label": "Lease", "content": "Clause 4.", "name": "Dana",
                                 "turn": 2, "chunk_id": 9, "document_id": 3}])
        # Dana's own [S3] reads as the exhibit it became.
        seats.publish("r2", entries=[{"turn": 2, "agent_id": "dana", "name": "Dana", "content": "See [S3]."}],
                      exhibits=[{"marker": "E1", "label": "Lease", "content": "Clause 4.", "name": "Dana",
                                 "turn": 2, "chunk_id": 9, "document_id": 3, "disclosed_by": "dana",
                                 "from_marker": "S3"}])
        assert live.transcript[-1]["markers"] == {"S3": "E1"}
        live.transcript.pop()
        view = seats.view(live, live.seats["sam"])
        text = repr(view)
        assert view["transcript"] == [{"turn": 1, "agent_id": "dana", "name": "Dana", "role": "Counsel",
                                       "content": "We offer 3%.", "markers": {}}]
        for secret in ("go to 5%", "They will take 4%", "Walk at 5%", "The mine is closing", "qwen"):
            assert secret not in text
        assert view["exhibits"][0]["marker"] == "E1" and "chunk_id" not in view["exhibits"][0]
        assert view["others"] == [{"id": "dana", "name": "Dana", "role": "Counsel", "person": False,
                                   "ready": True}]
        assert view["case_files"] == ["sam-file"] and not view["your_turn"]
    finally:
        seats.close("r2")


class _Sources:
    def places(self):
        return ["own"]

    def listing(self):
        return [{"marker": "S1", "label": "Deed", "private": True, "content": "..."}]


def test_a_persons_turn_waits_for_their_reply():
    async def go():
        live = seats.open_run("r3", _state(), {"sam": {"user": "u1", "name": "Pat"}}, 0)
        try:
            with pytest.raises(ValueError):
                seats.submit(live, "sam", "too early")
            reply = asyncio.create_task(seats.await_reply("r3", "sam", 2, "Answer Dana's question.", _Sources()))
            await asyncio.sleep(0)
            view = seats.view(live, live.seats["sam"])
            assert view["your_turn"] and view["turn"] == 2 and view["directive"] == "Answer Dana's question."
            assert view["can_search"] and view["found"][0]["marker"] == "S1" and view["deadline"] is None
            with pytest.raises(ValueError):
                seats.submit(live, "sam", "   ")
            seats.submit(live, "sam", "  We accept 4%, citing [S1].  ")
            assert await asyncio.wait_for(reply, 1) == "We accept 4%, citing [S1]."
            assert not seats.view(live, live.seats["sam"])["your_turn"]
        finally:
            seats.close("r3")
    asyncio.run(go())


def test_a_persons_turn_ends_on_the_time_limit_or_a_stop():
    async def go():
        live = seats.open_run("r4", _state(), {"sam": {"user": "u1", "name": "Pat"}}, 1)
        try:
            live.reply_minutes = 0.0005         # three hundredths of a second, for the test
            assert await seats.await_reply("r4", "sam", 1, "", None) is None
            live.reply_minutes = 0
            waiting = asyncio.create_task(seats.await_reply("r4", "sam", 2, "", None))
            await asyncio.sleep(0)
            assert seats.stop("r4") and await asyncio.wait_for(waiting, 1) is None
            assert live.stopped and await seats.wait_until_ready(live) is False
        finally:
            seats.close("r4")
        assert seats.get("r4") is None and not seats.stop("r4")
    asyncio.run(go())


def test_seats_are_found_by_the_person_holding_them():
    seats.open_run("r5", _state(), {"sam": {"user": "u7", "name": "Pat"}}, 0)
    try:
        assert [(lv.run_id, s.agent_id) for lv, s in seats.seats_of("u7")] == [("r5", "sam")]
        assert seats.seats_of("someone-else") == []
        assert seats.user_key(None) == "local" and seats.user_key({"id": 42}) == "42"
    finally:
        seats.close("r5")


def test_tool_calls_written_as_text_are_read_as_calls():
    from ailawlab.tools import text_tool_calls

    names = ["search_legal_sources"]
    plain = '{"arguments": {"query": "spring flow"}, "name": "search_legal_sources"}'
    assert text_tool_calls(plain, names) == [
        {"function": {"name": "search_legal_sources", "arguments": {"query": "spring flow"}}}]
    assert text_tool_calls(f"<tool_call>\n{plain}\n</tool_call>", names)[0]["function"]["name"] == names[0]
    # Speech, other tools, and JSON that merely mentions a tool are left alone.
    assert text_tool_calls("I will look that up.", names) == []
    assert text_tool_calls('{"name": "delete_everything", "arguments": {}}', names) == []
    assert text_tool_calls('{"offer": "3%"}', names) == []


def test_uncensored_models_are_not_offered():
    from ailawlab.models import hidden

    assert hidden("satgeze/qwen36-35b-uncensored-1m:latest") and hidden("huihui_ai/gemma-4-Abliterated:latest")
    assert not hidden("gemma4:latest") and not hidden("qwen3.6:latest") and not hidden("hermes3:latest")
