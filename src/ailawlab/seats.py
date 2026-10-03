"""Seats for people in a live role-play: roles typed by a person instead of a model.

A role-play run lives in this server process, so the people playing in it are coordinated
here, in memory: who holds each seat, whether they have finished their profile and are
ready, whether it is their turn, and the passages they have found while preparing it.
The engine (graphs/roleplay.py) waits on a seat when the moderator calls on its person;
the web layer (the play page) reads a seat's view and hands in the person's reply.

What a person sees is what an AI agent in the same seat would see, no more: the scenario,
their own profile, everything said aloud, and the exhibits on the record. The others'
private notes, reasoning, case files and the trace stay out of the seat's view until the
run ends, which is why the run page itself is closed to a player while the run is live.

A run is lost if the server restarts, as any run is; a person's reply is recorded only
once the engine has taken it.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

# Fields a person may fill in or change in their own profile before the run starts.
PROFILE_FIELDS = ("name", "role", "goal", "backstory", "priorities", "bottom_line", "confidential")


def user_key(user: dict | None) -> str:
    """How a seat names the person who holds it. Development mode has one user with no id."""
    return str(user["id"]) if user and user.get("id") else "local"


@dataclass
class Seat:
    agent_id: str
    user: str                     # user_key of the person holding it
    player_name: str
    agent: dict                   # their profile; editable until they are ready
    ready: bool = False
    turn: int = 0                 # the turn they are asked to take, while it is open
    directive: str = ""
    deadline: float | None = None  # time.time() by which they must reply, if limited
    sources: Any = None           # the turn's TurnSources (graphs/roleplay.py), while open
    reply: asyncio.Future | None = None

    @property
    def turn_open(self) -> bool:
        return self.reply is not None and not self.reply.done()


@dataclass
class LiveRun:
    run_id: str
    experiment: str               # the experiment's name, for telling people where they play
    scenario: str
    word_limit: int
    roster: list[dict]            # every participant: id, name, role, and who plays it
    seats: dict[str, Seat]
    reply_minutes: int = 0        # 0: wait for a person's reply as long as it takes
    phase: str = "joining"        # joining -> running -> assessing -> done
    speaking: str = ""            # agent id of whoever is taking the current turn
    transcript: list[dict] = field(default_factory=list)   # public entries only
    exhibits: list[dict] = field(default_factory=list)
    stopped: bool = False
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def touch(self) -> None:
        self.changed.set()
        self.changed = asyncio.Event()

    def everyone_ready(self) -> bool:
        return all(s.ready for s in self.seats.values())


_live: dict[str, LiveRun] = {}


def open_run(run_id: str, state: dict, players: dict[str, dict], reply_minutes: int,
             experiment: str = "") -> LiveRun:
    """Register a run's seats. `players` maps an agent id to {user, name}."""
    agents = state["agents"]
    seats = {a["id"]: Seat(agent_id=a["id"], user=str(players[a["id"]]["user"]),
                           player_name=str(players[a["id"]].get("name") or "a person"),
                           agent=dict(a))
             for a in agents if a["id"] in players}
    roster = [{"id": a["id"], "name": a.get("name", a["id"]), "role": a.get("role", ""),
               "person": a["id"] in seats} for a in agents]
    live = LiveRun(run_id=run_id, experiment=experiment, scenario=state.get("scenario", ""),
                   word_limit=int(state.get("word_limit") or 0), roster=roster, seats=seats,
                   reply_minutes=max(0, int(reply_minutes or 0)))
    _live[run_id] = live
    return live


def get(run_id: str) -> LiveRun | None:
    return _live.get(run_id)


def close(run_id: str) -> None:
    live = _live.pop(run_id, None)
    if live:
        live.phase = "done"
        for seat in live.seats.values():
            if seat.reply and not seat.reply.done():
                seat.reply.cancel()
        live.touch()


def stop(run_id: str) -> bool:
    """Wake anything waiting on a person: the run is ending."""
    live = _live.get(run_id)
    if not live:
        return False
    live.stopped = True
    for seat in live.seats.values():
        if seat.reply and not seat.reply.done():
            seat.reply.set_result(None)
    live.touch()
    return True


async def wait_until_ready(live: LiveRun) -> bool:
    """Wait for every person to be ready. False if the run was stopped first."""
    while not live.everyone_ready():
        if live.stopped:
            return False
        await live.changed.wait()
    live.phase = "running"
    live.touch()
    return not live.stopped


def set_profile(live: LiveRun, agent_id: str, fields: dict) -> dict:
    seat = live.seats[agent_id]
    if seat.ready:
        raise ValueError("Your profile is fixed once you are ready.")
    for key in PROFILE_FIELDS:
        if key in fields:
            value = " ".join(str(fields[key] or "").split()) if key in ("name", "role") \
                else str(fields[key] or "").strip()
            if value:
                seat.agent[key] = value
            elif key != "name":
                seat.agent.pop(key, None)
    for r in live.roster:
        if r["id"] == agent_id:
            r.update(name=seat.agent.get("name", agent_id), role=seat.agent.get("role", ""))
    live.touch()
    return seat.agent


def set_ready(live: LiveRun, agent_id: str) -> None:
    live.seats[agent_id].ready = True
    live.touch()


def publish(run_id: str, entries: list[dict] | None = None, exhibits: list[dict] | None = None,
            speaking: str | None = None, phase: str | None = None) -> None:
    """What was said aloud, the exhibits now on the record, and who is speaking. Private
    notes, reasoning and the sources a speaker was shown are stripped here."""
    live = _live.get(run_id)
    if not live:
        return
    for e in entries or []:
        public = {k: e[k] for k in ("turn", "agent_id", "name", "role", "content", "disclosed")
                  if k in e}
        # A speaker's [S7] means nothing to anyone else; what it disclosed is now [E1].
        public["markers"] = {x["from_marker"]: x["marker"] for x in exhibits or []
                             if x.get("turn") == e.get("turn") and x.get("disclosed_by") == e.get("agent_id")
                             and x.get("from_marker")}
        live.transcript.append(public)
    if exhibits is not None:
        live.exhibits = [{k: e.get(k) for k in ("marker", "label", "title", "corpus", "content",
                                                 "name", "turn", "private")} for e in exhibits]
    if speaking is not None:
        live.speaking = speaking
    if phase is not None:
        live.phase = phase
    live.touch()


async def await_reply(run_id: str, agent_id: str, turn: int, directive: str,
                      sources: Any) -> str | None:
    """Open a person's turn and wait for their reply. None if the time ran out or the run
    was stopped."""
    live = _live[run_id]
    seat = live.seats[agent_id]
    seat.turn, seat.directive, seat.sources = turn, directive, sources
    seat.reply = asyncio.get_running_loop().create_future()
    seat.deadline = time.time() + live.reply_minutes * 60 if live.reply_minutes else None
    live.speaking = agent_id
    live.touch()
    try:
        if live.stopped:
            return None
        timeout = live.reply_minutes * 60 if live.reply_minutes else None
        try:
            return await asyncio.wait_for(asyncio.shield(seat.reply), timeout)
        except TimeoutError:
            return None
    finally:
        if not seat.reply.done():
            seat.reply.cancel()
        seat.deadline = None
        live.touch()


def submit(live: LiveRun, agent_id: str, text: str) -> None:
    seat = live.seats[agent_id]
    if not seat.turn_open:
        raise ValueError("It is not your turn.")
    text = text.strip()
    if not text:
        raise ValueError("Write what you want to say first.")
    seat.reply.set_result(text)
    live.touch()


def seats_of(user: str) -> list[tuple[LiveRun, Seat]]:
    return [(live, s) for live in _live.values() for s in live.seats.values() if s.user == user]


def view(live: LiveRun, seat: Seat) -> dict[str, Any]:
    """Everything the person in `seat` may see now."""
    names = {r["id"]: r["name"] for r in live.roster}
    found = []
    if seat.turn_open and seat.sources is not None:
        found = seat.sources.listing()
    return {
        "phase": live.phase,
        "stopped": live.stopped,
        "agent_id": seat.agent_id,
        "profile": {k: seat.agent.get(k, "") for k in PROFILE_FIELDS},
        "case_files": list(seat.agent.get("libraries") or []),
        "ready": seat.ready,
        "others": [{**r, "ready": live.seats[r["id"]].ready if r["person"] else True}
                   for r in live.roster if r["id"] != seat.agent_id],
        "scenario": live.scenario,
        "word_limit": live.word_limit,
        "transcript": live.transcript,
        "exhibits": live.exhibits,
        "speaking": names.get(live.speaking, "") if live.speaking != seat.agent_id else "",
        "your_turn": seat.turn_open,
        "turn": seat.turn if seat.turn_open else None,
        "directive": seat.directive if seat.turn_open else "",
        "deadline": seat.deadline if seat.turn_open else None,
        "can_search": bool(seat.sources and seat.sources.places()) if seat.turn_open else False,
        "places": seat.sources.places() if seat.turn_open and seat.sources else [],
        "found": found,
    }
