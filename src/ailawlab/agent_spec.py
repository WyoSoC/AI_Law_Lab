"""Agent and experiment files: plain Markdown that a lawyer can edit in any text editor.

An agent file holds one agent or a whole cast. Each agent starts at a top-level heading with
the person's name, and each part of the character is a second-level heading under it:

    # Dana Reyes

    ## Role
    Lead counsel for the Provider

    ## Tendencies
    - Anchors hard early
    - Reframes every risk as a dollar figure

An experiment file is the same kind of file with the rest of a role-play above the cast,
under reserved top-level headings, so a whole role-play can be saved, shared and uploaded
again as one file:

    # Scenario
    A mediation over the renewal of state gravel leases...

    # Settings
    Max turns: 100
    Words per turn: 1000

    # Dana Reyes
    ...

"# Source" may record where the scenario came from and "# Cast" may introduce the people;
both are optional. Every agent file is also a valid upload wherever experiment files are
accepted: the reader tells them apart by whether a Scenario, Settings or Source section
appears.

The people writing these files are lawyers, not programmers, so the format asks for no
syntax beyond `#` and `-`, and the reader is deliberately forgiving: headings match
case-insensitively and by common synonyms, every section but the name is optional, and
text before the first name or inside <!-- comments --> is ignored so a file can carry its
own instructions. What the reader cannot place is never silently dropped. It is kept under
"Additional notes" and reported back, because a lawyer who wrote a section called
"Leverage" should find out it was not read as one of the standard sections.

"Bottom line" and "Confidential information" mirror how negotiators actually prepare (a
walk-away point, and facts the other side does not have). Studies of LLM negotiators found
that agents without a threshold to measure offers against tend to repeat themselves rather
than converge; see graphs/roleplay.py for how the engine uses them.

Ids are generated from names rather than asked for. An id is plumbing (it scopes each
agent's private memory), and inventing one is exactly the kind of step that trips up
someone who has never edited a configuration file.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    key: str                          # field name in the experiment config
    heading: str                      # heading written in downloaded files
    hint: str                         # plain-language guidance in the blank template
    synonyms: tuple[str, ...] = ()
    is_list: bool = False
    in_template: bool = True


SECTIONS: tuple[Section, ...] = (
    Section("role", "Role",
            "Who this person is in the matter, e.g. lead counsel for the Provider.",
            ("position", "title", "party")),
    Section("goal", "Objective",
            "What they are trying to achieve in this exchange.",
            ("objectives", "goal", "goals", "aim", "aims", "what they want")),
    Section("backstory", "Background",
            "Where they come from, and the experience that shapes how they act.",
            ("backstory", "back story", "history", "experience", "biography", "bio",
             "background and history")),
    Section("demeanor", "Demeanor",
            "How they come across in the room.",
            ("demeanour", "temperament", "manner", "style", "personality", "tone")),
    Section("tendencies", "Tendencies",
            "Habits and tactics. Put each one on its own line, starting with a dash (-).",
            ("behavioral tendencies", "behavioural tendencies", "behavior", "behaviour",
             "behaviors", "behaviours", "habits", "tactics"),
            is_list=True),
    Section("priorities", "Priorities",
            "The interests behind their position: what they care about most, and why.",
            ("interests", "what they care about", "what they care about most", "concerns",
             "values")),
    Section("bottom_line", "Bottom line",
            "Private. The point past which they would rather walk away than agree, and what "
            "they would do instead.",
            ("walk away point", "walkaway point", "walk away", "reservation point",
             "reservation price", "batna", "red lines", "limits")),
    Section("confidential", "Confidential information",
            "Private. Facts only this person knows. They guard them unless revealing one helps.",
            ("confidential", "private information", "private facts", "secrets",
             "secret information", "what only they know", "hidden information")),
    Section("notes", "Additional notes",
            "Anything else about this person.",
            ("notes", "other notes", "extra notes"),
            in_template=False),
    Section("system_prompt", "Full prompt (advanced)",
            "A complete prompt written by hand. If present, it replaces everything above.",
            ("custom prompt", "full prompt", "system prompt"),
            in_template=False),
    Section("id", "Short id",
            "Generated from the name when left out.",
            ("id",),
            in_template=False),
)

TEMPLATE_INTRO = """<!--
HOW TO USE THIS FILE

Describe each person in the role-play under a line that starts with "# " followed by
their name. Under the name, fill in the sections that start with "## ".

  - Only the name is required. Delete or leave empty any section you do not need.
  - To describe several people in one file, repeat the whole block: another "# Name"
    line followed by its sections.
  - Under "Tendencies", put each habit on its own line starting with "- ".
  - "Bottom line" and "Confidential information" are private. The other people in the
    role-play never see them.
  - Anything between these arrow markers is instructions and is ignored, like this
    whole paragraph. You can delete it.

Save as plain text (.md or .txt) and upload it on the New experiment page.
-->"""

EXPERIMENT_INTRO = """<!--
HOW TO USE THIS FILE

This file holds a whole role-play: the scenario, its settings, and the cast.

  - Under "# Scenario", describe what the exchange is about, in plain paragraphs.
  - Under "# Settings", keep the lines "Max turns:" and "Words per turn:" and change only
    the numbers. Max turns can be 2 to {max_turns_limit}; words per turn 40 to {word_limit_max}.
  - "# Source" is optional. It records the article or document the scenario came from.
  - Then describe each person under a line that starts with "# " followed by their name,
    with sections that start with "## ", exactly as in an agent file.
  - "Bottom line" and "Confidential information" are private. The other people in the
    role-play never see them.
  - Anything between these arrow markers is instructions and is ignored, like this
    whole paragraph. You can delete it.

Save as plain text (.md or .txt) and upload it on the New experiment page.
-->"""


def _norm(label: str) -> str:
    """Heading text reduced to a lookup key: 'Walk-away point (private):' -> 'walk away point'."""
    s = re.sub(r"\([^)]*\)", " ", label).lower().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


_BY_LABEL: dict[str, Section] = {}
for _s in SECTIONS:
    for _label in (_s.heading, _s.key.replace("_", " "), *_s.synonyms):
        _BY_LABEL.setdefault(_norm(_label), _s)
_LIST_KEYS = {s.key for s in SECTIONS if s.is_list}

# Top-level headings that belong to the experiment rather than naming a person.
_RESERVED = {
    "scenario": "scenario", "the scenario": "scenario",
    "settings": "settings", "role play settings": "settings", "roleplay settings": "settings",
    "experiment settings": "settings", "pacing": "settings",
    "source": "source", "drafted from": "source", "source material": "source",
    "cast": "cast", "the cast": "cast", "participants": "cast", "agents": "cast",
}
_SETTING_KEYS = {
    "max turns": "max_turns", "maximum turns": "max_turns", "turn limit": "max_turns",
    "maximum number of turns": "max_turns", "turns": "max_turns",
    "words per turn": "word_limit", "max words per turn": "word_limit",
    "maximum words per turn": "word_limit", "word limit": "word_limit", "words": "word_limit",
}
_SOURCE_KEYS = {
    "title": "title", "link": "url", "url": "url", "web address": "url", "address": "url",
    "site": "site", "website": "site", "publisher": "site", "published": "published",
    "publication date": "published", "date published": "published", "retrieved": "retrieved_at",
    "read": "retrieved_at", "accessed": "retrieved_at", "date read": "retrieved_at",
    "words": "words", "kind": "kind", "type": "kind", "shortened": "truncated",
    "truncated": "truncated", "sha 256": "sha256", "sha256": "sha256", "fingerprint": "sha256",
}

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# A space (or a letter) must follow the hash, so a line such as "#1 concern is cost" inside
# a section is not mistaken for a new person.
_H1 = re.compile(r"^#(?!#)(?:\s+|(?=[A-Za-z]))(.*?)\s*#*\s*$")
_H2 = re.compile(r"^#{2,6}(?:\s+|(?=[A-Za-z]))(.*?)\s*#*\s*$")
# People used to word processors reach for bold instead of "##"; accept that for known names.
_BOLD_HEADING = re.compile(r"^\s*(\*\*|__)(.+?)\1\s*:?\s*$")
_LABEL_LINE = re.compile(
    r"^\s*(?:\*\*|__)?([^:*_\n]{2,40}?)(?:\*\*|__)?\s*:\s*(?:\*\*|__)?\s*(.*\S)\s*$")
_RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+•‣◦]|\d{1,3}[.)])\s+(.*)$")
_ESCAPED = re.compile(r"^(\s*)\\([#*_+-])")


def _unescape(line: str) -> str:
    return _ESCAPED.sub(r"\1\2", line)


def _inline(text: str) -> str:
    """Heading or name text without surrounding bold markers or a trailing colon."""
    text = re.sub(r"^(\*\*|__)(.*)\1$", r"\2", text.strip()).strip()
    return text.rstrip(":").strip()


def _text(lines: Iterable[str]) -> str:
    out = [_unescape(line.strip()) for line in lines]
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def _items(lines: Iterable[str]) -> list[str]:
    """List entries from dashes, numbers or bullets -- or one per line if there are none."""
    lines = list(lines)
    bulleted = any(_BULLET.match(line) for line in lines)
    items: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if m := _BULLET.match(raw):
            items.append(m.group(1).strip())
        elif bulleted and items:
            items[-1] = f"{items[-1]} {line}"      # a long entry wrapped onto the next line
        else:
            items.append(line)
    return [_unescape(i) for i in items if i]


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9_]+", "-", s.lower()).strip("-")
    return s[:40].strip("-") or "agent"


def normalize_agent(agent: dict) -> dict:
    """A tidy copy of one agent: strings trimmed, tendencies as a list, empty fields dropped.

    Agents reach the engine from uploaded files, the builder, and hand-written JSON, so
    this is the one place their shape is made consistent.
    """
    out: dict = {}
    for key, value in agent.items():
        if key in _LIST_KEYS:
            if isinstance(value, str):
                value = _items(value.split("\n"))
            value = [" ".join(str(v).split()) for v in (value or []) if str(v).strip()]
        elif isinstance(value, str):
            value = value.strip()
        if value not in ("", None, []):
            out[key] = value
    return out


def assign_ids(agents: list[dict], taken: Iterable[str] = ()) -> list[str]:
    """Give every agent a unique id (from its name unless one was set). Returns warnings."""
    used = set(taken)
    warnings: list[str] = []
    for a in agents:
        base = slugify(a.get("id") or a.get("name") or "")
        candidate, n = base, 2
        while candidate in used:
            candidate = f"{base}-{n}"
            n += 1
        if candidate != base:
            warnings.append(
                f'Two agents would share the id "{base}", so "{a.get("name") or base}" was '
                f'given "{candidate}". Check that the same person was not added twice.')
        a["id"] = candidate
        used.add(candidate)
    return warnings


@dataclass
class ParseResult:
    agents: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scenario: str = ""
    settings: dict[str, int] = field(default_factory=dict)   # max_turns, word_limit
    source: dict | None = None
    is_experiment: bool = False     # the file had a Scenario, Settings or Source section


class _Reader:
    """Line-by-line reader for one file."""

    def __init__(self, source: str):
        self.source = source
        self.agents: list[dict] = []
        self.warnings: list[str] = []
        self.agent: dict | None = None
        self.heading: str | None = None    # current "##" section; None directly under the name
        self.lines: list[str] = []
        self.loose: list[str] = []
        # Experiment sections: which reserved block is open, and what has been read from them.
        self.block: str | None = None
        self.block_lines: list[str] = []
        self.scenario = ""
        self.settings: dict[str, int] = {}
        self.source_info: dict = {}
        self.is_experiment = False

    def warn(self, message: str) -> None:
        self.warnings.append(f"{self.source}: {message}")

    def feed(self, raw: str) -> None:
        if m := _H1.match(raw):
            self.finish_agent()
            self.finish_block()
            heading = _inline(m.group(1))
            block = _RESERVED.get(_norm(heading))
            if block:
                self.block = block
                self.is_experiment = self.is_experiment or block != "cast"
            else:
                self.start_agent(heading)
            return
        if self.block is not None:
            self.block_lines.append(raw)
            return
        if self.agent is None:
            return                          # before the first name: the file's own notes
        h2 = _H2.match(raw)
        bold = _BOLD_HEADING.match(raw)
        if h2 or (bold and _norm(bold.group(2)) in _BY_LABEL):
            self.finish_section()
            self.heading = _inline(h2.group(1) if h2 else bold.group(2))
            return
        if _RULE.match(raw):
            raw = ""                        # a horizontal rule is only a visual separator
        if self.heading is None:
            label = _LABEL_LINE.match(raw)
            if label and _norm(label.group(1)) in _BY_LABEL:
                self.store(label.group(1), [label.group(2)])
            elif raw.strip():
                self.loose.append(raw)
            return
        self.lines.append(raw)

    # ------------------------------------------------------------ experiment sections

    def finish_block(self) -> None:
        lines = ["" if _RULE.match(line) else line for line in self.block_lines]
        if self.block == "scenario" and (text := _text(lines)):
            if self.scenario:
                self.warn('there is more than one "# Scenario" section; all of them were kept.')
                self.scenario = f"{self.scenario}\n\n{text}"
            else:
                self.scenario = text
        elif self.block == "settings":
            self.read_settings(lines)
        elif self.block == "source":
            self.read_source(lines)
        self.block, self.block_lines = None, []

    def _label_lines(self, lines: list[str], section: str):
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if m := _BULLET.match(line):
                line = m.group(1)
            label = _LABEL_LINE.match(line)
            if not label:
                self.warn(f'the "# {section}" section has a line that could not be read: '
                          f'"{line}". Write each line as a name, a colon and a value.')
                continue
            yield label.group(1).strip(), label.group(2).strip()

    def read_settings(self, lines: list[str]) -> None:
        for label, value in self._label_lines(lines, "Settings"):
            key = _SETTING_KEYS.get(_norm(label))
            if key is None:
                self.warn(f'"{label}" is not a setting this app knows, so it was ignored. The '
                          'settings are "Max turns" and "Words per turn".')
                continue
            number = re.search(r"\d[\d,]*", value)
            if not number:
                self.warn(f'"{label}" needs a number, for example "{label}: 100".')
                continue
            self.settings[key] = int(number.group(0).replace(",", ""))

    def read_source(self, lines: list[str]) -> None:
        for label, value in self._label_lines(lines, "Source"):
            key = _SOURCE_KEYS.get(_norm(label))
            if key is None:
                self.warn(f'"{label}" is not something the Source section records, so it was '
                          'ignored. Use Title, Link, Site, Published or Retrieved.')
            elif key == "url" and not re.match(r"^https?://", value, re.IGNORECASE):
                self.warn(f'the source link "{value}" is not a web address starting with '
                          "http:// or https://, so it was left out.")
            elif key == "words":
                if number := re.search(r"\d[\d,]*", value):
                    self.source_info["words"] = int(number.group(0).replace(",", ""))
            elif key == "truncated":
                self.source_info["truncated"] = value.strip().lower() in ("yes", "true", "y")
            else:
                self.source_info[key] = value

    # ------------------------------------------------------------ people

    def start_agent(self, name: str) -> None:
        if not name:
            name = f"Unnamed agent {len(self.agents) + 1}"
            self.warn(f'a "#" line has no name after it, so that agent was called "{name}". '
                      "Add the person's name after the #.")
        self.agent = {"name": name}
        self.heading, self.lines, self.loose = None, [], []

    def finish_section(self) -> None:
        if self.agent is not None and self.heading is not None:
            self.store(self.heading, self.lines)
        self.heading, self.lines = None, []

    def finish_agent(self) -> None:
        if self.agent is None:
            return
        self.finish_section()
        if loose := _text(self.loose):
            self.warn(f'"{self.agent["name"]}" has text directly under the name, before any "##" '
                      'heading. It was kept under "Additional notes"; move it under a heading '
                      'such as "## Background" if that is where it belongs.')
            self.append_text("notes", loose)
        self.agents.append(normalize_agent(self.agent))
        self.agent = None

    def append_text(self, key: str, text: str) -> None:
        assert self.agent is not None
        self.agent[key] = f"{self.agent[key]}\n\n{text}" if self.agent.get(key) else text

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        if abs(len(a) - len(b)) > 2:
            return 3
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    @classmethod
    def section_for(cls, label: str) -> Section | None:
        """A standard section by heading, forgiving a small typo ("Demeanories", "Backround").

        The advanced full-prompt section is matched exactly only, since it replaces every
        other section and must never be picked up by a near miss.
        """
        key = _norm(label)
        if key in _BY_LABEL:
            return _BY_LABEL[key]
        if len(key) < 6:
            return None
        for known, section in _BY_LABEL.items():
            if section.key == "system_prompt" or len(known) < 6:
                continue
            if key.startswith(known) or cls._edit_distance(key, known) <= 2:
                return section
        return None

    def store(self, label: str, lines: list[str]) -> None:
        assert self.agent is not None
        name = self.agent["name"]
        section = self.section_for(label)
        if section is None:
            if text := _text(lines):
                self.warn(f'"{name}" has a section called "{label}", which is not one of the '
                          'standard sections. Its text was kept under "Additional notes".')
                self.append_text("notes", f"{label}:\n{text}" if "\n" in text else f"{label}: {text}")
            return
        if section.is_list:
            value: str | list[str] = _items(lines)
        else:
            value = _text(lines)
        if not value:
            return                          # an empty section, e.g. straight from the template
        if section.key in self.agent and section.key != "notes":
            self.warn(f'"{name}" has more than one "{section.heading}" section; all of them '
                      "were kept.")
            if section.is_list:
                self.agent[section.key] = [*self.agent[section.key], *value]
                return
        if section.is_list:
            self.agent[section.key] = value
        else:
            self.append_text(section.key, value)


_NAME_LINE = re.compile(r"^\s*(?:(\*\*|__)(?P<bold>.+?)\1|#{2,6}\s+(?P<head>.+?)\s*#*)\s*$")


def promote_name_lines(text: str) -> str:
    """Read a bold or sub-heading line holding a name as "# Name" when a section follows it.

    People used to word processors bold a person's name instead of typing "#", and models
    do the same when asked for this layout (gemma4 wrote "**Eleanor Vance**" above "## Role").
    To leave an ordinary bold sentence or an unknown "## Leverage" section alone, the line
    must look like a name (two to eight words, not a section name), and the next section
    must be "## Role" -- unless the text has no "# Name" lines at all, when any standard
    section will do.
    """
    lines = text.split("\n")
    has_names = any(_H1.match(line) for line in lines)
    for i, line in enumerate(lines):
        m = _NAME_LINE.match(line)
        if not m:
            continue
        label = _inline(m.group("bold") or m.group("head") or "")
        if not 2 <= len(label.split()) <= 8 or label.endswith(".") or _norm(label) in _BY_LABEL:
            continue
        following = next((x for x in lines[i + 1:] if x.strip()), "")
        h2, bold = _H2.match(following), _BOLD_HEADING.match(following)
        section = _BY_LABEL.get(_norm(_inline(h2.group(1) if h2 else bold.group(2) if bold else "")))
        if section and (section.key == "role" or not has_names):
            lines[i] = f"# {label}"
    return "\n".join(lines)


def parse_markdown(text: str, source: str = "file", taken: Iterable[str] = ()) -> ParseResult:
    """Read every agent, and any experiment sections, in one file.

    `taken` holds ids already used elsewhere in the cast.
    """
    reader = _Reader(source)
    text = _COMMENT.sub("", text.lstrip("﻿")).replace("\r\n", "\n").replace("\r", "\n")
    text = promote_name_lines(text)
    for line in text.split("\n"):
        reader.feed(line)
    reader.finish_agent()
    reader.finish_block()
    result = ParseResult(reader.agents, reader.warnings, reader.scenario, reader.settings,
                         reader.source_info or None, reader.is_experiment)
    if not result.agents:
        result.warnings.append(f'{source}: no agents found. Each person must start on a line '
                               'like "# Dana Reyes".')
    result.warnings += [f"{source}: {w}" for w in assign_ids(result.agents, taken)]
    return result


def parse_files(files: Iterable[tuple[str, str]], taken: Iterable[str] = ()) -> ParseResult:
    """Read several uploaded files as one cast, keeping ids unique across all of them.

    The scenario, settings and source come from the first experiment file; any later one
    only contributes its people, and says so.
    """
    combined = ParseResult()
    used = set(taken)
    for name, text in files:
        r = parse_markdown(text, source=name, taken=used)
        combined.agents += r.agents
        combined.warnings += r.warnings
        used |= {a["id"] for a in r.agents}
        if not r.is_experiment:
            continue
        if combined.is_experiment:
            combined.warnings.append(f"{name}: an experiment file was already read, so this "
                                     "file's scenario and settings were ignored; its people "
                                     "were added to the cast.")
        else:
            combined.is_experiment = True
            combined.scenario, combined.settings, combined.source = r.scenario, r.settings, r.source
    return combined


def apply_limits(settings: dict[str, int], max_turns_limit: int,
                 word_limit_max: int) -> tuple[dict[str, int], list[str]]:
    """Settings brought within the ranges the builder allows, with a note for each change."""
    bounds = {"max_turns": (2, max_turns_limit, "Max turns"),
              "word_limit": (40, word_limit_max, "Words per turn")}
    out: dict[str, int] = {}
    warnings: list[str] = []
    for key, value in settings.items():
        if key not in bounds:
            continue
        low, high, label = bounds[key]
        clamped = max(low, min(int(value), high))
        if clamped != value:
            warnings.append(f"{label} must be between {low} and {high:,}, so {value:,} was "
                            f"changed to {clamped:,}.")
        out[key] = clamped
    return out, warnings


def _escape(text: str) -> str:
    """Protect lines that would otherwise be read back as headings or separators."""
    return "\n".join(
        f"\\{line}" if line.startswith("#") or _RULE.match(line) or _BOLD_HEADING.match(line)
        else line
        for line in text.split("\n"))


def to_markdown(agents: list[dict]) -> str:
    """Write agents in the standard layout.

    Ids are left out: reading a file makes a unique id for every agent from its name, so
    a "Short id" section would only show lawyers plumbing they never need to edit.
    parse_markdown reads everything else back unchanged.
    """
    blocks: list[str] = []
    for raw in agents:
        a = normalize_agent(raw)
        name = a.get("name") or a.get("id") or "Unnamed agent"
        lines = [f"# {name}"]
        for s in SECTIONS:
            value = a.get(s.key)
            if not value or s.key == "id":
                continue
            body = "\n".join(f"- {item}" for item in value) if s.is_list else _escape(str(value))
            lines += ["", f"## {s.heading}", body]
        blocks.append("\n".join(lines))
    return "\n\n\n".join(blocks) + "\n"


_SOURCE_LINES = (("title", "Title"), ("kind", "Kind"), ("url", "Link"), ("site", "Site"),
                 ("published", "Published"), ("retrieved_at", "Retrieved"), ("words", "Words"),
                 ("truncated", "Shortened"), ("sha256", "SHA-256"))


def to_experiment_markdown(scenario: str, settings: dict[str, int], agents: list[dict],
                           source: dict | None = None, max_turns_limit: int = 100,
                           word_limit_max: int = 2000) -> str:
    """Write a whole role-play as one file. parse_markdown reads it back unchanged."""
    parts = [EXPERIMENT_INTRO.format(max_turns_limit=max_turns_limit,
                                     word_limit_max=f"{word_limit_max:,}"),
             "", "# Scenario",
             _escape(scenario.strip()) if scenario.strip()
             else "<!-- What the exchange is about: the dispute, the parties, and what this "
                  "meeting is meant to settle. -->",
             "", "# Settings"]
    if "max_turns" in settings:
        parts.append(f"Max turns: {settings['max_turns']}")
    if "word_limit" in settings:
        parts.append(f"Words per turn: {settings['word_limit']}")
    if source and (source.get("title") or source.get("url")):
        parts += ["", "# Source"]
        for key, label in _SOURCE_LINES:
            value = source.get(key)
            if isinstance(value, bool):
                value = "yes" if value else "no"
            if value not in (None, ""):
                parts.append(f"{label}: {' '.join(str(value).split())}")
    text = "\n".join(parts) + "\n"
    return f"{text}\n\n{to_markdown(agents)}" if agents else text


def _agent_template_lines() -> list[str]:
    lines = ["# Full name of the person"]
    for s in SECTIONS:
        if s.in_template:
            lines += ["", f"## {s.heading}", f"<!-- {s.hint} -->"]
    return lines


def template() -> str:
    """A blank, commented agent file to fill in."""
    return "\n".join([TEMPLATE_INTRO, "", *_agent_template_lines()]) + "\n"


def experiment_template(max_turns: int = 100, word_limit: int = 1000, max_turns_limit: int = 100,
                        word_limit_max: int = 2000) -> str:
    """A blank, commented experiment file: scenario, settings, and one person to copy."""
    head = to_experiment_markdown("", {"max_turns": max_turns, "word_limit": word_limit}, [],
                                  max_turns_limit=max_turns_limit, word_limit_max=word_limit_max)
    return head + "\n\n" + "\n".join(_agent_template_lines()) + "\n"


def check_cast(agents: list[dict]) -> list[dict]:
    """Problems a person can fix before running, in plain language.

    Levels: "error" stops a run from working at all, "warning" will likely spoil it, and
    "suggestion" is advice that tends to make the exchange more realistic.
    """
    issues: list[dict] = []

    def add(level: str, message: str, agent: str | None = None) -> None:
        issues.append({"level": level, "agent": agent, "message": message})

    if len(agents) < 2:
        add("error", "A role-play needs at least two agents." if agents
            else "Add at least two agents to the cast.")

    ids: set[str] = set()
    names: set[str] = set()
    goals: dict[str, str] = {}
    for a in (normalize_agent(x) for x in agents):
        aid, name = a.get("id"), a.get("name")
        label = name or aid or "An unnamed agent"
        if not aid:
            add("error", f"{label} has no short id.")
        elif aid in ids:
            add("error", f'Two agents share the id "{aid}". Each agent needs its own, because the '
                "id is what keeps its memory private.", aid)
        ids.add(aid or "")
        if not name:
            add("warning", f'Agent "{aid}" has no name.', aid)
        elif name.casefold() in names:
            add("warning", f'Two agents are both named "{name}". The others will not be able to '
                "tell them apart.", aid)
        names.add((name or "").casefold())

        if a.get("system_prompt"):
            continue                        # a hand-written prompt is its author's call
        if not a.get("goal"):
            add("warning", f"{label} has no objective, so they have nothing to argue for.", aid)
        else:
            key = " ".join(a["goal"].casefold().split())
            if key in goals:
                add("warning", f"{goals[key]} and {label} have the same objective. Was one agent "
                    "copied from another?", aid)
            goals.setdefault(key, label)
        if not a.get("role"):
            add("warning", f"{label} has no role.", aid)
        if not a.get("bottom_line"):
            add("suggestion", f"{label} has no bottom line. Without a point where they would "
                "rather walk away, agents tend to argue in circles or give in too easily.", aid)
    return issues
