"""Shared normalisation for chat-transcript connectors (Claude Code session
transcripts, claude.ai / ChatGPT conversation exports).

Both connectors write the SAME snapshot schema — one JSON object per session
or conversation — so a single extractor (``extractors/transcript.py``) can
render either. See ``claude_code.py`` and ``chat_export.py`` for the two
producers, and the module docstring in each for source-specific detail.

**What survives extraction.** Only prose ``text`` turns from the user and
the assistant are kept. Everything else — tool calls, tool results, internal
"thinking" blocks, and harness bookkeeping events (hooks, mode changes,
token-budget reminders, queue operations, ...) — is dropped and only
counted in ``stats``, never written, so the loss is visible in the note
rather than silent (AGENTS.md's "honest extraction").

**Redaction.** Obvious credential patterns (cloud/API keys, tokens, private
key blocks, bearer/basic/header auth, JWTs, ``.env``/JSON/YAML-style secret
assignments in any casing, quoted or not, with or without a leading
``export``/``set``, and URLs with embedded userinfo) are replaced with
``[REDACTED:<kind>]`` in every kept turn BEFORE it is written to the
snapshot payload — ``inbox/`` and ``archive/raw/`` are immutable per
AGENTS.md, so a secret that lands there is not something a later pass can
retract, and this must happen here, not downstream. The caller's home
directory is also swapped for ``~`` as a privacy/hygiene measure, since
absolute paths routinely appear in tool output and file references quoted
in prose. **This is a pattern-based net, not a guarantee: it catches the
shapes developers actually paste (env assignments, JSON/YAML keys,
Authorization/Cookie/X-Api-Key headers, vendor-prefixed tokens, a bare
40-hex/40-base64 high-entropy value, URL userinfo), not everything a human
might paste — over-redacting a non-secret-looking value is an accepted
false positive, silently keeping a real secret is not** (review N1).

Snapshot schema — one JSON object, all keys present:

    {"connector": "claude_code"|"claude_ai"|"chatgpt", "id": "...",
     "title": "...", "date": "2026-09-03", "context": "...",
     "turns": [{"role": "user"|"assistant"|"marker", "text": "..."}, ...],
     # "marker" turns are synthetic — never produced by a connector's own
     # parsing, only spliced in by cap_session_turns to record an elided
     # gap (review N5); the extractor renders them without a speaker label.
     "stats": {"kept_user_turns": int, "kept_assistant_turns": int,
               "dropped_thinking_blocks": int, "dropped_tool_uses": int,
               "dropped_tool_results": int, "dropped_other_events": int,
               "dropped_sidechain_lines": int, "redacted_secrets": int,
               "truncated_long_turns": int, "dropped_non_text_parts": int,
               "dropped_wrapper_turns": int, "dropped_overflow_turns": int}}
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, TypedDict

_MAX_TURN_CHARS = 6000

_SESSION_MAX_CHARS_ENV = "BRAIN_TRANSCRIPT_MAX_CHARS"
_DEFAULT_SESSION_MAX_CHARS = 60_000

# Bound a title before it becomes a filename slug (see ``filename_safe_title``
# below) — independent of the truncation cap on turn TEXT above.
_MAX_FILENAME_TITLE_CHARS = 120


class TranscriptSourceError(RuntimeError):
    """An explicitly configured source (a ``--path``/env-var-pointed file or
    directory) does not resolve. Raised rather than treated like "nothing
    configured" so a typo'd path is a loud failure, not a silent zero-item
    pull indistinguishable from "nothing new" (see review M2)."""


# Pure harness furniture: an injected wrapper with no standalone value once
# stripped out. A skill/command BODY dump (large plain prose with no wrapper
# tag) is deliberately NOT special-cased here — see the module docstring in
# claude_code.py — it survives as an oversized turn, bounded by the
# truncation cap below like any other long turn.
_WRAPPER_TAGS = (
    "system-reminder",
    "local-command-stdout",
    "local-command-stderr",
    "local-command-caveat",
    "task-notification",
    "bash-input",
    "bash-stdout",
    "bash-stderr",
)

# The bare slash-command / skill-invocation wrapper: some mix of these four
# elements and NOTHING else, in EITHER observed ordering — the harness emits
# ``command-name`` first for a typed slash command but ``command-message``
# first (plus a trailing ``skill-format``) for a skill invocation.
_COMMAND_ELEMENT_RE = re.compile(
    r"<(?P<tag>command-name|command-message|command-args|skill-format)>"
    r"(?P<val>.*?)</(?P=tag)>",
    re.DOTALL,
)

# After known wrapper tags are stripped and a bare-command match is ruled
# out, a remainder that STILL opens with a tag SHAPED LIKE ONE OF THIS
# HARNESS'S OWN WRAPPER FAMILIES is treated as an unrecognised-but-harness
# wrapper rather than kept verbatim — an allowlist-shaped safety net for
# wrapper tags this module doesn't yet name individually, restricted to the
# harness's own tag vocabulary (``local-command-*``, ``task-*``, ``bash-*``,
# ``command-*``, ``skill-*``, ``system-*``) rather than ANY tag-shaped
# ``<word>``. A bare ``<[a-z][a-z0-9-]*>`` net (the pre-fix shape) also
# matches real pasted HTML/JSX/XML that a developer opens a turn with —
# ``<div>``, ``<details>``, ``<svg>``, a ``<template>`` SFC snippet — and
# would drop that turn's entire content, which is unrecoverable once
# ``inbox/``/``archive/raw/`` are written (review N2).
_HARNESS_TAG_OPEN_RE = re.compile(
    r"\A<(?:local-command|task|bash|command|skill|system)-[a-z0-9-]*>"
)

# Vendor-prefixed / structurally-distinctive tokens, checked BEFORE the
# generic env-assignment/header nets below so a recognised shape keeps its
# specific label instead of being swallowed by the generic ``env-secret``/
# ``http-header`` catch-alls (and so the generic nets' bracket-exclusion
# guard, see below, never has to fire in practice for these).
_REDACTIONS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private-key-block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"
    )),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("openai-key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("github-pat-token", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("npm-token", re.compile(r"npm_[A-Za-z0-9]{20,}")),
    ("gitlab-token", re.compile(r"glpat-[A-Za-z0-9_-]{20,}")),
    ("google-api-key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer-token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-_.=]{20,}")),
    ("basic-auth", re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{16,}")),
    # A bare 40-hex GitHub "classic" token — checked before the more
    # general base64-ish secret pattern below so a pure-hex 40-char run
    # gets the more precise label (review N1).
    ("github-classic-token", re.compile(r"\b[0-9a-fA-F]{40}\b")),
    # A generic 40-char high-entropy base64-ish run (e.g. an AWS secret
    # access key) that isn't pure hex — the fallback for a vendor shape
    # this module doesn't name individually (review N1).
    ("aws-secret-key", re.compile(r"\b[A-Za-z0-9/+]{40}\b")),
]

# ``.env``/JSON/YAML-style ``SOME_API_KEY=value`` / ``"some_token": "value"``
# / ``some-secret: value`` assignments, in any casing, with or without a
# leading ``export``/``set``, quoted or not. The key name is kept (it is
# often useful context, e.g. "which var was unset") and only the value is
# redacted.
#
# ``lead`` matches either the start of a line (``^``, for a bare ``.env``
# line) OR a single separator character that commonly precedes a key in
# prose/JSON/YAML/shell — whitespace, an opening quote/bracket/brace/paren,
# a comma, a hyphen (a YAML list item's ``- key: value``), or a URL
# query-string ``?``/``&`` — since a strict line-start anchor alone misses
# every one of those (``export API_KEY=...``, ``{"api_key": "..."}``,
# ``x-api-key: ...``, ``?api_key=...``; review N1). The prefix run before
# the keyword is OPTIONAL (``*``, not ``[A-Za-z_]`` + ``*``) — a mandatory
# leading character means a bare ``API_KEY=``/``TOKEN=``/``SECRET=``/
# ``PASSWORD=`` (the commonest shapes in a pasted ``.env``) never matches at
# all, since there is nothing before the keyword for that first character
# to consume (see review C2). An optional quote/whitespace is allowed
# between the keyword and the ``:``/``=`` so a JSON key's closing quote
# (``"api_key": ...``) doesn't break the match.
#
# The value's character class excludes ``[``/``]`` (in addition to
# whitespace/quotes/comma/brace/ampersand) so this can never re-match an
# already-emitted ``[REDACTED:...]`` placeholder from an earlier pass in
# ``redact_secrets`` (e.g. "here is a secret: [REDACTED:bearer-token]" must
# not be re-redacted as ``env-secret``, clobbering the more specific label)
# — this is why the vendor-prefixed patterns above run FIRST.
_ENV_ASSIGN_RE = re.compile(
    r"(?im)(?P<lead>^|[\s\"'\[{(,?&-])"
    r"(?P<prefix>[A-Za-z0-9_.-]*"
    r"(?:SECRET|TOKEN|API[_-]?KEY|APIKEY|PASSWORD|PASSWD|PRIVATE[_-]?KEY)"
    r"[A-Za-z0-9_.-]*['\"]?[ \t]*[:=][ \t]*)"
    r"(?P<quote>['\"]?)(?P<value>[^\s'\"\[\]{}&]{6,})(?P=quote)"
)

# A full ``Authorization``/``Cookie``/``X-Api-Key`` header line — these
# don't always carry a SECRET/TOKEN/KEY-shaped value (e.g. a raw opaque
# Cookie string), so they're matched by header NAME rather than by
# ``_ENV_ASSIGN_RE``'s keyword list. The negative lookahead skips a value
# already redacted by an earlier pass (e.g. ``Authorization: Basic ...``
# already turned into ``Authorization: [REDACTED:basic-auth]`` by the
# vendor patterns above) so it isn't re-redacted under the generic label.
_HEADER_LINE_RE = re.compile(
    r"(?im)^(?P<prefix>[ \t]*(?:Authorization|Cookie|X-Api-Key)[ \t]*:[ \t]*)"
    r"(?P<value>(?!\[REDACTED:)\S.*)$"
)

# ``scheme://user:password@host`` URL-embedded credentials (e.g. a Postgres
# or Redis connection string pasted into a turn). The username is OPTIONAL
# (``*``, not ``+``) — ``redis://:password@host`` (no username) is a common
# real shape and previously required a non-empty username to match at all
# (see review N1).
_URL_CREDENTIALS_RE = re.compile(r"(?P<scheme>://)[^/\s:@]*:[^/\s@]{4,}@")


class TranscriptTurn(TypedDict):
    # "marker" is never produced by a connector's own parsing — it is
    # inserted ONLY by ``cap_session_turns`` to mark an elided gap between
    # the kept head and tail windows (review N5), and rendered without a
    # speaker label by the extractor.
    role: Literal["user", "assistant", "marker"]
    text: str


class TranscriptStatsDict(TypedDict):
    kept_user_turns: int
    kept_assistant_turns: int
    dropped_thinking_blocks: int
    dropped_tool_uses: int
    dropped_tool_results: int
    dropped_other_events: int
    dropped_sidechain_lines: int
    redacted_secrets: int
    truncated_long_turns: int
    dropped_non_text_parts: int
    dropped_wrapper_turns: int
    dropped_overflow_turns: int


class TranscriptPayload(TypedDict):
    connector: str
    id: str
    title: str
    date: str
    context: str
    turns: list[TranscriptTurn]
    stats: TranscriptStatsDict


@dataclass
class TranscriptStats:
    kept_user_turns: int = 0
    kept_assistant_turns: int = 0
    dropped_thinking_blocks: int = 0
    dropped_tool_uses: int = 0
    dropped_tool_results: int = 0
    dropped_other_events: int = 0
    dropped_sidechain_lines: int = 0
    redacted_secrets: int = 0
    truncated_long_turns: int = 0
    dropped_non_text_parts: int = 0
    dropped_wrapper_turns: int = 0
    dropped_overflow_turns: int = 0

    def to_dict(self) -> TranscriptStatsDict:
        return {
            "kept_user_turns": self.kept_user_turns,
            "kept_assistant_turns": self.kept_assistant_turns,
            "dropped_thinking_blocks": self.dropped_thinking_blocks,
            "dropped_tool_uses": self.dropped_tool_uses,
            "dropped_tool_results": self.dropped_tool_results,
            "dropped_other_events": self.dropped_other_events,
            "dropped_sidechain_lines": self.dropped_sidechain_lines,
            "redacted_secrets": self.redacted_secrets,
            "truncated_long_turns": self.truncated_long_turns,
            "dropped_non_text_parts": self.dropped_non_text_parts,
            "dropped_wrapper_turns": self.dropped_wrapper_turns,
            "dropped_overflow_turns": self.dropped_overflow_turns,
        }


def home_relative(path: str, home: str) -> str:
    """``path`` with a leading ``home`` swapped for ``~``. A project label
    kept in a snapshot must not embed the operator's full home directory."""
    if home and path.startswith(home):
        rest = path[len(home):]
        return "~" + rest if rest else "~"
    return path


def _redact_env_assignments(text: str) -> tuple[str, int]:
    def repl(m: re.Match[str]) -> str:
        return (f"{m.group('lead')}{m.group('prefix')}{m.group('quote')}"
                f"[REDACTED:env-secret]{m.group('quote')}")
    return _ENV_ASSIGN_RE.subn(repl, text)


def _redact_header_lines(text: str) -> tuple[str, int]:
    def repl(m: re.Match[str]) -> str:
        return f"{m.group('prefix')}[REDACTED:http-header]"
    return _HEADER_LINE_RE.subn(repl, text)


def _redact_url_credentials(text: str) -> tuple[str, int]:
    def repl(m: re.Match[str]) -> str:
        return f"{m.group('scheme')}[REDACTED:url-credentials]@"
    return _URL_CREDENTIALS_RE.subn(repl, text)


def redact_secrets(text: str) -> tuple[str, int]:
    """Replace obvious credential patterns with ``[REDACTED:<kind>]``.
    Returns the cleaned text and how many replacements were made.

    Order matters: the vendor-prefixed/structural patterns in
    ``_REDACTIONS`` run FIRST so a recognised shape (an AWS key, a bearer
    token, a JWT, ...) keeps its specific label; the header-line and
    generic env-assignment nets run after and are guarded (see their
    docstrings) against re-redacting a placeholder an earlier pass already
    emitted (review N1)."""
    total = 0
    for label, pattern in _REDACTIONS:
        text, n = pattern.subn(f"[REDACTED:{label}]", text)
        total += n
    text, n = _redact_header_lines(text)
    total += n
    text, n = _redact_env_assignments(text)
    total += n
    text, n = _redact_url_credentials(text)
    total += n
    return text, total


def _collapse_bare_command(stripped: str) -> str | None:
    """``stripped`` is a bare slash-command / skill-invocation wrapper —
    some mix of ``_COMMAND_ELEMENT_RE`` elements and NOTHING else — collapse
    it to a one-line marker. Returns ``None`` when ``stripped`` is not
    (purely) one of these wrappers, so the caller falls through to the
    general wrapper-tag stripping below."""
    matches = list(_COMMAND_ELEMENT_RE.finditer(stripped))
    if not matches:
        return None
    remainder = _COMMAND_ELEMENT_RE.sub("", stripped).strip()
    if remainder:
        return None
    vals = {m.group("tag"): m.group("val").strip() for m in matches}
    name = vals.get("command-name") or vals.get("command-message") or ""
    if not name:
        return None
    args = vals.get("command-args") or ""
    return f"_(ran `{name}{' ' + args if args else ''}`)_"


# The exact shape ``_collapse_bare_command`` produces — used to recognise a
# collapsed-command MARKER (not real prose) when picking a session's title
# fallback, so a marker like ``_(ran `/dream-pass`)_`` is never mistaken for
# the user's first real message (review C1 residual).
_COLLAPSED_COMMAND_MARKER_RE = re.compile(r"\A_\(ran `[^`]*`\)_\Z")


def is_collapsed_command_marker(text: str) -> bool:
    """Whether ``text`` is exactly a ``_collapse_bare_command`` marker
    (``_(ran `/dream-pass`)_``) rather than real surviving prose."""
    return bool(_COLLAPSED_COMMAND_MARKER_RE.match(text))


def _strip_harness_tags(text: str) -> str:
    """Strip harness wrapper tags from one turn's raw text.

    A bare slash-command / skill-invocation wrapper collapses to a one-line
    marker (``_collapse_bare_command``). Otherwise, any INDIVIDUAL
    command-element tag (``<command-name>``, ``<command-message>``,
    ``<command-args>``, ``<skill-format>``) mixed into otherwise-real prose
    — e.g. a leading ``<command-name>/foo</command-name>`` before the
    user's actual question — is stripped on its own rather than left
    sitting in front of the remainder, where the allowlist net below would
    otherwise mistake that remainder for an unrecognised wrapper and drop
    the WHOLE turn, prose included (review N2). Known wrapper tags
    (``_WRAPPER_TAGS``) are stripped outright next. Anything that STILL
    opens with a tag shaped like this harness's OWN wrapper vocabulary
    after that (``_HARNESS_TAG_OPEN_RE`` — ``local-command-*``, ``task-*``,
    ``bash-*``, ``command-*``, ``skill-*``, ``system-*``, not any
    ``<word>``) is treated as harness furniture too and the whole turn is
    dropped (an allowlist-shaped safety net restricted to a known tag
    FAMILY, not a bare denylist; see review C1/N2). Returns ``""`` in that
    case, same as a pure known-wrapper turn with nothing else in it —
    ``finalize_turn_text`` turns that into ``None``."""
    stripped = text.strip()
    collapsed = _collapse_bare_command(stripped)
    if collapsed is not None:
        return collapsed
    stripped = _COMMAND_ELEMENT_RE.sub("", stripped)
    for tag in _WRAPPER_TAGS:
        stripped = re.sub(rf"<{tag}>.*?</{tag}>", "", stripped, flags=re.DOTALL)
    stripped = stripped.strip()
    if stripped and _HARNESS_TAG_OPEN_RE.match(stripped):
        return ""
    return stripped


def finalize_turn_text(raw_text: str, home: str) -> tuple[str | None, int, bool]:
    """Clean one turn's raw prose: strip harness wrapper tags, scrub the
    home directory, redact credentials, and cap length.

    Returns ``(text, redacted_count, truncated)``; ``text`` is ``None`` when
    nothing survives stripping (e.g. a pure harness marker with no prose) —
    callers count that against ``dropped_wrapper_turns`` (see review C1/m3).
    """
    text = _strip_harness_tags(raw_text)
    text = text.strip()
    if not text:
        return None, 0, False
    if home:
        text = text.replace(home, "~")
    text, n_redacted = redact_secrets(text)
    truncated = False
    if len(text) > _MAX_TURN_CHARS:
        text = text[:_MAX_TURN_CHARS].rstrip() + "\n\n…[truncated, transcript continues]"
        truncated = True
    return text, n_redacted, truncated


def filename_safe_title(title: str) -> str:
    """Bound ``title`` before it is turned into a filename slug by
    ``meeting_filename`` — an unbounded user-controlled title (a long chat
    export name, or a session's opening line under a deep project path) can
    otherwise produce a slug long enough to blow the filesystem's filename
    length limit and raise an uncaught ``OSError`` (see review M1). Only the
    FILENAME is bounded; the full title is kept in the snapshot/note."""
    title = title.strip()
    if len(title) <= _MAX_FILENAME_TITLE_CHARS:
        return title
    return title[:_MAX_FILENAME_TITLE_CHARS].rstrip()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_HEAD_BUDGET_FRACTION = 0.25


def cap_session_turns(turns: list[TranscriptTurn], stats: TranscriptStats) -> list[TranscriptTurn]:
    """Enforce a total-character budget across one session/conversation's
    turns (``BRAIN_TRANSCRIPT_MAX_CHARS``, default 60,000) — the per-turn cap
    in ``finalize_turn_text`` bounds one turn but not a whole session, so a
    long-running live session can otherwise produce a megabyte-plus snapshot
    that is then sent to the summariser in full (see review M3).

    Keeps BOTH ends — the oldest turns (up to 25% of the budget) and the
    newest turns (the remaining budget) — and drops the middle, rather than
    only ever dropping the oldest turns: a note that opens mid-conversation
    is missing exactly the framing request/decision that motivated the
    session, and could previously reach zero surviving USER turns (all of
    them elided) while still passing the caller's "no real exchange"
    check — that check only ran BEFORE this cap (review N5). When a gap is
    dropped, a synthetic ``role: "marker"`` turn recording how many turns
    were elided is spliced in between the two windows (the extractor
    renders it without a speaker label) so the loss is visible IN the
    transcript, not just in ``stats.dropped_overflow_turns``. At least one
    real turn always survives (the newest, even if it alone exceeds the
    budget) — this never returns an empty list for a non-empty input.
    """
    max_chars = _int_env(_SESSION_MAX_CHARS_ENV, _DEFAULT_SESSION_MAX_CHARS)
    if max_chars <= 0 or not turns or sum(len(t["text"]) for t in turns) <= max_chars:
        return turns

    head_budget = int(max_chars * _HEAD_BUDGET_FRACTION)
    tail_budget = max_chars - head_budget

    head: list[TranscriptTurn] = []
    running = 0
    for turn in turns:
        turn_len = len(turn["text"])
        if head and running + turn_len > head_budget:
            break
        running += turn_len
        head.append(turn)

    remaining = turns[len(head):]
    tail: list[TranscriptTurn] = []
    running = 0
    for turn in reversed(remaining):
        turn_len = len(turn["text"])
        if tail and running + turn_len > tail_budget:
            break
        running += turn_len
        tail.append(turn)
    tail.reverse()

    dropped_middle = turns[len(head):len(turns) - len(tail)]
    if not dropped_middle:
        return turns

    for turn in dropped_middle:
        if turn["role"] == "user":
            stats.kept_user_turns -= 1
        else:
            stats.kept_assistant_turns -= 1
    stats.dropped_overflow_turns += len(dropped_middle)

    marker: TranscriptTurn = {
        "role": "marker",
        "text": f"…[{len(dropped_middle)} turn(s) elided to stay under the size cap]…",
    }
    return head + [marker] + tail


def build_transcript_payload(
    *, connector: str, native_id: str, title: str, date: str, context: str,
    turns: list[TranscriptTurn], stats: TranscriptStats,
) -> TranscriptPayload:
    return {
        "connector": connector,
        "id": native_id,
        "title": title,
        "date": date,
        "context": context,
        "turns": turns,
        "stats": stats.to_dict(),
    }
