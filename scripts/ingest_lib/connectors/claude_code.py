"""Claude Code session-transcript connector (local-first).

Reads local Claude Code CLI session transcripts under
``~/.claude/projects/<slugged-cwd>/*.jsonl`` (one JSONL file per session,
inspected directly on this machine to learn the shape below) and normalises
each into the common transcript snapshot schema (see
``_transcript_common.py``), written to ``inbox/transcripts/claude_code/``.
No API, no auth — the "fetch" is a local file read.

**Snapshot unit: one session.** A session (one top-level ``<uuid>.jsonl``
file) is the unit a human would search for later ("that session where we
decided X"), and Claude Code already assigns every session a stable uuid to
key on. Splitting into individual messages would fragment context across
hundreds of tiny notes; grouping by day or by project would blur unrelated
work together into one.

**Title.** Preferred from the session's own ``ai-title`` event line (the
harness's own short, human-written summary of the session) when present;
otherwise the first surviving user turn that is NEITHER a harness wrapper
(which never survives as a turn's text — see below) NOR a collapsed
bare-command MARKER (``_(ran `/dream-pass`)_`` — a real turn's text, since
``_collapse_bare_command`` deliberately keeps it as one, but not prose a
human wrote; see review C1/N7). When every kept user turn is such a
marker — a session that only ever ran slash commands — the title falls
back to the slugged project path and date instead (see review C1 residual).

**What is dropped, deliberately** (counted in the snapshot's ``stats``,
never silently):

- ``thinking`` blocks — Claude's internal reasoning, not something either
  party "said"; frequently redacted/empty in this harness anyway.
- ``tool_use`` / ``tool_result`` blocks — the command/file/search
  machinery. High volume, low value once the decision is already captured
  in the surrounding prose.
- Non-conversation event lines: ``attachment`` (hook output, token-budget
  reminders, skill/agent listings, ...), ``system`` (hook summaries, turn
  timers), ``mode``, ``queue-operation``, ``atis-latch``, ``bridge-session``,
  ``last-prompt``, ``file-history-snapshot``/``-delta``, ``permission-mode``,
  ``cost-state`` — pure harness bookkeeping.
- Lines with ``isSidechain: true``, and everything under a session's
  ``subagents/`` folder — delegated subagent work, not the primary
  human/assistant dialogue.
- A bare slash-command invocation (``<command-message>``/``<command-name>``
  wrapper with nothing else) collapses to a one-line marker
  (``_(ran `/dream-pass`)_``) instead of being kept verbatim.
- Harness bookkeeping WRAPPED AROUND a user/assistant turn's own text —
  ``<system-reminder>``, ``<local-command-caveat>``, ``<task-notification>``,
  ``<bash-input>``/``<bash-stdout>``/``<bash-stderr>`` and (as a fallback
  allowlist net, restricted to this harness's own tag vocabulary, for
  shapes not yet named above — see review N2) any other
  ``local-command-*``/``task-*``/``bash-*``/``command-*``/``skill-*``/
  ``system-*``-shaped wrapper that leaves nothing else behind once
  stripped — dropped in ``_transcript_common._strip_harness_tags`` and
  counted as ``dropped_wrapper_turns``, never kept as the turn's text (see
  review C1). A bare slash-command marker (the item above) is DIFFERENT
  from this: it DOES survive as the turn's text, which is exactly why the
  title fallback below has to skip past it explicitly (review C1/N7).

Kept as-is, NOT specially detected: a large prose turn — including an
injected skill-file body (Claude Code inlines the skill's Markdown as a
plain user turn with no wrapper tag around it). It survives extraction like
any other long turn, bounded only by the shared truncation cap.

**Bounded by default** (see the env vars below) so a first pull on a
machine with hundreds of sessions across many projects cannot flood
``inbox/``. A session whose file was modified more recently than the
"settle" threshold (``BRAIN_CLAUDE_CODE_SETTLE_HOURS``, default 24 hours) is
skipped outright. This does double duty: it excludes the very session doing
the pulling (still being appended to, so a 5-minute grace would suffice for
that alone), but the longer default also stops a session that spans several
calendar days from being re-snapshotted whole on every pull while it is
still open — each such session settles into one snapshot after a day of
inactivity instead of duplicating megabytes into the immutable archive once
per day it stays open (see review M4).
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, TYPE_CHECKING

from ._transcript_common import (
    TranscriptSourceError,
    TranscriptStats,
    TranscriptTurn,
    build_transcript_payload,
    cap_session_turns,
    filename_safe_title,
    finalize_turn_text,
    home_relative,
    is_collapsed_command_marker,
)
from .base import Snapshot, meeting_filename

if TYPE_CHECKING:
    from ._transcript_common import TranscriptPayload
    from .state import ConnectorState

_GENERIC_PATH_ENV = "BRAIN_PULL_PATH"
_DIR_ENV = "BRAIN_CLAUDE_CODE_DIR"
_SINCE_DAYS_ENV = "BRAIN_CLAUDE_CODE_SINCE_DAYS"
_MAX_SESSIONS_ENV = "BRAIN_CLAUDE_CODE_MAX_SESSIONS"
_SETTLE_HOURS_ENV = "BRAIN_CLAUDE_CODE_SETTLE_HOURS"
_DEFAULT_DIR = "~/.claude/projects"
_DEFAULT_SINCE_DAYS = 14
_DEFAULT_MAX_SESSIONS = 50
_DEFAULT_SETTLE_HOURS = 24


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _iter_candidate_files(base: Path) -> list[Path]:
    """Top-level ``*.jsonl`` files directly under each project directory.
    Deliberately non-recursive: a ``subagents/`` sub-folder's transcripts
    are delegated work, not a session (see module docstring)."""
    files: list[Path] = []
    for project_dir in sorted(base.iterdir()):
        if project_dir.is_dir():
            files.extend(sorted(project_dir.glob("*.jsonl")))
    return files


def _line_text(content: object) -> tuple[str, int, int]:
    """Kept prose text from one message's ``content`` field, plus counts of
    dropped thinking blocks and dropped tool blocks. ``content`` is either a
    plain string or a list of typed blocks (``text``/``thinking``/
    ``tool_use``/``tool_result``)."""
    if isinstance(content, str):
        return content, 0, 0
    if not isinstance(content, list):
        return "", 0, 0
    texts: list[str] = []
    dropped_thinking = 0
    dropped_tool = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
        elif btype == "thinking":
            dropped_thinking += 1
        elif btype in ("tool_use", "tool_result"):
            dropped_tool += 1
    return "\n\n".join(texts), dropped_thinking, dropped_tool


def _parse_session(path: Path, home: str) -> TranscriptPayload | None:
    turns: list[TranscriptTurn] = []
    stats = TranscriptStats()
    project_cwd = ""
    first_date = ""
    ai_title = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj: object = json.loads(raw_line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue

            ts = obj.get("timestamp")
            if not first_date and isinstance(ts, str) and len(ts) >= 10:
                first_date = ts[:10]
            cwd = obj.get("cwd")
            if not project_cwd and isinstance(cwd, str):
                project_cwd = cwd

            otype = obj.get("type")
            if otype == "ai-title":
                # The harness already computed a clean, human-written title
                # for most sessions — captured like `cwd`/`timestamp` above
                # (not a dropped event) and preferred over the first prose
                # turn below when present.
                at = obj.get("aiTitle")
                if not ai_title and isinstance(at, str) and at.strip():
                    ai_title = at.strip()
                continue
            if otype != "user" and otype != "assistant":
                stats.dropped_other_events += 1
                continue
            if obj.get("isSidechain"):
                stats.dropped_sidechain_lines += 1
                continue

            message = obj.get("message")
            if not isinstance(message, dict):
                continue
            role_raw = message.get("role")
            role: Literal["user", "assistant"]
            if role_raw == "user":
                role = "user"
            elif role_raw == "assistant":
                role = "assistant"
            else:
                continue

            raw_text, dropped_thinking, dropped_tool = _line_text(message.get("content"))
            stats.dropped_thinking_blocks += dropped_thinking
            if role == "user":
                stats.dropped_tool_results += dropped_tool
            else:
                stats.dropped_tool_uses += dropped_tool
            if not raw_text.strip():
                # No text block at all. Only count this as a dropped
                # "wrapper turn" when nothing else already explains the
                # emptiness — a tool-result-only/tool-use-only/
                # thinking-only line is already counted via
                # dropped_tool_results/dropped_tool_uses/
                # dropped_thinking_blocks above, and double-counting it here
                # too made a rendered note claim far more "harness wrapper
                # turns" were dropped than actually happened (review N4).
                if dropped_thinking == 0 and dropped_tool == 0:
                    stats.dropped_wrapper_turns += 1
                continue

            text, n_redacted, truncated = finalize_turn_text(raw_text, home)
            if text is None:
                # Pure harness wrapper (see _strip_harness_tags) with no
                # surviving prose — never written, but counted (review C1/m3).
                stats.dropped_wrapper_turns += 1
                continue
            stats.redacted_secrets += n_redacted
            if truncated:
                stats.truncated_long_turns += 1
            if role == "user":
                stats.kept_user_turns += 1
            else:
                stats.kept_assistant_turns += 1
            turns.append({"role": role, "text": text})

    if stats.kept_user_turns == 0 or stats.kept_assistant_turns == 0:
        # No real exchange survived (aborted session, pure tool-only run,
        # or everything was harness machinery) — not worth a note.
        return None

    # Prefer the harness's own ai-title when the session has one; otherwise
    # fall back to the first surviving user turn that is real prose — a
    # harness WRAPPER never reaches `turns` as text at all
    # (finalize_turn_text collapses/drops it), but a bare-command MARKER
    # (``_(ran `/dream-pass`)_``) does survive as a turn's text and must be
    # skipped explicitly, or a session that opens with a slash command gets
    # the marker itself as its title (review C1 residual/N7). Computed on
    # the FULL turn list before the session cap below. When every kept user
    # turn is such a marker, fall back to the slugged project path + date
    # rather than a marker string.
    if ai_title:
        title_line = ai_title[:80]
    else:
        first_real_user_text = next(
            (t["text"] for t in turns
             if t["role"] == "user" and not is_collapsed_command_marker(t["text"])),
            None,
        )
        if first_real_user_text is not None:
            title_line = first_real_user_text.strip().splitlines()[0][:80]
        else:
            title_line = first_date or "session"
    project_label = home_relative(project_cwd, home) if project_cwd else ""
    title = f"{project_label}: {title_line}" if project_label else title_line

    turns = cap_session_turns(turns, stats)
    if stats.kept_user_turns == 0 or stats.kept_assistant_turns == 0:
        # The head/tail cap (review N5) can still leave a session with no
        # real exchange in the surviving window (e.g. every user turn fell
        # in the elided middle) — re-run the same guard the parser already
        # applied before capping, so that isn't written as a note either.
        return None

    return build_transcript_payload(
        connector="claude_code", native_id=path.stem, title=title, date=first_date,
        context=project_label, turns=turns, stats=stats,
    )


class ClaudeCodeConnector:
    name = "claude_code"

    def pull(self, state: ConnectorState) -> Iterator[Snapshot]:
        # An explicit ``--path`` (which pull.py sets as BRAIN_PULL_PATH) is
        # the more specific, more recently expressed configuration, so it
        # wins over BRAIN_CLAUDE_CODE_DIR rather than being silently
        # ignored — pull.py --path previously only worked for chat_export
        # (review N6). For this connector ``--path`` names a SESSIONS
        # DIRECTORY (the same shape as BRAIN_CLAUDE_CODE_DIR), not a file.
        generic = os.environ.get(_GENERIC_PATH_ENV)
        dir_raw = generic or os.environ.get(_DIR_ENV)
        base = Path(dir_raw or _DEFAULT_DIR).expanduser()
        if not base.is_dir():
            if dir_raw:
                # Explicitly configured (not just the default missing on a
                # machine without Claude Code) — a typo'd dir must not look
                # like "nothing new" (review M2).
                env_name = _GENERIC_PATH_ENV if generic else _DIR_ENV
                raise TranscriptSourceError(
                    f"{env_name} points at a path that is not a directory: {base}"
                )
            return
        home = str(Path.home())
        since_days = _int_env(_SINCE_DAYS_ENV, _DEFAULT_SINCE_DAYS)
        max_sessions = _int_env(_MAX_SESSIONS_ENV, _DEFAULT_MAX_SESSIONS)
        settle_seconds = _int_env(_SETTLE_HOURS_ENV, _DEFAULT_SETTLE_HOURS) * 3600
        now = time.time()
        cutoff = now - since_days * 86400 if since_days > 0 else None

        candidates: list[tuple[float, Path]] = []
        for f in _iter_candidate_files(base):
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            if settle_seconds > 0 and now - mtime < settle_seconds:
                continue
            if cutoff is not None and mtime < cutoff:
                continue
            candidates.append((mtime, f))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        if max_sessions > 0:
            candidates = candidates[:max_sessions]

        for _, f in candidates:
            try:
                payload = _parse_session(f, home)
            except OSError:
                continue
            if payload is None:
                continue
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            yield Snapshot(
                source_class="transcripts/claude_code", native_id=payload["id"],
                filename=meeting_filename(
                    payload["date"], filename_safe_title(payload["title"]), payload["id"]
                ),
                payload=body,
            )
