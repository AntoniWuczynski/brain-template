#!/usr/bin/env python3
"""Bridge Claude Code activity into the brain vault.

Three hook entry points, selected by argv[1]:

  mark   — PostToolUse(Write|Edit|MultiEdit). Sets per-session "dirty" flags in
           the temp dir: a MEMORY flag if the edited file lives under a Claude
           Code memory store (~/.claude/.../memory/), and a PROJECT flag if the
           edited file lives inside the current working project (excluding VCS /
           dependency / build noise).
  check  — Stop. If the MEMORY flag is set AND the brain MCP server is
           reachable, nudge the model to drop a *distilled* fact-note into
           knowledge/assistant/inbox/. If the PROJECT flag is set AND the server
           is reachable, nudge the model to append a distilled entry to this
           project's brain log (knowledge/projects/<slug>/log/<date>.md). Both
           nudges can fire in the same turn. If the server is DOWN the flags are
           left in place, so the nudge fires on the first Stop after recovery
           instead of being lost. Flags are consumed at emit time (one-shot).
  session — SessionStart. In a real project (not the brain repo itself): when
           the brain MCP server is reachable, inject one line of
           additionalContext nudging the model to load prior context via
           mcp__brain__memory_search (and TODO.md when present) and to log via
           the brain-project-note skill at the end; when the server is DOWN,
           inject a one-line outage notice instead so the model can tell the
           user memory is unavailable rather than silently knowing nothing.

Design (matches the agreed contract):
  - One-shot: dirty flags are cleared when their nudge is emitted, and we honour
    `stop_hook_active` so the harness never loops on us. Flags deliberately
    SURVIVE a server-down Stop — durability beats strict once-per-turn there.
  - Selective, not a mirror: the nudges ask the MODEL to judge what is durable
    knowledge. Operational / Claude-Code-only preferences are explicitly
    excluded — the brain's consolidation pass, not this hook, does promotion.
  - Fail-open: any error means "let the turn end". A memory-sync nudge must
    never block the user's work or crash their turn. The only signal emitted on
    failure is the SessionStart outage notice, which is informational context,
    never a block.

Configured in ~/.claude/settings.json (SessionStart -> session,
PostToolUse -> mark, Stop -> check).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

# A path is a Claude Code memory file when it sits under .claude/.../memory/.
# Both the per-project store (.claude/projects/<proj>/memory/MEMORY.md and its
# fact files) and any future .claude/memory/ store match this pair of markers.
_CLAUDE_MARKER = "/.claude/"
_MEMORY_MARKER = "/memory/"

# Directory names that are "project files" on disk but never worth a brain log.
_NOISE_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build", ".next", "__pycache__", ".cache"}

# /health is the server's only unauthenticated endpoint (mcp_server/auth.py),
# so a bare GET is a safe liveness probe. In a devcontainer set this to the
# host you reach the MCP at.
_HEALTH_URL = os.environ.get("BRAIN_MCP_HEALTH_URL", "http://127.0.0.1:8765/health")
_HEALTH_TIMEOUT_S = 1.0

# Subprocess budget: the session path can run up to two git calls plus the
# health probe, and the whole hook must finish inside its 10s settings.json
# timeout — keep git short so a slow disk degrades to the basename fallback
# instead of killing the nudge entirely.
_GIT_TIMEOUT_S = 2.0

_NUDGE = (
    "You updated your Claude Code memory this session. If any of it is DURABLE "
    "knowledge about a person, organisation, project, or meeting that belongs "
    "in the brain vault — not a Claude-Code operational preference or "
    "session-only detail — drop one distilled fact-note into "
    "knowledge/assistant/inbox/ via the brain MCP tools "
    "(mcp__brain__vault_create_note, following the "
    "knowledge/index/templates/memory-fact.md shape: a promote: mapping with "
    "target / relations / fact / source). The vault's consolidation pass will "
    "promote it deterministically. Do NOT mirror operational or preference "
    "memory into the vault. If nothing qualifies, say so in one line and stop."
)

_PROJECT_NUDGE = (
    "You did project work this session (edited files under the '{slug}' project). "
    "If anything DURABLE resulted — a decision, a research finding, a milestone, or "
    "new project state worth recalling in a future session — append a short, "
    "distilled entry to this project's brain log via the brain MCP tools: "
    "mcp__brain__vault_append_to_note on knowledge/projects/{slug}/log/{date}.md "
    "(or mcp__brain__vault_create_note if it does not exist yet), and ensure a "
    "knowledge/projects/{slug}/{slug}.md overview note exists. Use pointers to repo "
    "files, not copies; skip purely operational or trivial edits. If nothing durable "
    "happened, just stop without writing."
)

# SessionStart context-load nudge. Deterministic firing (the hook), model-driven
# action (the actual memory_search) — matches the _check philosophy: we point at
# the tool, the model judges what to do.
_SESSION_NUDGE = (
    "The brain MCP vault is connected and this looks like the '{slug}' project. "
    "Before substantive work, call mcp__brain__memory_search for prior context "
    "on this project (try query \"{slug}\", and recent decisions / people / "
    "meetings as relevant).{todo_clause} If the user's opening message asks for "
    "a status recap (\"where did we leave off\", \"what's the status\"), also "
    "read the latest note under knowledge/projects/{slug}/log/ via "
    "mcp__brain__vault_read and answer with a three-line last-session / "
    "current-status / next-actions recap. At the end of a session in which "
    "decisions were made, features shipped, or plans changed, save a session "
    "note with the brain-project-note skill. The brain is MCP-only — never "
    "write vault files directly; if nothing relevant comes back, just proceed."
)

_TODO_CLAUSE = (
    " The project root has a {name} — read it before reporting any task or "
    "project status."
)

_OUTAGE_NUDGE = (
    "The brain MCP vault appears to be DOWN (health check failed at {url}), so "
    "durable cross-session memory is unavailable this session. Mention this to "
    "the user at the first natural moment and suggest restarting the brain "
    "server; continue without vault context and do not call mcp__brain__* "
    "tools until it is back."
)


def _git_out(cwd: str, *args: str) -> str:
    """Run a read-only git command in ``cwd``; return stripped stdout or '' on
    any failure (not a repo, git missing, timeout). Never raises — the hook is
    fail-open by contract."""
    try:
        r = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, check=False,
        )
    except Exception:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _repo_top(cwd: str) -> str:
    """Git toplevel for ``cwd``, or ``cwd`` itself when not a repo. Computed
    once per hook run and threaded through, so the session path stays inside
    its subprocess budget."""
    return _git_out(cwd, "rev-parse", "--show-toplevel") or cwd


def _project_slug(cwd: str, top: str = "") -> str:
    """Project identity, matching the brain-project-note rule: the git remote
    basename, else the repo-root basename, else the cwd basename — slugified."""
    seg = ""
    url = _git_out(cwd, "remote", "get-url", "origin")
    if url:
        seg = url.rstrip("/").rsplit("/", 1)[-1]
        if seg.endswith(".git"):
            seg = seg[:-4]
    if not seg:
        t = top or _git_out(cwd, "rev-parse", "--show-toplevel")
        seg = os.path.basename(t) if t else ""
    if not seg:
        seg = os.path.basename(os.path.abspath(cwd))
    return _slugify(seg)


def _is_brain_repo(cwd: str, top: str = "") -> bool:
    """True when ``cwd`` is inside the brain vault itself — detected by the
    server-code marker so we don't nudge the brain to search the brain."""
    t = top or _repo_top(cwd)
    return os.path.exists(os.path.join(t, "mcp_server", "app.py"))


def _todo_file(cwd: str, top: str = "") -> str:
    """Name of the project's task file when one exists at the repo root (or
    cwd for non-repos), else ''. Only well-known names — this feeds a one-line
    nudge, not a search."""
    t = top or _repo_top(cwd)
    for name in ("TODO.md", "TODO.txt", "TASKS.md"):
        if os.path.exists(os.path.join(t, name)):
            return name
    return ""


def _is_project_edit(path: str, cwd: str) -> bool:
    """True when ``path`` is a file inside the working project ``cwd`` that is
    worth a brain log — i.e. under cwd and not in a VCS/dependency/build
    directory."""
    if not path or not cwd:
        return False
    try:
        ap = os.path.abspath(path)
        ac = os.path.abspath(cwd)
    except Exception:
        return False
    if ap != ac and not ap.startswith(ac + os.sep):
        return False
    rel_parts = ap[len(ac):].split(os.sep)
    return not any(p in _NOISE_DIRS for p in rel_parts)


def _session_payload(
    data: dict,
    *,
    server_up,
    project_slug,
    is_brain_repo,
    todo_file=lambda cwd: "",
) -> dict | None:
    """Pure decision for SessionStart: the additionalContext payload to print,
    or None to stay silent. Dependencies are injected so this is testable
    without git or a live server."""
    if data.get("source") == "compact":
        return None  # don't re-nudge on every context compaction
    cwd = str(data.get("cwd") or "")
    if not cwd:
        return None
    if is_brain_repo(cwd):
        return None
    if not server_up():
        # Surface the outage instead of silently knowing nothing — this is the
        # "can't recall prior context" confusion the vault exists to prevent.
        return {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": _OUTAGE_NUDGE.format(url=_HEALTH_URL),
            }
        }
    slug = project_slug(cwd)
    if not slug:
        return None
    todo = todo_file(cwd)
    todo_clause = _TODO_CLAUSE.format(name=todo) if todo else ""
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": _SESSION_NUDGE.format(slug=slug, todo_clause=todo_clause),
        }
    }


def _session(data: dict) -> int:
    """SessionStart: emit the context-load (or outage) nudge when appropriate."""
    cwd = str(data.get("cwd") or "")
    top = _repo_top(cwd) if cwd else ""
    payload = _session_payload(
        data,
        server_up=_server_up,
        project_slug=lambda c: _project_slug(c, top),
        is_brain_repo=lambda c: _is_brain_repo(c, top),
        todo_file=lambda c: _todo_file(c, top),
    )
    if payload is not None:
        print(json.dumps(payload))
    return 0


def _flag_for(prefix: str, session_id: str) -> str:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:128] or "unknown"
    return os.path.join(tempfile.gettempdir(), f"{prefix}{safe}")


def _flag_path(session_id: str) -> str:
    """Memory-dirty flag (Claude Code memory-store edits)."""
    return _flag_for("brain-mem-dirty-", session_id)


def _proj_flag_path(session_id: str) -> str:
    """Project-dirty flag (working-repo edits)."""
    return _flag_for("brain-proj-dirty-", session_id)


def _touch(path: str) -> None:
    try:
        open(path, "w").close()
    except OSError:
        pass  # fail-open: a missed flag just means no nudge this turn


def _read_stdin() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _mark(data: dict) -> int:
    """PostToolUse: flag the session when a memory file or project file was
    written."""
    tool_input = data.get("tool_input") or {}
    path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    sid = str(data.get("session_id") or "")
    if _CLAUDE_MARKER in path and _MEMORY_MARKER in path:
        _touch(_flag_path(sid))
    if _is_project_edit(path, str(data.get("cwd") or "")):
        _touch(_proj_flag_path(sid))
    return 0  # no stdout -> normal flow


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(_HEALTH_URL, timeout=_HEALTH_TIMEOUT_S) as resp:
            return 200 <= getattr(resp, "status", resp.getcode()) < 300
    except Exception:
        return False


def _project_nudge_text(data: dict) -> str:
    cwd = str(data.get("cwd") or "")
    slug = (_project_slug(cwd) if cwd else "") or "this"
    try:
        date = datetime.date.today().isoformat()
    except Exception:
        date = "<today>"
    return _PROJECT_NUDGE.format(slug=slug, date=date)


def _check(data: dict) -> int:
    """Stop: emit the nudge(s) once if memory/project files changed and the
    brain is reachable."""
    # Already continuing because of a stop hook -> never re-block (loop guard).
    if data.get("stop_hook_active"):
        return 0
    sid = str(data.get("session_id") or "")
    mem_flag = _flag_path(sid)
    proj_flag = _proj_flag_path(sid)
    mem = os.path.exists(mem_flag)
    proj = os.path.exists(proj_flag)
    if not mem and not proj:
        return 0
    # Server down: leave the flags in place so the nudge fires on the first
    # Stop after the server recovers, instead of being silently lost (the
    # exit-144 crash scenario). Loop safety still holds — nothing is emitted.
    if not _server_up():
        return 0
    # Consume the flags at emit time: guarantees one-shot and is loop-safe
    # even if stop_hook_active were ever absent.
    for f in (mem_flag, proj_flag):
        try:
            os.remove(f)
        except OSError:
            pass
    reasons = []
    if mem:
        reasons.append(_NUDGE)
    if proj:
        reasons.append(_project_nudge_text(data))
    print(json.dumps({"decision": "block", "reason": "\n\n".join(reasons)}))
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    data = _read_stdin()
    if mode == "mark":
        return _mark(data)
    if mode == "check":
        return _check(data)
    if mode == "session":
        return _session(data)
    return 0  # unknown mode: no-op, fail-open


if __name__ == "__main__":
    sys.exit(main())
