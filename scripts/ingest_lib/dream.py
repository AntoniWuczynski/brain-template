"""Dream-pass gate: deterministic prep for the LLM memory-consolidation
session.

The dream pass itself is an LLM session (``.claude/skills/dream-pass/``).
This module is everything deterministic around it:

- state: ``metadata/dream.json`` (``last_run``, ``last_commit``), advanced
  only by :func:`mark_done`, plus the ``metadata/dream.pending`` marker
  used by sweep to detect stalled runs.
- gate: has enough new information landed since the last dream?
- packet: the session's worklist, computed reproducibly from git history,
  ``metadata/index.jsonl`` and ``metadata/connections.jsonl``.

No LLM, no wall clock — callers inject ``as_of``/``now``.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

from .config import VaultPaths
from .connections import load_edges
from .metadata import latest_records_by_path
from .notes import _atomic_write

_STATE_NAME = "dream.json"
_PENDING_NAME = "dream.pending"


class GitError(RuntimeError):
    """A git subprocess failed or timed out."""


def _git(root: Path, *args: str, timeout: float = 30.0) -> str:
    """Run git with explicit argv (no shell), mirroring mcp_server/git_ops."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out after {timeout}s") from exc
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _iso_z(ts: datetime) -> str:
    """The vault's timestamp form: UTC, second precision, trailing Z."""
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(raw: str) -> datetime | None:
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


@dataclass(frozen=True)
class DreamState:
    """metadata/dream.json — where the last completed dream left off."""

    last_run: str    # trailing-Z UTC timestamp
    last_commit: str  # full sha at the moment mark_done ran


def load_state(paths: VaultPaths) -> DreamState | None:
    state_path = paths.metadata / _STATE_NAME
    if not state_path.is_file():
        return None
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    last_run = raw.get("last_run")
    last_commit = raw.get("last_commit")
    if not isinstance(last_run, str) or not isinstance(last_commit, str):
        return None
    return DreamState(last_run=last_run, last_commit=last_commit)


def mark_done(paths: VaultPaths, *, now: datetime, logger: logging.Logger) -> DreamState:
    """Advance the state to HEAD and clear the pending marker. Only the
    gate CLI calls this (after a successful dream) — never the LLM
    directly writing files.

    Race: a write landing between the dream's last commit and this stamp
    falls outside the next gate window (behind ``last_commit``, before
    ``last_run``) and will only be dreamed over if touched again — an
    accepted trade-off for a seconds-wide window at a quiet hour."""
    head = _git(paths.root, "rev-parse", "HEAD").strip()
    state = DreamState(last_run=_iso_z(now), last_commit=head)
    payload = json.dumps(asdict(state), ensure_ascii=False, sort_keys=True) + "\n"
    _atomic_write(paths.metadata / _STATE_NAME, payload)
    (paths.metadata / _PENDING_NAME).unlink(missing_ok=True)
    logger.info("dream state advanced: last_run=%s last_commit=%s", state.last_run, head)
    return state


def record_pending(paths: VaultPaths, *, now: datetime) -> None:
    """Stamp 'the gate fired'. First detection wins: an existing marker is
    left alone so its timestamp keeps pointing at the first unanswered
    gate pass, which is what the sweep staleness check needs."""
    pending = paths.metadata / _PENDING_NAME
    if pending.exists():
        return
    _atomic_write(pending, json.dumps({"since": _iso_z(now)}, sort_keys=True) + "\n")


def load_pending_since(paths: VaultPaths) -> datetime | None:
    pending = paths.metadata / _PENDING_NAME
    if not pending.is_file():
        return None
    try:
        raw = json.loads(pending.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    since = raw.get("since")
    if not isinstance(since, str):
        return None
    return _parse_ts(since)


@dataclass(frozen=True)
class GateVerdict:
    """Outcome of the change-volume gate."""

    should_dream: bool
    reason: str
    since: str                       # trailing-Z UTC: window start for new_sources
    base_commit: str                 # "" when diffing against all tracked files
    changed_notes: tuple[str, ...]   # vault-relative knowledge/**/*.md
    new_sources: tuple[str, ...]     # index.jsonl relative_path values
    days_since_last: int | None      # None on first run


def _changed_knowledge_notes(root: Path, base_commit: str) -> tuple[str, ...]:
    if base_commit:
        out = _git(root, "diff", "--name-only", f"{base_commit}..HEAD", "--", "knowledge")
    else:
        out = _git(root, "ls-files", "--", "knowledge")
    return tuple(sorted({ln.strip() for ln in out.splitlines() if ln.strip().endswith(".md")}))


def _commit_before(root: Path, cutoff: datetime) -> str:
    out = _git(root, "rev-list", "-1", "--before", cutoff.isoformat(), "HEAD").strip()
    return out


def _new_sources(paths: VaultPaths, *, since: datetime) -> tuple[str, ...]:
    fresh: list[str] = []
    for rel, rec in latest_records_by_path(paths.metadata_index_jsonl).items():
        created = _parse_ts(rec.created_at) if rec.created_at else None
        if created is not None and created > since:
            fresh.append(rel)
    return tuple(sorted(fresh))


def evaluate_gate(
    paths: VaultPaths,
    *,
    logger: logging.Logger,
    as_of: datetime,
    threshold: int = 5,
    stale_days: int = 7,
) -> GateVerdict:
    """Dream when >= threshold knowledge notes / sources changed since the
    last dream, or when anything changed and stale_days have passed.
    First run (no state): dream on any change within the last stale_days."""
    state = load_state(paths)
    window_start = as_of - timedelta(days=stale_days)
    if state is None:
        base = _commit_before(paths.root, window_start)
        since = window_start
        days: int | None = None
    else:
        base = state.last_commit
        parsed = _parse_ts(state.last_run)
        since = parsed if parsed is not None else window_start
        days = (as_of - since).days

    try:
        changed = _changed_knowledge_notes(paths.root, base)
    except GitError:
        # base commit vanished (rebase/gc): fall back to the time window
        logger.warning("dream gate: base commit %s unreachable, using %s-day window", base, stale_days)
        base = _commit_before(paths.root, window_start)
        changed = _changed_knowledge_notes(paths.root, base)

    sources = _new_sources(paths, since=since)
    total = len(changed) + len(sources)

    if state is None:
        should = total >= 1
        reason = f"first run: {total} change(s) in the last {stale_days} day(s)"
    elif total >= threshold:
        should = True
        reason = f"{total} change(s) >= threshold {threshold}"
    elif total >= 1 and days is not None and days >= stale_days:
        should = True
        reason = f"{total} change(s), {days} day(s) since last dream (stale override)"
    else:
        should = False
        reason = f"{total} change(s) < threshold {threshold}, not stale"

    logger.info("dream gate: %s -> %s", reason, "dream" if should else "skip")
    return GateVerdict(
        should_dream=should,
        reason=reason,
        since=_iso_z(since),
        base_commit=base,
        changed_notes=changed,
        new_sources=sources,
        days_since_last=days,
    )


class PairCandidate(TypedDict):
    """A semantic edge with no written-down relationship yet."""

    a: str
    b: str
    weight: float


class DreamPacket(TypedDict):
    """The dream session's entire worklist, computed deterministically."""

    since: str
    base_commit: str
    head_commit: str
    changed_notes: list[str]
    new_sources: list[str]
    candidate_pairs: list[PairCandidate]
    active_entities: list[str]
    existing_dream_notes: list[str]


_ENTITY_ROOTS = ("knowledge/people/", "knowledge/projects/", "knowledge/organisations/")


def _candidate_pairs(paths: VaultPaths, *, top_k: int) -> list[PairCandidate]:
    """Semantic-but-never-cooccurring pairs: the embedding space thinks
    these concepts relate but no note records why. Weight-ranked."""
    edges = load_edges(paths)
    linked = {(e.a, e.b) for e in edges if e.kind != "semantic"}
    semantic = [e for e in edges if e.kind == "semantic" and (e.a, e.b) not in linked]
    semantic.sort(key=lambda e: (-e.weight, e.a, e.b))
    return [PairCandidate(a=e.a, b=e.b, weight=e.weight) for e in semantic[:top_k]]


def _active_entities(changed_notes: tuple[str, ...]) -> list[str]:
    """Entity identifiers touched by the changeset. People notes are flat
    files (knowledge/people/anna.md -> knowledge/people/anna); projects and
    organisations are folders (knowledge/projects/brain/log/x.md ->
    knowledge/projects/brain)."""
    entities: set[str] = set()
    for note in changed_notes:
        for prefix in _ENTITY_ROOTS:
            if note.startswith(prefix):
                parts = note.split("/")
                if len(parts) > 3:
                    entities.add("/".join(parts[:3]))
                else:
                    entities.add(note.removesuffix(".md"))
    return sorted(entities)


def _existing_dream_notes(paths: VaultPaths) -> list[str]:
    dreams_dir = paths.knowledge / "notes" / "dreams"
    if not dreams_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(paths.root)) for p in dreams_dir.rglob("*.md") if p.is_file()
    )


def build_packet(
    paths: VaultPaths,
    *,
    verdict: GateVerdict,
    top_k: int = 10,
    logger: logging.Logger,
) -> DreamPacket:
    head = _git(paths.root, "rev-parse", "HEAD").strip()
    packet = DreamPacket(
        since=verdict.since,
        base_commit=verdict.base_commit,
        head_commit=head,
        changed_notes=list(verdict.changed_notes),
        new_sources=list(verdict.new_sources),
        candidate_pairs=_candidate_pairs(paths, top_k=top_k),
        active_entities=_active_entities(verdict.changed_notes),
        existing_dream_notes=_existing_dream_notes(paths),
    )
    logger.info(
        "dream packet: %d changed note(s), %d new source(s), %d candidate pair(s)",
        len(packet["changed_notes"]), len(packet["new_sources"]), len(packet["candidate_pairs"]),
    )
    return packet
