"""Dream-pass state and gate: has enough new information landed since the
last dream?

Split out of ``dream.py`` (AUD-121). State: ``metadata/dream.json``
(``last_run``, ``last_commit``), advanced only by :func:`mark_done`, plus
the ``metadata/dream.pending`` marker used by sweep to detect stalled
runs. No LLM, no wall clock — callers inject ``as_of``/``now``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import VaultPaths
from .dream_compiled import _strip_compiled_truth, _text_at, marker_state
from .dream_git import GitError, _git
from .metadata import latest_records_by_path
from .notes import _atomic_write, _split_frontmatter  # private helper, module-internal

_STATE_NAME = "dream.json"
_PENDING_NAME = "dream.pending"


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


# knowledge/concepts/ and knowledge/index/ are regenerated wholesale by the
# derived-notes refresher (concept notes, dashboards, the sweep report) —
# same generated/hand-written split as knowledge.KNOWLEDGE_NOTE_DIRS.
_GENERATED_PATH_PREFIXES = ("knowledge/concepts/", "knowledge/index/")
# The assistant inbox: a deterministic pass PROPOSING a fact (the wikilink
# pass, this module's contradiction detector) writes here with
# ``written_via: script``. A proposal is a question addressed to a human,
# never new information about the vault — counting it would let a proposer
# manufacture the quorum for the very dream that reads its output.
_PROPOSAL_INBOX_PREFIX = "knowledge/assistant/inbox/"


def _is_generated_note(root: Path, rel: str) -> bool:
    if rel.startswith(_GENERATED_PATH_PREFIXES):
        return True
    try:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False  # deleted/renamed since HEAD: not path-generated, count it
    frontmatter, _body = _split_frontmatter(text)
    if frontmatter.get("generated_by") == "dream-pass":
        return True
    return (
        rel.startswith(_PROPOSAL_INBOX_PREFIX)
        and frontmatter.get("written_via") == "script"
    )


def _only_compiled_truth_changed(root: Path, base_commit: str, rel: str) -> bool:
    """True when the ONLY thing that changed on this note since
    ``base_commit`` is the text between the compiled-truth markers.

    The compiled-truth job rewrites entity notes in place, and an entity
    note carries no ``generated_by:`` — so without this a dream's own three
    compiled-truth refreshes would count toward the next night's threshold
    of five. Same promise ``_changed_knowledge_notes`` already makes for
    connection notes and digests, extended to the one output the dream
    writes INSIDE a hand-written note."""
    if not base_commit:
        return False
    try:
        new_text = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if marker_state(new_text) != "ok":
        return False
    old_text = _text_at(root, base_commit, rel)
    if old_text is None:
        return False
    return _strip_compiled_truth(old_text) == _strip_compiled_truth(new_text)


def _changed_knowledge_notes(root: Path, base_commit: str) -> tuple[str, ...]:
    """Vault-relative changed notes, excluding generated ones (a note under
    ``knowledge/concepts/`` or ``knowledge/index/``, one stamped
    ``generated_by: dream-pass``, a script-written proposal in the assistant
    inbox, or a note whose only change is its compiled-truth block) so a
    derived-notes refresh, a deterministic proposer or a dream's own output
    can't trip the gate that is supposed to detect new information."""
    if base_commit:
        out = _git(root, "diff", "--name-only", f"{base_commit}..HEAD", "--", "knowledge")
    else:
        out = _git(root, "ls-files", "--", "knowledge")
    candidates = {ln.strip() for ln in out.splitlines() if ln.strip().endswith(".md")}
    return tuple(sorted(
        rel for rel in candidates
        if not _is_generated_note(root, rel)
        and not _only_compiled_truth_changed(root, base_commit, rel)
    ))


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
