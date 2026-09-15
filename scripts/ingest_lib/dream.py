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
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, TypedDict

from .config import VaultPaths
from .connections import load_edges
from .metadata import latest_records_by_path
from .notes import _atomic_write, _split_frontmatter  # private helper, module-internal
from .propose import ProposeResult, propose_fact
from .relations import (  # private helpers: one definition of "what is a ## Log bullet"
    _HEADING_RE,
    _LOG_HEADING,
    Relation,
    _code_fence_flags,
    node_id_for_note,
    parse_relations,
)

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
    except subprocess.TimeoutExpired:
        # `from None`, and stderr capped, for the same reason as git_ops:
        # this text reaches the scheduled-run log via dream_gate, and raw git
        # stderr can carry remote URLs / ssh hints.
        raise GitError(f"git {' '.join(args)} timed out after {timeout}s") from None
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()[:300]}")
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


def _strip_compiled_truth(text: str) -> str:
    """The note with its compiled-truth block emptied, so two revisions that
    differ only inside the fence compare equal — and so no reader of the
    note's prose (``_log_bullets`` above all) can mistake the dream's own
    summary for evidence the vault gained."""
    span = compiled_truth_span(text)
    if span is None:
        return text
    start, end = span
    return text[:start] + text[end:]


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


class PairCandidate(TypedDict):
    """A semantic edge with no written-down relationship yet."""

    a: str
    b: str
    weight: float


class CompiledTruthCandidate(TypedDict):
    """An entity note whose compiled-truth block is worth refreshing.

    ``marker_state`` is ``absent`` (no block yet - the job writes the first
    one) or ``ok`` (a block exists between the markers and is replaced in
    place). ``broken`` notes never become candidates; they land in the
    packet's ``compiled_truth_blocked`` list instead.

    The three evidence fields are what makes a note a candidate: the block
    is a synthesis of the entity's ``## Log`` AND its open relations, so
    either moving is new evidence. ``new_log_bullets`` counts bullets added
    anywhere under the entity (a folder-backed project's evidence lands in
    ``log/<date>.md``, not in the overview's own ``## Log``),
    ``changed_notes`` counts the changeset's notes under the entity, and
    ``relations_changed`` says the rendered open relations differ from
    ``base_commit`` — a closed relation appends no Log bullet at all.
    """

    node_id: str            # canonical id: people/anna, projects/brain/brain
    note_path: str          # vault-relative, including .md
    marker_state: str       # "absent" | "ok"
    log_bullets: int        # total ## Log bullets on the entity note
    new_log_bullets: int    # ## Log bullets added under the entity since base_commit
    changed_notes: int      # changeset notes under the entity, besides its own note
    relations_changed: bool  # open relations differ from base_commit
    open_relations: list[str]   # "<rel> -> <target> (since <date>)", no valid_until


class ProposalRelation(TypedDict):
    """One ``promote.relations`` entry on a contradiction proposal.

    Every key is present (empty string when unset) so the mapping converts
    to a :class:`~ingest_lib.relations.Relation` — and back — without
    optional-key narrowing, and so the proposal's content hash is stable.
    """

    rel: str
    target: str
    valid_from: str
    valid_until: str
    source: str


class ContradictionCandidate(TypedDict):
    """Two claims on one entity note that look like they disagree.

    A HINT, never a verdict: the detectors are lexical and interval
    arithmetic, so the LLM adjudicates and a human gates the result. The
    job may only propose - it never edits the entity note.
    """

    kind: str               # relation_interval_conflict | relation_vs_log | restated_fact
    node_id: str
    note_path: str
    left: str               # the first claim, verbatim
    right: str              # the second claim, verbatim
    detail: str             # why the detector paired them
    proposal_path: str      # vault-relative fact note to create (incl. .md)
    proposal_fact: str      # the promote.fact line that path is addressed by
    # The closure a human's approval would apply: the same (rel, target)
    # carrying valid_until. Empty when the finding implies no closure.
    promote_relations: list[ProposalRelation]


class EditBudget(TypedDict):
    """How the SKILL's per-run edit cap is split across the jobs.

    ``total_note_edits`` mirrors the skill's tripwire (replaces of
    pre-existing notes). Compiled truth gets a fixed slice so it cannot
    starve digests, consolidation and the questions rewrite.
    ``contradiction_proposals`` bounds a different resource: fact-note
    CREATIONS, which the cap does not count.
    """

    total_note_edits: int
    compiled_truth: int
    reserved_for_other_jobs: int
    contradiction_proposals: int


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
    compiled_truth: list[CompiledTruthCandidate]
    compiled_truth_blocked: list[str]
    contradictions: list[ContradictionCandidate]
    edit_budget: EditBudget


_ENTITY_ROOTS = ("knowledge/people/", "knowledge/projects/", "knowledge/organisations/")


def _written_connection_stems(paths: VaultPaths) -> set[str]:
    """File stems of the connection notes the dream has already written.
    Matched whole (never split on ``--``, which can occur inside a slug)."""
    conn_dir = paths.knowledge / "notes" / "dreams" / "connections"
    if not conn_dir.is_dir():
        return set()
    return {p.stem for p in conn_dir.glob("*.md") if p.is_file()}


def _candidate_pairs(paths: VaultPaths, *, top_k: int) -> list[PairCandidate]:
    """Semantic-but-never-cooccurring pairs: the embedding space thinks
    these concepts relate but no note records why. Weight-ranked.

    Pairs whose connection note already exists are dropped before the
    top_k cut — in either name order, since notes predating the edge
    list's ``a <= b`` invariant are stored as ``<b>--<a>`` — so handled
    pairs cannot permanently occupy the packet's slots."""
    edges = load_edges(paths)
    written = _written_connection_stems(paths)
    linked = {(e.a, e.b) for e in edges if e.kind != "semantic"}
    semantic = [e for e in edges if e.kind == "semantic" and (e.a, e.b) not in linked]
    semantic.sort(key=lambda e: (-e.weight, e.a, e.b))
    fresh = [e for e in semantic if f"{e.a}--{e.b}" not in written and f"{e.b}--{e.a}" not in written]
    return [PairCandidate(a=e.a, b=e.b, weight=e.weight) for e in fresh[:top_k]]


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


# ---------------------------------------------------------------------------
# compiled truth: a marker-guarded "current best understanding" block
# ---------------------------------------------------------------------------

# The block the compiled-truth job owns is FENCED by these two markers, the
# same idea as the concept notes' AUTO-GENERATED-END marker
# (mcp_server.tools.tool_update_concept_user_section): everything outside is
# hand-written and survives untouched. A fence rather than a single marker
# because an entity note has hand-written prose on BOTH sides — frontmatter
# and the note's own body above, the ## Log below — where a concept note has
# only generated-above / hand-written-below.
COMPILED_TRUTH_START: Final = "<!-- COMPILED-TRUTH-START -->"
COMPILED_TRUTH_END: Final = "<!-- COMPILED-TRUTH-END -->"

# The skill's tripwire: at most this many pre-existing notes replaced per run.
EDIT_CAP: Final = 10
# Compiled truth's slice of it. Deliberately small: digests, consolidation
# and the questions rewrite need the rest, and a compiled-truth block that
# lags one night is harmless.
COMPILED_TRUTH_BUDGET: Final = 3
# Contradiction findings are note CREATIONS (fact notes in the assistant
# inbox), which the edit cap does not count — this bounds the packet and the
# human's review queue instead.
CONTRADICTION_BUDGET: Final = 5


def _marker_offsets(text: str) -> tuple[list[int], list[int]]:
    """Offsets of every START / END marker that is real Markdown, in file
    order.

    Occurrences inside a fenced code block do not count — a note DOCUMENTING
    the fence (this skill's own step 6 example, an AGENTS.md excerpt) carries
    both markers in a ```` ``` ```` block, and reading those as the note's own
    fence would have the tool overwrite the example. Same fence detection
    ``_log_bullets`` already uses, so "what counts as real Markdown" has one
    definition in this module.
    """
    lines = text.split("\n")
    in_fence = _code_fence_flags(lines)
    starts: list[int] = []
    ends: list[int] = []
    pos = 0
    for i, line in enumerate(lines):
        if not in_fence[i]:
            for marker, found in ((COMPILED_TRUTH_START, starts), (COMPILED_TRUTH_END, ends)):
                at = line.find(marker)
                while at >= 0:
                    found.append(pos + at)
                    at = line.find(marker, at + 1)
        pos += len(line) + 1
    return starts, ends


def marker_state(text: str) -> str:
    """``"absent"`` | ``"ok"`` | ``"broken"`` for a note's compiled-truth fence.

    ``broken`` is any fence a rewrite could not land on safely: one marker
    without the other, END before START, or a duplicated marker. Mirrors
    concepts.py's refusal to regenerate a note with ambiguous markers —
    guessing where the block ends is how a hand-written section gets eaten.
    """
    starts, ends = _marker_offsets(text)
    if not starts and not ends:
        return "absent"
    if len(starts) != 1 or len(ends) != 1:
        return "broken"
    return "ok" if starts[0] < ends[0] else "broken"


def compiled_truth_span(text: str) -> tuple[int, int] | None:
    """``(first byte of the block, offset of the END marker)`` for a note
    whose fence is ``ok``; ``None`` otherwise.

    The one place that decides WHERE the block is: ``_strip_compiled_truth``
    here and ``mcp_server.tools.tool_update_compiled_truth`` both slice on
    it, so neither can land on a marker the other would have ignored (one
    inside a code fence, say).
    """
    starts, ends = _marker_offsets(text)
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        return None
    return starts[0] + len(COMPILED_TRUTH_START), ends[0]


def _log_bullets(text: str) -> list[str]:
    """The note's ``## Log`` bullets, stripped, in file order.

    Same section detection ``relations.append_fact_to_log`` writes with (and
    ``mcp_server.tools._log_section`` guards with): the first non-fenced line
    that is exactly ``## Log``, ending at the next heading of any level.
    """
    lines = text.split("\n")
    in_fence = _code_fence_flags(lines)
    heading_idx = -1
    for i, line in enumerate(lines):
        if not in_fence[i] and line.strip() == _LOG_HEADING:
            heading_idx = i
            break
    if heading_idx < 0:
        return []
    end = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        if not in_fence[j] and _HEADING_RE.match(lines[j]):
            end = j
            break
    return [
        lines[j].rstrip("\r").strip()
        for j in range(heading_idx + 1, end)
        if lines[j].rstrip("\r").strip().startswith("- ")
    ]


def _entity_note_path(root: Path, active_entity: str) -> str | None:
    """``knowledge/people/anna`` -> ``knowledge/people/anna.md``;
    ``knowledge/projects/brain`` -> ``knowledge/projects/brain/brain.md``.

    Resolution is by existence only (AGENTS.md 'folder-backed entities'):
    a project's canonical note is the overview INSIDE its folder, and the
    flat path is checked first because a person note is flat.
    """
    flat = f"{active_entity}.md"
    if (root / flat).is_file():
        return flat
    slug = active_entity.rsplit("/", 1)[-1]
    overview = f"{active_entity}/{slug}.md"
    if (root / overview).is_file():
        return overview
    return None


def _text_at(root: Path, commit: str, rel: str) -> str | None:
    """The file's content at ``commit``, or None if it wasn't there."""
    if not commit:
        return None
    try:
        return _git(root, "show", f"{commit}:{rel}")
    except GitError:
        return None


def _render_open_relations(relations: list[Relation]) -> list[str]:
    """``"<rel> -> <target>"``, with ``(since <date>)`` when the entry says
    so — the block is asked to state since when, and re-deriving that from
    the note is work the packet can do once, deterministically."""
    return sorted(
        f"{r.rel} -> {r.target}" + (f" (since {r.valid_from})" if r.valid_from else "")
        for r in relations if not r.valid_until
    )


def _new_log_bullets(root: Path, base_commit: str, rel: str, text: str) -> int:
    """``## Log`` bullets on ``rel`` that were not there at ``base_commit``.

    Both revisions are read with the compiled-truth block stripped. The
    fence lives above ``## Log`` on a well-formed note, but nothing enforces
    that — and a block that ends up INSIDE the Log section would otherwise
    be counted as new evidence by the very job that wrote it.
    """
    bullets = _log_bullets(_strip_compiled_truth(text))
    if not bullets:
        return 0
    old_text = _text_at(root, base_commit, rel)
    if old_text is None:
        return len(bullets)
    seen = set(_log_bullets(_strip_compiled_truth(old_text)))
    return sum(1 for b in bullets if b not in seen)


def _entity_changeset(
    entity: str, note_rel: str, changed_notes: tuple[str, ...]
) -> list[str]:
    """The changeset's notes belonging to this entity, other than its own
    note: a folder-backed project's evidence is its ``log/<date>.md`` and
    curated sub-notes, which is where every real project's activity lands."""
    prefix = f"{entity}/"
    return [n for n in changed_notes if n != note_rel and n.startswith(prefix)]


def _compiled_truth_candidates(
    paths: VaultPaths,
    *,
    active_entities: list[str],
    changed_notes: tuple[str, ...],
    base_commit: str,
    budget: int,
) -> tuple[list[CompiledTruthCandidate], list[str]]:
    """Entity notes worth a compiled-truth refresh, plus the blocked ones.

    A note qualifies when it has something to compile AND new evidence to
    compile it from. The block synthesises the ``## Log`` PLUS the open
    relations, so evidence is either of them moving:

    - ``## Log`` bullets added since ``base_commit`` anywhere under the
      entity — the overview's own Log, and every changed note under a
      folder-backed project;
    - any other changed note under the entity (a project's ``log/<date>.md``
      carries its bullets under its own headings, not under a ``## Log``);
    - a difference in the rendered open relations, which is how a CLOSED
      relation registers: ``upsert_relation_in_text`` edits frontmatter and
      appends no bullet, so without this the block would assert a fact the
      note has since retired, for ever.

    A quiet entity — nothing added, nothing closed — never burns an edit,
    whether or not it already has a block. Ranked by how much new evidence
    landed, then by node id; truncated to ``budget``.
    """
    candidates: list[CompiledTruthCandidate] = []
    blocked: list[str] = []
    for entity in active_entities:
        rel = _entity_note_path(paths.root, entity)
        if rel is None:
            continue
        try:
            text = (paths.root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        state = marker_state(text)
        if state == "broken":
            blocked.append(rel)
            continue
        bullets = _log_bullets(_strip_compiled_truth(text))
        frontmatter, _body = _split_frontmatter(text)
        relations, _problems = parse_relations(frontmatter)
        open_rels = _render_open_relations(relations)

        new_bullets = _new_log_bullets(paths.root, base_commit, rel, text)
        members = _entity_changeset(entity, rel, changed_notes)
        for member in members:
            try:
                member_text = (paths.root / member).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            new_bullets += _new_log_bullets(paths.root, base_commit, member, member_text)

        old_text = _text_at(paths.root, base_commit, rel)
        if old_text is None:
            relations_changed = bool(open_rels)
        else:
            old_fm, _old_body = _split_frontmatter(old_text)
            old_relations, _old_problems = parse_relations(old_fm)
            relations_changed = _render_open_relations(old_relations) != open_rels

        if not new_bullets and not members and not relations_changed:
            continue    # no new evidence: a refresh would say the same thing
        if state == "absent" and not (bullets or open_rels or new_bullets or members):
            # Nothing to compile a first block out of: no Log, no open
            # relations, and nothing under the entity to synthesise from.
            # (An entity whose relations all just CLOSED has something to
            # say, but only once it has a block saying the opposite.)
            continue

        candidates.append(CompiledTruthCandidate(
            node_id=node_id_for_note(rel),
            note_path=rel,
            marker_state=state,
            log_bullets=len(bullets),
            new_log_bullets=new_bullets,
            changed_notes=len(members),
            relations_changed=relations_changed,
            open_relations=open_rels,
        ))
    candidates.sort(key=lambda c: (
        -(c["new_log_bullets"] + c["changed_notes"]),
        not c["relations_changed"],
        c["node_id"],
    ))
    return candidates[:budget], sorted(blocked)


# ---------------------------------------------------------------------------
# contradiction detection: propose-only hints
# ---------------------------------------------------------------------------

_DATE_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BULLET_DATE_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}\s*[\u2014\u2013-]\s*")
# The same leading date, read off a bullet that still carries its "- ".
_BULLET_LEADING_DATE_RE: Final = re.compile(r"^-\s*(\d{4}-\d{2}-\d{2})\b")
_WIKILINK_RE: Final = re.compile(r"\[\[[^\]]*\]\]")
_WORD_RE: Final = re.compile(r"[a-z0-9]{3,}")

# Closed lexicon of "this ended" cues. Lexical only: 'left the meeting' will
# match too, which is why the finding is a proposal a human gates, never an
# edit. Kept small on purpose — a long list is a long false-positive list.
_TERMINATION_CUES: Final[tuple[str, ...]] = (
    "departed",
    "ended",
    "former",
    "left",
    "leaving",
    "no longer",
    "resigned",
    "stepped down",
    "stopped working",
    "wound down",
)
_CUE_RE: Final = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in _TERMINATION_CUES) + r")\b"
)

# The relations a termination cue can contradict: an edge that is meant to
# RUN until something ends it. The rest of the closed vocabulary is
# point-in-time (``attended``, ``met_at`` — the meeting happened, and then
# it was over) or deliberately timeless (``related_to``); an open entry
# there is the correct state, not a contradiction waiting to be flagged.
_CLOSABLE_RELS: Final[frozenset[str]] = frozenset(
    {"works_at", "member_of", "stakeholder_in", "collaborator_on"}
)

# Function words carry no claim: with them in the token set, "ran the dream
# pass" and "ran the dream gate" share 3 of 5 words and clear the bar. Kept
# to the words that actually show up in a dated one-line Log fact.
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "and", "are", "for", "from", "had", "has", "have", "into", "not", "our",
    "that", "the", "this", "was", "were", "with",
})
# Two bullets sharing at least this share of their words are the same claim
# told twice often enough to be worth a human's glance.
_RESTATEMENT_MIN_OVERLAP: Final = 0.6
# Pairwise comparison is quadratic; a Log longer than this is compared over
# its most recent bullets only (newest last), which is where restatements land.
_RESTATEMENT_WINDOW: Final = 200
# The window bounds the INPUTS; a Log of near-identical bullets still emits
# O(n^2) findings out of them, and the packet's budget cut happens much
# later. This bounds the outputs per note.
_RESTATEMENT_MAX_FINDINGS: Final = 50
# Severity order for the packet's truncation: a self-contradicting relation
# interval is a hard error, a lexical cue is a hint, a restatement is tidying.
_KIND_RANK: Final[dict[str, int]] = {
    "relation_interval_conflict": 0,
    "relation_vs_log": 1,
    "restated_fact": 2,
}


# The ``kind`` this job proposes under, in ingest_lib.propose's sense: it
# seeds the fact note's filename and its ``author: script:...`` stamp, and
# keeps these proposals from ever colliding with another pass's.
PROPOSAL_KIND: Final = "dream-contradiction"
_PROPOSAL_KIND: Final = PROPOSAL_KIND
# Where consolidate.py MOVES a fact note once a human has approved it.
_ASSISTANT_ARCHIVE_REL: Final = "knowledge/assistant/archive"
# Every promoted contradiction line starts with this, and the detectors skip
# any ``## Log`` bullet that does. Without it the pass eats its own output:
# consolidate appends promote.fact to the Log, that line quotes the offending
# bullet (so it carries the target mention AND the termination cue), and the
# next run re-detects the settled contradiction against different text —
# a different content hash, so neither the inbox nor the archive check
# suppresses it. Stable by contract: changing this string re-arms every
# proposal a human has already promoted.
CONTRADICTION_FLAG_PREFIX: Final = "contradiction flagged by the dream pass"


def _relation_from_entry(entry: ProposalRelation) -> Relation:
    return Relation(
        rel=entry["rel"],
        target=entry["target"],
        valid_from=entry["valid_from"],
        valid_until=entry["valid_until"],
        source=entry["source"],
    )


@dataclass(frozen=True)
class _RawFinding:
    """A detector hit, before it is addressed to a fact note."""

    kind: str
    node_id: str
    note_path: str
    left: str
    right: str
    detail: str
    # The closure a human's approval would apply, if the finding implies one.
    relations: tuple[ProposalRelation, ...] = ()
    # Why this finding carries no closure, when the reason is worth a human's
    # attention. Rides in the fact text, so it reaches the entity's ## Log.
    flag: str = ""


def _finding(
    kind: str, node_id: str, note_path: str, left: str, right: str, detail: str,
    relations: tuple[ProposalRelation, ...] = (),
    flag: str = "",
) -> _RawFinding:
    return _RawFinding(
        kind=kind, node_id=node_id, note_path=note_path,
        left=left, right=right, detail=detail, relations=relations, flag=flag,
    )


def _claim_text(claim: str) -> str:
    """A claim as it reads inside a one-line sentence (## Log bullets arrive
    with their ``- `` marker, which would nest a bullet inside a bullet)."""
    return claim.removeprefix("- ").strip()


def _proposal_fact(raw: _RawFinding) -> str:
    """The one line a human's approval would append to the entity's ``## Log``.

    It records THAT the two claims disagree, never a resolution in prose:
    the resolution a ``relation_vs_log`` finding implies rides in
    ``promote.relations`` instead, as the closure of the relation the Log
    line says ended — so approving the proposal actually settles it.
    Deterministic, and it carries both claims, so it is also what makes one
    finding's proposal path differ from another's on the same note.
    """
    line = (
        f"{CONTRADICTION_FLAG_PREFIX} ({raw.kind}): "
        f'"{_claim_text(raw.left)}" vs "{_claim_text(raw.right)}"'
    )
    return f"{line} — {raw.flag}" if raw.flag else line


def _is_contradiction_flag(bullet: str) -> bool:
    """True for a ``## Log`` bullet this job's own proposal put there."""
    text = _BULLET_DATE_RE.sub("", _claim_text(bullet))
    return text.startswith(CONTRADICTION_FLAG_PREFIX)


def _proposal_already_handled(paths: VaultPaths, proposal_path: str) -> bool:
    """True when this exact proposal has already left the inbox.

    ``propose_fact(dry_run=True)`` answers the inbox half. This is the other
    half: ``consolidate.py`` MOVES a fact note out of the inbox into
    ``knowledge/assistant/archive/<YYYY-MM>/`` — when a human approved it,
    and equally when it went unanswered long enough to be digested. Either
    way the proposal has been PUT to the human once; re-proposing it every
    night is how a standing finding starves the budget. The glob also covers
    the ``-N`` suffix ``consolidate._archive_destination`` adds on a name
    collision.
    """
    archive = paths.root / _ASSISTANT_ARCHIVE_REL
    if not archive.is_dir():
        return False
    stem = proposal_path.rsplit("/", 1)[-1].removesuffix(".md")
    return any(archive.glob(f"*/{stem}*.md"))


def _describe_relation(relation: Relation) -> str:
    span = relation.valid_from or "(open start)"
    span += f"..{relation.valid_until}" if relation.valid_until else ".. (open)"
    return f"{relation.rel} -> {relation.target} [{span}]"


def _interval_conflicts(
    node_id: str, note_path: str, relations: list[Relation]
) -> list[_RawFinding]:
    """One (rel, target) both closed and continuously open across the closure.

    Superseding is legal and common — close the old interval, append a fresh
    one starting later. The contradiction is the other shape: an open entry
    that starts BEFORE (or without) a closure of the same edge, so the note
    asserts the relation both ended on a date and never stopped.
    """
    found: list[_RawFinding] = []
    by_edge: dict[tuple[str, str], list[Relation]] = {}
    for r in relations:
        by_edge.setdefault((r.rel, r.target), []).append(r)
    for entries in by_edge.values():
        closed = [r for r in entries if _DATE_RE.match(r.valid_until)]
        for open_entry in (r for r in entries if not r.valid_until):
            for closed_entry in closed:
                # A malformed valid_from ("last spring") would compare
                # lexically against the closure date and silently decide
                # the question; only a real date can rule the pair out.
                if (
                    _DATE_RE.match(open_entry.valid_from)
                    and open_entry.valid_from >= closed_entry.valid_until
                ):
                    continue    # a fresh interval starting after the closure
                found.append(_finding(
                    "relation_interval_conflict", node_id, note_path,
                    _describe_relation(open_entry), _describe_relation(closed_entry),
                    f"the note closes {open_entry.rel} -> {open_entry.target} on "
                    f"{closed_entry.valid_until} and also carries it as open from "
                    f"{open_entry.valid_from or '(open start)'}",
                ))
    return found


def _mentions_target(lowered_bullet: str, target: str) -> bool:
    """Does this ``## Log`` line talk about the relation's target?

    Two surface forms: the node id as a wikilink writes it (a substring —
    it is already a distinctive path), and the slug as words, matched on
    word boundaries the way ``_CUE_RE`` matches cues. Without the boundary,
    a project slugged ``check`` matches "checklist", "checked" and
    "checking" — and this vault has slugs called ``check``, ``agent``,
    ``brain``, ``claude`` and ``server``.
    """
    if target.lower() in lowered_bullet:
        return True
    words = target.rsplit("/", 1)[-1].replace("-", " ").lower()
    if not words:
        return False
    return re.search(rf"\b{re.escape(words)}\b", lowered_bullet) is not None


def _closure_relation(
    relation: Relation, bullet: str
) -> tuple[tuple[ProposalRelation, ...], str]:
    """The closure this finding implies — the same edge, ending on the date
    the ``## Log`` line carries — plus a flag when it implies none.

    It rides in ``promote.relations``, so a human approving the proposal
    hands ``consolidate.py`` a real ``valid_until`` and
    ``upsert_relation_in_text`` closes the open span — supersede-never-
    delete, done by the one code path that owns it. Without a date on the
    bullet there is nothing to close it ON, and the proposal carries the
    disagreement alone.

    A date EARLIER than the relation's own ``valid_from`` is not a closure
    either: that is the rejoin shape (left in June, came back in August, the
    old span already closed), and writing it would hand consolidate an
    incoherent ``from 2026-08-01 until 2026-06-02``. Propose nothing and say
    why in the fact text, so the human sees the stale Log line rather than a
    silently mangled interval.
    """
    match = _BULLET_LEADING_DATE_RE.match(bullet)
    if match is None:
        return (), ""
    ended = match.group(1)
    if _DATE_RE.match(relation.valid_from) and ended < relation.valid_from:
        return (), (
            f"no closure proposed: the Log line's date {ended} precedes the "
            f"relation's own start {relation.valid_from}"
        )
    return (ProposalRelation(
        rel=relation.rel,
        target=relation.target,
        valid_from=relation.valid_from,
        valid_until=ended,
        source=relation.source,
    ),), ""


def _relation_vs_log(
    node_id: str, note_path: str, relations: list[Relation], bullets: list[str]
) -> list[_RawFinding]:
    """An open relation while a Log line about that same target says it ended.

    Only the CLOSABLE relations: ``attended`` and ``met_at`` are
    point-in-time by design (the closed vocabulary's own reading), so such
    an edge is always open and correctly so — "the meeting ended" is not a
    contradiction, and one busy person note carries dozens of them.
    """
    found: list[_RawFinding] = []
    for relation in relations:
        if relation.valid_until or relation.rel not in _CLOSABLE_RELS:
            continue
        for bullet in bullets:
            lowered = bullet.lower()
            if not _mentions_target(lowered, relation.target):
                continue
            cue = _CUE_RE.search(lowered)
            if cue is None:
                continue
            closure, flag = _closure_relation(relation, bullet)
            found.append(_finding(
                "relation_vs_log", node_id, note_path,
                _describe_relation(relation), bullet,
                f"relation is open while a ## Log line about {relation.target} "
                f"carries the termination cue {cue.group(0)!r}",
                closure, flag,
            ))
    return found


def _bullet_tokens(bullet: str) -> frozenset[str]:
    stripped = bullet.removeprefix("- ").strip()
    stripped = _BULLET_DATE_RE.sub("", stripped)
    stripped = _WIKILINK_RE.sub(" ", stripped)
    return frozenset(
        w for w in _WORD_RE.findall(stripped.lower()) if w not in _STOPWORDS
    )


def _restated_facts(
    node_id: str, note_path: str, bullets: list[str]
) -> list[_RawFinding]:
    """The same fact asserted twice in one Log, worded differently.

    Bounded at both ends: the window caps how many bullets are compared,
    ``_RESTATEMENT_MAX_FINDINGS`` caps how many pairs one note may emit —
    a Log of near-identical bullets is quadratic in findings long before
    the packet's budget cut would notice.
    """
    window = bullets[-_RESTATEMENT_WINDOW:]
    tokens = [_bullet_tokens(b) for b in window]
    found: list[_RawFinding] = []
    for i in range(len(window)):
        if not tokens[i]:
            continue
        for j in range(i + 1, len(window)):
            if len(found) >= _RESTATEMENT_MAX_FINDINGS:
                return found
            if not tokens[j] or window[i] == window[j]:
                continue
            union = tokens[i] | tokens[j]
            overlap = len(tokens[i] & tokens[j]) / len(union)
            if overlap < _RESTATEMENT_MIN_OVERLAP:
                continue
            found.append(_finding(
                "restated_fact", node_id, note_path, window[i], window[j],
                f"{overlap:.2f} word overlap between two ## Log lines",
            ))
    return found


def _address_proposals(
    paths: VaultPaths, raws: list[_RawFinding], *, budget: int
) -> list[ContradictionCandidate]:
    """Give each hit the fact note it would become, dropping handled ones.

    The destination comes from :func:`ingest_lib.propose.propose_fact` in
    ``dry_run`` mode — the shared proposer owns the inbox filename contract,
    so this job and every other deterministic proposer address the same
    candidate to the same file. Dry-run because the packet must not write:
    these detectors are lexical, most hits are false positives, and the
    LLM half adjudicates before anything reaches the human's queue.

    Skipping proposals that already exist (inbox) or were already promoted
    (archive) is what keeps the budget useful: a standing high-severity
    finding would otherwise occupy the same slot every night and starve
    everything detected since — the guard ``_candidate_pairs`` already
    applies to connection notes.
    """
    addressed: list[ContradictionCandidate] = []
    seen: set[str] = set()
    for raw in raws:
        if len(addressed) >= budget:
            break
        fact = _proposal_fact(raw)
        result = propose_fact(
            paths,
            kind=_PROPOSAL_KIND,
            title=f"Contradiction on {raw.node_id}",
            target=raw.node_id,
            source=raw.note_path.removesuffix(".md"),
            reason=raw.detail,
            relations=[_relation_from_entry(e) for e in raw.relations],
            fact=fact,
            dry_run=True,
        )
        if result.action == "exists" or _proposal_already_handled(paths, result.path):
            continue
        # Two findings can address ONE file (a note declaring the same edge
        # twice detects the same conflict twice); without this the packet
        # spends two budget slots on one proposal.
        if result.path in seen:
            continue
        seen.add(result.path)
        addressed.append(ContradictionCandidate(
            kind=raw.kind,
            node_id=raw.node_id,
            note_path=raw.note_path,
            left=raw.left,
            right=raw.right,
            detail=raw.detail,
            proposal_path=result.path,
            proposal_fact=fact,
            promote_relations=list(raw.relations),
        ))
    return addressed


def _contradictions(
    paths: VaultPaths, *, active_entities: list[str], budget: int
) -> list[ContradictionCandidate]:
    """Contradiction hints across the changeset's entity notes.

    Scope is deliberately the entities the changeset touched — the dream pass
    is change-driven, and a whole-vault pairwise scan is neither bounded nor
    something the edit budget could act on. Findings are sorted by severity
    then note, then addressed to fact notes and truncated to ``budget``.
    """
    found: list[_RawFinding] = []
    for entity in active_entities:
        rel = _entity_note_path(paths.root, entity)
        if rel is None:
            continue
        try:
            text = (paths.root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        node_id = node_id_for_note(rel)
        frontmatter, _body = _split_frontmatter(text)
        relations, _problems = parse_relations(frontmatter)
        # A promoted proposal's own Log line quotes the claims that produced
        # it; reading it back as evidence is how the detector re-arms itself
        # against a contradiction a human has already answered.
        # Stripped of the compiled-truth block first: it is this pass's own
        # summary of the Log, so reading it back as a Log line manufactures a
        # ``restated_fact`` against every bullet it summarises.
        bullets = [
            b for b in _log_bullets(_strip_compiled_truth(text))
            if not _is_contradiction_flag(b)
        ]
        found.extend(_interval_conflicts(node_id, rel, relations))
        found.extend(_relation_vs_log(node_id, rel, relations, bullets))
        found.extend(_restated_facts(node_id, rel, bullets))
    found.sort(key=lambda r: (_KIND_RANK[r.kind], r.node_id, r.left, r.right))
    return _address_proposals(paths, found, budget=budget)


def build_packet(
    paths: VaultPaths,
    *,
    verdict: GateVerdict,
    top_k: int = 10,
    compiled_truth_budget: int = COMPILED_TRUTH_BUDGET,
    contradiction_budget: int = CONTRADICTION_BUDGET,
    logger: logging.Logger,
) -> DreamPacket:
    head = _git(paths.root, "rev-parse", "HEAD").strip()
    active = _active_entities(verdict.changed_notes)
    compiled, blocked = _compiled_truth_candidates(
        paths,
        active_entities=active,
        changed_notes=verdict.changed_notes,
        base_commit=verdict.base_commit,
        budget=compiled_truth_budget,
    )
    packet = DreamPacket(
        since=verdict.since,
        base_commit=verdict.base_commit,
        head_commit=head,
        changed_notes=list(verdict.changed_notes),
        new_sources=list(verdict.new_sources),
        candidate_pairs=_candidate_pairs(paths, top_k=top_k),
        active_entities=active,
        existing_dream_notes=_existing_dream_notes(paths),
        compiled_truth=compiled,
        compiled_truth_blocked=blocked,
        contradictions=_contradictions(
            paths, active_entities=active, budget=contradiction_budget
        ),
        edit_budget=EditBudget(
            total_note_edits=EDIT_CAP,
            compiled_truth=compiled_truth_budget,
            reserved_for_other_jobs=EDIT_CAP - compiled_truth_budget,
            contradiction_proposals=contradiction_budget,
        ),
    )
    logger.info(
        "dream packet: %d changed note(s), %d new source(s), %d candidate pair(s), "
        "%d compiled-truth candidate(s), %d contradiction hint(s)",
        len(packet["changed_notes"]), len(packet["new_sources"]),
        len(packet["candidate_pairs"]), len(packet["compiled_truth"]),
        len(packet["contradictions"]),
    )
    return packet


# ---------------------------------------------------------------------------
# writing the survivors the LLM kept
# ---------------------------------------------------------------------------

class AdjudicatedFinding(TypedDict):
    """One packet contradiction the dream session decided is real.

    Every field except ``title``/``body`` comes from the packet verbatim —
    the proposal's path is a hash of its own content, so a reworded
    ``proposal_fact`` addresses a different file and the idempotency the
    inbox/archive checks depend on is gone. The LLM's own words go in the
    title and the body, which the path does not depend on.
    """

    node_id: str
    note_path: str
    proposal_path: str
    proposal_fact: str
    promote_relations: list[ProposalRelation]
    title: str
    body: str


class ProposalOutcome(TypedDict):
    """What :func:`write_proposals` did with one adjudicated finding."""

    proposal_path: str
    action: str     # "proposed" (written now) | "exists" (already in the inbox)


def _as_mapping(raw: object, where: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a JSON object")
    return raw


def _require_str(mapping: dict[str, object], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key!r} must be a non-empty string")
    return value


def _parse_promote_relations(raw: object, where: str) -> list[ProposalRelation]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{where}: 'promote_relations' must be a list")
    entries: list[ProposalRelation] = []
    for i, item in enumerate(raw):
        entry = _as_mapping(item, f"{where} relation {i}")
        rel = _require_str(entry, "rel", f"{where} relation {i}")
        target = _require_str(entry, "target", f"{where} relation {i}")
        optional: dict[str, str] = {}
        for key in ("valid_from", "valid_until", "source"):
            value = entry.get(key, "")
            if not isinstance(value, str):
                raise ValueError(f"{where} relation {i}: {key!r} must be a string")
            optional[key] = value
        entries.append(ProposalRelation(
            rel=rel, target=target,
            valid_from=optional["valid_from"],
            valid_until=optional["valid_until"],
            source=optional["source"],
        ))
    return entries


def parse_adjudicated(raw: object) -> list[AdjudicatedFinding]:
    """Validate the JSON the dream session hands back after adjudication.

    Strict on purpose: this is the boundary between an LLM's output and a
    deterministic writer, so a missing or mistyped field is an error the
    caller sees, never a proposal written with half a contract.
    """
    if not isinstance(raw, list):
        raise ValueError("adjudicated findings must be a JSON array")
    findings: list[AdjudicatedFinding] = []
    for i, item in enumerate(raw):
        where = f"finding {i}"
        entry = _as_mapping(item, where)
        findings.append(AdjudicatedFinding(
            node_id=_require_str(entry, "node_id", where),
            note_path=_require_str(entry, "note_path", where),
            proposal_path=_require_str(entry, "proposal_path", where),
            proposal_fact=_require_str(entry, "proposal_fact", where),
            promote_relations=_parse_promote_relations(
                entry.get("promote_relations"), where
            ),
            title=_require_str(entry, "title", where),
            body=_require_str(entry, "body", where),
        ))
    return findings


def write_proposals(
    paths: VaultPaths,
    findings: list[AdjudicatedFinding],
    *,
    now: datetime,
    logger: logging.Logger,
) -> list[ProposalOutcome]:
    """Write the adjudicated survivors as fact notes, through the shared
    proposer.

    This is the deterministic half doing the writing, which is what makes
    AGENTS.md's "the fact-note proposer ... used by the dream pass's
    contradiction detector" true: every proposal carries the memory-fact
    contract (``approved: false``, ``memory_status: unconsolidated``) and
    the honest ``written_via: script`` / ``author: script:<kind>`` stamp
    that only a script may claim. The LLM adjudicates; it does not author
    the frontmatter.

    A finding whose recomputed path differs from the packet's is refused
    outright: the only way that happens is a reworded ``proposal_fact`` or
    an edited ``promote_relations``, and the packet's inbox/archive dedup
    (and therefore the budget) is derived from that path. Every finding is
    checked BEFORE any of them is written, so a refused run leaves no half
    of itself in the inbox.
    """
    def _propose(finding: AdjudicatedFinding, *, dry_run: bool) -> ProposeResult:
        return propose_fact(
            paths,
            kind=PROPOSAL_KIND,
            title=finding["title"],
            target=finding["node_id"],
            source=finding["note_path"].removesuffix(".md"),
            reason=finding["body"],
            relations=[_relation_from_entry(e) for e in finding["promote_relations"]],
            fact=finding["proposal_fact"],
            now=now,
            dry_run=dry_run,
        )

    for finding in findings:
        addressed = _propose(finding, dry_run=True)
        if addressed.path != finding["proposal_path"]:
            raise ValueError(
                f"proposal for {finding['node_id']} addresses {addressed.path!r}, "
                f"not the packet's {finding['proposal_path']!r} — the fact or "
                "its relations were edited; send them back unaltered"
            )

    outcomes: list[ProposalOutcome] = []
    for finding in findings:
        result = _propose(finding, dry_run=False)
        logger.info("dream proposal %s: %s", result.action, result.path)
        outcomes.append(ProposalOutcome(
            proposal_path=result.path, action=result.action
        ))
    return outcomes
