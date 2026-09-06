"""Promote processed meeting snapshots into first-class meeting notes.

A Granola/justREC snapshot already becomes a searchable SOURCE note
(``extractors/meeting.py`` -> ``knowledge/index/…``), and there it stopped:
no ``knowledge/meetings/<YYYY>/<YYYY-MM-DD>-<slug>.md`` node, no ``attended``
edges, so the meeting never joined the graph AGENTS.md describes
("Meetings are first-class notes").

This pass closes that gap deterministically, and it splits the write in two
along the vault's defining line:

- **The meeting note is derived output.** It is regenerated from archived
  bytes the way a concept note or a dashboard is, so this module writes it
  directly. What it re-writes on a later run is bounded by a marker fence
  (``<!-- MEETING-DERIVED-START -->`` … ``<!-- MEETING-DERIVED-END -->``, the
  idiom ``concepts.py`` and ``dream.py`` already use): only the attendee and
  source bullets inside that fence, and the derived frontmatter keys, are
  refreshed. Agenda/Notes/Decisions/Actions, anything a human wrote outside
  the fence, and a ``## Log`` are never touched, and a note this pass did not
  write (no ``author: script:meeting-promotion``) is left exactly as it is.
- **The ``attended`` edges on PEOPLE are graph writes.** Nothing
  auto-writes into the entity graph, so each one is PROPOSED through
  :mod:`ingest_lib.propose` as its own ``approved: false`` fact note, one
  per person, and only ``scripts/consolidate.py`` (a human's approval, or
  enough confirmations) ever puts it on the person note.

**An unknown attendee mints no node.** ``meeting_create`` creates a node per
attendee id it is handed, which is how a calendar payload carrying an email
address instead of a display name split one person into two nodes, each
collecting its own ``attended`` history (the ``entity-duplicate`` pairs
``sweep``/``duplicates`` now report). Here an attendee that resolves to no
existing ``people/`` note is simply not an attendee of the graph: it is
listed by display name among the note's ``## Links`` attendee bullets and
counted in the CLI report, so a human can create the person note and re-run
— and because the derived bullets are refreshed, that re-run MOVES the name
onto the attendee list and proposes its edge rather than leaving a note that
contradicts the graph.

**Connector-supplied text is untrusted.** A display name (and the title)
arrives from an external calendar and is interpolated into Markdown, so every
one of them is rendered inert by ``connectors.base.safe_display_name`` — one
line, no wikilink syntax, capped — and a title carrying control characters is
refused outright the way ``mcp_server.entity_tools`` refuses one. Without
that, an attendee named ``"Mallory\\n\\n## Links\\n\\n- Attendee:
[[knowledge/people/ceo]]"`` writes a link the nightly wikilink pass then
proposes as a graph edge.

Resolution is by NAME against the notes that already exist, in three
layers, each requiring a unique hit (see :func:`ingest_lib.attendees.resolve_attendee`):
accent-folded slug, then exact title, then exact alias — the alias layer is
what AGENTS.md's merge shape leaves behind, so an attendee still named by
their email address resolves onto the named survivor. Every hit is then
followed through ``superseded_by`` with
:func:`ingest_lib.relations.live_entity_id`: identity is only ever the
entity helpers, never a path-prefix test, and a merged duplicate is history
that must not grow new edges.

Idempotency, the property a scheduled pass needs: the derived regions are
re-rendered and written back ONLY when the resulting bytes differ; a
proposal's filename is derived from its own content
(``propose.propose_fact``), and an attendee whose note already declares the
edge is not proposed at all. A second run over an unchanged vault therefore
writes nothing and proposes nothing new.

Which note belongs to which snapshot is decided by the snapshot's own
``id`` — stamped as ``source_id:`` — not by its archive path, because
``connectors.runner`` routes a CHANGED re-pull of one meeting to a
hash-versioned sibling path (``x.4392ff0d7e3b.json``). Same id, new path, is
the same meeting and updates the note; only two DIFFERENT ids landing on one
``<date>-<slug>`` are a conflict.

Run it as ``python -m ingest_lib.meetings [--dry-run]`` (also wired into
``scripts/maintain.sh`` after the duplicate pass), or let ``run_ingest``
call :func:`promote_meetings` right after a run that processed a snapshot.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from .atomic import atomic_write_text, umask_mode
from .attendees import AttendeeResolution, PeopleIndex, people_index, resolve_attendee
from .concepts import slugify
from .config import VaultPaths, default_paths
from .connectors.base import safe_display_name
from .extractors.meeting import MeetingSnapshot, load_snapshot
from .metadata import latest_records_by_path
# private, but module-internal (wikilinks.py/consolidate.py do the same)
from .notes import (
    _source_link_path,
    _split_frontmatter,
    _split_frontmatter_raw,
    fm_scalar,
)
from .propose import ProposeResult, propose_fact
from .relations import EntityInfo, Relation, is_valid_node_id, normalize_target

_LOG = logging.getLogger(__name__)

_KIND = "meeting-promotion"
_AUTHOR = f"script:{_KIND}"
_EXTRACTOR = "meeting"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# The fence around the only body region this pass owns: the attendee and
# source bullets inside ``## Links``. Everything outside it — the note's
# other sections, a hand-written tail, a ``## Log`` — is somebody else's
# bytes and survives every re-run. Same idea as concepts.py's
# AUTO-GENERATED-END marker and dream.py's COMPILED-TRUTH fence, and a fence
# rather than a single marker for dream.py's reason: a meeting note has
# hand-written content on BOTH sides of the block.
_DERIVED_START = "<!-- MEETING-DERIVED-START -->"
_DERIVED_END = "<!-- MEETING-DERIVED-END -->"

# The frontmatter keys this pass derives and may refresh in place. Every
# other key (project, topics, anything a human added) keeps its bytes.
_DERIVED_KEYS = ("attendees", "source_file", "source_id")

_LOG_HEADING_RE = re.compile(r"^##\s+Log\s*$", re.MULTILINE)
# ``connectors.runner._versioned_relpath`` inserts a 12-hex content tag
# before the extension when a changed re-pull must not overwrite the
# snapshot already in the inbox.
_VERSION_TAG_RE = re.compile(r"\.[0-9a-f]{12}(?=\.[^./]+$)")
_CONTROL_CHARS = frozenset(chr(c) for c in range(0x20)) | {"\x7f"}

# What this run did to one meeting note. "updated" is a note this pass wrote
# earlier whose DERIVED regions no longer matched the snapshot (a re-pull
# added an attendee, or a person note now exists for one it could not
# resolve); its hand-written regions are untouched.
NoteAction = Literal["written", "updated", "exists", "conflict"]


@dataclass(frozen=True)
class MeetingCandidate:
    """One archived snapshot, ready to become a meeting note."""

    node_id: str        # meetings/<YYYY>/<YYYY-MM-DD>-<slug>
    rel_path: str       # knowledge/meetings/<YYYY>/<YYYY-MM-DD>-<slug>.md
    title: str
    date: str
    summary: str
    connector: str
    # The snapshot's own source-native id ("" when it carries none). This,
    # not the archive path, is what says whether an existing note is THIS
    # meeting: a changed re-pull of one meeting lands at a second archive
    # path (connectors.runner._versioned_relpath) with the same id.
    native_id: str
    source_file: str    # archive/raw/... — the snapshot itself
    source_note: str    # knowledge/index/....json.md, "" if unrecorded
    attendees: tuple[AttendeeResolution, ...]

    @property
    def resolved_ids(self) -> tuple[str, ...]:
        """Live ``people/`` node ids, deduped, in snapshot order."""
        out: list[str] = []
        for a in self.attendees:
            if a.status == "resolved" and a.node_id not in out:
                out.append(a.node_id)
        return tuple(out)

    @property
    def unresolved_names(self) -> tuple[str, ...]:
        """Display names that resolved to no live node, deduped, in order."""
        out: list[str] = []
        for a in self.attendees:
            if a.status != "resolved" and a.display_name not in out:
                out.append(a.display_name)
        return tuple(out)


@dataclass(frozen=True)
class SkippedSnapshot:
    """A snapshot that cannot become a meeting note, and why. Reported
    rather than guessed at: AGENTS.md's canonical meeting path needs a real
    date and a sluggable title, and inventing either would be a lie about
    when the meeting happened."""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class MeetingPromotion:
    """What one candidate's promotion did."""

    candidate: MeetingCandidate
    note_action: NoteAction
    # Set only when note_action == "conflict": the source_file the existing
    # note names (empty when it names none, e.g. a hand-written note).
    conflict_source: str
    # Set only when note_action == "conflict": the source_id the existing
    # note names, i.e. the OTHER meeting's native id ("" when it names none).
    conflict_source_id: str
    proposals: tuple[ProposeResult, ...]
    # Attendees whose note ALREADY declares the attended edge, so nothing
    # was proposed for them.
    already_related: tuple[str, ...]


@dataclass(frozen=True)
class PromotionReport:
    """One whole run: what was promoted, and what could not be."""

    promotions: tuple[MeetingPromotion, ...]
    skipped: tuple[SkippedSnapshot, ...]


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def _is_canonical_date(value: str) -> bool:
    """Strict YYYY-MM-DD, round-tripped so ``2026-1-1`` and ``2026-02-31``
    are both refused — the date lands in a filename and a node id, where one
    canonical spelling keeps everything sortable."""
    if not _DATE_RE.match(value):
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%d") == value


def _candidate_from_snapshot(
    snapshot: MeetingSnapshot,
    *,
    source_file: str,
    source_note: str,
    index: PeopleIndex,
) -> tuple[MeetingCandidate | None, str]:
    """Build the candidate, or ``(None, reason)`` when the snapshot cannot
    name a node this pass is willing to write."""
    if not _is_canonical_date(snapshot.date):
        return None, (
            f"no canonical YYYY-MM-DD date — date={snapshot.date!r} "
            f"title={snapshot.title!r}"
        )
    # A title is connector-supplied text that ends up in a heading, a YAML
    # scalar and a node id. Control characters in one are refused outright,
    # the way mcp_server.entity_tools.tool_meeting_create refuses them: a
    # multi-line "title" is not a title, and collapsing it would silently
    # name the meeting after whatever was smuggled in. What survives is still
    # rendered inert (no link syntax, capped) before it is used anywhere.
    if _CONTROL_CHARS.intersection(snapshot.title):
        return None, f"title contains control characters — title={snapshot.title!r}"
    title = safe_display_name(snapshot.title)
    # The MEETING slug is concepts.slugify, the same function
    # ``mcp_server.entity_tools.tool_meeting_create`` uses — every meeting
    # note in the live vault is named by it, so promoting a meeting an agent
    # already created must land on that exact path (and be recognised as
    # already present) rather than beside it under a second spelling.
    # Attendee lookup keeps the accent-FOLDED slug: that is how the person
    # notes are named, and the two rules genuinely differ
    # ("Antoni Wuczyński" -> meetings/…-antoni-wuczy-ski but
    # people/antoni-wuczynski).
    slug = slugify(title)
    if not slug:
        return None, f"no sluggable title — title={snapshot.title!r}"
    node_id = f"meetings/{snapshot.date[:4]}/{snapshot.date}-{slug}"
    if not is_valid_node_id(node_id):
        return None, f"{node_id!r} is not a valid node id"
    return MeetingCandidate(
        node_id=node_id,
        rel_path=f"knowledge/{node_id}.md",
        title=title,
        date=snapshot.date,
        summary=snapshot.summary,
        connector=snapshot.connector,
        native_id=snapshot.native_id,
        source_file=source_file,
        source_note=source_note,
        attendees=tuple(resolve_attendee(a, index) for a in snapshot.attendees),
    ), ""


def _find_candidates(
    paths: VaultPaths, index: PeopleIndex
) -> tuple[list[MeetingCandidate], list[SkippedSnapshot]]:
    """Every archived meeting snapshot that could become a meeting note.

    ``metadata/index.jsonl`` is the source of truth for what has been
    processed (AGENTS.md rule 4), so the pass is driven by the records whose
    ``extractor`` was the meeting one — never by walking ``archive/raw/``,
    which would also claim files no ingest has accepted. A record whose
    extraction failed (``manual_review``) is not promoted: there is no
    trustworthy meeting behind it.

    Deterministic: records are read in sorted path order and the result
    keeps it. A snapshot the run cannot turn into a node is returned in the
    second list with its reason rather than dropped silently.

    One meeting yields one candidate even when the archive holds it twice.
    A CHANGED re-pull is archived BESIDE the first snapshot
    (``connectors.runner._versioned_relpath``), so ``index.jsonl`` carries
    both versions and both name the same node; only the newest may own the
    note, or the two would take turns rewriting its ``source_file`` on every
    run forever. Newest is the record ingest last updated, ties broken by
    path so the choice is deterministic even for records with no timestamp.
    Two DIFFERENT native ids on one node id are NOT one meeting and both
    survive: that is the collision :func:`_write_note` reports as a conflict.
    """
    found: list[tuple[str, str, MeetingCandidate]] = []
    skipped: list[SkippedSnapshot] = []
    records = latest_records_by_path(paths.metadata_index_jsonl)
    for rel in sorted(records):
        record = records[rel]
        if record.extractor != _EXTRACTOR:
            continue
        if record.status not in ("processed", "partial"):
            skipped.append(SkippedSnapshot(rel, f"record status is {record.status}"))
            continue
        src = paths.root / record.raw_path
        snapshot, error = load_snapshot(src)
        if snapshot is None:
            skipped.append(SkippedSnapshot(rel, error))
            continue
        candidate, reason = _candidate_from_snapshot(
            snapshot,
            source_file=record.raw_path,
            source_note=record.index_note_path or "",
            index=index,
        )
        if candidate is None:
            skipped.append(SkippedSnapshot(rel, reason))
            continue
        found.append((rel, record.updated_at, candidate))
    newest: dict[tuple[str, str], tuple[str, str, MeetingCandidate]] = {}
    for rel, updated_at, candidate in found:
        # A snapshot naming no id of its own is identified by its archive
        # path with the re-pull version tag stripped, the same fallback
        # _is_same_meeting uses.
        key = (candidate.node_id, candidate.native_id or _unversioned(rel))
        previous = newest.get(key)
        if previous is None or (updated_at, rel) > (previous[1], previous[0]):
            newest[key] = (rel, updated_at, candidate)
    candidates = [c for _rel, _updated, c in sorted(newest.values(), key=lambda e: e[0])]
    return candidates, skipped


def find_meeting_candidates(
    paths: VaultPaths,
) -> tuple[list[MeetingCandidate], list[SkippedSnapshot]]:
    """:func:`_find_candidates` with its own freshly-read people index."""
    return _find_candidates(paths, people_index(paths))


# ---------------------------------------------------------------------------
# the meeting note
# ---------------------------------------------------------------------------

def _yaml_squote(value: str) -> str:
    """Single-quoted YAML scalar, the style
    ``mcp_server.entity_tools._meeting_note_text`` writes (single quotes
    escape by doubling). Used for the keys that tool also writes —
    ``title:`` and the ``attendees:`` list — so the frontmatter is
    byte-comparable with a meeting note ``meeting_create`` wrote:
    ``notes.fm_scalar`` leaves a plain title UNQUOTED and ``notes.fm_list``,
    though equally valid YAML, line-wraps a list of four or more node ids,
    which no existing meeting note does."""
    return "'" + value.replace("'", "''") + "'"


def _derived_block(candidate: MeetingCandidate) -> str:
    """The ``## Links`` bullets this pass owns, in one block.

    Resolved attendees link to their node; a name that resolved to no live
    ``people/`` note is listed as ordinary text on the same list rather than
    in a section of its own — ``knowledge/index/templates/meeting.md`` and
    ``meeting_create`` both write Agenda/Notes/Decisions/Actions/Links and
    nothing else, and a promoted meeting is not a second body shape. The
    source links close the block, and they are derived too: a re-pull of the
    same meeting changes ``source_file``.
    """
    lines = [f"- Attendee: [[knowledge/{node_id}]]" for node_id in candidate.resolved_ids]
    lines += [
        f"- {name} (unresolved attendee: no people/ note)"
        for name in candidate.unresolved_names
    ]
    # Extensionless, via the same helper the source notes use: a bare
    # [[archive/raw/.../x.json]] is read as a reference to a literal .json
    # file and resolves to nothing. The index note is the opposite case and
    # keeps its full ``.json.md`` name for exactly the same reason.
    lines.append(f"- Source: [[{_source_link_path(candidate.source_file)}]]")
    if candidate.source_note:
        lines.append(f"- Source note: [[{candidate.source_note}]]")
    return "\n".join(lines)


def render_meeting_note(candidate: MeetingCandidate, *, now_iso: str) -> str:
    """The meeting note text.

    Same shape ``mcp_server.entity_tools._meeting_note_text`` writes — the
    frontmatter keys of ``knowledge/index/templates/meeting.md`` and the
    Agenda/Notes/Decisions/Actions/Links body, spelled with the same
    single-quoted scalars — plus what an ingest-side note owes AGENTS.md
    rule 3: ``source_file`` and a "Links -> Source:" wikilink back to the
    snapshot, and the ``source_id`` that says WHICH meeting the note is for
    when a re-pull moves the snapshot. Provenance is stamped the way
    :mod:`ingest_lib.propose` stamps it (``written_via: script``,
    ``author: script:<kind>``): honest about being machine-written without
    claiming to have passed through the MCP server.

    Deterministic apart from ``created``/``updated``, which are frontmatter
    only (AGENTS.md rule 7). Sections with nothing in them keep their
    heading and say ``_(empty)_``. The attendee/source bullets sit inside
    the marker fence :func:`_regenerated_note` re-renders on a later run;
    everything else is written once and then belongs to whoever edits it.
    """
    return (
        "---\n"
        f"title: {_yaml_squote(candidate.title)}\n"
        "type: meeting\n"
        f"date: {fm_scalar(candidate.date)}\n"
        f"{_derived_key_line('attendees', candidate)}\n"
        "project: ''\n"
        "topics: []\n"
        f"{_derived_key_line('source_file', candidate)}\n"
        f"{_derived_key_line('source_id', candidate)}\n"
        f"created: {fm_scalar(now_iso)}\n"
        f"updated: {fm_scalar(now_iso)}\n"
        f"author: {fm_scalar(_AUTHOR)}\n"
        "written_via: script\n"
        "---\n"
        "\n"
        f"# {candidate.title}\n"
        "\n"
        "## Agenda\n"
        "\n"
        "_(empty)_\n"
        "\n"
        "## Notes\n"
        "\n"
        f"{candidate.summary or '_(empty)_'}\n"
        "\n"
        "## Decisions\n"
        "\n"
        "_(empty)_\n"
        "\n"
        "## Actions\n"
        "\n"
        "_(empty)_\n"
        "\n"
        "## Links\n"
        "\n"
        f"{_DERIVED_START}\n"
        f"{_derived_block(candidate)}\n"
        f"{_DERIVED_END}\n"
    )


# ---------------------------------------------------------------------------
# refreshing an existing note's derived regions
# ---------------------------------------------------------------------------

def _fm_str(frontmatter: Mapping[str, object], key: str) -> str:
    raw = frontmatter.get(key)
    return raw.strip() if isinstance(raw, str) else ""


def _derived_key_line(key: str, candidate: MeetingCandidate) -> str:
    """One derived frontmatter line, spelled the one way — so a note this
    pass writes and a note it later refreshes cannot drift apart."""
    if key == "attendees":
        ids = ", ".join(_yaml_squote(a) for a in candidate.resolved_ids)
        return f"attendees: [{ids}]"
    if key == "source_file":
        return f"source_file: {fm_scalar(candidate.source_file)}"
    return f"source_id: {_yaml_squote(candidate.native_id)}"


def _unversioned(source_file: str) -> str:
    """``archive/raw/…/x.4392ff0d7e3b.json`` -> ``archive/raw/…/x.json``.

    ``connectors.runner._resolve_dest`` routes a CHANGED re-pull of one
    native id to a hash-versioned sibling rather than overwriting the
    snapshot already in the inbox, so one meeting can reach the archive under
    two paths. Only the FALLBACK identity test, for a snapshot carrying no
    ``id`` of its own — otherwise ``source_id`` answers this exactly.
    """
    return _VERSION_TAG_RE.sub("", source_file, count=1)


def _is_same_meeting(frontmatter: Mapping[str, object], candidate: MeetingCandidate) -> bool:
    """Whether the note already at this path is THIS snapshot's meeting.

    The snapshot's own id decides it whenever both sides carry one: a
    re-pulled meeting is the same meeting under a new archive path, and
    calling that a collision made every later run report a conflict that was
    never real. Failing that, the archive path decides — with the re-pull
    version tag stripped from both sides — and a note naming no source at all
    is this meeting, the live vault's normal case (every note
    ``meeting_create`` wrote stamps no ``source_file``, and it sits at exactly
    this meeting's ``<date>-<slug>``).
    """
    existing_id = _fm_str(frontmatter, "source_id")
    if existing_id and candidate.native_id:
        return existing_id == candidate.native_id
    existing_source = _fm_str(frontmatter, "source_file")
    if not existing_source:
        return True
    return _unversioned(existing_source) == _unversioned(candidate.source_file)


def _derived_span(text: str) -> tuple[int, int] | None:
    """``(first byte after START, offset of END)`` for a note carrying
    exactly one well-ordered marker pair; None otherwise.

    Ambiguity is refused rather than guessed at, as ``concepts.py`` and
    ``dream.marker_state`` refuse it: guessing where a generated block ends
    is how a hand-written section gets eaten.
    """
    if text.count(_DERIVED_START) != 1 or text.count(_DERIVED_END) != 1:
        return None
    start = text.index(_DERIVED_START) + len(_DERIVED_START)
    end = text.index(_DERIVED_END)
    return (start, end) if start < end else None


def _splice_fm_key(text: str, line: str) -> str:
    """``text`` with one top-level frontmatter key rewritten to ``line``
    (appended after the existing keys when the key is absent).

    Line surgery, not a YAML round-trip, for the reason
    ``relations._splice_relations`` gives: safe_load/safe_dump is YAML 1.1
    and silently rewrites values (``yes`` -> ``true``, ``010`` -> ``8``) the
    note never asked it to. Every other line keeps its source bytes.
    """
    raw = _split_frontmatter_raw(text)
    if raw is None:
        return text
    yaml_block, body = raw
    fm_lines = yaml_block.splitlines(keepends=True)
    all_lines = text.splitlines(keepends=True)
    open_fence, close_fence = all_lines[0], all_lines[1 + len(fm_lines)]
    key = line.split(":", 1)[0]
    prefix = f"{key}:"
    start = next(
        (i for i, ln in enumerate(fm_lines) if ln.rstrip("\r\n") == prefix
         or ln.startswith(f"{prefix} ")),
        len(fm_lines),
    )
    # A folded/nested value runs to the last indented line (or column-0
    # sequence dash) after the key; blank lines neither extend nor end it.
    end = start + 1
    for j in range(start + 1, len(fm_lines)):
        stripped = fm_lines[j].strip()
        if not stripped:
            continue
        if not (fm_lines[j][0].isspace() or stripped.startswith("-")):
            break
        end = j + 1
    return (
        open_fence
        + "".join(fm_lines[:start])
        + line + "\n"
        + "".join(fm_lines[end:])
        + close_fence
        + body
    )


def _regenerated_note(
    text: str, candidate: MeetingCandidate, *, now_iso: str
) -> str | None:
    """``text`` with only this pass's derived regions refreshed, or None when
    there is nothing to change (or nowhere safe to change it).

    The derived regions are exactly two: the frontmatter keys in
    ``_DERIVED_KEYS`` and the bullets between the markers. Agenda, Notes,
    Decisions, Actions, a hand-written tail and a ``## Log`` are other
    people's bytes — a ``## Log`` that has been moved INSIDE the fence stops
    the rewrite altogether rather than being re-rendered away, since that
    section is append-only history (AGENTS.md).
    """
    span = _derived_span(text)
    if span is None or _split_frontmatter_raw(text) is None:
        return None
    start, end = span
    if _LOG_HEADING_RE.search(text[start:end]):
        return None
    refreshed = text[:start] + "\n" + _derived_block(candidate) + "\n" + text[end:]
    for key in _DERIVED_KEYS:
        refreshed = _splice_fm_key(refreshed, _derived_key_line(key, candidate))
    if refreshed == text:
        return None
    # Only now is this a real write, so only now does ``updated`` move: a
    # run that changes nothing must leave the note byte-identical.
    return _splice_fm_key(refreshed, f"updated: {fm_scalar(now_iso)}")


def _write_note(
    paths: VaultPaths, candidate: MeetingCandidate, *, dry_run: bool, now: datetime
) -> tuple[NoteAction, str, str]:
    """Write or refresh the meeting node's note. Returns
    ``(action, conflict_source, conflict_source_id)``.

    What happens to a note already on disk depends on the only hard evidence
    available:

    - it is a DIFFERENT meeting (:func:`_is_same_meeting` says so, i.e. a
      different snapshot id landed on this same ``<date>-<slug>``):
      ``"conflict"``, with nothing written and nothing proposed — hanging
      this snapshot's attendees off the other meeting's note would be a
      graph lie.
    - it is this meeting but this pass did not write it (no
      ``author: script:meeting-promotion``): ``"exists"``, untouched. The
      live vault's normal case — a note ``meeting_create`` or a human wrote,
      whose body is theirs. The attendee proposals still run: an attendee the
      earlier note missed is a real gap, and it is a PROPOSAL a human gates.
    - it is this meeting and this pass wrote it: the derived regions are
      re-rendered. ``"updated"`` when that changes bytes (a re-pull added an
      attendee; a person note now exists for a name that resolved to none
      last time), ``"exists"`` when it does not. This is what keeps the note
      from contradicting the graph: without it, creating the missing person
      note and re-running proposed the edge while the note went on listing
      that person as unresolved forever.
    """
    dest = paths.root / candidate.rel_path
    if dest.exists():
        try:
            text = dest.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _LOG.warning("meetings: cannot read %s (%s)", candidate.rel_path, exc)
            return "conflict", "", ""
        frontmatter, _body = _split_frontmatter(text)
        if not _is_same_meeting(frontmatter, candidate):
            return (
                "conflict",
                _fm_str(frontmatter, "source_file"),
                _fm_str(frontmatter, "source_id"),
            )
        if _fm_str(frontmatter, "author") != _AUTHOR:
            return "exists", "", ""
        if _derived_span(text) is None:
            _LOG.warning(
                "meetings: %s carries this pass's provenance but no usable %s … %s "
                "fence — leaving it untouched",
                candidate.rel_path, _DERIVED_START, _DERIVED_END,
            )
            return "exists", "", ""
        refreshed = _regenerated_note(
            text, candidate, now_iso=now.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        if refreshed is None:
            return "exists", "", ""
        if not dry_run:
            atomic_write_text(
                dest, refreshed, prefix=".meeting-", suffix=".md", mode=umask_mode(),
            )
        return "updated", "", ""
    if dry_run:
        return "written", "", ""
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        dest,
        render_meeting_note(candidate, now_iso=now_iso),
        prefix=".meeting-",
        suffix=".md",
        mode=umask_mode(),
    )
    return "written", "", ""


# ---------------------------------------------------------------------------
# the attended proposals
# ---------------------------------------------------------------------------

def _already_attended(info: EntityInfo, node_id: str) -> bool:
    """True when this person note already declares ``attended`` -> the
    meeting. Targets are re-normalised so the tolerated ``knowledge/...``
    spelling suppresses the proposal exactly as the canonical one does."""
    return any(
        r.rel == "attended" and normalize_target(r.target) == node_id
        for r in info.relations
    )


def _reason(candidate: MeetingCandidate, *, person_id: str) -> str:
    unresolved = candidate.unresolved_names
    tail = (
        ""
        if not unresolved
        else (
            " The snapshot also named attendees this pass could resolve to no "
            "existing people/ note; no node was minted for them, so they are "
            "recorded here (and in the pass's own output) rather than in the "
            "graph: "
            + ", ".join(f"`{name}`" for name in unresolved)
            + "."
        )
    )
    return (
        "Deterministic meeting-promotion pass "
        "(scripts/ingest_lib/meetings.py): the "
        f"{candidate.connector or 'meeting'} snapshot "
        f"[[{_source_link_path(candidate.source_file)}]] lists this person among "
        f"the attendees of "
        f"[[knowledge/{candidate.node_id}]] on {candidate.date}, and the person "
        "note declares no `attended` relation to that meeting yet. The name was "
        "resolved against the people/ notes that already exist — no node was "
        f"created for it — giving `{person_id}`." + tail
    )


def _propose_attendance(
    paths: VaultPaths,
    candidate: MeetingCandidate,
    *,
    index: PeopleIndex,
    dry_run: bool,
    now: datetime | None,
) -> tuple[tuple[ProposeResult, ...], tuple[str, ...]]:
    meeting_link = f"knowledge/{candidate.node_id}"
    results: list[ProposeResult] = []
    already: list[str] = []
    for person_id in candidate.resolved_ids:
        info = index.entities.get(person_id)
        if info is not None and _already_attended(info, candidate.node_id):
            already.append(person_id)
            continue
        relation = Relation(
            rel="attended",
            target=candidate.node_id,
            valid_from=candidate.date,
            source=meeting_link,
        )
        results.append(propose_fact(
            paths,
            kind=_KIND,
            title=f"attended candidate: {person_id} -> {candidate.node_id}",
            target=person_id,
            source=meeting_link,
            reason=_reason(candidate, person_id=person_id),
            relations=[relation],
            now=now,
            dry_run=dry_run,
        ))
    return tuple(results), tuple(already)


def promote_meetings(
    paths: VaultPaths, *, dry_run: bool = False, now: datetime | None = None
) -> PromotionReport:
    """Promote every processed meeting snapshot, and propose its attendees.

    Writes the meeting note when it is absent and refreshes its derived
    regions when they have gone stale (derived output), and proposes one
    ``attended`` fact note per resolved attendee (graph write, so it goes
    through the inbox and waits for a human). Re-running over an unchanged
    vault writes nothing and proposes nothing new.
    """
    index = people_index(paths)
    when = now or datetime.now(UTC)
    candidates, skipped = _find_candidates(paths, index)
    promotions: list[MeetingPromotion] = []
    for candidate in candidates:
        action, conflict_source, conflict_source_id = _write_note(
            paths, candidate, dry_run=dry_run, now=when
        )
        if action == "conflict":
            _LOG.warning(
                "meetings: conflict %s — note already exists for a different "
                "meeting (source_id %r, source_file %r), snapshot is source_id "
                "%r at %r; nothing written or proposed",
                candidate.rel_path, conflict_source_id, conflict_source,
                candidate.native_id, candidate.source_file,
            )
            promotions.append(MeetingPromotion(
                candidate=candidate, note_action=action,
                conflict_source=conflict_source,
                conflict_source_id=conflict_source_id,
                proposals=(), already_related=(),
            ))
            continue
        proposals, already = _propose_attendance(
            paths, candidate, index=index, dry_run=dry_run, now=now
        )
        _LOG.info(
            "meetings: %s %s (%d attendee(s) resolved, %d unresolved, "
            "%d proposal(s), %d already related)",
            action, candidate.node_id, len(candidate.resolved_ids),
            len(candidate.unresolved_names), len(proposals), len(already),
        )
        promotions.append(MeetingPromotion(
            candidate=candidate, note_action=action, conflict_source="",
            conflict_source_id="", proposals=proposals, already_related=already,
        ))
    return PromotionReport(promotions=tuple(promotions), skipped=tuple(skipped))


# ---------------------------------------------------------------------------
# CLI
#
# Same arrangement as ingest_lib.wikilinks / ingest_lib.duplicates: the CLI
# lives in the module and is invoked as ``python -m ingest_lib.meetings``
# (ingest_lib is an installed package, so ``-m`` resolves the relative
# imports above correctly, unlike running this file by path).
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="brain-meetings",
        description=(
            "Deterministic (zero-LLM) pass: promote processed Granola/justREC "
            "snapshots to first-class knowledge/meetings/<YYYY>/ notes and "
            "propose one attended relation per resolved attendee into "
            "knowledge/assistant/inbox/. Never writes to a person note "
            "directly, and never creates a person note for an unknown "
            "attendee — scripts/consolidate.py gates promotion."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what WOULD be written and proposed without writing anything.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    paths = default_paths()
    report = promote_meetings(paths, dry_run=args.dry_run)
    promotions, skipped = report.promotions, report.skipped

    written = sum(1 for p in promotions if p.note_action == "written")
    updated = sum(1 for p in promotions if p.note_action == "updated")
    exists = sum(1 for p in promotions if p.note_action == "exists")
    conflicts = [p for p in promotions if p.note_action == "conflict"]
    proposals = [r for p in promotions for r in p.proposals]
    proposed = sum(1 for r in proposals if r.action == "proposed")
    in_inbox = sum(1 for r in proposals if r.action == "exists")
    archived = sum(1 for r in proposals if r.action == "archived")
    already = sum(len(p.already_related) for p in promotions)
    resolved = sum(len(p.candidate.resolved_ids) for p in promotions)
    # Unresolved names across the run, each with the meetings it appeared
    # in — the resolution rate is the number a human needs before deciding
    # whether to create the missing person notes.
    unresolved: dict[str, list[str]] = {}
    for promotion in promotions:
        for name in promotion.candidate.unresolved_names:
            unresolved.setdefault(name, []).append(promotion.candidate.node_id)

    print(f"meeting promotion ({'dry-run' if args.dry_run else 'real'}):")
    print(f"  snapshots found        : {len(promotions) + len(skipped)}")
    print(f"  meeting notes written  : {written}")
    print(f"  notes refreshed        : {updated}")
    print(f"  notes already present  : {exists}")
    print(f"  id conflicts           : {len(conflicts)}")
    print(f"  snapshots skipped      : {len(skipped)}")
    print(f"  attendees resolved     : {resolved}")
    print(f"  relations proposed     : {proposed}")
    print(f"  already in inbox       : {in_inbox}")
    print(f"  already archived       : {archived}")
    print(f"  already a relation     : {already}")
    print(f"  attendees unresolved   : {len(unresolved)}")
    for name in sorted(unresolved):
        print(f"    - {name} ({', '.join(unresolved[name])})")
    for promotion in conflicts:
        print(
            f"  conflict: {promotion.candidate.rel_path} already exists for a "
            f"different meeting (source_id={promotion.conflict_source_id or 'none'}, "
            f"source_file={promotion.conflict_source or 'none'}); snapshot "
            f"{promotion.candidate.source_file} not promoted"
        )
    for item in skipped:
        print(f"  skipped: {item.relative_path} — {item.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AttendeeResolution",
    "MeetingCandidate",
    "MeetingPromotion",
    "PromotionReport",
    "SkippedSnapshot",
    "find_meeting_candidates",
    "main",
    "promote_meetings",
    "render_meeting_note",
]
