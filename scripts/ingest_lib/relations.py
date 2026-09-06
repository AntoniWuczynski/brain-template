"""Typed, dated relationships between knowledge entities.

Entity notes (people, organisations, projects, meetings, …) can declare
how they relate to each other in frontmatter::

    relations:
      - rel: works_at
        target: organisations/acme
        valid_from: "2025-03-01"      # optional, YYYY-MM-DD
        valid_until: ""               # optional; absent/empty = currently valid
        source: knowledge/meetings/2026/2026-06-12-kern-call   # provenance

This module owns three things, all deterministic and LLM-free:

- **Parsing**: a tolerant reader (:func:`parse_relations`) that skips
  malformed entries with a problem string instead of raising, and a vault
  scanner (:func:`entity_notes`) that turns every hand-edited knowledge
  note into an :class:`EntityInfo`.
- **Identity**: an entity's node id is its ``knowledge/``-relative path
  without the ``.md`` extension (``people/anna-kowalska``). Node ids
  always contain a ``/`` (the subdirectory), so they can never collide
  with concept slugs, which never do. A *folder-backed* entity — a
  project laid out as ``knowledge/projects/<slug>/`` with an overview
  note beside its curated notes — is identified by that overview note's
  own path, ``projects/<slug>/<slug>``; the folder itself is not a node.
  :func:`normalize_target` resolves the ambiguous short form
  (``projects/<slug>``) to it when given a ``vault_root`` to check.
  :func:`canonical_entity_id` is the single answer to "is this note a
  graph node, and which" (a project's ``log/`` entry is not a node of its
  own — it belongs to the project's overview — while a curated note beside
  the overview keeps its own id, as AGENTS.md says it does),
  :func:`owning_entity_id` answers the neighbouring question "which entity
  does prose written in this note belong to", and :func:`live_entity_id`
  follows ``superseded_by`` from a merged duplicate to the survivor.
  Nothing should re-derive any of them from a path prefix.
- **Pure text editing**: :func:`upsert_relation_in_text` and
  :func:`append_fact_to_log` are ``text -> text`` so the MCP entity tools
  and the consolidation pass (separate stages) can apply them to any note
  body without touching the filesystem here. History is never edited or
  deleted — closing an entry and adding a new one is the only mutation
  (supersede, don't delete).

Frontmatter editing strategy: parse with ``notes._split_frontmatter`` to
read and mutate the relation entries, then splice ONLY the ``relations:``
block back into the fence textually (the same line surgery
``mcp_server/provenance.py`` does for its own keys). Every other key keeps
its source bytes — quote style, key order, comments, and timestamps in the
vault's ``…Z`` form — and so does the body below the fence. A whole-fence
``yaml.safe_dump`` round-trip would rewrite all of that on every relation
write.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import yaml

from .config import VaultPaths
from .knowledge import KNOWLEDGE_NOTE_DIRS
from .notes import (  # private helpers, but module-internal
    _split_frontmatter,
    _split_frontmatter_raw,
)

# Assistant history: promoted facts (archive/) and swept digests are the
# audit trail, NOT live entities. They must not become graph nodes — a query
# would otherwise resolve to an archived fact or a "Memory digest 2026-06"
# node in vault_related, contradicting AGENTS.md ("treat archive/ and
# digests/ as the audit trail, not live memory"). Mirrors the exclusion
# scan_knowledge / semantic.upsert_notes already apply.
_ASSISTANT_HISTORY_PREFIXES = (
    "knowledge/assistant/archive/",
    "knowledge/assistant/digests/",
)

_LOG = logging.getLogger(__name__)

# Closed vocabulary. Unknown rel values are reported, not stored: a fixed
# set keeps the graph queryable by deterministic tooling (no LLM needed to
# decide whether "employed_by" means "works_at").
RELATION_VOCAB: Final[frozenset[str]] = frozenset(
    {
        "works_at",
        "member_of",
        "attended",
        "stakeholder_in",
        "collaborator_on",
        "met_at",
        "related_to",
    }
)

# Closed key set for one relation entry. Order matters here (used verbatim
# in the "allowed: ..." problem message) — not alphabetical, the shape as
# documented in the module docstring.
_RELATION_ENTRY_KEYS: Final[tuple[str, ...]] = (
    "rel", "target", "valid_from", "valid_until", "source",
)

_LOG_HEADING = "## Log"
_HEADING_RE = re.compile(r"^#{1,6}\s")
# Blank sections carry this placeholder (AGENTS.md: "leave the heading and
# write _(empty)_"); the first real bullet replaces it.
_EMPTY_PLACEHOLDER = "_(empty)_"
# A Markdown code fence: up to three leading spaces, then 3+ backticks or
# tildes. Lines inside one are content, not document structure.
_CODE_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def normalize_target(raw: str, *, vault_root: Path | None = None) -> str:
    """Canonicalise a relation target to a node id (``people/x``).

    Tolerant of the forms humans (and earlier notes) actually write:
    ``knowledge/people/x``, ``people/x.md``, ``[[knowledge/people/x]]``,
    ``[[knowledge/people/x|Anna]]``, ``[[people/x#Log]]`` — all collapse
    to ``people/x``. A ``#…`` fragment names a heading INSIDE a note, not
    a node of its own, so it is dropped (F19).

    Pure string work by default. Pass ``vault_root`` (the vault root, not
    ``knowledge/``) to additionally resolve the ambiguous short form of a
    folder-backed entity against the filesystem: ``projects/server``
    becomes ``projects/server/server`` when ``knowledge/projects/server.md``
    does NOT exist but ``knowledge/projects/server/server.md`` does. A
    flat note always wins, so an id that already names a real note is
    returned untouched, and an id resolving to neither is left alone for
    the caller to report.
    """
    t = raw.strip()
    if t.startswith("[[") and t.endswith("]]"):
        t = t[2:-2].strip()
    t = t.split("|", 1)[0].strip()  # wikilink display alias
    t = t.split("#", 1)[0].strip()  # heading fragment
    if t.endswith(".md"):
        t = t[: -len(".md")]
    t = t.lstrip("/")
    if t.startswith("knowledge/"):
        t = t[len("knowledge/") :]
    if vault_root is None:
        return t
    return _resolve_folder_entity(t, vault_root)


# A canonical node id is one or more path segments of [a-z0-9._-] joined by
# single slashes, e.g. ``people/anna-kowalska`` or ``meetings/2026/2026-06-12-x``.
# It must NOT contain a ``.``/``..`` path segment: those survive
# ``normalize_target`` unchanged and, once expanded to ``knowledge/<id>.md``,
# resolve OUTSIDE the knowledge/ tree (``people/../../archive/...``), which
# would let a relation target or meeting attendee escape the write allowlist.
_NODE_ID_RE = re.compile(r"^(?![./])[a-z0-9._/-]+$")


def is_valid_node_id(node_id: str) -> bool:
    """True iff ``node_id`` is a safe, canonical node id (see _NODE_ID_RE).

    Rejects empty ids, leading ``/`` or ``.``, and any ``.``/``..`` segment,
    so a caller can never smuggle a path traversal through a node id.
    """
    if not node_id or "/" not in node_id:
        return False
    segments = node_id.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        return False
    return bool(_NODE_ID_RE.match(node_id))


def node_id_for_note(rel_path: str) -> str:
    """Vault-relative note path -> node id.

    ``knowledge/people/anna.md`` -> ``people/anna``.
    """
    return normalize_target(rel_path)


def note_path_for_node(node_id: str) -> str:
    """Node id -> vault-relative note path (inverse of node_id_for_note).

    ``people/anna`` -> ``knowledge/people/anna.md``.
    """
    return f"knowledge/{node_id}.md"


def folder_overview_id(node_id: str) -> str:
    """The node id of ``node_id``'s folder overview note.

    ``projects/server`` -> ``projects/server/server``. Purely textual —
    says nothing about whether either note exists.
    """
    return f"{node_id}/{node_id.rsplit('/', 1)[-1]}"


def _resolve_folder_entity(node_id: str, vault_root: Path) -> str:
    """Short form -> folder-overview node id, when that is what exists.

    A project entity lives in a FOLDER (``knowledge/projects/<slug>/``)
    whose overview note is ``<slug>.md`` inside it, so its honest id is
    ``projects/<slug>/<slug>``. Notes (and the brain-project-note skill,
    historically) also wrote the bare ``projects/<slug>``, which expands
    to ``knowledge/projects/<slug>.md`` — a path with no note behind it,
    so a relation or a ``promote.target`` using it can never resolve.

    Resolution order, both by existence, never by guesswork:

    1. a flat note at ``knowledge/<node_id>.md`` wins outright — a real
       note keeps its own id (``projects/forward-deployed-engineer.md``);
    2. otherwise the folder overview ``knowledge/<node_id>/<slug>.md``,
       returned as ``<node_id>/<slug>``;
    3. otherwise ``node_id`` unchanged, so the caller reports the miss.

    Ids that aren't valid node ids (``..`` traversal, no ``/``) are
    returned untouched WITHOUT any filesystem access: nothing here may
    build a path out of a segment ``is_valid_node_id`` would reject.
    """
    if not is_valid_node_id(node_id):
        return node_id
    if (vault_root / note_path_for_node(node_id)).is_file():
        return node_id
    overview = folder_overview_id(node_id)
    if (vault_root / note_path_for_node(overview)).is_file():
        return overview
    return node_id


_FLAT_ENTITY_DIRS: Final[tuple[str, ...]] = ("people", "organisations")
_YEAR_RE = re.compile(r"^\d{4}$")
# The cross-project folder under knowledge/projects/ holds topics shared by
# several projects (AGENTS.md, "Folder-backed entities"); it is not a project
# and owns no overview.
_SHARED_PROJECT_DIR: Final[str] = "shared"
# A project's dated diary entries: knowledge/projects/<slug>/log/<date>.md.
_PROJECT_LOG_DIR: Final[str] = "log"


def _project_overview(slug: str, *, vault_root: Path) -> str | None:
    """``projects/<slug>/<slug>`` when that overview note is on disk."""
    overview = folder_overview_id(f"projects/{slug}")
    if (vault_root / note_path_for_node(overview)).is_file():
        return overview
    return None


def canonical_entity_id(rel_path: str, *, vault_root: Path) -> str | None:
    """The graph node a knowledge note belongs to, or None if it is not one.

    THE single definition of "is this note an entity node, and which" —
    every consumer (graph builders, the wikilink pass, sweep, the dream
    pass) must ask here rather than re-deriving it from a path prefix, which
    is how dated project log notes ended up as nodes of their own.

    ``rel_path`` is vault-relative and includes ``.md``
    (``knowledge/people/anna.md``). Given the layout in AGENTS.md
    ("Entity notes and typed relations", "Folder-backed entities"):

    - a flat note directly under ``knowledge/people/`` or
      ``knowledge/organisations/`` -> its own node id;
    - a meeting at ``knowledge/meetings/<YYYY>/<note>.md`` -> its own node id;
    - ``knowledge/projects/<slug>/<slug>.md`` -> ``projects/<slug>/<slug>``;
    - a curated note beside it, ``knowledge/projects/<slug>/<note>.md`` ->
      its OWN id: AGENTS.md is explicit that "curated notes beside it keep
      their own path (``projects/fps/stack-and-architecture``)", and 36 of
      the live vault's stored relation targets are exactly such notes;
    - a dated diary entry, ``knowledge/projects/<slug>/log/<note>.md`` ->
      the project's overview id, because a log entry is not a node of its
      own — provided the overview note exists (checked on disk, the way
      :func:`_resolve_folder_entity` checks it); otherwise None, because
      there is no node to speak for it;
    - a flat ``knowledge/projects/<slug>.md`` -> its own id when that file
      really is on disk (AGENTS.md tolerates the shape, and one such
      project is live). The same two-segment id with a FOLDER behind it
      instead is the short form ``normalize_target`` rewrites, and no note
      at all, so it stays None;
    - ``knowledge/projects/shared/…`` -> None (the cross-project folder is
      not a project — AGENTS.md, "Folder-backed entities");
    - anything nested deeper under a project than those documented shapes,
      and everything else — concepts, index, notes, assistant, university,
      research, anything outside ``knowledge/`` — None.
    """
    node_id = node_id_for_note(rel_path)
    if not rel_path.strip().lstrip("/").startswith("knowledge/"):
        return None
    if not is_valid_node_id(node_id):
        return None
    parts = node_id.split("/")
    if len(parts) == 2 and parts[0] in _FLAT_ENTITY_DIRS:
        return node_id
    if len(parts) == 3 and parts[0] == "meetings" and _YEAR_RE.match(parts[1]):
        return node_id
    if parts[0] != "projects":
        return None
    if len(parts) == 2:
        return node_id if (vault_root / note_path_for_node(node_id)).is_file() else None
    if parts[1] == _SHARED_PROJECT_DIR:
        return None
    if len(parts) == 4 and parts[2] == _PROJECT_LOG_DIR:
        return _project_overview(parts[1], vault_root=vault_root)
    if len(parts) == 3:
        return node_id
    return None


def owning_entity_id(rel_path: str, *, vault_root: Path) -> str | None:
    """The entity a note's CONTENT is attributed to, or None.

    :func:`canonical_entity_id` answers "which node IS this note" — the
    identity question a link target, a ``promote.target`` or an evidence
    hint asks. This answers the other one: "which entity does something
    written in this note belong to". They differ for exactly one shape, a
    project's own notes: ``projects/kern/stack.md`` is a node in its own
    right, but a sentence written there is a statement about the PROJECT,
    so a pass mining that prose attributes it to ``projects/kern/kern``
    (with the note itself kept as provenance). Everything else — people,
    organisations, meetings, a flat project note — owns its own content.

    A project note whose folder has no overview has no project to be
    attributed to, so it keeps its own id rather than losing its content.
    """
    node_id = canonical_entity_id(rel_path, vault_root=vault_root)
    if node_id is None:
        return None
    parts = node_id.split("/")
    if len(parts) == 3 and parts[0] == "projects":
        return _project_overview(parts[1], vault_root=vault_root) or node_id
    return node_id


# A superseded_by chain is a hand-written pointer; cap the walk so a long
# (or looping) chain can never spin. Merges are one or two hops in practice.
_MAX_SUPERSEDED_HOPS: Final[int] = 16


def live_entity_id(node_id: str, entities: Mapping[str, EntityInfo]) -> str | None:
    """Follow ``superseded_by`` from ``node_id`` to the live entity.

    Returns None when ``node_id`` names no entity at all. Otherwise walks
    the merge chain (AGENTS.md: supersede, never delete) and returns the
    survivor — the last node that is still an entity and no longer points
    onwards. A pointer to a node that isn't in ``entities`` is a dead end,
    so the walk stops at the last live node rather than returning something
    unresolvable; a cycle, and any chain longer than
    ``_MAX_SUPERSEDED_HOPS``, stops the same way instead of looping.
    """
    info = entities.get(node_id)
    if info is None:
        return None
    seen = {node_id}
    for _ in range(_MAX_SUPERSEDED_HOPS):
        successor = info.superseded_by
        if successor is None or successor in seen:
            break
        nxt = entities.get(successor)
        if nxt is None:
            break
        seen.add(successor)
        info = nxt
    return info.node_id


@dataclass(frozen=True)
class Relation:
    """One typed edge declared on an entity note.

    ``target`` is normalised to a node id on construction, so every
    consumer (graph builder, upsert matcher) compares canonical forms.
    Dates are kept as strings — format validation lives in the sweep
    tool, not here.
    """

    rel: str
    target: str
    valid_from: str = ""
    valid_until: str = ""
    source: str = ""    # provenance note, vault-relative without extension

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", normalize_target(self.target))


@dataclass(frozen=True)
class EntityInfo:
    """One knowledge note seen as a graph node."""

    node_id: str
    rel_path: str                    # vault-relative path incl. .md
    title: str
    type: str
    aliases: tuple[str, ...]
    relations: tuple[Relation, ...]
    updated: str
    # AGENTS.md "Merging a duplicate entity": a merged duplicate keeps its
    # note (its closed intervals are the history) but points at the survivor
    # via ``superseded_by``. Normalised to a node id, or None when the note
    # is live. :func:`live_entity_id` follows it.
    superseded_by: str | None = None


def _coerce_str(raw: object) -> str:
    """Frontmatter value -> stripped string. YAML may hand us dates or
    numbers (unquoted ``valid_from: 2025-03-01`` parses as a date); their
    ``str()`` is the ISO form we want anyway. ``None`` -> empty."""
    if raw is None:
        return ""
    return str(raw).strip()


def parse_relations(frontmatter: Mapping[str, object]) -> tuple[list[Relation], list[str]]:
    """Read ``relations:`` from a note's frontmatter, tolerantly.

    Returns ``(relations, problems)``. Malformed entries (non-list value,
    non-dict items, missing rel/target, unknown entry keys) are skipped
    with a problem string — never silently stored (F4: a typo'd key such
    as ``valid_to`` must not be dropped, or a closed span reads as open).
    Unknown rel values are EXCLUDED from the result but reported — the
    closed vocabulary keeps tooling deterministic.

    Targets are held to the SAME node-id invariants the write path
    enforces (:func:`is_valid_node_id`, checked by ``entity_upsert_relation``
    before it will store an edge): a slash-less, uppercase, non-ASCII or
    traversing target is reported and excluded, not silently carried into
    the graph as a node the write path could never address (F19).
    """
    raw = frontmatter.get("relations")
    if raw is None or raw == [] or raw == "":
        return [], []
    if not isinstance(raw, list):
        return [], [f"relations: expected a list, got {type(raw).__name__}"]

    relations: list[Relation] = []
    problems: list[str] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            problems.append(f"relations[{i}]: expected a mapping, got {type(entry).__name__}")
            continue
        unknown_keys = [str(k) for k in entry if str(k) not in _RELATION_ENTRY_KEYS]
        if unknown_keys:
            allowed = ", ".join(_RELATION_ENTRY_KEYS)
            for key in unknown_keys:
                problems.append(f"relations[{i}]: unknown key '{key}' (allowed: {allowed})")
            continue
        rel = _coerce_str(entry.get("rel"))
        target = normalize_target(_coerce_str(entry.get("target")))
        if not rel:
            problems.append(f"relations[{i}]: missing rel")
            continue
        if not target:
            problems.append(f"relations[{i}]: missing target")
            continue
        if not is_valid_node_id(target):
            problems.append(f"relations[{i}]: invalid target node id '{target}'")
            continue
        if rel not in RELATION_VOCAB:
            problems.append(f"relations[{i}]: unknown rel '{rel}'")
            continue
        relations.append(
            Relation(
                rel=rel,
                target=target,
                valid_from=_coerce_str(entry.get("valid_from")),
                valid_until=_coerce_str(entry.get("valid_until")),
                source=_coerce_str(entry.get("source")),
            )
        )
    return relations, problems


def _coerce_aliases(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw.strip(),) if raw.strip() else ()
    if isinstance(raw, list):
        return tuple(str(a).strip() for a in raw if str(a).strip())
    return ()


def _coerce_superseded_by(raw: object, vault_root: Path) -> str | None:
    """``superseded_by:`` frontmatter -> survivor node id, or None.

    Written by hand (and by the merge described in AGENTS.md) in the
    tolerated ``knowledge/people/<slug>`` form, so it goes through
    :func:`normalize_target` with the vault root — the same resolution a
    relation target gets, folder-backed projects included. A value that is
    no valid node id is dropped: a broken pointer must not become an edge.
    """
    value = _coerce_str(raw)
    if not value:
        return None
    resolved = normalize_target(value, vault_root=vault_root)
    return resolved if is_valid_node_id(resolved) else None


def entity_notes(paths: VaultPaths) -> dict[str, EntityInfo]:
    """Scan the hand-edited knowledge areas and build one EntityInfo per
    Markdown note, keyed by node id. Deterministic ordering (sorted by
    node id); unreadable files are skipped with a warning, mirroring
    ``scan_knowledge``'s tolerance."""
    found: dict[str, EntityInfo] = {}
    failures = 0
    for sub in KNOWLEDGE_NOTE_DIRS:
        base = paths.knowledge / sub
        if not base.is_dir():
            continue
        for md in sorted(base.rglob("*.md")):
            if not md.is_file():
                continue
            if md.relative_to(paths.root).as_posix().startswith(
                _ASSISTANT_HISTORY_PREFIXES
            ):
                continue  # historical audit trail, not a live entity node
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue  # vanished between rglob and read: genuinely gone
            except OSError as exc:
                failures += 1
                _LOG.warning("relations: failed to read %s (%s)", md, exc)
                continue
            if not text.strip():
                continue
            frontmatter, _body = _split_frontmatter(text)
            rel_path = md.relative_to(paths.root).as_posix()
            node_id = node_id_for_note(rel_path)
            relations, problems = parse_relations(frontmatter)
            for problem in problems:
                _LOG.warning("relations: %s: %s", rel_path, problem)
            title = _coerce_str(frontmatter.get("title")) or md.stem
            found[node_id] = EntityInfo(
                node_id=node_id,
                rel_path=rel_path,
                title=title,
                type=_coerce_str(frontmatter.get("type")),
                aliases=_coerce_aliases(frontmatter.get("aliases")),
                relations=tuple(relations),
                updated=_coerce_str(frontmatter.get("updated")),
                superseded_by=_coerce_superseded_by(
                    frontmatter.get("superseded_by"), paths.root
                ),
            )
    if failures:
        _LOG.warning("relations: %d knowledge note(s) unreadable — entity map incomplete", failures)
    return {node_id: found[node_id] for node_id in sorted(found)}


# ---------------------------------------------------------------------------
# pure text -> text editing (callers handle file I/O atomically)
# ---------------------------------------------------------------------------

def _relation_entry(relation: Relation) -> dict[str, str]:
    """Serialisable frontmatter entry. Optional keys are emitted only when
    set, matching the frontmatter contract (absent == currently valid)."""
    entry: dict[str, str] = {"rel": relation.rel, "target": relation.target}
    if relation.valid_from:
        entry["valid_from"] = relation.valid_from
    if relation.valid_until:
        entry["valid_until"] = relation.valid_until
    if relation.source:
        entry["source"] = relation.source
    return entry


def _entry_matches(entry: dict[str, object], relation: Relation) -> bool:
    return (
        _coerce_str(entry.get("rel")) == relation.rel
        and normalize_target(_coerce_str(entry.get("target"))) == relation.target
    )


# Top-level ``relations:`` key line. Quote-tolerant, like provenance.py's
# strip matcher: ``"relations":`` parses as the same key, and a splice that
# missed it would append a SECOND relations key and lose every entry.
_RELATIONS_KEY_RE = re.compile(r"""^["']?relations["']?\s*:""")


def _dump_relations(entries: list[object]) -> str:
    """The ``relations:`` block on its own, as block-style YAML.

    ``width`` is effectively unbounded: a folded long ``source:`` or
    ``target:`` round-trips fine but churns the file and defeats the
    line-oriented tools (grep, provenance's line surgery) that read it.
    """
    dumped = yaml.safe_dump(
        {"relations": entries},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=2**31 - 1,
    )
    # PyYAML ships no stubs, so mypy sees Any; without ``stream=`` it always
    # returns str. Narrowed here the way ``notes._yaml_dump_str`` does (that
    # helper only takes bool kwargs, so it can't carry ``width``).
    if not isinstance(dumped, str):
        raise TypeError(f"yaml.safe_dump did not return str: {type(dumped)!r}")
    return dumped


def _splice_relations(text: str, entries: list[object]) -> str:
    """``text`` with only its ``relations:`` block rewritten (F10).

    Every other frontmatter line and the whole body keep their source
    bytes; a note with no fence at all gets a fresh one carrying just the
    relations block, and a note whose fence has no ``relations:`` key yet
    gets the block appended after the existing keys.
    """
    block = _dump_relations(entries)
    raw = _split_frontmatter_raw(text)
    if raw is None:
        return f"---\n{block}---\n{text}"
    yaml_block, body = raw
    fm_lines = yaml_block.splitlines(keepends=True)
    all_lines = text.splitlines(keepends=True)
    open_fence, close_fence = all_lines[0], all_lines[1 + len(fm_lines)]

    start = next(
        (i for i, line in enumerate(fm_lines) if _RELATIONS_KEY_RE.match(line)),
        len(fm_lines),
    )
    # The block runs to the last indented line or column-0 sequence dash
    # after the key ("- rel:" at the key's own indent is legal YAML).
    # Blank lines neither extend it nor end it.
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
        + block
        + "".join(fm_lines[end:])
        + close_fence
        + body
    )


def _frontmatter_fence_error(text: str) -> str | None:
    """None if ``text`` has no ``---`` fence at all (nothing to check here);
    otherwise a diagnostic string if the fence fails to parse as a YAML
    mapping, else None for a fence that parses cleanly.

    Mirrors ``notes._split_frontmatter``'s own fence-extraction so the two
    stay in lockstep: this only ever reports on the same inputs for which
    that function falls back to its "no mapping" identity return.
    """
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return None
    lines = text.splitlines(keepends=True)
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return "no closing '---' fence found"
    yaml_block = "".join(lines[1:end])
    try:
        loaded = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as exc:
        return str(exc)
    if not isinstance(loaded, dict):
        return f"top-level frontmatter is a {type(loaded).__name__}, not a mapping"
    return None


_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _as_day(raw: str) -> date | None:
    """A relation bound (or ``as_of``) as a date, or None when it isn't one.

    Tolerant of the one non-canonical form YAML itself produces: an
    unquoted ``valid_from: 2025-03-01T00:00:00Z`` loads as a datetime whose
    ``str()`` is ``'2025-03-01 00:00:00+00:00'``, so the leading ten
    characters are still the day. Anything else — ``2025-1-1``, prose — is
    None: comparing those as raw text silently mis-answers ``as_of`` (F12).
    """
    head = raw.strip()[:10]
    if not _DAY_RE.match(head):
        return None
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


def _day_before(iso_date: str) -> str:
    """Best-effort ISO "day before" ``iso_date``, used to auto-close a
    superseded open span the moment a new one starts (see
    ``upsert_relation_in_text``). Falls back to ``iso_date`` unchanged when
    it isn't a parseable date — format validation is the sweep's job, not
    this module's (mirrors the trade-off in ``Relation``'s docstring)."""
    try:
        return (date.fromisoformat(iso_date) - timedelta(days=1)).isoformat()
    except ValueError:
        return iso_date


def upsert_relation_in_text(text: str, relation: Relation) -> tuple[str, str]:
    """Apply one relation to a note's frontmatter. Pure: text -> (text, action).

    Semantics (matching = same rel + same normalised target):

    - ``relation.valid_until`` set and an OPEN matching entry exists ->
      close that entry (set valid_until, keep its valid_from): the entry
      whose ``valid_from`` equals ``relation.valid_from``, or — when
      ``relation.valid_from`` is empty — the open entry with the latest
      ``valid_from``, never blindly the first open match. Result: ``"closed"``.
    - an identical entry already exists (same valid_from — both empty
      counts — and same open/closed state) -> ``"noop"``, text unchanged.
    - adding a new OPEN entry (``relation.valid_until`` empty) while an
      open entry already exists for the same (rel, target): that existing
      entry is superseded — its ``valid_until`` is set to the day before
      the new entry's ``valid_from`` — so two open spans for one edge
      never coexist. (One sentence, per the module's supersede-never-
      delete contract: the OLD span's end is inferred, never its start.)
    - otherwise append the relation as a new entry: ``"added"``.

    Existing entries — including malformed or unknown-rel ones — are never
    rewritten or removed; only a close/supersede sets one new key on one
    entry. A note with no ``---`` fence at all gets a fresh one prepended.
    A note whose fence IS present but does not parse as a YAML mapping is
    refused (raises ``ValueError``) instead of being silently buried under
    a second fence (F2) — every other key would otherwise drop out of the
    frontmatter with the tool reporting success.
    """
    fence_error = _frontmatter_fence_error(text)
    if fence_error is not None:
        raise ValueError(
            "frontmatter fence present but does not parse as a YAML "
            f"mapping: {fence_error}"
        )
    frontmatter, _body = _split_frontmatter(text)

    raw = frontmatter.get("relations")
    if isinstance(raw, list):
        entries: list[object] = raw
    elif raw in (None, "", []):
        entries = []
    else:
        # Truthy non-list (someone wrote a scalar): wrap it instead of
        # discarding — history is never deleted, even malformed history.
        entries = [raw]

    matches = [e for e in entries if isinstance(e, dict) and _entry_matches(e, relation)]

    if relation.valid_until:
        open_matches = [e for e in matches if not _coerce_str(e.get("valid_until"))]
        target_entry: dict[str, object] | None = None
        if open_matches:
            if relation.valid_from:
                # Close the open span the caller actually named, never
                # blindly the first (F3): the one whose own valid_from
                # equals relation.valid_from.
                target_entry = next(
                    (e for e in open_matches
                     if _coerce_str(e.get("valid_from")) == relation.valid_from),
                    None,
                )
            else:
                # No valid_from given ("this ended on Y", the natural close
                # form): with several open spans (legacy bad data), close
                # the one that started most recently, not entries[0].
                target_entry = max(
                    open_matches, key=lambda e: _coerce_str(e.get("valid_from"))
                )
        if target_entry is not None:
            target_entry["valid_until"] = relation.valid_until
            return _splice_relations(text, entries), "closed"
        for e in matches:
            if _coerce_str(e.get("valid_until")) != relation.valid_until:
                continue
            # A close carrying NO valid_from ("this ended on Y") is the
            # natural way to close: an empty relation.valid_from must match
            # the existing entry's own valid_from (whatever it is) so a
            # *retried* close is a noop, not a bogus duplicate. An explicit
            # different valid_from still distinguishes (falls through to
            # append), preserving the exact-match path.
            if not relation.valid_from or (
                _coerce_str(e.get("valid_from")) == relation.valid_from
            ):
                return text, "noop"   # this closed span already recorded
    else:
        for e in matches:
            if _coerce_str(e.get("valid_until")):
                continue   # ended span: a new open entry may coexist
            if _coerce_str(e.get("valid_from")) == relation.valid_from:
                return text, "noop"
        if relation.valid_from:
            # Adding a new OPEN span: any existing open entry for this
            # (rel, target) is superseded, never left open alongside it
            # (F3 — never two open spans for one edge). Close it the day
            # before this one starts.
            for e in matches:
                if _coerce_str(e.get("valid_until")):
                    continue
                e["valid_until"] = _day_before(relation.valid_from)

    entries.append(_relation_entry(relation))
    return _splice_relations(text, entries), "added"


@dataclass(frozen=True)
class RelationHit:
    """One relation entry matched by :func:`query_relations`, carrying the
    declaring entity (node id) plus the edge's fields and provenance."""

    entity: str          # node id of the note that declares the relation
    rel: str
    target: str
    valid_from: str
    valid_until: str
    source: str


def query_relations(
    paths: VaultPaths,
    *,
    rel: str | None = None,
    entity: str | None = None,
    target: str | None = None,
    as_of: str | None = None,
    include_closed: bool = False,
    limit: int = 50,
) -> list[RelationHit]:
    """Structured, time-aware query over the typed relation graph.

    Filters (all optional, ANDed):
    - ``rel``: relation name (closed vocabulary).
    - ``entity`` / ``target``: node ids (tolerant of the ``knowledge/`` and
      ``.md`` forms via ``normalize_target``); ``entity`` is the declaring
      note, ``target`` the pointed-at node (reverse lookup).
    - ``as_of`` (YYYY-MM-DD): keep only relations whose interval CONTAINS the
      date — the supersede-never-delete history made queryable ("where did X
      work last spring?"). Bounds are parsed to real dates before comparison,
      so a bound YAML handed back as ``'2025-03-01 00:00:00+00:00'`` still
      answers correctly; a relation carrying a bound that is no date at all
      is reported and excluded rather than compared as text (F12). Open
      bounds are treated as ±infinity. When ``as_of`` is given it decides
      membership; ``include_closed`` is ignored. ``ValueError`` if ``as_of``
      itself is not YYYY-MM-DD.
    - ``include_closed`` (no ``as_of``): include relations that have a
      ``valid_until`` (ended). Default returns only currently-open relations.

    Deterministic: pure read over ``entity_notes`` + ``parse_relations``.
    Results are sorted and capped at ``limit``.
    """
    ent_norm = normalize_target(entity) if entity else None
    tgt_norm = normalize_target(target) if target else None
    as_of_day = _as_day(as_of) if as_of else None
    if as_of and as_of_day is None:
        raise ValueError(f"as_of must be YYYY-MM-DD, got {as_of!r}")

    hits: list[RelationHit] = []
    for node_id, info in entity_notes(paths).items():
        if ent_norm is not None and node_id != ent_norm:
            continue
        for r in info.relations:
            if rel is not None and r.rel != rel:
                continue
            if tgt_norm is not None and r.target != tgt_norm:
                continue
            if as_of_day is not None:
                start = _as_day(r.valid_from) if r.valid_from else None
                end = _as_day(r.valid_until) if r.valid_until else None
                if (r.valid_from and start is None) or (r.valid_until and end is None):
                    _LOG.warning(
                        "relations: %s: %s -> %s has a bound that is not "
                        "YYYY-MM-DD ([%s..%s]) — excluded from the as_of query",
                        info.rel_path, r.rel, r.target,
                        r.valid_from or "open", r.valid_until or "open",
                    )
                    continue
                if start is not None and start > as_of_day:
                    continue
                if end is not None and end < as_of_day:
                    continue
            elif not include_closed and r.valid_until:
                continue  # ended relation, and no as_of asked for history
            hits.append(RelationHit(
                entity=node_id, rel=r.rel, target=r.target,
                valid_from=r.valid_from, valid_until=r.valid_until, source=r.source,
            ))
    hits.sort(key=lambda h: (h.entity, h.rel, h.target, h.valid_from, h.valid_until))
    return hits[: max(0, limit)]


def _code_fence_flags(lines: list[str]) -> list[bool]:
    """Per-line "this line belongs to a fenced code block" flags.

    The fence lines themselves count as inside, so neither the opener nor
    the closer can be mistaken for a heading or a bullet. A fence closes on
    a line of the same character, at least as long, with nothing after it;
    an unterminated fence runs to end of text, exactly as Markdown reads it.
    """
    flags: list[bool] = []
    marker = ""
    for line in lines:
        match = _CODE_FENCE_RE.match(line)
        if not marker:
            marker = match.group(1) if match is not None else ""
            flags.append(bool(marker))
            continue
        flags.append(True)
        if (
            match is not None
            and match.group(1)[0] == marker[0]
            and len(match.group(1)) >= len(marker)
            and not line.strip()[len(match.group(1)):]
        ):
            marker = ""
    return flags


def append_fact_to_log(text: str, fact_line: str) -> str:
    """Append ``- <fact_line>`` as the last bullet of the ``## Log``
    section; create the section at end of file if absent.

    Idempotent on an EXACT-duplicate bullet: if the rendered ``- <fact_line>``
    already appears verbatim as a line inside the existing ``## Log`` section,
    the text is returned unchanged. This is crash-/retry-safety — a
    consolidation pass that crashes mid-flight and reruns, or a retried
    ``entity_append_fact``, must not double-apply the same fact. Distinct
    fact lines (different date, text, or source) still both append. Pure:
    text -> text.

    Fenced code blocks are content, not structure (F11): a ``#`` comment or
    a ``## Log`` line inside one never ends or starts the section, so a
    bullet can't land inside a fence — ahead of the real last fact, which
    would break "newest LAST, never rewritten". The first bullet REPLACES a
    lone ``_(empty)_`` placeholder: a section with a fact in it is not empty.
    """
    bullet = f"- {fact_line}"

    lines = text.split("\n")
    in_fence = _code_fence_flags(lines)
    heading_idx = -1
    for i, line in enumerate(lines):
        if not in_fence[i] and line.strip() == _LOG_HEADING:
            heading_idx = i
            break

    if heading_idx < 0:
        base = text.rstrip("\n")
        if base:
            return f"{base}\n\n{_LOG_HEADING}\n\n{bullet}\n"
        return f"{_LOG_HEADING}\n\n{bullet}\n"

    # Section ends at the next heading (any level) or end of file — a '#'
    # line inside a fenced block is code, not a heading.
    end = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        if not in_fence[j] and _HEADING_RE.match(lines[j]):
            end = j
            break

    body_range = range(heading_idx + 1, end)

    # Idempotency: the exact bullet already recorded in THIS section is a
    # no-op (crash-safe consolidation reruns / retried appends). Compare with
    # a trailing '\r' stripped so a note that picked up CRLF endings (Windows
    # editor / some sync tools) doesn't defeat the dedup and double-apply.
    # A copy inside a code fence is example text, not a recorded fact.
    if any(
        not in_fence[j] and lines[j].rstrip("\r") == bullet for j in body_range
    ):
        return text

    # First real bullet: take the placeholder's place rather than sitting
    # under it, so the section never claims to be both empty and dated.
    if not any(
        not in_fence[j] and lines[j].lstrip().startswith("- ") for j in body_range
    ):
        for j in body_range:
            if not in_fence[j] and lines[j].strip() == _EMPTY_PLACEHOLDER:
                lines[j] = bullet
                return "\n".join(lines)

    # Insert after the section's last non-blank line; an empty section
    # gets a blank line between heading and first bullet.
    for j in range(end - 1, heading_idx, -1):
        if lines[j].strip():
            lines.insert(j + 1, bullet)
            break
    else:
        lines.insert(heading_idx + 1, "")
        lines.insert(heading_idx + 2, bullet)
    return "\n".join(lines)


__all__ = [
    "RELATION_VOCAB",
    "Relation",
    "EntityInfo",
    "normalize_target",
    "is_valid_node_id",
    "node_id_for_note",
    "note_path_for_node",
    "folder_overview_id",
    "canonical_entity_id",
    "owning_entity_id",
    "live_entity_id",
    "parse_relations",
    "entity_notes",
    "RelationHit",
    "query_relations",
    "upsert_relation_in_text",
    "append_fact_to_log",
]
