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
  with concept slugs, which never do.
- **Pure text editing**: :func:`upsert_relation_in_text` and
  :func:`append_fact_to_log` are ``text -> text`` so the MCP entity tools
  and the consolidation pass (separate stages) can apply them to any note
  body without touching the filesystem here. History is never edited or
  deleted — closing an entry and adding a new one is the only mutation
  (supersede, don't delete).

Frontmatter editing strategy: parse with ``notes._split_frontmatter``,
mutate the dict, re-serialise the WHOLE frontmatter with ``yaml.safe_dump``
preserving key insertion order. Trade-off: other keys' quote style may
normalise on the first edit, but entity notes are machine-managed enough
that this beats fragile textual splicing — and the body below the fence is
preserved byte-for-byte.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final

import yaml

from .config import VaultPaths
from .knowledge import KNOWLEDGE_NOTE_DIRS
from .notes import _split_frontmatter  # private helper, but module-internal

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

_LOG_HEADING = "## Log"
_HEADING_RE = re.compile(r"^#{1,6}\s")


def normalize_target(raw: str) -> str:
    """Canonicalise a relation target to a node id (``people/x``).

    Tolerant of the forms humans (and earlier notes) actually write:
    ``knowledge/people/x``, ``people/x.md``, ``[[knowledge/people/x]]``,
    ``[[knowledge/people/x|Anna]]`` — all collapse to ``people/x``.
    """
    t = raw.strip()
    if t.startswith("[[") and t.endswith("]]"):
        t = t[2:-2].strip()
    t = t.split("|", 1)[0].strip()  # wikilink display alias
    if t.endswith(".md"):
        t = t[: -len(".md")]
    t = t.lstrip("/")
    if t.startswith("knowledge/"):
        t = t[len("knowledge/") :]
    return t


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


def _coerce_str(raw: object) -> str:
    """Frontmatter value -> stripped string. YAML may hand us dates or
    numbers (unquoted ``valid_from: 2025-03-01`` parses as a date); their
    ``str()`` is the ISO form we want anyway. ``None`` -> empty."""
    if raw is None:
        return ""
    return str(raw).strip()


def parse_relations(frontmatter: dict[str, object]) -> tuple[list[Relation], list[str]]:
    """Read ``relations:`` from a note's frontmatter, tolerantly.

    Returns ``(relations, problems)``. Malformed entries (non-list value,
    non-dict items, missing rel/target) are skipped with a problem string.
    Unknown rel values are EXCLUDED from the result but reported — the
    closed vocabulary keeps tooling deterministic.
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
        rel = _coerce_str(entry.get("rel"))
        target = normalize_target(_coerce_str(entry.get("target")))
        if not rel:
            problems.append(f"relations[{i}]: missing rel")
            continue
        if not target:
            problems.append(f"relations[{i}]: missing target")
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


def _serialise(frontmatter: dict[str, object], body: str) -> str:
    dumped = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    return f"---\n{dumped}---\n{body}"


def upsert_relation_in_text(text: str, relation: Relation) -> tuple[str, str]:
    """Apply one relation to a note's frontmatter. Pure: text -> (text, action).

    Semantics (matching = same rel + same normalised target):

    - ``relation.valid_until`` set and an OPEN matching entry exists ->
      close that entry (set valid_until, keep its valid_from): ``"closed"``.
    - an identical entry already exists (same valid_from — both empty
      counts — and same open/closed state) -> ``"noop"``, text unchanged.
    - otherwise append the relation as a new entry: ``"added"``.

    Existing entries — including malformed or unknown-rel ones — are never
    rewritten or removed; only a close sets one new key on one entry.
    Notes without (or with unparseable) frontmatter get a fresh fence
    prepended; the body is preserved byte-for-byte either way.
    """
    frontmatter, body = _split_frontmatter(text)

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
        if open_matches:
            open_matches[0]["valid_until"] = relation.valid_until
            frontmatter["relations"] = entries
            return _serialise(frontmatter, body), "closed"
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

    entries.append(_relation_entry(relation))
    frontmatter["relations"] = entries
    return _serialise(frontmatter, body), "added"


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
      work last spring?"). Open bounds are treated as ±infinity. When
      ``as_of`` is given it decides membership; ``include_closed`` is ignored.
    - ``include_closed`` (no ``as_of``): include relations that have a
      ``valid_until`` (ended). Default returns only currently-open relations.

    Deterministic: pure read over ``entity_notes`` + ``parse_relations``.
    Results are sorted and capped at ``limit``.
    """
    ent_norm = normalize_target(entity) if entity else None
    tgt_norm = normalize_target(target) if target else None

    hits: list[RelationHit] = []
    for node_id, info in entity_notes(paths).items():
        if ent_norm is not None and node_id != ent_norm:
            continue
        for r in info.relations:
            if rel is not None and r.rel != rel:
                continue
            if tgt_norm is not None and r.target != tgt_norm:
                continue
            if as_of is not None:
                # Interval containment (YYYY-MM-DD compares lexicographically).
                if r.valid_from and r.valid_from > as_of:
                    continue
                if r.valid_until and r.valid_until < as_of:
                    continue
            elif not include_closed and r.valid_until:
                continue  # ended relation, and no as_of asked for history
            hits.append(RelationHit(
                entity=node_id, rel=r.rel, target=r.target,
                valid_from=r.valid_from, valid_until=r.valid_until, source=r.source,
            ))
    hits.sort(key=lambda h: (h.entity, h.rel, h.target, h.valid_from, h.valid_until))
    return hits[: max(0, limit)]


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
    """
    bullet = f"- {fact_line}"

    lines = text.split("\n")
    heading_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == _LOG_HEADING:
            heading_idx = i
            break

    if heading_idx < 0:
        base = text.rstrip("\n")
        if base:
            return f"{base}\n\n{_LOG_HEADING}\n\n{bullet}\n"
        return f"{_LOG_HEADING}\n\n{bullet}\n"

    # Section ends at the next heading (any level) or end of file.
    end = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        if _HEADING_RE.match(lines[j]):
            end = j
            break

    # Idempotency: the exact bullet already recorded in THIS section is a
    # no-op (crash-safe consolidation reruns / retried appends). Compare with
    # a trailing '\r' stripped so a note that picked up CRLF endings (Windows
    # editor / some sync tools) doesn't defeat the dedup and double-apply.
    if any(lines[j].rstrip("\r") == bullet for j in range(heading_idx + 1, end)):
        return text

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
    "parse_relations",
    "entity_notes",
    "RelationHit",
    "query_relations",
    "upsert_relation_in_text",
    "append_fact_to_log",
]
