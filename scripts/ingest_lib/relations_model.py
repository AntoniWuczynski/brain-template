"""Entity/relation data model and tolerant frontmatter parsing.

Split out of :mod:`ingest_lib.relations` (AUD-121): the module docstring
there still owns the full picture. This slice is the "Parsing" bullet —
a tolerant reader (:func:`parse_relations`) that skips malformed entries
with a problem string instead of raising, and a vault scanner
(:func:`entity_notes`) that turns every hand-edited knowledge note into
an :class:`EntityInfo` — plus :func:`live_entity_id`, which walks the
``superseded_by`` chain those ``EntityInfo`` records carry.

Depends on :mod:`ingest_lib.relations_identity` for node-id canonicalisation;
must not import :mod:`ingest_lib.relations` (that would be a cycle back to
the module this was split out of) or :mod:`ingest_lib.relations_text`.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .config import VaultPaths
from .knowledge import KNOWLEDGE_NOTE_DIRS
from .notes import _split_frontmatter  # private helper, but module-internal
from .relations_identity import is_valid_node_id, node_id_for_note, normalize_target

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

# Hardcoded (not __name__) so log records keep the pre-split logger name
# "ingest_lib.relations" regardless of which sibling module a call site
# lives in — relations_text.py's query_relations shares this same logger.
_LOG = logging.getLogger("ingest_lib.relations")

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
