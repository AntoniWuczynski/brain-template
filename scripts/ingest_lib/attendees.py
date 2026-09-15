"""Resolve a snapshot attendee's display NAME to a live ``people/`` node.

Split out of :mod:`ingest_lib.meetings` (which owns the promotion pass
itself) because the two halves answer different questions and the module was
past the repo's 1000-line ceiling: this one only ever READS
``knowledge/people/`` and never writes anything, anywhere.

The rule the whole file exists to keep: **an unknown attendee mints no
node.** ``meeting_create`` creates a node per attendee id it is handed, which
is how a calendar payload carrying an email address instead of a display name
split one person into two nodes, each collecting its own ``attended`` history
(the ``entity-duplicate`` pairs ``sweep``/``duplicates`` now report). Here a
name that matches no existing note resolves to nothing at all, and a name
that matches two resolves to nothing rather than to a guess — a wrong
``attended`` edge is worse than a missing one a human can add.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .config import VaultPaths
from .extractors.meeting import attendee_slug
from .relations import (
    EntityInfo,
    canonical_entity_id,
    entity_notes,
    is_valid_node_id,
    live_entity_id,
)

_PEOPLE_PREFIX = "people/"

# Why an attendee did not become a graph edge. "resolved" is the success
# case; the other two are reported by display name and mint nothing.
AttendeeStatus = Literal["resolved", "no-note", "ambiguous"]


@dataclass(frozen=True)
class AttendeeResolution:
    """One snapshot attendee, resolved (or not) to a live ``people/`` node."""

    display_name: str
    node_id: str  # "" unless status == "resolved"
    status: AttendeeStatus


def _name_key(name: str) -> str:
    """Comparison key for a display name: whitespace collapsed, casefolded."""
    return " ".join(name.split()).casefold()


@dataclass(frozen=True)
class PeopleIndex:
    """The ``knowledge/people/`` nodes, indexed by the three things an
    attendee display name can be matched against."""

    entities: Mapping[str, EntityInfo]
    by_title: Mapping[str, tuple[str, ...]]
    by_alias: Mapping[str, tuple[str, ...]]


def people_index(paths: VaultPaths) -> PeopleIndex:
    entities = entity_notes(paths)
    titles: dict[str, list[str]] = {}
    aliases: dict[str, list[str]] = {}
    for node_id, info in entities.items():
        if not node_id.startswith(_PEOPLE_PREFIX):
            continue
        # Identity is the entity helper's answer, never the path prefix:
        # a note nested deeper under people/ is no node of its own.
        if canonical_entity_id(info.rel_path, vault_root=paths.root) != node_id:
            continue
        title_key = _name_key(info.title)
        if title_key:
            titles.setdefault(title_key, []).append(node_id)
        for alias in info.aliases:
            alias_key = _name_key(alias)
            if alias_key:
                aliases.setdefault(alias_key, []).append(node_id)
    return PeopleIndex(
        entities=entities,
        by_title={k: tuple(sorted(v)) for k, v in titles.items()},
        by_alias={k: tuple(sorted(v)) for k, v in aliases.items()},
    )


def _live_person(node_id: str, index: PeopleIndex) -> str | None:
    """``node_id`` followed through ``superseded_by``, if the survivor is
    still a person node. A merge that pointed somewhere else entirely is a
    pointer this pass will not guess at.

    A chain that DEAD-ENDS is refused too. ``live_entity_id`` stops at the
    last node that is still an entity, so a half-merged duplicate whose
    ``superseded_by`` names a note that is not on disk (or a cycle) comes
    back as itself — and hanging a new ``attended`` edge off it is exactly
    the email-twin growth AGENTS.md's merge shape exists to stop. The test
    is the SURVIVOR's own pointer: a node that still points onwards after
    the walk is a node the walk could not complete.
    """
    live = live_entity_id(node_id, index.entities)
    if live is None or not live.startswith(_PEOPLE_PREFIX):
        return None
    survivor = index.entities.get(live)
    if survivor is None or survivor.superseded_by is not None:
        return None
    return live if is_valid_node_id(live) else None


def resolve_attendee(name: str, index: PeopleIndex) -> AttendeeResolution:
    """Resolve one display name to a live ``people/`` node, or say why not.

    Three ways a name can match, each following ``superseded_by`` to the
    live survivor before anything is compared:

    1. the accent-folded slug — ``Antoni Wuczyński`` -> ``people/antoni-wuczynski``;
    2. the exact note title — this is what catches an attendee named by
       their email address, whose node is slugged without separators
       (``people/alexasymmetricsecuritycom``) but titled with the address
       verbatim;
    3. an exact ``aliases:`` entry — AGENTS.md's merge keeps the address as
       an alias on the named survivor, so a snapshot still carrying it
       resolves onto the person rather than the merged twin.

    The three are UNIONED rather than tried in precedence order, and the
    union must come out at exactly one live node. Precedence would let the
    narrowest match hide a real ambiguity: two notes both titled "Alice
    Smith" are two people with one name, and the fact that one of them also
    happens to own the ``people/alice-smith`` slug is no evidence about
    which of them was in the room. A tie is reported ``ambiguous`` and
    resolves to nothing — a wrong ``attended`` edge is worse than a missing
    one a human can add.
    """
    found: set[str] = set()
    slug = attendee_slug(name)
    if slug and f"{_PEOPLE_PREFIX}{slug}" in index.entities:
        live = _live_person(f"{_PEOPLE_PREFIX}{slug}", index)
        if live is not None:
            found.add(live)
    key = _name_key(name)
    for table in (index.by_title, index.by_alias):
        for hit in table.get(key, ()):
            live = _live_person(hit, index)
            if live is not None:
                found.add(live)
    if len(found) == 1:
        return AttendeeResolution(name, found.pop(), "resolved")
    if found:
        return AttendeeResolution(name, "", "ambiguous")
    return AttendeeResolution(name, "", "no-note")


__all__ = [
    "AttendeeResolution",
    "AttendeeStatus",
    "PeopleIndex",
    "people_index",
    "resolve_attendee",
]
