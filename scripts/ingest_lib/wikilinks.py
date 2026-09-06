"""Wikilink-derived relation candidates (TODO.md, Proposals, 2026-08-28
GBrain group): a deterministic, zero-LLM pass that turns a prose wikilink
between two entity notes into a candidate ``related_to`` relation.

Today an agent writing a note links entities in prose —
``[[knowledge/people/anna-kowalska]]`` — but that link never becomes a
typed relation unless the agent separately calls
``entity_upsert_relation``. GBrain extracts such edges from wikilinks
deterministically at write time; this pass keeps the vault's propose-only
posture instead of copying GBrain's auto-merge: it finds candidates and
drops them into ``knowledge/assistant/inbox/`` via
:mod:`ingest_lib.propose`, and only ``scripts/consolidate.py`` (gated by
a human's approval or enough confirmations) ever turns one into a real
relation.

Scope, each rule chosen to keep the pass exact rather than noisy:

- Both ends of a candidate edge are entity node ids, resolved by
  :func:`ingest_lib.relations.owning_entity_id` — "which entity does
  prose written in this note belong to". A dated session log
  (``projects/agent/log/2026-07-27``) is a diary entry, and a curated note
  beside a project's overview (``projects/kern/stack``) is a node of its
  own but still a document ABOUT the project: prose in either belongs to
  ``projects/agent/agent`` / ``projects/kern/kern``, so a link found there
  is attributed to the project, with the note itself recorded as the
  relation's ``source:`` provenance. (Identity — what a link may POINT at
  — is :func:`ingest_lib.relations.canonical_entity_id`; this pass asks
  the ownership question instead, because a relation proposed on a
  project's stack note is really a claim about the project.) A note that
  is no entity at all (a concept, a research note, and — the
  feedback-loop guard — this pass's own output under
  ``knowledge/assistant/inbox/``, whose body quotes the very links that
  produced it) yields no candidates in either direction. Measured before
  this rule, 188 of the live vault's 308 candidates had a dated log note
  at one end.
- A link resolving to a node with no on-disk note is skipped outright —
  nothing to attach a candidate to.
- Both ends are then followed through ``superseded_by`` with
  :func:`ingest_lib.relations.live_entity_id`: a merged duplicate is
  history that must not grow new edges, and the link means the survivor.
  19 live candidates pointed at the owner's already-merged email twin
  before this rule.
- A link inside one project — its log or curated notes pointing at the
  overview or a sibling — collapses to ``source == target`` once both
  ends are canonicalised and is dropped there: that is folder
  navigation, not a relation. 193 of the 308 were exactly this.
- A link already covered by ANY typed relation (not only an existing
  ``related_to``) is skipped, in EITHER direction: a ``works_at`` edge on
  the source, or an ``attended`` edge declared back on the target,
  already says everything a ``related_to`` proposal would add. Stored
  relation targets are re-normalised against the vault before comparing,
  so the short folder form AGENTS.md tolerates (``projects/kern``)
  suppresses a candidate for ``projects/kern/kern`` as it should.
- A meeting linking one of its own ``attendees:`` is skipped: the link is
  the generated "- Attendee: [[…]]" bullet rendering that frontmatter, and
  the edge is already declared twice — as ``attendees:`` here and as
  ``attended`` on the person. All 64 live meeting candidates were this.
- Frontmatter and fenced code blocks are not prose: structural keys
  (``project: "[[knowledge/projects/kern/kern]]"``) and YAML examples
  inside a fence are stripped before the scan.
- ``related_to`` is the only rel ever proposed — it is the one honest
  relation for a bare prose link; guessing ``works_at`` from surrounding
  text would need an LLM, which this pass deliberately has none of.
- ``related_to`` is UNDIRECTED — the pass's own suppression rule treats an
  edge declared either way round as covering the pair — so one pair is one
  candidate, not two. A and B each linking the other used to produce two
  mirrored proposals for one edge (4 of the live vault's 10, and 30
  proposals for 15 edges among 6 cross-linked people); the pair is now
  keyed unordered and emitted on the end that sorts first, with the first
  note seen kept as provenance.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import VaultPaths, default_paths
# private, but module-internal (relations.py/sweep.py do the same)
from .notes import (
    _split_frontmatter_raw,
    _top_level_raw_blocks,
    _try_parse_frontmatter_block,
)
from .propose import ProposeResult, propose_fact
from .relations import (
    EntityInfo,
    Relation,
    _code_fence_flags,  # private, but module-internal (see relations.py)
    entity_notes,
    is_valid_node_id,
    live_entity_id,
    normalize_target,
    note_path_for_node,
    owning_entity_id,
)

_LOG = logging.getLogger(__name__)

# Same pattern sweep.py's dangling-wikilink check uses.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")

_KIND = "wikilink-relation"


@dataclass(frozen=True)
class WikilinkCandidate:
    """One prose link between two entity nodes with no typed relation yet.

    ``source``/``target`` are the pair in its canonical (sorted) order, not
    the direction the link happened to be written in: ``related_to`` is
    undirected, so the same pair found from either end is one candidate.
    ``found_on`` is the end the link was actually written on, which is what
    the evidence prose describes.
    """

    source: str       # node id the proposal is written on (the lower of the pair)
    target: str       # the other end of the pair
    found_on: str     # node id of the entity the link was written on
    source_note: str  # node id of the note the link was actually found in
    raw_link: str     # the link text as written, for the evidence trail


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _LOG.warning("wikilinks: failed to read %s (%s)", path, exc)
        return None


def _frontmatter_mapping(yaml_block: str) -> dict[str, object]:
    """The frontmatter keys of an already-fenced-out block.

    A block that parses whole is used as it is. One that does not is read
    key by key instead of thrown away: a single malformed value (an
    unclosed ``topics: [a, b``) must not silently cost the ``attendees:``
    list the attendee-suppression rule depends on — losing it turns a
    meeting's own generated "- Attendee: [[…]]" bullets back into
    discovered links on exactly the notes least worth trusting.
    """
    parsed = _try_parse_frontmatter_block(yaml_block)
    if parsed is not None:
        return parsed
    salvaged: dict[str, object] = {}
    for raw_key_block in _top_level_raw_blocks(yaml_block).values():
        one = _try_parse_frontmatter_block(raw_key_block)
        if one is not None:
            salvaged.update(one)
    return salvaged


def _prose(text: str) -> tuple[dict[str, object], str]:
    """(frontmatter, scannable prose) for one note.

    Frontmatter keys are structure, not prose — ``project:
    "[[knowledge/projects/kern/kern]]"`` states the note's own folder, and
    reading it as a discovered link proposes an edge the layout already
    implies. Fenced code blocks are content: a YAML example inside one is
    documentation, not an assertion about this vault. Both are dropped
    before the link scan — the fence textually, so a note whose YAML does
    not parse still has its frontmatter stripped.
    """
    raw = _split_frontmatter_raw(text)
    if raw is None:
        frontmatter: dict[str, object] = {}
        body = text
    else:
        yaml_block, body = raw
        # The fence is stripped from the RAW split, which succeeds whether or
        # not the YAML parses. _split_frontmatter hands the whole text back as
        # the body when the block is malformed, which would feed the
        # frontmatter into the link scan — on exactly the notes least worth
        # trusting.
        frontmatter = _frontmatter_mapping(yaml_block)
    lines = body.splitlines()
    flags = _code_fence_flags(lines)
    return frontmatter, "\n".join(
        line for line, fenced in zip(lines, flags, strict=True) if not fenced
    )


def _attendee_ids(
    frontmatter: Mapping[str, object],
    *,
    entities: Mapping[str, EntityInfo],
    vault_root: Path,
) -> set[str]:
    """The live node ids a meeting's ``attendees:`` frontmatter names.

    A meeting note's "- Attendee: [[…]]" bullets are generated FROM this
    key, so a link matching one of them is the frontmatter read back, not a
    discovered edge. Anything that isn't a string, or doesn't normalise to
    a valid node id, is dropped rather than guessed at; a merged duplicate
    listed here resolves to the same survivor a link to it would.
    """
    raws = frontmatter.get("attendees")
    if not isinstance(raws, list):
        return set()
    entries: list[object] = raws
    found: set[str] = set()
    for raw in entries:
        if not isinstance(raw, str):
            continue
        node_id = normalize_target(raw, vault_root=vault_root)
        if not is_valid_node_id(node_id):
            continue
        found.add(live_entity_id(node_id, entities) or node_id)
    return found


def _related_ids(info: EntityInfo, *, vault_root: Path) -> set[str]:
    """The node ids an entity note already declares a typed relation to.

    ``Relation.target`` is normalised WITHOUT the vault root, so a target
    stored in the short folder form AGENTS.md tolerates (``projects/kern``)
    stays short while a candidate's target resolves to
    ``projects/kern/kern`` — re-normalising here is what makes the two
    compare equal, so an already-declared edge really does suppress the
    candidate.
    """
    return {normalize_target(r.target, vault_root=vault_root) for r in info.relations}


def find_candidates(paths: VaultPaths) -> list[WikilinkCandidate]:
    """Scan every knowledge note that belongs to a graph node for wikilinks
    into another node, and return the edges not already declared as a typed
    relation in either direction.

    Deterministic and sorted; a note this run can't read is skipped rather
    than raising (relations.entity_notes's own tolerance). One candidate per
    distinct UNORDERED pair — ``related_to`` is undirected, so A linking B
    and B linking A are one edge, not two — evidenced by the first note in
    node-id order that carries it, and emitted on whichever end sorts
    first. Several notes in one project folder linking the same entity are
    likewise one edge on the project.
    """
    entities = entity_notes(paths)
    root = paths.root
    related: dict[str, set[str]] = {}

    def related_ids(node_id: str) -> set[str]:
        cached = related.get(node_id)
        if cached is None:
            cached = _related_ids(entities[node_id], vault_root=root)
            related[node_id] = cached
        return cached

    candidates: list[WikilinkCandidate] = []
    seen: set[tuple[str, str]] = set()
    for node_id, info in entities.items():
        # THE gate: only a note that belongs to a graph node can carry an
        # edge, and the node it belongs to — not the note — is the source.
        owner = owning_entity_id(info.rel_path, vault_root=root)
        if owner is None:
            continue
        source = live_entity_id(owner, entities)
        if source is None:
            continue  # the overview note exists but is empty: no node to speak for it
        text = _read(root / info.rel_path)
        if text is None:
            continue
        frontmatter, prose = _prose(text)
        attendees = _attendee_ids(frontmatter, entities=entities, vault_root=root)
        for raw in _WIKILINK_RE.findall(prose):
            raw = raw.strip()
            if not raw or raw.startswith(("http://", "https://")):
                continue
            linked = normalize_target(raw, vault_root=root)
            if not is_valid_node_id(linked):
                continue
            linked_info = entities.get(linked)
            if linked_info is None:
                continue  # resolves to no on-disk entity note
            owned_by = owning_entity_id(linked_info.rel_path, vault_root=root)
            if owned_by is None:
                continue  # links a note that is not (and does not belong to) a node
            # A merged duplicate (AGENTS.md ``superseded_by``) is history, not
            # a node to grow new edges on: the link means the survivor.
            target = live_entity_id(owned_by, entities)
            if target is None or target == source:
                continue  # a link inside one project is navigation, not a relation
            # related_to is undirected and the suppression rule below already
            # treats an edge declared either way round as covering the pair,
            # so the pair is keyed unordered: the mirror link found from the
            # other end is the same candidate, not a second one.
            pair = (source, target) if source < target else (target, source)
            if pair in seen:
                continue
            if target in related_ids(source) or source in related_ids(target):
                continue  # already a typed relation, in one direction or the other
            if target in attendees:
                continue  # the meeting's own attendees: frontmatter already says this
            seen.add(pair)
            candidates.append(WikilinkCandidate(
                source=pair[0], target=pair[1], found_on=source,
                source_note=node_id, raw_link=raw,
            ))
    candidates.sort(key=lambda c: (c.source, c.target))
    return candidates


def _reason(
    candidate: WikilinkCandidate,
    *,
    source_link: str,
    target_link: str,
    found_link: str,
    linked_link: str,
    note_link: str,
) -> str:
    """Evidence prose for the proposal body.

    Describes the link as it was actually found — which note, on which
    entity, pointing where — and then says which end of the pair the
    proposal is written on, since the pair is emitted in sorted order and
    that need not be the end the link was written on.

    Every wikilink written here is the CANONICAL form, never the link as
    typed: the pass deliberately accepts the short folder form on input
    (``[[projects/server]]``), and echoing that into a note would mint a
    fresh ``wikilink-not-canonical`` (or ``dangling-wikilink``) sweep
    finding per proposal. The form as written is still on the record —
    quoted as inline code, with no brackets, so it is evidence rather than
    a live link.
    """
    # When the link was written on the entity note itself, "that note
    # belongs to the entity X" would restate the sentence it follows.
    belongs = "" if note_link == found_link else (
        f" That note belongs to the entity [[{found_link}]]."
    )
    return (
        "Deterministic wikilink pass (scripts/ingest_lib/wikilinks.py): "
        f"[[{note_link}]] links [[{linked_link}]] (written as `{candidate.raw_link}`) "
        f"in prose.{belongs} Neither entity declares a typed relation to the "
        "other, and `related_to` is undirected, so the pair is proposed once — "
        f"on [[{source_link}]], with [[{target_link}]] as the target. "
        "`related_to` is the only honest relation for a bare prose link — see "
        "AGENTS.md's wikilink-derived relation candidates."
    )


def propose_wikilink_relations(
    paths: VaultPaths, *, dry_run: bool = False
) -> list[ProposeResult]:
    """Run the pass and propose one ``related_to`` fact note per candidate.

    Returns one :class:`ProposeResult` per candidate found this run —
    ``action="exists"`` for ones already sitting in the inbox from an
    earlier run, ``"archived"`` for ones consolidate has since moved out of
    it, ``"proposed"`` for ones newly written (or, under ``dry_run``, what
    would be newly written). Re-running over an unchanged vault therefore
    proposes nothing new: every candidate lands on the same
    content-derived filename as before.
    """
    results: list[ProposeResult] = []
    for candidate in find_candidates(paths):
        source_link = note_path_for_node(candidate.source)[: -len(".md")]
        target_link = note_path_for_node(candidate.target)[: -len(".md")]
        found_link = note_path_for_node(candidate.found_on)[: -len(".md")]
        # The end the link actually pointed at, which is the other one.
        linked = (
            candidate.target if candidate.found_on == candidate.source
            else candidate.source
        )
        linked_link = note_path_for_node(linked)[: -len(".md")]
        # Provenance is the note the link was actually found in — often a
        # dated log under the project, never the project overview it is
        # attributed to. The relation carries it so an approved edge can be
        # traced back to the sentence that produced it.
        note_link = note_path_for_node(candidate.source_note)[: -len(".md")]
        relation = Relation(rel="related_to", target=candidate.target, source=note_link)
        result = propose_fact(
            paths,
            kind=_KIND,
            title=f"related_to candidate: {candidate.source} -> {candidate.target}",
            target=candidate.source,
            source=note_link,
            reason=_reason(
                candidate,
                source_link=source_link,
                target_link=target_link,
                found_link=found_link,
                linked_link=linked_link,
                note_link=note_link,
            ),
            relations=[relation],
            dry_run=dry_run,
        )
        _LOG.info(
            "wikilinks: %s %s -> %s (from %s, %s)",
            result.action, candidate.source, candidate.target,
            candidate.source_note, result.path,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
#
# Every other ingest_lib module keeps its CLI in a separate top-level
# scripts/*.py wrapper (consolidate.py, dream_gate.py, sweep.py) so the
# logic module keeps clean relative imports. That wrapper is deliberately
# NOT added here; the CLI lives in this module instead, invoked as
# ``python -m ingest_lib.wikilinks`` (ingest_lib is an installed package —
# see pyproject's hatch config — so ``-m`` resolves the relative imports
# above correctly, unlike running this file by path).
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="brain-wikilinks",
        description=(
            "Deterministic (zero-LLM) pass: propose related_to relation "
            "candidates into knowledge/assistant/inbox/ for wikilinks between "
            "entity notes (people/organisations/projects/meetings) that are "
            "not already a typed relation. Never writes to an entity note "
            "directly — scripts/consolidate.py gates promotion."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what WOULD be proposed without writing anything.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    paths = default_paths()
    results = propose_wikilink_relations(paths, dry_run=args.dry_run)
    proposed = sum(1 for r in results if r.action == "proposed")
    exists = sum(1 for r in results if r.action == "exists")
    archived = sum(1 for r in results if r.action == "archived")

    print(f"wikilink relation candidates ({'dry-run' if args.dry_run else 'real'}):")
    print(f"  candidates found : {len(results)}")
    print(f"  proposed         : {proposed}")
    print(f"  already in inbox : {exists}")
    # Declined or digested: put to the human once and moved to the archive,
    # so the file named in the log line is NOT in the inbox.
    print(f"  already archived : {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["WikilinkCandidate", "find_candidates", "propose_wikilink_relations", "main"]
