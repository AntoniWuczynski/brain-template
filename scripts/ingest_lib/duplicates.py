"""Duplicate-entity merge candidates (TODO.md, Proposals, 2026-08-28 GBrain
group): a deterministic pass that turns ``sweep``'s ``entity-duplicate``
detection into merge proposals under ``knowledge/assistant/inbox/``.

GBrain auto-merges duplicate entities (an email-slugged node beside a named
node for the same person or organisation). The vault keeps its propose-only
posture instead: ``sweep._check_entity_duplicates`` (refactored here into
:func:`ingest_lib.sweep.find_duplicate_entity_pairs`, an importable function
so this module and the sweep agree on exactly what counts as a duplicate and
what counts as already merged) only REPORTS the pair. This module is the
proposal half — same shape as :mod:`ingest_lib.wikilinks`, built on the same
:func:`ingest_lib.propose.propose_fact`.

A proposal's ``promote.target`` is the NAMED node — the one that survives a
merge. ``promote.fact`` states plainly that the email-slugged node appears to
be the same entity and names the merge steps AGENTS.md's "Merging a
duplicate entity" paragraph spells out (address as an alias, `related_to`
both ways, supersede the email node). ``promote.relations`` is deliberately
empty: a merge is not a single typed relation, and ``scripts/consolidate.py``
cannot execute one today — the note body says so, so a human who approves
the proposal knows the merge itself is still theirs (or an agent's) to
perform per AGENTS.md, not something approval alone completes.

Once a pair has actually been merged (any part of the AGENTS.md shape
present), :func:`ingest_lib.sweep.find_duplicate_entity_pairs` stops
returning it — reused here rather than re-derived, so this pass goes quiet
on a merged pair exactly when the sweep does, and a nightly re-run over an
unmerged pair proposes nothing new (the same content hashes to the same
inbox filename every time).
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import VaultPaths, default_paths
from .propose import ProposeResult, propose_fact
from .sweep import DuplicateEntityPair, find_duplicate_entity_pairs

_LOG = logging.getLogger(__name__)

_KIND = "entity-duplicate-merge"

_MERGE_STEPS = (
    "merge them by superseding, never deleting (AGENTS.md, 'Merging a "
    "duplicate entity'): copy the duplicate's relations onto the named "
    "node, close every relation on the duplicate with valid_until, add "
    "superseded_by: knowledge/{named} to its frontmatter, and keep the "
    "address as an aliases: entry on the named node"
)


def _fact(pair: DuplicateEntityPair, named_node_id: str) -> str:
    return (
        f"'{pair.email_node.title}' ({pair.email_node.node_id}) appears to be "
        f"the same entity as {named_node_id} — "
        + _MERGE_STEPS.format(named=named_node_id)
        + "."
    )


def _reason(pair: DuplicateEntityPair, named_node_id: str) -> str:
    return (
        "Deterministic duplicate-entity pass (scripts/ingest_lib/duplicates.py), "
        "reusing sweep's `entity-duplicate` detection: "
        f"[[knowledge/{pair.email_node.node_id}]] is titled a bare email "
        f"address and [[knowledge/{named_node_id}]] names the same local "
        "part, with none of the AGENTS.md merge shape present yet between "
        "them. Proposed relations are deliberately empty — a merge is not a "
        "single typed relation, and scripts/consolidate.py cannot execute "
        "one today. Approving this note records that the merge is believed "
        "correct; performing the merge (superseding the duplicate, copying "
        "its relations, adding the alias) is still a human's or an agent's "
        "job per AGENTS.md."
    )


def propose_duplicate_merges(
    paths: VaultPaths, *, dry_run: bool = False
) -> list[ProposeResult]:
    """Run the pass and propose one merge candidate per email-slugged node
    :func:`ingest_lib.sweep.find_duplicate_entity_pairs` still reports with
    exactly one named candidate. A pair with more than one named candidate
    (the same local part matching two distinct named nodes) is ambiguous —
    there is no single survivor to name as ``promote.target`` — and is left
    for a human to disambiguate by hand rather than guessed at; the sweep
    report still flags it. Returns one :class:`ProposeResult` per proposal —
    ``"exists"`` for ones already in the inbox from an earlier run,
    ``"proposed"`` for new ones (or, under ``dry_run``, what would be newly
    written)."""
    results: list[ProposeResult] = []
    for pair in find_duplicate_entity_pairs(paths):
        if len(pair.named_candidates) != 1:
            continue
        named = pair.named_candidates[0]
        result = propose_fact(
            paths,
            kind=_KIND,
            title=(
                f"duplicate-entity merge candidate: {pair.email_node.node_id} "
                f"-> {named.node_id}"
            ),
            target=named.node_id,
            source=pair.email_node.node_id,
            reason=_reason(pair, named.node_id),
            fact=_fact(pair, named.node_id),
            dry_run=dry_run,
        )
        _LOG.info(
            "duplicates: %s %s -> %s (%s)",
            result.action, pair.email_node.node_id, named.node_id, result.path,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
#
# Same convention as ingest_lib.wikilinks: the CLI lives in this module,
# invoked as ``python -m ingest_lib.duplicates`` (ingest_lib is an installed
# package, so -m resolves the relative imports above correctly).
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="brain-duplicates",
        description=(
            "Deterministic (zero-LLM) pass: propose duplicate-entity merge "
            "candidates into knowledge/assistant/inbox/ for email-slugged "
            "person/organisation nodes sitting beside a named node for the "
            "same entity. Never writes to an entity note directly, and never "
            "performs the merge itself — a human (or an agent) still has to, "
            "per AGENTS.md."
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
    results = propose_duplicate_merges(paths, dry_run=args.dry_run)
    proposed = sum(1 for r in results if r.action == "proposed")
    exists = sum(1 for r in results if r.action == "exists")
    archived = sum(1 for r in results if r.action == "archived")

    print(f"duplicate-entity merge candidates ({'dry-run' if args.dry_run else 'real'}):")
    print(f"  candidates found : {len(results)}")
    print(f"  proposed         : {proposed}")
    print(f"  already in inbox : {exists}")
    print(f"  already archived : {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["propose_duplicate_merges", "main"]
