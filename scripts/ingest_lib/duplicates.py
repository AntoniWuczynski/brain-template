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
merge — and the proposal carries ``promote.merge: {duplicate, survivor}``,
the shape ``scripts/consolidate.py`` executes on ``approved: true``
(FOUNDER_DECISIONS.md IMP-021). ``promote.relations`` stays empty: a merge is
not a single typed relation, it is the four steps AGENTS.md's "Merging a
duplicate entity" paragraph spells out, and ``promote.merge`` is how they are
named. ``promote.fact`` and the note body still say plainly what approving
will do — copy the duplicate's relations onto the named node, close every
relation on the duplicate with ``valid_until``, stamp ``superseded_by``, keep
the address as an alias — so a human approves a described act, not a blank
cheque. Nothing happens without that ``approved: true``: this pass only ever
writes into ``knowledge/assistant/inbox/``, never into an entity note.

Once a pair has actually been merged (any part of the AGENTS.md shape
present), :func:`ingest_lib.sweep.find_duplicate_entity_pairs` stops
returning it — reused here rather than re-derived, so this pass goes quiet
on a merged pair exactly when the sweep does.

A nightly re-run over an unmerged pair proposes nothing new either, and that
holds across edits to this file: the inbox filename hashes the pair's
IDENTITY (``propose._content_key``'s ``identity``), not the generated
wording, and before proposing anything the pass looks for a note it already
wrote for the same pair and reconciles that one in place
(:func:`propose_duplicate_merges`). One pair is one note in the human's
queue, however many times the pass runs and however its prose is reworded.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from .atomic import atomic_write_text, umask_mode
from .concepts import slugify
from .config import VaultPaths, default_paths
from .notes import _split_frontmatter  # private helper, package-internal
# _INBOX_REL / _SLUG_MAX_CHARS are propose.py's own filename contract, imported
# rather than re-declared so the reconciliation glob below can never drift out
# of step with the names propose_fact actually writes.
from .propose import (
    _INBOX_REL,
    _SLUG_MAX_CHARS,
    ProposeAction,
    ProposeResult,
    propose_fact,
)
from .relations import normalize_target, note_path_for_node
from .sweep import DuplicateEntityPair, find_duplicate_entity_pairs

_LOG = logging.getLogger(__name__)

_KIND = "entity-duplicate-merge"

# Quote-tolerant frontmatter key matches, the same shape as
# ``relations._RELATIONS_KEY_RE``: PyYAML round-trips ``"promote":`` and
# ``'source':`` as ordinary keys, so a splice that only recognised the bare
# spelling would silently give up on a note it should have healed.
_PROMOTE_KEY_RE = re.compile(r"""^["']?promote["']?\s*:""")
# Exactly the two-space indent yaml.safe_dump gives a key of the promote
# mapping — never the deeper indent of a key INSIDE a relations entry, which
# has a ``source:`` of its own.
_SOURCE_KEY_RE = re.compile(r"""^ {2}["']?source["']?\s*:""")

# promote.fact lands VERBATIM in the survivor's ## Log, and it only ever
# lands there once consolidate has actually performed the merge (a refused
# merge writes no fact line at all). So it is phrased as the record of a
# completed merge, not as a promise — the proposal's body carries the
# future tense, for the human deciding whether to approve.
_MERGE_STEPS = (
    "merged into it by superseding, never deleting (AGENTS.md, 'Merging a "
    "duplicate entity'): the duplicate's relations copied onto the named "
    "node, its open relations closed with valid_until, superseded_by: "
    "knowledge/{named} added to its frontmatter, and the address kept as an "
    "aliases: entry on the named node"
)


def _fact(pair: DuplicateEntityPair, named_node_id: str) -> str:
    return (
        f"'{pair.email_node.title}' ({pair.email_node.node_id}) is the same "
        f"entity as {named_node_id} — "
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
        "single typed relation; the four AGENTS.md steps are named by "
        "promote.merge instead, and scripts/consolidate.py performs exactly "
        "those on approval (FOUNDER_DECISIONS.md IMP-021). What approving "
        f"this note will do: copy [[knowledge/{pair.email_node.node_id}]]'s "
        f"relations onto [[knowledge/{named_node_id}]], close every open "
        "relation on the duplicate with valid_until set to that run's date, "
        f"add superseded_by: knowledge/{named_node_id} to the duplicate's "
        f"frontmatter, and keep '{pair.email_node.title}' as an alias on the "
        "named node. The duplicate note stays on disk — its closed intervals "
        "are the history of what was believed when. Leave this note "
        "unapproved and nothing happens to either entity."
    )


def _merge_block(duplicate: str, survivor: str) -> list[str]:
    """The ``promote.merge`` lines, indented for the promote mapping. Node
    ids match ``relations._NODE_ID_RE`` ([a-z0-9._/-]), so none of them can
    need YAML quoting."""
    return [
        "  merge:",
        f"    duplicate: {duplicate}",
        f"    survivor: {survivor}",
    ]


def _frontmatter_close(lines: list[str]) -> int:
    """Index of the closing ``---`` of the frontmatter block, or -1."""
    if not lines or lines[0].strip() != "---":
        return -1
    return next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), -1)


def _with_merge_block(text: str, *, duplicate: str, survivor: str) -> str:
    """``text`` with ``promote.merge`` spliced in, or ``text`` unchanged.

    ``propose.propose_fact`` renders the four documented promote keys and
    nothing else, and it is shared machinery — the wikilink pass writes
    through it too — so the merge key is spliced in here, by the one pass
    that has a merge to name, rather than widening the shared contract.
    Unchanged when the key is already there (so re-running is a no-op, and
    a note left behind by a crash between the propose write and this one
    self-heals) and when the note is not the shape this pass writes: no
    frontmatter block, or no ``promote:`` key inside it.
    """
    frontmatter, _body = _split_frontmatter(text)
    promote = frontmatter.get("promote")
    if isinstance(promote, dict) and "merge" in promote:
        return text
    lines = text.split("\n")
    close = _frontmatter_close(lines)
    if close < 0:
        return text
    key = next(
        (i for i in range(1, close) if _PROMOTE_KEY_RE.match(lines[i])), -1
    )
    if key < 0:
        return text
    return "\n".join(
        lines[: key + 1] + _merge_block(duplicate, survivor) + lines[key + 1 :]
    )


def _with_canonical_source(text: str, *, source: str) -> str:
    """``text`` with ``promote.source`` rewritten to ``source``.

    An earlier run of this pass wrote the duplicate's bare NODE ID there.
    That id lands verbatim inside the ``[[…]]`` of the survivor's Log line
    on promotion, where it dangles (the vault root is the repo, not
    ``knowledge/``), so an existing proposal is corrected in place rather
    than left to promote a broken link. Only the first ``source:`` line
    inside the ``promote:`` block is touched; every other byte is kept.
    """
    lines = text.split("\n")
    close = _frontmatter_close(lines)
    if close < 0:
        return text
    key = next(
        (i for i in range(1, close) if _PROMOTE_KEY_RE.match(lines[i])), -1
    )
    if key < 0:
        return text
    line = f"  source: {source}"
    for i in range(key + 1, close):
        if lines[i] and not lines[i][0].isspace():
            break               # out of the promote block, back at top level
        if _SOURCE_KEY_RE.match(lines[i]):
            if lines[i] == line:
                return text
            return "\n".join(lines[:i] + [line] + lines[i + 1 :])
    return text


def _reconcile_note(
    path: Path, *, duplicate: str, survivor: str, source: str, dry_run: bool = False
) -> bool:
    """Bring one existing proposal note up to the current contract. True if
    it was written (or, under ``dry_run``, would have been).

    Frontmatter this pass does not own — ``approved:``, ``confirmations:``,
    ``created:`` — and the whole body are left alone: reconciling must never
    quietly un-approve a note a human has already said yes to.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    new_text = _with_canonical_source(
        _with_merge_block(text, duplicate=duplicate, survivor=survivor),
        source=source,
    )
    if new_text == text:
        return False
    if not dry_run:
        atomic_write_text(
            path, new_text, prefix=".propose-", suffix=".md", mode=umask_mode()
        )
    return True


def _names_duplicate(text: str, *, duplicate: str) -> bool:
    """Is this note a proposal about ``duplicate``?

    The filename only carries the SURVIVOR (it is the proposal's
    ``promote.target``), so a glob by target can pick up a proposal about a
    different email-slugged node merging into the same person. The
    duplicate is confirmed from the note's own content: its
    ``promote.merge`` when it has one, else the ``promote.source`` it was
    derived from, else the wikilink the body names it by.
    """
    frontmatter, body = _split_frontmatter(text)
    promote = frontmatter.get("promote")
    if isinstance(promote, dict):
        merge = promote.get("merge")
        if isinstance(merge, dict):
            named = merge.get("duplicate")
            return isinstance(named, str) and normalize_target(named) == duplicate
        source = promote.get("source")
        if isinstance(source, str) and normalize_target(source) == duplicate:
            return True
    return f"[[knowledge/{duplicate}]]" in body


def _existing_proposals(
    paths: VaultPaths, *, target: str, duplicate: str
) -> list[Path]:
    """Inbox notes this pass has already written for this exact pair.

    Globs the halves of the filename that are NOT the content digest —
    which is precisely why the digest must hash identity rather than
    wording (``propose._content_key``'s ``identity``): a note written by an
    earlier, differently worded version of this pass is still found here
    and reconciled, instead of being missed and duplicated.
    """
    inbox = paths.root / _INBOX_REL
    if not inbox.is_dir():
        return []
    kind_slug = slugify(_KIND)[:_SLUG_MAX_CHARS]
    target_slug = slugify(target)[:_SLUG_MAX_CHARS]
    return [
        found
        for found in sorted(inbox.glob(f"{kind_slug}--{target_slug}--*.md"))
        if _names_duplicate(
            found.read_text(encoding="utf-8", errors="replace"), duplicate=duplicate
        )
    ]


def propose_duplicate_merges(
    paths: VaultPaths, *, dry_run: bool = False
) -> list[ProposeResult]:
    """Run the pass and propose one merge candidate per email-slugged node
    :func:`ingest_lib.sweep.find_duplicate_entity_pairs` still reports with
    exactly one named candidate. A pair with more than one named candidate
    (the same local part matching two distinct named nodes) is ambiguous —
    there is no single survivor to name as ``promote.target`` — and is left
    for a human to disambiguate by hand rather than guessed at; the sweep
    report still flags it.

    A pair the inbox ALREADY holds a proposal for is never proposed a second
    time. This pass runs nightly against a queue a human reads at their own
    pace, so a second note for a pair already waiting is not a harmless
    duplicate: it is two notes to approve, and approving the older one
    performs whatever the older one says. The earlier note is therefore
    RECONCILED in place instead — ``promote.merge`` spliced in if it predates
    that key, ``promote.source`` canonicalised — and the propose skipped
    (:func:`_existing_proposals`, :func:`_reconcile_note`). Everything a
    human owns on that note (``approved:``, ``confirmations:``, the body) is
    left exactly as it is.

    Returns one :class:`ProposeResult` per candidate: ``"proposed"`` for a
    new note (or, under ``dry_run``, what would be written), ``"reconciled"``
    for an existing note this run brought up to the current contract,
    ``"exists"`` for one already correct, and ``"archived"`` for a candidate
    the human has already dealt with. Every proposal that is (or becomes) an
    inbox note carries ``promote.merge``."""
    results: list[ProposeResult] = []
    for pair in find_duplicate_entity_pairs(paths):
        if len(pair.named_candidates) != 1:
            continue
        named = pair.named_candidates[0]
        duplicate = pair.email_node.node_id
        # promote.source is a VAULT-relative path, not a node id: it lands
        # verbatim inside the [[…]] of the target's Log line, and a bare node
        # id there is a dangling wikilink (the vault root is the repo, not
        # knowledge/). Same form wikilinks.py passes.
        source = note_path_for_node(duplicate)[: -len(".md")]

        existing = _existing_proposals(paths, target=named.node_id, duplicate=duplicate)
        if existing:
            for found in existing:
                rel = found.relative_to(paths.root).as_posix()
                written = _reconcile_note(
                    found,
                    duplicate=duplicate,
                    survivor=named.node_id,
                    source=source,
                    dry_run=dry_run,
                )
                action: ProposeAction = "reconciled" if written else "exists"
                _LOG.info(
                    "duplicates: %s %s -> %s (%s)",
                    action, duplicate, named.node_id, rel,
                )
                results.append(ProposeResult(path=rel, action=action))
            continue

        result = propose_fact(
            paths,
            kind=_KIND,
            title=(
                f"duplicate-entity merge candidate: {duplicate} "
                f"-> {named.node_id}"
            ),
            target=named.node_id,
            source=source,
            reason=_reason(pair, named.node_id),
            fact=_fact(pair, named.node_id),
            # The merge pair IS the proposal's identity: re-wording _fact or
            # _reason must not re-hash a live proposal into a second file
            # beside the one already in the inbox.
            identity=(duplicate, named.node_id),
            dry_run=dry_run,
        )
        if not dry_run and result.action == "proposed":
            _reconcile_note(
                paths.root / result.path,
                duplicate=duplicate,
                survivor=named.node_id,
                source=source,
            )
        _LOG.info(
            "duplicates: %s %s -> %s (%s)",
            result.action, duplicate, named.node_id, result.path,
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
            "performs the merge itself: each proposal carries promote.merge, "
            "and scripts/consolidate.py performs the AGENTS.md merge steps "
            "only once a human sets approved: true on the note."
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
    reconciled = sum(1 for r in results if r.action == "reconciled")
    exists = sum(1 for r in results if r.action == "exists")
    archived = sum(1 for r in results if r.action == "archived")

    print(f"duplicate-entity merge candidates ({'dry-run' if args.dry_run else 'real'}):")
    print(f"  candidates found : {len(results)}")
    print(f"  proposed         : {proposed}")
    print(f"  reconciled       : {reconciled}")
    print(f"  already in inbox : {exists}")
    print(f"  already archived : {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["propose_duplicate_merges", "main"]
