"""Deterministic fact-note proposer for ``knowledge/assistant/inbox/``.

Nothing in the codebase writes fact notes today — only agents do, over MCP,
by calling a generic ``vault_create_note`` against the inbox. This module is
the first non-agent writer, and it is built as shared machinery on purpose:
any deterministic pass that wants to PROPOSE a typed relation or a Log fact
(the wikilink pass below, and the duplicate-entity merge proposer described
alongside it in TODO.md) drops a note through here rather than hand-rolling
the frontmatter contract again.

The vault's defining posture — nothing auto-writes into the entity graph;
an LLM or a deterministic pass may only PROPOSE — is enforced by construction:
this module writes exactly the shape
``knowledge/index/templates/memory-fact.md`` documents (``approved: false``,
``memory_status: unconsolidated``), and only ``scripts/consolidate.py``
(a human's ``approved: true``, or enough ``confirmations``) ever promotes
one into an entity note.

Determinism is the property that makes a *scheduled* proposer usable at
all: the inbox filename is derived from the proposal's own content (kind +
target + relations + fact + source), so running the same pass twice over an
unchanged vault proposes nothing new — it just finds the same file already
there and no-ops. Two structurally different proposals about the same
target never collide because the content hash differs; the same proposal
computed twice always lands on the same file. That holds over the whole
deployment lifecycle, not just inside one 30-day window: a proposal
``consolidate.py`` has since moved into ``knowledge/assistant/archive/``
counts as already proposed too, so a candidate a human declined to
approve is never re-proposed (see :func:`propose_fact` for the resulting
rejection path).

Provenance — ``author`` / ``written_via`` — is server-asserted for MCP
writes (``mcp_server/provenance.py``) and simply absent from hand-edited
notes (AGENTS.md). A note written by THIS module is neither: it is real
provenance (something did write it, deterministically, from real evidence),
but it did not pass through the MCP server, so ``written_via: mcp`` would be
the exact provenance lie AGENTS.md warns against for hand-faked keys. It
also is not attributable to any human or agent, so ``author: 'agent:...'``
would misattribute a belief to someone who never asserted it. This module
therefore stamps ``written_via: script`` and ``author: 'script:<kind>'`` —
honest about being machine-written, and about which deterministic pass
wrote it, without borrowing either the MCP or the agent identity.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypedDict

from .atomic import atomic_write_text, umask_mode
from .concepts import slugify
from .config import VaultPaths
from .notes import _yaml_dump_str  # private, but module-internal (consolidate.py/sweep.py do the same)
from .relations import Relation, _relation_entry  # private, but module-internal (see relations.py)

_INBOX_REL = "knowledge/assistant/inbox"
# Where ``scripts/consolidate.py`` MOVES a fact note once it has been dealt
# with — promoted after approval, or digested after ``stale_days``. Kept in
# step with ``consolidate._ARCHIVE_REL``.
_ARCHIVE_REL = "knowledge/assistant/archive"
# ``consolidate._archive_destination`` appends ``-2``, ``-3``, … on a name
# collision in a month directory; the archived twin of ``<stem>.md`` is
# therefore ``<stem>.md`` or ``<stem>-<n>.md`` and nothing else.
_ARCHIVE_SUFFIX_RE = re.compile(r"^(-\d+)?$")
# Hex chars of the content hash kept in the filename: enough to make an
# accidental collision between two DIFFERENT proposals negligible for a
# personal vault, short enough that the filename stays readable.
_HASH_LEN = 16
# Kept short: two-argument slugify(kind) + slugify(target) + a 16-hex digest
# is already close to typical filesystem limits once knowledge/assistant/
# inbox/ is prefixed; this caps each half independently.
_SLUG_MAX_CHARS = 60


# What one propose_fact call did. Closed set, so a caller branching on it
# cannot compare against a typo'd string.
ProposeAction = Literal["proposed", "exists", "archived"]


@dataclass(frozen=True)
class ProposeResult:
    """Outcome of one :func:`propose_fact` call."""

    path: str  # vault-relative path (incl. .md) of the proposal note
    # "proposed": this call wrote (or, under dry_run, WOULD write) a new
    # note, at ``path``. "exists": a proposal with this exact content is
    # sitting in the inbox at ``path``. "archived": it was put to the human
    # once and consolidate has since MOVED it out of the inbox — ``path``
    # is where it actually is now, under knowledge/assistant/archive/, not
    # the inbox path it no longer occupies. The last two are the defining
    # idempotency property, not a failure; they are kept apart because a
    # caller that prints or logs ``path`` must not claim a file is in the
    # inbox when it is not.
    action: ProposeAction


def _content_key(
    *, kind: str, target: str, relations: Sequence[Relation], fact: str, source: str
) -> str:
    """Canonical string identifying one proposal's content.

    Two calls that would produce the SAME promote mapping must hash to the
    same key regardless of relation list order upstream never varying it —
    callers pass relations in a stable order, and this does not re-sort,
    so a caller that wants order-independence sorts before calling. \x1f
    (unit separator) can't appear in any field's normal content, so the
    join can't collide two different tuples onto one key.
    """
    rel_parts = "\x1e".join(
        f"{r.rel}\x1f{r.target}\x1f{r.valid_from}\x1f{r.valid_until}\x1f{r.source}"
        for r in relations
    )
    return "\x1f".join([kind, target, rel_parts, fact, source])


def _proposal_filename(*, kind: str, target: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_HASH_LEN]
    kind_slug = slugify(kind)[:_SLUG_MAX_CHARS] or "proposal"
    target_slug = slugify(target)[:_SLUG_MAX_CHARS] or "target"
    return f"{kind_slug}--{target_slug}--{digest}.md"


class _PromoteBlock(TypedDict):
    """The ``promote:`` mapping ``consolidate.py`` reads back off the note."""

    target: str
    relations: list[dict[str, str]]
    fact: str
    source: str


class _ProposalFrontmatter(TypedDict):
    """Exactly the frontmatter ``knowledge/index/templates/memory-fact.md``
    documents, spelled out so a missing or misspelt key is a type error
    here rather than a note ``consolidate.py`` silently skips."""

    title: str
    type: str
    created: str
    author: str
    written_via: str
    memory_status: str
    confirmations: int
    approved: bool
    promote: _PromoteBlock


def _archived_path(paths: VaultPaths, filename: str) -> str | None:
    """Where this exact proposal sits in the archive, or None.

    A proposal has already been PUT to the human and dealt with when
    ``consolidate.py`` has MOVED it out of the inbox into
    ``knowledge/assistant/archive/<YYYY-MM>/``. Testing the inbox alone is
    not enough: consolidate moves a fact note out both when it is promoted
    after approval and when it goes unanswered past ``stale_days`` and is
    digested. Without this half, a scheduled proposer rewrites every
    digested candidate on its next run and the same finding is re-proposed
    on a monthly cycle, forever.

    The vault-relative path is returned rather than a bare bool so the
    caller can report where the note actually is; matches are sorted so
    the answer does not depend on directory order when the same proposal
    was archived in two different months.
    """
    archive = paths.root / _ARCHIVE_REL
    if not archive.is_dir():
        return None
    stem = filename.removesuffix(".md")
    # slugify() output plus a hex digest: no glob metacharacters, so the
    # stem is safe to interpolate into the pattern.
    for found in sorted(archive.glob(f"*/{stem}*.md")):
        if _ARCHIVE_SUFFIX_RE.match(found.name.removesuffix(".md")[len(stem):]):
            return found.relative_to(paths.root).as_posix()
    return None


def _render_note(
    *,
    title: str,
    created_iso: str,
    author: str,
    target: str,
    relations: Sequence[Relation],
    fact: str,
    source: str,
    body: str,
) -> str:
    """Build the note text field for field against
    ``knowledge/index/templates/memory-fact.md``. This is a FRESH note (no
    existing frontmatter to preserve byte-for-byte, unlike
    ``relations.upsert_relation_in_text``), so a single ``yaml.safe_dump``
    over the whole mapping is both simpler and safer than hand-assembling
    lines: it quotes a title containing a colon, an empty ``fact: ''``, and
    a ``created:`` timestamp that would otherwise round-trip as a datetime
    instead of a string — exactly the escaping ``relations._dump_relations``
    already trusts it for.
    """
    frontmatter: _ProposalFrontmatter = {
        "title": title,
        "type": "memory_fact",
        "created": created_iso,
        "author": author,
        "written_via": "script",
        "memory_status": "unconsolidated",
        "confirmations": 0,
        "approved": False,
        "promote": {
            "target": target,
            "relations": [_relation_entry(r) for r in relations],
            "fact": fact,
            "source": source,
        },
    }
    yaml_block = _yaml_dump_str(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    return f"---\n{yaml_block}---\n\n{body.rstrip(chr(10))}\n"


def propose_fact(
    paths: VaultPaths,
    *,
    kind: str,
    title: str,
    target: str,
    source: str,
    reason: str,
    relations: Sequence[Relation] = (),
    fact: str = "",
    now: datetime | None = None,
    dry_run: bool = False,
) -> ProposeResult:
    """Propose one fact/relation candidate into ``knowledge/assistant/inbox/``.

    ``kind`` names the deterministic pass making the proposal (e.g.
    ``"wikilink-relation"``) — it seeds the filename and the ``author``
    stamp, so two different passes proposing about the same target never
    collide on one file. ``target`` is the entity note that would receive
    ``relations``/``fact`` on promotion (``promote.target``); ``source`` is
    the vault-relative, extensionless path of the note the proposal was
    derived from (``promote.source`` and, for each relation lacking its
    own, effectively the same provenance). ``reason`` is free-form prose —
    why this pass believes the candidate, with its evidence — written
    verbatim into the note body, exactly as
    ``knowledge/index/templates/memory-fact.md`` asks for.

    Never overwrites: the destination filename is pure content (kind +
    target + relations + fact + source), so a note already there for the
    same content means this exact candidate was already proposed —
    returned as ``action="exists"`` without touching the file. This is
    what makes a scheduled pass idempotent: re-running it over an
    unchanged vault proposes nothing new.

    "Already proposed" is a vault-wide fact, not an inbox-directory one:
    ``knowledge/assistant/archive/`` is checked too (see
    :func:`_archived_path`), because ``consolidate.py`` moves a fact
    note there once it has been promoted OR digested. That case answers
    ``action="archived"`` with ``path`` naming the archived note, not the
    inbox filename it has vacated. That is what gives a human an honest
    way to say NO to a candidate — the inbox is a queue, not a to-do list:

    - leave an unwanted proposal alone. After ``stale_days`` (30)
      ``consolidate.py`` records it in that month's digest and moves it
      into the archive; from then on this function answers ``"archived"``
      and the candidate is never proposed again.
    - to reject it at once, move the file by hand out of the inbox into
      ``knowledge/assistant/archive/<YYYY-MM>/`` (move, never delete —
      deleting it only makes the next run write it back).

    ``now`` is injectable so callers (and tests) can pin the ``created:``
    timestamp instead of depending on wall-clock time; it only affects a
    freshly-written note; an existing one keeps its own ``created:``.
    ``dry_run`` computes the destination and the content but never writes,
    still distinguishing ``"exists"`` from what would be ``"proposed"``.
    """
    key = _content_key(kind=kind, target=target, relations=relations, fact=fact, source=source)
    filename = _proposal_filename(kind=kind, target=target, key=key)
    rel_path = f"{_INBOX_REL}/{filename}"
    dest = paths.root / rel_path

    if dest.exists():
        return ProposeResult(path=rel_path, action="exists")
    archived = _archived_path(paths, filename)
    if archived is not None:
        return ProposeResult(path=archived, action="archived")

    if dry_run:
        return ProposeResult(path=rel_path, action="proposed")

    when = now or datetime.now(UTC)
    created_iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    text = _render_note(
        title=title,
        created_iso=created_iso,
        author=f"script:{kind}",
        target=target,
        relations=relations,
        fact=fact,
        source=source,
        body=reason,
    )
    atomic_write_text(dest, text, prefix=".propose-", suffix=".md", mode=umask_mode())
    return ProposeResult(path=rel_path, action="proposed")


__all__ = ["ProposeAction", "ProposeResult", "propose_fact"]
