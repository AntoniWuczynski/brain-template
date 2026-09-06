"""Deterministic consolidation of assistant memory facts.

The lifecycle (the "dream pass", minus the mysticism):

1. The assistant drops structured fact notes into
   ``knowledge/assistant/inbox/`` over MCP. Each note carries
   ``memory_status: unconsolidated``, ``confirmations: N``,
   ``approved: bool`` and a ``promote:`` mapping
   (``{target, relations: [...], fact, source}``, plus an optional
   ``merge: {duplicate, survivor}``) — the contract is
   ``knowledge/index/templates/memory-fact.md``.
2. THIS pass — counters and thresholds, NO LLM — promotes confirmed
   facts into their target entity notes (relations merged into
   frontmatter, the fact line appended to ``## Log``), performs an
   approved ``promote.merge`` as AGENTS.md's "Merging a duplicate
   entity" spells it out, stamps the fact note ``memory_status:
   consolidated``, and MOVES it to
   ``knowledge/assistant/archive/<YYYY-MM>/``. Moved, never deleted:
   the original wording stays reviewable forever.
3. Facts that linger unconsolidated past ``stale_days`` are swept into
   a monthly digest under ``knowledge/assistant/digests/`` (one bullet
   each, linking the archived note) instead of accumulating forever.
4. Everything else waits in the inbox for more confirmations or a
   human's ``approved: true``.

WHO MAY APPROVE — ``approved:`` and ``confirmations:`` in the fact note's
own frontmatter are taken at face value. Note the standing limitation the
2026-09-04 audit named: an agent that writes a fact note also writes those
keys, so this gate is advisory, not enforced. Reverting the out-of-reach
approval ledger was a deliberate owner decision (FOUNDER_DECISIONS.md
IMP-016) on the grounds that the path has never fired in practice.

Every decision here derives from frontmatter + ``as_of`` + the
thresholds, so two runs over identical inputs produce
byte-identical results. The only timestamps written are frontmatter
values derived from ``as_of`` (AGENTS.md rule 7).

Run this pass when the MCP server is idle. It takes NO cross-process
lock: it rewrites entity notes and unlinks inbox copies directly on
disk, so a server writing the same entity note (or mid-edit of an inbox
note this pass unlinks) at the same instant can lose or duplicate a
write. For a single-user vault that just means "don't consolidate while
actively talking to the assistant" — not a locking problem worth
solving in code.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .concepts import slugify
from .config import VaultPaths
from .notes import (  # _-prefixed helpers are private but module-internal
    _atomic_write,
    _split_frontmatter,
    fm_list,
)
from .relations import (
    Relation,
    append_fact_to_log,
    canonical_entity_id,
    folder_overview_id,
    is_valid_node_id,
    normalize_target,
    note_path_for_node,
    parse_relations,
    upsert_relation_in_text,
)

_LOG = logging.getLogger(__name__)

_INBOX_REL = "knowledge/assistant/inbox"
_ARCHIVE_REL = "knowledge/assistant/archive"
_DIGESTS_REL = "knowledge/assistant/digests"
_SNIPPET_MAX_CHARS = 200


@dataclass(frozen=True)
class ConsolidateStats:
    """Outcome of one consolidation pass. ``moved`` pairs are
    ``(old, new)`` vault-relative paths including ``.md`` — exactly what
    ``semantic.upsert_notes`` wants (sources drop their index rows, the
    archived destinations gain fresh ones)."""

    promoted: int
    digested: int
    unresolved: int
    skipped: int
    touched_entity_paths: tuple[str, ...]
    moved: tuple[tuple[str, str], ...]
    problems: tuple[str, ...]
    # Vault-relative ``.md`` path of the digest note written this run, or
    # "" when none was. The CLI reindexes it so a fresh monthly digest is
    # searchable immediately instead of after the next full rebuild.
    digest_path: str = ""


# ---------------------------------------------------------------------------
# tolerant frontmatter-value coercion (YAML hands us bools, ints, dates,
# strings or garbage — never raise over a malformed inbox note)
# ---------------------------------------------------------------------------

def _coerce_str(raw: object) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _coerce_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    return _coerce_str(raw).lower() in {"true", "yes", "1", "on"}


def _coerce_int(raw: object) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    try:
        return int(_coerce_str(raw))
    except ValueError:
        return 0




def _parse_date(raw: object) -> date | None:
    """``created:`` -> date, tolerating YAML date/datetime objects, plain
    ``YYYY-MM-DD`` strings and full ISO timestamps. None when hopeless —
    callers decide whether that blocks the note or falls back."""
    if isinstance(raw, datetime):   # datetime is a date subclass: check first
        return raw.date()
    if isinstance(raw, date):
        return raw
    s = _coerce_str(raw)
    if len(s) < 10:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# frontmatter line surgery
#
# Same approach as mcp_server/provenance.py (which stamps author /
# written_via on writes): edit single top-level ``key: value`` lines
# inside the fence, leaving every other byte of the note alone.
# Reimplemented here rather than imported — vault scripts must not
# depend on the server package. Keep the two in sync by hand.
# ---------------------------------------------------------------------------

def _frontmatter_close(lines: list[str]) -> int:
    """Index of the closing ``---`` fence line, or -1 when the text has
    no parseable fence at all."""
    if not lines or lines[0].strip() != "---":
        return -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return i
    return -1


def _same_promotion(a: str, b: str) -> bool:
    """True iff two archived-note texts are the same promotion, ignoring the
    volatile ``consolidated:`` date stamp. Used so a crash-rerun on a later
    day still recognises its own already-archived copy."""
    norm_a = _set_frontmatter_key(a, "consolidated", "''")
    norm_b = _set_frontmatter_key(b, "consolidated", "''")
    return norm_a == norm_b


def _set_frontmatter_key(text: str, key: str, value: str) -> str:
    """Set one TOP-LEVEL scalar frontmatter key by textual surgery.

    Column-0 match only, so nested keys (``promote.target``) are never
    touched. Existing line replaced in place; missing key inserted just
    before the closing fence; a note without a fence gets a minimal one
    prepended (tolerant-parsing spirit: never refuse to stamp)."""
    line = f"{key}: {value}"
    lines = text.split("\n")
    close = _frontmatter_close(lines)
    if close < 0:
        return f"---\n{line}\n---\n{text}"
    for i in range(1, close):
        if lines[i].startswith(f"{key}:"):
            lines[i] = line
            return "\n".join(lines)
    lines.insert(close, line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------

def _normalize_source_link(raw: str) -> str:
    """``promote.source`` -> vault-relative no-extension wikilink target.

    Unlike ``relations.normalize_target`` this KEEPS any ``knowledge/``
    prefix: Log lines link full vault paths; node ids are the
    ``knowledge/``-relative shorthand."""
    t = raw.strip()
    if t.startswith("[[") and t.endswith("]]"):
        t = t[2:-2].strip()
    t = t.split("|", 1)[0].strip()
    if t.endswith(".md"):
        t = t[: -len(".md")]
    return t.lstrip("/")


def _first_paragraph(body: str, max_chars: int = _SNIPPET_MAX_CHARS) -> str:
    """First non-heading paragraph, whitespace-collapsed. Local parallel
    of ``knowledge._first_paragraph`` with the digest's tighter 200-char
    cap — kept separate so digest bullets can't widen if the knowledge
    snippet contract changes."""
    for block in body.split("\n\n"):
        lines = [
            ln for ln in block.strip().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if not lines:
            continue
        collapsed = " ".join(" ".join(lines).split())
        if len(collapsed) > max_chars:
            collapsed = collapsed[: max_chars - 1] + "…"
        return collapsed
    return ""


def _archive_destination(archive_dir: Path, filename: str, reserved: set[Path]) -> Path:
    """Collision-free destination in the month's archive dir. Existing
    files AND destinations already claimed this run (dry or real) get a
    ``-2``, ``-3``, … suffix — never overwrite, never merge."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = archive_dir / filename
    n = 2
    while candidate.exists() or candidate in reserved:
        candidate = archive_dir / f"{stem}-{n}{suffix}"
        n += 1
    reserved.add(candidate)
    return candidate


def _digest_skeleton(month: str, as_of: date) -> str:
    iso = as_of.isoformat()
    return (
        "---\n"
        f"title: Memory digest {month}\n"
        "type: digest\n"
        f"created: '{iso}'\n"
        f"updated: '{iso}'\n"
        "---\n"
        "\n"
        f"# Memory digest {month}\n"
    )


# ---------------------------------------------------------------------------
# duplicate-entity merge (AGENTS.md, "Merging a duplicate entity")
#
# A merge is not a typed relation, so it rides its own ``promote.merge:
# {duplicate, survivor}`` key rather than ``promote.relations``. On an
# approved fact this pass performs exactly the four documented steps and
# nothing else: copy the duplicate's relations onto the survivor, close
# every open relation on the duplicate with ``valid_until``, stamp
# ``superseded_by`` on the duplicate, keep its title as an alias on the
# survivor. Supersede, never delete — the duplicate note stays on disk.
#
# ATOMICITY. Two notes change, and no filesystem gives us one rename for
# both. Everything is COMPUTED first, so any refusal (bad ids, missing or
# unparseable notes) happens before a single byte is written. Of the two
# orderings FOUNDER_DECISIONS.md IMP-021 allowed, this takes the second:
# write the survivor, then the duplicate, and if the duplicate write fails
# leave the fact in the inbox with a problem line naming the half-applied
# state. Rolling the survivor back is not actually available — a crash
# between two renames rolls nothing back either — whereas both halves are
# idempotent (an already-copied relation upserts to a noop, an
# already-present alias is not re-added), so the next run finishes the
# merge instead of doubling it. The survivor goes first because it is the
# half that PRESERVES information: superseding the duplicate before its
# relations exist anywhere else would, for the window between the writes,
# leave the graph believing a live edge had ended.
# ---------------------------------------------------------------------------

_MERGE_KEYS: tuple[str, str] = ("duplicate", "survivor")


@dataclass(frozen=True)
class _MergePlan:
    """Both halves of one merge, fully computed before anything is written.

    ``duplicate_text`` is None when the duplicate already carries
    ``superseded_by`` — the merge has already happened, so this run writes
    nothing and simply lets the fact archive (idempotent re-run)."""

    duplicate_rel: str            # vault-relative path incl. .md
    duplicate_text: str | None
    survivor_text: str
    problems: tuple[str, ...]


@dataclass(frozen=True)
class _MergeRefusal:
    """Why this merge will NOT be performed. The fact stays in the inbox."""

    problem: str


def _existing_aliases(frontmatter: dict[str, object]) -> list[str]:
    """``aliases:`` as a list of non-empty strings. A bare string counts as
    a one-element list; anything else as none — the same tolerance
    ``relations._coerce_aliases`` applies."""
    raw = frontmatter.get("aliases")
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [_coerce_str(a) for a in raw if _coerce_str(a)]
    return []


def _frontmatter_key_re(key: str) -> re.Pattern[str]:
    """Quote-tolerant match for one top-level frontmatter key.

    PyYAML round-trips ``"aliases": [Dan]`` as an ordinary ``aliases`` key,
    so a bare ``startswith(f"{key}:")`` test misses it and appends a SECOND
    key of the same name — the note then carries two, and which one wins is
    PyYAML's business, not the caller's. Same tolerance, and the same
    reason, as ``relations._RELATIONS_KEY_RE``.
    """
    return re.compile(rf"""^["']?{re.escape(key)}["']?\s*:""")


def _set_frontmatter_seq(text: str, key: str, values: list[str]) -> str:
    """Set one TOP-LEVEL sequence-valued frontmatter key to ``values``.

    ``_set_frontmatter_key`` replaces a single line, which would orphan the
    ``- item`` lines of a block-style list under the key. This replaces the
    whole block — the key line plus every following indented or ``-`` line,
    exactly the extent ``relations._splice_relations`` uses — with one
    inline line (``aliases: [a, b]``, the form the templates and the live
    vault already use). Every other frontmatter line and the whole body
    keep their bytes."""
    line = f"{key}: {fm_list(values)}"
    lines = text.split("\n")
    close = _frontmatter_close(lines)
    if close < 0:
        return f"---\n{line}\n---\n{text}"
    start = next(
        (i for i in range(1, close) if _frontmatter_key_re(key).match(lines[i])), -1
    )
    if start < 0:
        lines.insert(close, line)
        return "\n".join(lines)
    end = start + 1
    for j in range(start + 1, close):
        stripped = lines[j].strip()
        if not stripped:
            continue
        if not (lines[j][0].isspace() or stripped.startswith("-")):
            break
        end = j + 1
    return "\n".join(lines[:start] + [line] + lines[end:])


# A value that is nothing but an email address — the same test
# ``sweep._EMAIL_TITLE_RE`` applies when it detects a duplicate entity.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def _email_slug_forms(address: str) -> set[str]:
    """The slugs an email address becomes as the last segment of a node id:
    ``concepts.slugify`` (punctuation to ``-``) and the punctuation-stripped
    form the MCP write path produced for the nodes in the live vault
    (``dana@example.com`` -> ``danaexamplecom``)."""
    lowered = address.casefold()
    return {slugify(lowered), re.sub(r"[^a-z0-9]", "", lowered)}


def _email_node_address(frontmatter: dict[str, object], node_id: str) -> str:
    """The bare email address this node is NAMED FOR, or ``""``.

    Two shapes count, and both come off the note itself rather than a guess
    about what a slug might once have been: a ``title:`` that is nothing but
    an address, and a node id whose last segment is the slug of an address
    the note keeps as an alias.
    """
    title = " ".join(_coerce_str(frontmatter.get("title")).split())
    if _EMAIL_RE.match(title):
        return title
    slug = node_id.rsplit("/", 1)[-1]
    for alias in _existing_aliases(frontmatter):
        if _EMAIL_RE.match(alias) and slug in _email_slug_forms(alias):
            return alias
    return ""


def _plan_merge(
    raw: object,
    *,
    root: Path,
    target: str,
    survivor_text: str,
    as_of: date,
) -> _MergePlan | _MergeRefusal:
    """Validate ``promote.merge`` and compute both notes' new text.

    Refuses (nothing is written, the fact stays in the inbox) when the
    mapping is malformed, the two ids name one node, the survivor is not
    ``promote.target``, either id is not a graph node by
    ``relations.canonical_entity_id`` — never a path-prefix test — the
    duplicate note is missing or unreadable, or the survivor is itself
    already superseded (merging into a dead node would strand the
    relations one hop further from the live entity).
    """
    if not isinstance(raw, dict):
        return _MergeRefusal(
            f"promote.merge is a {type(raw).__name__}, not a mapping"
        )
    unknown = sorted(str(k) for k in raw if str(k) not in _MERGE_KEYS)
    if unknown:
        return _MergeRefusal(
            f"promote.merge has unknown key(s) {', '.join(unknown)} "
            f"(allowed: {', '.join(_MERGE_KEYS)})"
        )
    duplicate = normalize_target(_coerce_str(raw.get("duplicate")), vault_root=root)
    survivor = normalize_target(_coerce_str(raw.get("survivor")), vault_root=root)
    if not duplicate or not survivor:
        return _MergeRefusal(
            "promote.merge needs both a duplicate: and a survivor: node id"
        )
    if duplicate == survivor:
        return _MergeRefusal(
            f"promote.merge names {survivor} as both duplicate and survivor"
        )
    if survivor != target:
        return _MergeRefusal(
            f"promote.merge survivor {survivor!r} is not promote.target "
            f"{target!r} — the survivor is the note the fact promotes into"
        )
    for role, node in (("duplicate", duplicate), ("survivor", survivor)):
        if canonical_entity_id(note_path_for_node(node), vault_root=root) != node:
            return _MergeRefusal(
                f"promote.merge {role} {node!r} is not an entity node"
            )

    duplicate_rel = note_path_for_node(duplicate)
    duplicate_path = root / duplicate_rel
    if not duplicate_path.is_file():
        return _MergeRefusal(
            f"promote.merge duplicate {duplicate!r} resolves to no note "
            f"({duplicate_rel})"
        )
    try:
        duplicate_text = duplicate_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _MergeRefusal(
            f"promote.merge duplicate note {duplicate_rel} unreadable ({exc})"
        )

    survivor_fm, _survivor_body = _split_frontmatter(survivor_text)
    survivor_superseded = _coerce_str(survivor_fm.get("superseded_by"))
    if survivor_superseded:
        return _MergeRefusal(
            f"promote.merge survivor {survivor} is itself superseded "
            f"(superseded_by: {survivor_superseded}) — merge into the "
            "survivor of that merge instead"
        )

    address = _email_node_address(survivor_fm, survivor)
    if address:
        # The one shape a duplicate-entity merge must never take. An
        # email-slugged node exists because a calendar payload carried an
        # address instead of a display name; naming it the SURVIVOR buries
        # the named entity under the address, closes the named node's live
        # relations, and leaves the graph pointing at the accident.
        return _MergeRefusal(
            f"promote.merge survivor {survivor} is the email-slugged node "
            f"for {address!r} — the merge is the other way round: write "
            f"duplicate: {survivor}, survivor: {duplicate}"
        )

    duplicate_fm, _duplicate_body = _split_frontmatter(duplicate_text)
    already = _coerce_str(duplicate_fm.get("superseded_by"))
    if already:
        if normalize_target(already, vault_root=root) != survivor:
            # Merged into a DIFFERENT node already. Not an idempotent
            # re-run: nothing here can be applied (the duplicate's
            # relations now live on that other survivor, and this note's
            # promote.fact is phrased as a completed merge), so performing
            # the rest would write a false record into this survivor's Log
            # while the duplicate still points elsewhere. Refuse, and name
            # the node the merge would have to be re-aimed at.
            return _MergeRefusal(
                f"promote.merge duplicate {duplicate} is already superseded by "
                f"{already}, not {survivor} — merge into that survivor instead"
            )
        # Already merged into THIS survivor: a genuine idempotent no-op.
        return _MergePlan(
            duplicate_rel=duplicate_rel,
            duplicate_text=None,
            survivor_text=survivor_text,
            problems=(),
        )

    relations, relation_problems = parse_relations(duplicate_fm)
    problems: list[str] = [
        f"promote.merge duplicate {duplicate}: {p}" for p in relation_problems
    ]

    new_survivor = survivor_text
    try:
        for relation in relations:
            new_survivor, _action = upsert_relation_in_text(new_survivor, relation)
    except ValueError as exc:
        return _MergeRefusal(
            f"promote.merge survivor note has unparseable frontmatter ({exc})"
        )

    new_duplicate = duplicate_text
    closing = as_of.isoformat()
    try:
        for relation in relations:
            if relation.valid_until:
                continue        # already closed: its span is history already
            new_duplicate, _action = upsert_relation_in_text(
                new_duplicate,
                Relation(
                    rel=relation.rel,
                    target=relation.target,
                    valid_from=relation.valid_from,
                    valid_until=closing,
                    source=relation.source,
                ),
            )
    except ValueError as exc:
        return _MergeRefusal(
            f"promote.merge duplicate note {duplicate_rel} has unparseable "
            f"frontmatter ({exc})"
        )

    alias = " ".join(_coerce_str(duplicate_fm.get("title")).split())
    if alias:
        aliases = _existing_aliases(survivor_fm)
        if alias.casefold() not in {a.casefold() for a in aliases}:
            new_survivor = _set_frontmatter_seq(
                new_survivor, "aliases", [*aliases, alias]
            )
    else:
        problems.append(
            f"promote.merge duplicate {duplicate} has no title — no alias "
            f"kept on {survivor}"
        )
    new_duplicate = _set_frontmatter_key(
        new_duplicate, "superseded_by", f"knowledge/{survivor}"
    )
    return _MergePlan(
        duplicate_rel=duplicate_rel,
        duplicate_text=new_duplicate,
        survivor_text=new_survivor,
        problems=tuple(problems),
    )


# ---------------------------------------------------------------------------
# the pass itself
# ---------------------------------------------------------------------------

def consolidate(
    paths: VaultPaths,
    *,
    logger: logging.Logger,
    as_of: date,
    stale_days: int = 30,
    min_confirmations: int = 3,
    dry_run: bool = False,
) -> ConsolidateStats:
    """Run one consolidation pass over ``knowledge/assistant/inbox/``.

    Per note (sorted, top-level ``*.md`` only — the inbox is flat):

    - PROMOTE when ``approved:`` is truthy or ``confirmations >=
      min_confirmations``, and the target entity note exists. Missing
      target -> the note STAYS in the inbox and counts ``unresolved``
      (a human or the agent resolves it).
    - DIGEST when still ``unconsolidated``, unapproved, and ``created:``
      is more than ``stale_days`` old relative to ``as_of``.
    - Otherwise ``skipped`` — still fresh, still unapproved; it waits.

    ``dry_run`` computes the full plan but mutates nothing; the stats
    reflect what WOULD happen.
    """
    inbox = paths.root / _INBOX_REL
    month = as_of.strftime("%Y-%m")
    archive_dir = paths.root / _ARCHIVE_REL / month
    digest_path = paths.root / _DIGESTS_REL / f"{month}.md"

    promoted = digested = unresolved = skipped = 0
    touched: list[str] = []
    moved: list[tuple[str, str]] = []
    problems: list[str] = []
    reserved: set[Path] = set()
    digest_bullets: list[str] = []
    # F15: stale-note moves are DEFERRED until after the digest is written,
    # so the durable record always precedes the move. (md, dest) pairs.
    deferred_moves: list[tuple[Path, Path]] = []

    notes = sorted(inbox.glob("*.md")) if inbox.is_dir() else []
    for md in notes:
        rel = md.relative_to(paths.root).as_posix()
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped += 1
            problems.append(f"{rel}: unreadable ({exc}) — skipped")
            continue

        fm, body = _split_frontmatter(text)
        promote_raw = fm.get("promote")
        status = _coerce_str(fm.get("memory_status"))
        if not isinstance(promote_raw, dict) and not status:
            skipped += 1
            problems.append(
                f"{rel}: not a memory fact note (no promote mapping, no memory_status) — skipped"
            )
            continue

        created = _parse_date(fm.get("created"))
        approved = _coerce_bool(fm.get("approved"))
        confirmations = _coerce_int(fm.get("confirmations"))

        # -- PROMOTE -------------------------------------------------------
        if approved or confirmations >= min_confirmations:
            promote = promote_raw if isinstance(promote_raw, dict) else {}
            raw_target = _coerce_str(promote.get("target"))
            # vault_root: resolve a folder-backed project's ambiguous short
            # form ("projects/server") to the overview note that actually
            # exists ("projects/server/server"). Without it such a target
            # never resolves and the fact sits in the inbox until it is
            # digested unpromoted (F9).
            target = normalize_target(raw_target, vault_root=paths.root)
            if not target:
                unresolved += 1
                problems.append(
                    f"{rel}: promote-eligible but promote.target is missing — left in inbox"
                )
                continue
            # promote.target is agent-controlled. Gate it exactly like the MCP
            # server gates every entity target (entity_tools.py) so a '..'
            # traversal can't escape knowledge/ and overwrite an arbitrary file.
            if not is_valid_node_id(target):
                unresolved += 1
                problems.append(
                    f"{rel}: promote.target {target!r} is not a valid node id "
                    "('..'/traversal or malformed) — left in inbox"
                )
                continue
            entity_rel = note_path_for_node(target)
            entity_path = paths.root / entity_rel
            if not entity_path.is_file():
                unresolved += 1
                tried = [entity_rel]
                if is_valid_node_id(target):
                    tried.append(note_path_for_node(folder_overview_id(target)))
                problems.append(
                    f"{rel}: promote.target {raw_target!r} resolves to no note "
                    f"(tried {', '.join(tried)}) — left in inbox"
                )
                continue
            try:
                entity_text = entity_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                unresolved += 1
                problems.append(f"{rel}: target note {entity_rel} unreadable ({exc}) — left in inbox")
                continue

            # Compute the archived note's final bytes FIRST, so a crash-rerun
            # (archive write landed, inbox unlink didn't) can reuse the plain
            # destination instead of minting a '-2' suffix. The suffix would
            # change the fallback source link below and defeat
            # append_fact_to_log's exact-duplicate dedup, double-applying the
            # fact and duplicating the archive copy.
            stamped = _set_frontmatter_key(text, "memory_status", "consolidated")
            stamped = _set_frontmatter_key(stamped, "consolidated", f"'{as_of.isoformat()}'")

            plain_dest = archive_dir / md.name
            if plain_dest not in reserved and plain_dest.is_file():
                try:
                    already = plain_dest.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    already = None
            else:
                already = None
            # The archived copy differs from `stamped` only in the volatile
            # `consolidated: <as_of>` date, so a byte-exact match would fail
            # on any rerun the day AFTER the crash — minting a '-2' dest and
            # (source unset) double-appending the fact line. Normalise that
            # one line before comparing so the reuse fires on any day.
            if already is not None and _same_promotion(already, stamped):
                dest = plain_dest       # crash-rerun: archive write already done
                reserved.add(plain_dest)
            else:
                dest = _archive_destination(archive_dir, md.name, reserved)
            dest_rel = dest.relative_to(paths.root).as_posix()

            # promote.merge (optional): a duplicate-entity merge, executed
            # here rather than left to a human (FOUNDER_DECISIONS.md
            # IMP-021). Planned BEFORE anything is written, so a refusal
            # leaves both notes and the fact exactly as they were.
            merge: _MergePlan | None = None
            if "merge" in promote:
                # Presence, not truthiness: `merge:` with no value parses as
                # None, and treating that as "no merge asked for" would
                # promote a note whose whole point is the merge as a plain
                # fact line — recording a merge in the Log that never
                # happened. _plan_merge refuses it by type.
                planned = _plan_merge(
                    promote.get("merge"),
                    root=paths.root,
                    target=target,
                    survivor_text=entity_text,
                    as_of=as_of,
                )
                if isinstance(planned, _MergeRefusal):
                    unresolved += 1
                    problems.append(f"{rel}: {planned.problem} — left in inbox")
                    continue
                merge = planned
                problems.extend(f"{rel}: {p}" for p in merge.problems)

            # promote.relations share the entity-relations shape, so the
            # same tolerant parser applies: bad entries become problems,
            # good ones land.
            relations, rel_problems = parse_relations(
                {"relations": promote.get("relations")}
            )
            for p in rel_problems:
                problems.append(f"{rel}: promote.{p}")

            new_text = merge.survivor_text if merge is not None else entity_text
            try:
                for relation in relations:
                    new_text, _action = upsert_relation_in_text(new_text, relation)
            except ValueError as exc:
                # The target note's own frontmatter does not parse; writing
                # into it would bury the breakage (AUD-001 class). Leave the
                # fact in the inbox for a human to repair the entity note.
                unresolved += 1
                problems.append(
                    f"{rel}: target note {entity_rel} has unparseable frontmatter ({exc}) — left in inbox"
                )
                continue

            # Collapse internal whitespace: a multi-line block-scalar fact
            # would otherwise land as several raw lines in ## Log, a
            # '#'-prefixed line could terminate the section, and the
            # multi-line bullet could never match the exact-duplicate check.
            fact = " ".join(_coerce_str(promote.get("fact")).split())
            if fact:
                if created is None:
                    problems.append(
                        f"{rel}: created date missing/unparseable — fact line dated {as_of.isoformat()}"
                    )
                fact_date = (created or as_of).isoformat()
                source = _normalize_source_link(_coerce_str(promote.get("source")))
                if not source:
                    # No declared source: the archived fact note IS the
                    # provenance (AGENTS.md rule 3 — outputs link sources).
                    source = dest_rel[: -len(".md")]
                new_text = append_fact_to_log(
                    new_text, f"{fact_date} — {fact} ([[{source}]])"
                )

            if not dry_run:
                if new_text != entity_text:
                    _atomic_write(entity_path, new_text)
                if merge is not None and merge.duplicate_text is not None:
                    try:
                        _atomic_write(
                            paths.root / merge.duplicate_rel, merge.duplicate_text
                        )
                    except OSError as exc:
                        # The survivor IS on disk with the copied relations,
                        # the alias and (below) the fact line: record it as
                        # touched BEFORE bailing out, or scripts/consolidate.py
                        # never reindexes a note this run rewrote.
                        if new_text != entity_text:
                            touched.append(entity_rel)
                        # Half-applied, and deliberately so (see the merge
                        # section's ATOMICITY note): the fact stays in the
                        # inbox and both halves are idempotent, so the next
                        # run finishes the merge rather than doubling it.
                        unresolved += 1
                        problems.append(
                            f"{rel}: merge HALF-APPLIED — survivor {entity_rel} "
                            f"written but duplicate {merge.duplicate_rel} could "
                            f"not be ({exc}); fact left in inbox, re-run to finish"
                        )
                        logger.error(
                            "consolidate: merge half-applied for %s (duplicate %s: %s)",
                            rel, merge.duplicate_rel, exc,
                        )
                        continue
                archive_dir.mkdir(parents=True, exist_ok=True)
                _atomic_write(dest, stamped)
                md.unlink()
            if new_text != entity_text:
                touched.append(entity_rel)
            if merge is not None and merge.duplicate_text is not None:
                touched.append(merge.duplicate_rel)
                logger.info(
                    "consolidate: merged %s into %s (superseded, relations copied)%s",
                    merge.duplicate_rel, entity_rel,
                    " [dry-run]" if dry_run else "",
                )
            moved.append((rel, dest_rel))
            promoted += 1
            logger.info(
                "consolidate: promoted %s -> %s (target %s)%s",
                rel, dest_rel, target, " [dry-run]" if dry_run else "",
            )
            continue

        # -- DIGEST --------------------------------------------------------
        if status == "unconsolidated" and not approved:
            if created is None:
                skipped += 1
                problems.append(
                    f"{rel}: created date missing/unparseable — staleness unknown, left in inbox"
                )
                continue
            if (as_of - created).days > stale_days:
                dest = _archive_destination(archive_dir, md.name, reserved)
                dest_rel = dest.relative_to(paths.root).as_posix()
                # Collapse whitespace so a multi-line title can't break the
                # single-line digest bullet (mirror the promote fact fix).
                title = " ".join((_coerce_str(fm.get("title")) or md.stem).split())
                bullet = (
                    f"- {created.isoformat()} — {title} -> [[{dest_rel[: -len('.md')]}]]"
                )
                snippet = _first_paragraph(body)
                if snippet:
                    bullet += f" — {snippet}"
                digest_bullets.append(bullet)
                # F15: DEFER the move. The digest file records WHERE this
                # note went; it must be on disk BEFORE the note leaves the
                # inbox, or a crash between the move and the digest write
                # orphans the note (moved out, no bullet pointing at it).
                # The move happens after the loop, once every bullet is
                # durably written — see the post-loop digest block.
                deferred_moves.append((md, dest))
                moved.append((rel, dest_rel))
                digested += 1
                logger.info(
                    "consolidate: digested %s -> %s%s",
                    rel, dest_rel, " [dry-run]" if dry_run else "",
                )
                continue

        # -- WAIT ------------------------------------------------------------
        skipped += 1
        logger.debug("consolidate: %s waits (fresh/unapproved)", rel)

    digest_rel = ""
    if digest_bullets and not dry_run:
        if digest_path.is_file():
            digest_text = digest_path.read_text(encoding="utf-8", errors="replace")
            digest_text = _set_frontmatter_key(
                digest_text, "updated", f"'{as_of.isoformat()}'"
            )
        else:
            digest_text = _digest_skeleton(month, as_of)
        # Dedup against bullets already in the digest: a crash-rerun (bullet
        # written, deferred move not yet done) would otherwise append the same
        # durable line twice. Mirrors append_fact_to_log's idempotency.
        existing_lines = set(digest_text.splitlines())
        fresh = [b for b in digest_bullets if b not in existing_lines]
        digest_text = (
            digest_text.rstrip("\n") + "\n" + "\n".join(fresh) + "\n"
            if fresh else digest_text
        )
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        # F15: write the durable record FIRST. Only once every bullet is on
        # disk do we move the notes out of the inbox (below). A crash before
        # this leaves all stale notes in the inbox (the next run re-digests
        # them); a crash during the moves leaves every moved note already
        # recorded in the digest — so no stale note is ever orphaned.
        _atomic_write(digest_path, digest_text)
        digest_rel = digest_path.relative_to(paths.root).as_posix()
        logger.info(
            "consolidate: %d bullet(s) appended to %s",
            len(digest_bullets), digest_rel,
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        for src_md, dest in deferred_moves:
            src_md.replace(dest)   # atomic rename: bytes preserved exactly

    logger.info(
        "consolidate: promoted=%d digested=%d unresolved=%d skipped=%d problems=%d%s",
        promoted, digested, unresolved, skipped, len(problems),
        " [dry-run]" if dry_run else "",
    )
    return ConsolidateStats(
        promoted=promoted,
        digested=digested,
        unresolved=unresolved,
        skipped=skipped,
        touched_entity_paths=tuple(dict.fromkeys(touched)),
        moved=tuple(moved),
        problems=tuple(problems),
        digest_path=digest_rel,
    )


__all__ = ["ConsolidateStats", "consolidate"]
