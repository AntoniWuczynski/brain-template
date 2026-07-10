"""Deterministic consolidation of assistant memory facts.

The lifecycle (the "dream pass", minus the mysticism):

1. The assistant drops structured fact notes into
   ``knowledge/assistant/inbox/`` over MCP. Each note carries
   ``memory_status: unconsolidated``, ``confirmations: N``,
   ``approved: bool`` and a ``promote:`` mapping
   (``{target, relations: [...], fact, source}``) — the contract is
   ``knowledge/index/templates/memory-fact.md``.
2. THIS pass — counters and thresholds, NO LLM — promotes confirmed
   facts into their target entity notes (relations merged into
   frontmatter, the fact line appended to ``## Log``), stamps the fact
   note ``memory_status: consolidated``, and MOVES it to
   ``knowledge/assistant/archive/<YYYY-MM>/``. Moved, never deleted:
   the original wording stays reviewable forever.
3. Facts that linger unconsolidated past ``stale_days`` are swept into
   a monthly digest under ``knowledge/assistant/digests/`` (one bullet
   each, linking the archived note) instead of accumulating forever.
4. Everything else waits in the inbox for more confirmations or a
   human's ``approved: true``.

LLMs may *propose* facts; only deterministic code or the human promotes
them. Every decision here derives from frontmatter + ``as_of`` + the
thresholds, so two runs over identical inputs produce byte-identical
results. The only timestamps written are frontmatter values derived
from ``as_of`` (AGENTS.md rule 7).

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
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .config import VaultPaths
from .notes import _atomic_write, _split_frontmatter  # private helpers, module-internal
from .relations import (
    append_fact_to_log,
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

    - PROMOTE when ``approved`` is truthy OR
      ``confirmations >= min_confirmations`` and the target entity note
      exists. Missing target -> the note STAYS in the inbox and counts
      ``unresolved`` (a human or the agent resolves it).
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

        approved = _coerce_bool(fm.get("approved"))
        confirmations = _coerce_int(fm.get("confirmations"))
        created = _parse_date(fm.get("created"))

        # -- PROMOTE -------------------------------------------------------
        if approved or confirmations >= min_confirmations:
            promote = promote_raw if isinstance(promote_raw, dict) else {}
            target = normalize_target(_coerce_str(promote.get("target")))
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
                problems.append(
                    f"{rel}: target note {entity_rel} does not exist — left in inbox"
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

            # promote.relations share the entity-relations shape, so the
            # same tolerant parser applies: bad entries become problems,
            # good ones land.
            relations, rel_problems = parse_relations(
                {"relations": promote.get("relations")}
            )
            for p in rel_problems:
                problems.append(f"{rel}: promote.{p}")

            new_text = entity_text
            for relation in relations:
                new_text, _action = upsert_relation_in_text(new_text, relation)

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
                archive_dir.mkdir(parents=True, exist_ok=True)
                _atomic_write(dest, stamped)
                md.unlink()
            if new_text != entity_text:
                touched.append(entity_rel)
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
