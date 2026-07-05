"""Tests for ingest_lib.consolidate — the deterministic memory
consolidation pass: promote confirmed facts into entity notes, archive
the originals, digest stale leftovers. No LLM, no wall-clock: every test
injects ``as_of`` and asserts purely on file contents and stats."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.consolidate import consolidate
from ingest_lib.notes import _split_frontmatter

_LOG = logging.getLogger("test")
AS_OF = date(2026, 6, 12)

_INBOX = "knowledge/assistant/inbox"
_ARCHIVE_MONTH = "knowledge/assistant/archive/2026-06"
_DIGEST = "knowledge/assistant/digests/2026-06.md"

_ENTITY = """---
title: Anna Kowalska
type: person
created: '2026-01-01'
updated: '2026-01-01'
---

# Anna Kowalska

## Log

- 2026-01-01 — note created
"""

# The entity AFTER a promote that crashed before stamping+moving the inbox
# note: the relation is already merged into frontmatter and the fact line
# already sits in ## Log. The matching inbox note (``_fact_note(approved)``)
# is still present, so a rerun re-promotes it. Used by the F13 tests.
_ENTITY_AFTER_PROMOTE = """---
title: Anna Kowalska
type: person
created: '2026-01-01'
updated: '2026-01-01'
relations:
- rel: works_at
  target: organisations/acme
  valid_from: '2026-06-01'
---

# Anna Kowalska

## Log

- 2026-01-01 — note created
- 2026-06-10 — Anna works at Acme ([[knowledge/meetings/2026/2026-06-01-call]])
"""


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    for sub in (
        "knowledge/assistant/inbox",
        "knowledge/assistant/archive",
        "knowledge/assistant/digests",
        "knowledge/people",
        "knowledge/organisations",
    ):
        (paths.root / sub).mkdir(parents=True, exist_ok=True)
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _read(paths: VaultPaths, rel: str) -> str:
    return (paths.root / rel).read_text(encoding="utf-8")


def _fact_note(
    *,
    title: str = "Anna works at Acme",
    created: str = "2026-06-10",
    confirmations: int = 0,
    approved: bool = False,
    target: str = "people/anna",
    fact: str = "Anna works at Acme",
    source: str = "knowledge/meetings/2026/2026-06-01-call",
    body: str = "Anna mentioned her new role twice during the call.",
) -> str:
    """Hand-built note matching knowledge/index/templates/memory-fact.md."""
    return (
        "---\n"
        f'title: "{title}"\n'
        "type: memory_fact\n"
        f'created: "{created}"\n'
        'author: "claude"\n'
        'written_via: "mcp"\n'
        "memory_status: unconsolidated\n"
        f"confirmations: {confirmations}\n"
        f"approved: {'true' if approved else 'false'}\n"
        "promote:\n"
        f'  target: "{target}"\n'
        "  relations:\n"
        "    - rel: works_at\n"
        "      target: organisations/acme\n"
        '      valid_from: "2026-06-01"\n'
        f'  fact: "{fact}"\n'
        f'  source: "{source}"\n'
        "---\n"
        "\n"
        f"{body}\n"
    )


def _snapshot(root: Path) -> dict[str, bytes | None]:
    """Full tree snapshot: file bytes, plus directories as None so a
    dry run creating an empty archive month dir would also be caught."""
    out: dict[str, bytes | None] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel] = p.read_bytes() if p.is_file() else None
    return out


# ---------------------------------------------------------------------------
# promotion
# ---------------------------------------------------------------------------

def test_approved_fact_promotes(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _ENTITY)
    _write(paths, f"{_INBOX}/fact-anna.md", _fact_note(approved=True))

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    assert stats.digested == stats.unresolved == stats.skipped == 0
    assert stats.touched_entity_paths == ("knowledge/people/anna.md",)
    assert stats.moved == (
        (f"{_INBOX}/fact-anna.md", f"{_ARCHIVE_MONTH}/fact-anna.md"),
    )

    # Relation merged into the entity's frontmatter.
    entity = _read(paths, "knowledge/people/anna.md")
    fm, body = _split_frontmatter(entity)
    assert fm["relations"] == [
        {"rel": "works_at", "target": "organisations/acme", "valid_from": "2026-06-01"}
    ]
    # Fact line in ## Log: fact note's created date + declared source link.
    assert (
        "- 2026-06-10 — Anna works at Acme "
        "([[knowledge/meetings/2026/2026-06-01-call]])" in body
    )
    # Pre-existing log history untouched.
    assert "- 2026-01-01 — note created" in body

    # Note moved (not deleted), stamped consolidated.
    assert not (paths.root / _INBOX / "fact-anna.md").exists()
    archived = _read(paths, f"{_ARCHIVE_MONTH}/fact-anna.md")
    afm, abody = _split_frontmatter(archived)
    assert afm["memory_status"] == "consolidated"
    assert str(afm["consolidated"]) == "2026-06-12"
    # Free-form context preserved verbatim through the move.
    assert "Anna mentioned her new role twice during the call." in abody


def test_confirmations_threshold_promotes_below_waits(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _ENTITY)
    _write(paths, f"{_INBOX}/confirmed.md", _fact_note(confirmations=3))
    _write(paths, f"{_INBOX}/unconfirmed.md", _fact_note(confirmations=2))

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF, min_confirmations=3)

    assert stats.promoted == 1
    assert stats.skipped == 1
    assert not (paths.root / _INBOX / "confirmed.md").exists()
    assert (paths.root / _ARCHIVE_MONTH / "confirmed.md").is_file()
    # Below threshold: untouched, byte-for-byte.
    assert _read(paths, f"{_INBOX}/unconfirmed.md") == _fact_note(confirmations=2)


def test_unresolved_target_stays_in_inbox(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    original = _fact_note(approved=True, target="people/ghost")
    _write(paths, f"{_INBOX}/fact-ghost.md", original)

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.unresolved == 1
    assert stats.promoted == 0
    assert stats.moved == ()
    # Left in place, byte-for-byte, with the missing path named.
    assert _read(paths, f"{_INBOX}/fact-ghost.md") == original
    assert any("knowledge/people/ghost.md" in p for p in stats.problems)


# ---------------------------------------------------------------------------
# digestion + waiting
# ---------------------------------------------------------------------------

def test_stale_unconsolidated_is_digested(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    stale = _fact_note(
        title="Old whisper",
        created="2026-04-01",
        body="Something half-remembered about a conference hallway chat.",
    )
    _write(paths, f"{_INBOX}/old-whisper.md", stale)

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF, stale_days=30)

    assert stats.digested == 1
    assert stats.promoted == stats.unresolved == stats.skipped == 0
    # Moved verbatim — digestion changes nothing inside the note.
    assert not (paths.root / _INBOX / "old-whisper.md").exists()
    assert _read(paths, f"{_ARCHIVE_MONTH}/old-whisper.md") == stale

    digest = _read(paths, _DIGEST)
    dfm, dbody = _split_frontmatter(digest)
    assert dfm["title"] == "Memory digest 2026-06"
    assert dfm["type"] == "digest"
    assert str(dfm["created"]) == "2026-06-12"
    assert (
        "- 2026-04-01 — Old whisper -> "
        f"[[{_ARCHIVE_MONTH}/old-whisper]] — "
        "Something half-remembered about a conference hallway chat." in dbody
    )


def test_fresh_unapproved_waits_untouched(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    fresh = _fact_note(created="2026-06-10")
    _write(paths, f"{_INBOX}/fresh.md", fresh)
    # A stray non-fact note is skipped with a problem, never moved.
    _write(paths, f"{_INBOX}/scratch.md", "# Just a scratchpad\n\nNot a fact.\n")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.skipped == 2
    assert stats.promoted == stats.digested == stats.unresolved == 0
    assert stats.moved == ()
    assert _read(paths, f"{_INBOX}/fresh.md") == fresh
    assert any("not a memory fact note" in p for p in stats.problems)
    assert not (paths.root / _DIGEST).exists()


# ---------------------------------------------------------------------------
# dry run + collisions + determinism
# ---------------------------------------------------------------------------

def test_dry_run_mutates_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _ENTITY)
    _write(paths, f"{_INBOX}/promotable.md", _fact_note(approved=True))
    _write(paths, f"{_INBOX}/stale.md", _fact_note(created="2026-04-01"))

    before = _snapshot(paths.root)
    stats = consolidate(paths, logger=_LOG, as_of=AS_OF, dry_run=True)
    after = _snapshot(paths.root)

    assert after == before
    # Stats still reflect the full plan.
    assert stats.promoted == 1
    assert stats.digested == 1
    assert stats.touched_entity_paths == ("knowledge/people/anna.md",)
    assert len(stats.moved) == 2


def test_archive_collision_gets_numeric_suffix(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _ENTITY)
    _write(paths, f"{_ARCHIVE_MONTH}/fact.md", "earlier archived note\n")
    _write(paths, f"{_INBOX}/fact.md", _fact_note(approved=True))

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.moved == ((f"{_INBOX}/fact.md", f"{_ARCHIVE_MONTH}/fact-2.md"),)
    # Never overwrite: the earlier occupant is intact.
    assert _read(paths, f"{_ARCHIVE_MONTH}/fact.md") == "earlier archived note\n"
    fm, _ = _split_frontmatter(_read(paths, f"{_ARCHIVE_MONTH}/fact-2.md"))
    assert fm["memory_status"] == "consolidated"


def test_determinism_byte_identical_across_copies(tmp_path: Path) -> None:
    def seed(root: Path) -> VaultPaths:
        paths = _vault(root)
        _write(paths, "knowledge/people/anna.md", _ENTITY)
        _write(paths, f"{_INBOX}/approved.md", _fact_note(approved=True))
        _write(paths, f"{_INBOX}/confirmed.md", _fact_note(confirmations=5, source=""))
        _write(paths, f"{_INBOX}/stale.md", _fact_note(created="2026-03-15"))
        _write(paths, f"{_INBOX}/fresh.md", _fact_note(created="2026-06-11"))
        return paths

    paths_a = seed(tmp_path / "a")
    paths_b = seed(tmp_path / "b")

    stats_a = consolidate(paths_a, logger=_LOG, as_of=AS_OF)
    stats_b = consolidate(paths_b, logger=_LOG, as_of=AS_OF)

    assert stats_a == stats_b
    assert _snapshot(paths_a.root) == _snapshot(paths_b.root)
    # And the empty-source fact fell back to linking its own archived path.
    log_body = _read(paths_a, "knowledge/people/anna.md")
    assert f"([[{_ARCHIVE_MONTH}/confirmed]])" in log_body


# ---------------------------------------------------------------------------
# digest reindex hook (F9) + crash-idempotent promote (F13)
# ---------------------------------------------------------------------------

def test_digest_path_set_only_when_digest_written(tmp_path: Path) -> None:
    """F9: the CLI reindexes ``stats.digest_path`` so a fresh monthly digest
    is searchable immediately, not after the next full rebuild. The field
    names the digest note when one was written this run, and is "" otherwise."""
    with_digest = _vault(tmp_path / "with")
    _write(with_digest, f"{_INBOX}/stale.md", _fact_note(created="2026-04-01"))
    s1 = consolidate(with_digest, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert s1.digested == 1
    assert s1.digest_path == _DIGEST

    without_digest = _vault(tmp_path / "without")
    _write(without_digest, f"{_INBOX}/fresh.md", _fact_note(created="2026-06-11"))
    s2 = consolidate(without_digest, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert s2.digested == 0
    assert s2.digest_path == ""


def test_crashed_promote_rerun_relation_upsert_is_noop(tmp_path: Path) -> None:
    """F13: a promote that crashed AFTER updating the entity but BEFORE
    stamping+moving the inbox note leaves the entity already carrying the
    relation while the inbox note is still present. A rerun must re-promote
    (archive the inbox copy) without duplicating the relation — the upsert
    is already idempotent on an exact match."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _ENTITY_AFTER_PROMOTE)
    _write(paths, f"{_INBOX}/fact-anna.md", _fact_note(approved=True))

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    assert not (paths.root / _INBOX / "fact-anna.md").exists()
    fm, _ = _split_frontmatter(_read(paths, "knowledge/people/anna.md"))
    assert fm["relations"] == [
        {"rel": "works_at", "target": "organisations/acme", "valid_from": "2026-06-01"}
    ]


def test_crashed_promote_rerun_does_not_duplicate_fact_line(tmp_path: Path) -> None:
    """F13: the same crashed-promote rerun must not append the fact line a
    SECOND time — relies on ``relations.append_fact_to_log`` being idempotent
    on an exact-duplicate bullet (asserted directly in test_relations)."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _ENTITY_AFTER_PROMOTE)
    _write(paths, f"{_INBOX}/fact-anna.md", _fact_note(approved=True))

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    _, body = _split_frontmatter(_read(paths, "knowledge/people/anna.md"))
    line = (
        "- 2026-06-10 — Anna works at Acme "
        "([[knowledge/meetings/2026/2026-06-01-call]])"
    )
    assert body.count(line) == 1
