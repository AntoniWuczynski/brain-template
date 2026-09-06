"""Tests for ingest_lib.consolidate — the deterministic memory
consolidation pass: promote confirmed facts into entity notes, archive
the originals, digest stale leftovers. No LLM, no wall-clock: every test
injects ``as_of`` and asserts purely on file contents and stats."""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

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


def _approve(paths: VaultPaths, *names: str) -> None:
    """Mark inbox fact notes approved. The gate is the note's own
    ``approved:`` frontmatter — FOUNDER_DECISIONS.md IMP-016 reverted the
    out-of-reach ledger — so approving is flipping that key."""
    for name in names:
        note = paths.root / _INBOX / name
        text = note.read_text(encoding="utf-8")
        assert "approved:" in text, f"{name} is not a fact note"
        note.write_text(text.replace("approved: false", "approved: true", 1), encoding="utf-8")


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
    provenance: bool = True,
) -> str:
    """Note matching knowledge/index/templates/memory-fact.md.

    ``provenance=True`` (the default, and the live case) carries the
    server-asserted keys the MCP server stamps on every write, marking the
    note agent-written: its own ``approved:``/``confirmations:`` are then
    advisory and only the human ledger (``_approve``) promotes it.
    ``provenance=False`` is a note the human wrote by hand — no server
    provenance, so the plain frontmatter gate applies."""
    return (
        "---\n"
        f'title: "{title}"\n'
        "type: memory_fact\n"
        f'created: "{created}"\n'
        + ("author: 'agent:claude-code'\nwritten_via: \"mcp\"\n" if provenance else "")
        + "memory_status: unconsolidated\n"
        + f"confirmations: {confirmations}\n"
        + f"approved: {'true' if approved else 'false'}\n"
        + "promote:\n"
        + f'  target: "{target}"\n'
        + "  relations:\n"
        + "    - rel: works_at\n"
        + "      target: organisations/acme\n"
        + '      valid_from: "2026-06-01"\n'
        + f'  fact: "{fact}"\n'
        + f'  source: "{source}"\n'
        + "---\n"
        + "\n"
        + f"{body}\n"
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
    _approve(paths, "fact-anna.md")

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
    # Hand-written notes (no server provenance): their own confirmations
    # counter IS the human's, so the threshold gate applies to them.
    _write(paths, f"{_INBOX}/confirmed.md", _fact_note(confirmations=3, provenance=False))
    _write(paths, f"{_INBOX}/unconfirmed.md", _fact_note(confirmations=2, provenance=False))

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF, min_confirmations=3)

    assert stats.promoted == 1
    assert stats.skipped == 1
    assert not (paths.root / _INBOX / "confirmed.md").exists()
    assert (paths.root / _ARCHIVE_MONTH / "confirmed.md").is_file()
    # Below threshold: untouched, byte-for-byte.
    assert _read(paths, f"{_INBOX}/unconfirmed.md") == _fact_note(
        confirmations=2, provenance=False
    )


def test_unresolved_target_stays_in_inbox(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    original = _fact_note(approved=True, target="people/ghost")
    _write(paths, f"{_INBOX}/fact-ghost.md", original)
    _approve(paths, "fact-ghost.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.unresolved == 1
    assert stats.promoted == 0
    assert stats.moved == ()
    # Left in place, byte-for-byte, with the missing path named.
    assert _read(paths, f"{_INBOX}/fact-ghost.md") == original
    assert any("knowledge/people/ghost.md" in p for p in stats.problems)


def test_promote_target_traversal_is_rejected(tmp_path: Path) -> None:
    # promote.target is agent-controlled; a '..' traversal must never let the
    # promoter write outside knowledge/. AGENTS.md exists at the repo root, so
    # this would overwrite it (the .is_file() guard only blocks new files).
    paths = _vault(tmp_path)
    sentinel = paths.root / "AGENTS.md"
    sentinel.write_text("ORIGINAL AGENTS CONTENT\n", encoding="utf-8")
    original = _fact_note(approved=True, target="../AGENTS")
    _write(paths, f"{_INBOX}/fact-evil.md", original)
    _approve(paths, "fact-evil.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 0
    assert stats.unresolved == 1
    # The out-of-tree file is untouched and the note stays in the inbox.
    assert sentinel.read_text(encoding="utf-8") == "ORIGINAL AGENTS CONTENT\n"
    assert _read(paths, f"{_INBOX}/fact-evil.md") == original
    assert any("not a valid node id" in p for p in stats.problems)


# ---------------------------------------------------------------------------
# folder-backed project targets (F9)
# ---------------------------------------------------------------------------

_PROJECT = """---
title: Home server
type: project
created: '2026-01-01'
updated: '2026-01-01'
---

# Home server

## Log

- 2026-01-01 — note created
"""


def test_short_form_project_target_promotes_into_folder_overview(tmp_path: Path) -> None:
    """F9: a project lives in a folder, so ``projects/server`` names no note
    and the live inbox fact could never promote. The overview note
    ``knowledge/projects/server/server.md`` is the entity, and the short form
    resolves to it."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/server/server.md", _PROJECT)
    _write(paths, f"{_INBOX}/fact-server.md",
           _fact_note(approved=True, target="projects/server",
                      fact="Antoni runs the brain on the home server"))
    _approve(paths, "fact-server.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    assert stats.unresolved == 0
    assert stats.touched_entity_paths == ("knowledge/projects/server/server.md",)
    overview = _read(paths, "knowledge/projects/server/server.md")
    assert "- 2026-06-10 — Antoni runs the brain on the home server" in overview
    # No stray flat note was created next to the folder.
    assert not (paths.root / "knowledge/projects/server.md").exists()
    assert not (paths.root / f"{_INBOX}/fact-server.md").exists()


def test_flat_project_note_wins_over_folder_overview(tmp_path: Path) -> None:
    """Resolution is by existence, not by shape: a real note at the short
    path keeps its own id even when a same-named folder overview exists."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/rex.md", _PROJECT)
    _write(paths, "knowledge/projects/rex/rex.md", _PROJECT)
    _write(paths, f"{_INBOX}/fact-rex.md",
           _fact_note(approved=True, target="projects/rex", fact="Rex shipped"))
    _approve(paths, "fact-rex.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    assert stats.touched_entity_paths == ("knowledge/projects/rex.md",)
    assert "Rex shipped" in _read(paths, "knowledge/projects/rex.md")
    assert "Rex shipped" not in _read(paths, "knowledge/projects/rex/rex.md")


def test_folder_without_overview_stays_unresolved_and_names_both_paths(
    tmp_path: Path,
) -> None:
    """A project folder holding only curated notes has no entity note, so the
    fact stays in the inbox — and the problem names every path that was
    tried, not just the flat one."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/ghost/design.md", _PROJECT)
    original = _fact_note(approved=True, target="projects/ghost")
    _write(paths, f"{_INBOX}/fact-ghost-project.md", original)
    _approve(paths, "fact-ghost-project.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 0
    assert stats.unresolved == 1
    assert _read(paths, f"{_INBOX}/fact-ghost-project.md") == original
    problem = next(p for p in stats.problems if "fact-ghost-project.md" in p)
    assert "'projects/ghost'" in problem
    assert "knowledge/projects/ghost.md" in problem
    assert "knowledge/projects/ghost/ghost.md" in problem


# ---------------------------------------------------------------------------
# anti-self-approval (F1)
# ---------------------------------------------------------------------------


def test_hand_written_note_promotes_on_its_own_approved(tmp_path: Path) -> None:
    """A note with NO server provenance was written by the human, so its own
    ``approved: true`` is the human's decision — no ledger entry needed."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _ENTITY)
    _write(paths, f"{_INBOX}/hand.md", _fact_note(approved=True, provenance=False))

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    assert stats.problems == ()
    assert (paths.root / _ARCHIVE_MONTH / "hand.md").is_file()

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
    _approve(paths, "promotable.md")

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
    _approve(paths, "fact.md")

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
        _write(paths, f"{_INBOX}/confirmed.md", _fact_note(
            confirmations=5, source="", provenance=False,
        ))
        _approve(paths, "approved.md")
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
    _approve(paths, "fact-anna.md")

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
    _approve(paths, "fact-anna.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    _, body = _split_frontmatter(_read(paths, "knowledge/people/anna.md"))
    line = (
        "- 2026-06-10 — Anna works at Acme "
        "([[knowledge/meetings/2026/2026-06-01-call]])"
    )
    assert body.count(line) == 1


def test_crashed_promote_rerun_next_day_no_duplicate(tmp_path: Path) -> None:
    # The crash-rerun guard must survive a rerun on a LATER day. The archived
    # copy carries a `consolidated: <as_of>` stamp, so a byte-exact match
    # fails cross-day, minting a '-2' archive dest and (source unset) a
    # second Log bullet whose fallback source is that '-2' path.
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", _ENTITY)
    fact = _fact_note(approved=True, source="")  # source unset -> dest fallback
    _write(paths, f"{_INBOX}/fact-anna.md", fact)
    _approve(paths, "fact-anna.md")

    day1 = date(2026, 6, 12)
    s1 = consolidate(paths, logger=_LOG, as_of=day1)
    assert s1.promoted == 1

    # Simulate the crash: archive write + entity update landed, but the inbox
    # unlink didn't — so the note is still pending on the next run.
    _write(paths, f"{_INBOX}/fact-anna.md", fact)

    day2 = date(2026, 6, 13)  # recovery run happens the next day
    s2 = consolidate(paths, logger=_LOG, as_of=day2)
    assert s2.promoted == 1

    _, body = _split_frontmatter(_read(paths, "knowledge/people/anna.md"))
    assert body.count("— Anna works at Acme (") == 1
    archived = sorted(
        p.name for p in (paths.root / _ARCHIVE_MONTH).glob("fact-anna*.md")
    )
    assert archived == ["fact-anna.md"]  # no 'fact-anna-2.md' duplicate


def test_promote_into_unparseable_entity_note_is_left_unresolved(tmp_path: Path) -> None:
    """A target note whose frontmatter fence does not parse must not be
    written into (it would be buried under a second fence); the fact stays
    in the inbox and the run reports why."""
    paths = _vault(tmp_path)
    broken = "---\ntitle: fps — \"Quarterly Review: a fever dream\"\ntype: person\n---\n\n# Anna\n"
    _write(paths, "knowledge/people/anna.md", broken)
    _write(paths, f"{_INBOX}/fact-anna.md", _fact_note(approved=True))
    _approve(paths, "fact-anna.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 0
    assert stats.unresolved == 1
    assert (paths.root / _INBOX / "fact-anna.md").exists()
    assert _read(paths, "knowledge/people/anna.md") == broken
    assert any("unparseable frontmatter" in p for p in stats.problems)


# ---------------------------------------------------------------------------
# duplicate-entity merge (promote.merge) — FOUNDER_DECISIONS.md IMP-021
# Fixtures are the three live pairs (alicia/nate/peter beside their
# email-address twins), anonymised: same frontmatter key order, quoting and
# meeting_create-minted ``attended`` relation, because those exact bytes are
# what the merge has to survive.
# ---------------------------------------------------------------------------

_MEETING = "meetings/2026/2026-07-02-offsite"

_LIVE_PAIRS = (
    ("dana-scott", "Dana Scott", "danaexamplecom", "dana@example.com"),
    ("milo-frey", "Milo Frey", "miloexamplecom", "milo@example.com"),
    ("ravi-patel", "Ravi Patel", "raviexamplecom", "ravi@example.com"),
)


def _named_note(name: str) -> str:
    return (
        f'---\ntitle: "{name}"\ntype: person\naliases: []\ntopics: []\n'
        "relations: []\nauthor: 'agent:default'\nwritten_via: mcp\n"
        "---\n\n## Log\n"
    )


def _email_note(email: str) -> str:
    return (
        f"---\ntitle: {email}\ntype: person\naliases: []\ntopics: []\n"
        f"relations:\n- rel: attended\n  target: {_MEETING}\n"
        f"  valid_from: '2026-07-02'\n  source: knowledge/{_MEETING}\n"
        "last_written_by: 'agent:default'\nwritten_via: mcp\n"
        "author: 'agent:default'\n---\n\n## Log\n"
    )


def _merge_fact(
    *,
    duplicate: str,
    survivor: str,
    target: str | None = None,
    merge_block: str | None = None,
    fact: str = "",
    created: str = "2026-06-10",
) -> str:
    """A memory fact carrying ``promote.merge``, the shape
    ``ingest_lib.duplicates`` proposes; ``merge_block`` overrides the
    mapping verbatim so a malformed one can be tested."""
    block = merge_block if merge_block is not None else (
        "  merge:\n"
        f"    duplicate: {duplicate}\n"
        f"    survivor: {survivor}\n"
    )
    return (
        "---\n"
        'title: "merge candidate"\n'
        "type: memory_fact\n"
        f"created: '{created}'\n"
        "memory_status: unconsolidated\n"
        "confirmations: 0\n"
        "approved: false\n"
        "promote:\n"
        + block
        + f"  target: {target if target is not None else survivor}\n"
        + "  relations: []\n"
        + f'  fact: "{fact}"\n'
        + f"  source: knowledge/{duplicate}\n"
        + "---\n"
        + "\n"
        + "Proposed by the deterministic duplicate-entity pass.\n"
    )


def _seed_pair(paths: VaultPaths, slug: str, name: str, email_slug: str, email: str) -> None:
    _write(paths, f"knowledge/people/{slug}.md", _named_note(name))
    _write(paths, f"knowledge/people/{email_slug}.md", _email_note(email))


# The two notes after the merge, byte for byte. Written out rather than
# derived, so a change in how either half is spliced fails loudly.
_MERGED_SURVIVOR = """---
title: "Dana Scott"
type: person
aliases: [dana@example.com]
topics: []
relations:
- rel: attended
  target: meetings/2026/2026-07-02-offsite
  valid_from: '2026-07-02'
  source: knowledge/meetings/2026/2026-07-02-offsite
author: 'agent:default'
written_via: mcp
---

## Log

- 2026-06-10 — merged dana@example.com ([[knowledge/people/danaexamplecom]])
"""

_MERGED_DUPLICATE = """---
title: dana@example.com
type: person
aliases: []
topics: []
relations:
- rel: attended
  target: meetings/2026/2026-07-02-offsite
  valid_from: '2026-07-02'
  source: knowledge/meetings/2026/2026-07-02-offsite
  valid_until: '2026-06-12'
last_written_by: 'agent:default'
written_via: mcp
author: 'agent:default'
superseded_by: knowledge/people/dana-scott
---

## Log
"""


def test_approved_merge_performs_the_four_agents_md_steps(tmp_path: Path) -> None:
    """Relations copied over, open spans closed at the run date,
    superseded_by stamped, address kept as an alias."""
    paths = _vault(tmp_path)
    _seed_pair(paths, *_LIVE_PAIRS[0])
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(
            duplicate="people/danaexamplecom",
            survivor="people/dana-scott",
            fact="merged dana@example.com",
        ),
    )
    _approve(paths, "merge-dana.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    assert stats.unresolved == 0
    assert stats.problems == ()
    assert _read(paths, "knowledge/people/dana-scott.md") == _MERGED_SURVIVOR
    assert _read(paths, "knowledge/people/danaexamplecom.md") == _MERGED_DUPLICATE
    # Both halves are reindexed, so search stops resolving the dead node.
    assert stats.touched_entity_paths == (
        "knowledge/people/dana-scott.md", "knowledge/people/danaexamplecom.md",
    )
    assert not (paths.root / _INBOX / "merge-dana.md").exists()


def test_approved_merge_over_the_three_live_shaped_pairs(tmp_path: Path) -> None:
    """The live distribution: three email twins minted by meeting_create,
    all three merged in one pass."""
    paths = _vault(tmp_path)
    for pair in _LIVE_PAIRS:
        _seed_pair(paths, *pair)
    for slug, _name, email_slug, _email in _LIVE_PAIRS:
        _write(
            paths, f"{_INBOX}/merge-{slug}.md",
            _merge_fact(
                duplicate=f"people/{email_slug}", survivor=f"people/{slug}"
            ),
        )
        _approve(paths, f"merge-{slug}.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 3
    assert stats.problems == ()
    for slug, _name, email_slug, email in _LIVE_PAIRS:
        survivor_fm, _ = _split_frontmatter(_read(paths, f"knowledge/people/{slug}.md"))
        assert survivor_fm["aliases"] == [email]
        assert survivor_fm["relations"] == [{
            "rel": "attended",
            "target": _MEETING,
            "valid_from": "2026-07-02",
            "source": f"knowledge/{_MEETING}",
        }]
        dup_fm, _ = _split_frontmatter(
            _read(paths, f"knowledge/people/{email_slug}.md")
        )
        assert dup_fm["superseded_by"] == f"knowledge/people/{slug}"
        assert dup_fm["relations"] == [{
            "rel": "attended",
            "target": _MEETING,
            "valid_from": "2026-07-02",
            "source": f"knowledge/{_MEETING}",
            "valid_until": "2026-06-12",
        }]


def test_merge_rerun_is_idempotent(tmp_path: Path) -> None:
    """A second approved fact for an already-merged pair changes neither
    note (the duplicate carries superseded_by) but still archives."""
    paths = _vault(tmp_path)
    _seed_pair(paths, *_LIVE_PAIRS[0])
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "merge-dana.md")
    consolidate(paths, logger=_LOG, as_of=AS_OF)
    after_first = {
        rel: _read(paths, rel)
        for rel in ("knowledge/people/dana-scott.md", "knowledge/people/danaexamplecom.md")
    }

    _write(
        paths, f"{_INBOX}/merge-dana-again.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "merge-dana-again.md")
    stats = consolidate(paths, logger=_LOG, as_of=date(2026, 6, 20))

    assert stats.promoted == 1
    assert stats.unresolved == 0
    assert stats.touched_entity_paths == ()
    assert {rel: _read(paths, rel) for rel in after_first} == after_first
    assert not (paths.root / _INBOX / "merge-dana-again.md").exists()


def test_merge_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths, *_LIVE_PAIRS[0])
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "merge-dana.md")
    before = _snapshot(paths.root)

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF, dry_run=True)

    assert stats.promoted == 1
    assert _snapshot(paths.root) == before


def test_merge_closes_only_still_open_relations(tmp_path: Path) -> None:
    """An already-closed span keeps its own valid_until."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/dana-scott.md", _named_note("Dana Scott"))
    _write(
        paths, "knowledge/people/danaexamplecom.md",
        "---\n"
        "title: dana@example.com\n"
        "type: person\n"
        "relations:\n"
        "- rel: works_at\n"
        "  target: organisations/acme\n"
        "  valid_from: '2025-01-01'\n"
        "  valid_until: '2025-12-31'\n"
        "- rel: attended\n"
        f"  target: {_MEETING}\n"
        "  valid_from: '2026-07-02'\n"
        "---\n\n## Log\n",
    )
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "merge-dana.md")

    consolidate(paths, logger=_LOG, as_of=AS_OF)

    dup_fm, _ = _split_frontmatter(_read(paths, "knowledge/people/danaexamplecom.md"))
    assert dup_fm["relations"] == [
        {
            "rel": "works_at", "target": "organisations/acme",
            "valid_from": "2025-01-01", "valid_until": "2025-12-31",
        },
        {
            "rel": "attended", "target": _MEETING,
            "valid_from": "2026-07-02", "valid_until": "2026-06-12",
        },
    ]
    # Both spans copied to the survivor; the closed one keeps its own date
    # and the open one arrives still open.
    survivor_fm, _ = _split_frontmatter(_read(paths, "knowledge/people/dana-scott.md"))
    assert survivor_fm["relations"] == [
        {
            "rel": "works_at", "target": "organisations/acme",
            "valid_from": "2025-01-01", "valid_until": "2025-12-31",
        },
        {"rel": "attended", "target": _MEETING, "valid_from": "2026-07-02"},
    ]


def test_merge_appends_alias_beside_existing_ones(tmp_path: Path) -> None:
    """A survivor that already has aliases keeps them, and a re-merge does
    not add the address twice."""
    paths = _vault(tmp_path)
    _write(
        paths, "knowledge/people/dana-scott.md",
        "---\ntitle: Dana Scott\ntype: person\naliases:\n- D. Scott\n---\n\n## Log\n",
    )
    _write(paths, "knowledge/people/danaexamplecom.md", _email_note("dana@example.com"))
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "merge-dana.md")

    consolidate(paths, logger=_LOG, as_of=AS_OF)

    fm, _ = _split_frontmatter(_read(paths, "knowledge/people/dana-scott.md"))
    assert fm["aliases"] == ["D. Scott", "dana@example.com"]


# -- refusals ---------------------------------------------------------------
# Each case seeds a vault and returns the fact note text; the assertions are
# shared, because a refused merge must leave both entity notes AND the fact
# byte-for-byte as it found them.

def _refuse_survivor_missing(paths: VaultPaths) -> str:
    _write(paths, "knowledge/people/danaexamplecom.md", _email_note("dana@example.com"))
    return _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott")


def _refuse_survivor_superseded(paths: VaultPaths) -> str:
    _seed_pair(paths, *_LIVE_PAIRS[0])
    _write(
        paths, "knowledge/people/dana-scott.md",
        "---\ntitle: Dana Scott\ntype: person\n"
        "superseded_by: knowledge/people/dana-scott-2\n---\n\n## Log\n",
    )
    return _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott")


def _refuse_same_node(paths: VaultPaths) -> str:
    _seed_pair(paths, *_LIVE_PAIRS[0])
    return _merge_fact(duplicate="people/dana-scott", survivor="people/dana-scott")


def _refuse_duplicate_not_a_node(paths: VaultPaths) -> str:
    _seed_pair(paths, *_LIVE_PAIRS[0])
    _write(paths, "knowledge/concepts/graphs.md", "---\ntitle: Graphs\n---\n\n## Log\n")
    return _merge_fact(duplicate="concepts/graphs", survivor="people/dana-scott")


def _refuse_survivor_not_a_node(paths: VaultPaths) -> str:
    _seed_pair(paths, *_LIVE_PAIRS[0])
    _write(paths, "knowledge/concepts/graphs.md", "---\ntitle: Graphs\n---\n\n## Log\n")
    return _merge_fact(duplicate="people/danaexamplecom", survivor="concepts/graphs")


def _refuse_survivor_is_not_the_target(paths: VaultPaths) -> str:
    _seed_pair(paths, *_LIVE_PAIRS[0])
    _seed_pair(paths, *_LIVE_PAIRS[1])
    return _merge_fact(
        duplicate="people/danaexamplecom", survivor="people/dana-scott",
        target="people/milo-frey",
    )


def _refuse_duplicate_note_missing(paths: VaultPaths) -> str:
    _write(paths, "knowledge/people/dana-scott.md", _named_note("Dana Scott"))
    return _merge_fact(duplicate="people/ghostexamplecom", survivor="people/dana-scott")


def _refuse_merge_not_a_mapping(paths: VaultPaths) -> str:
    _seed_pair(paths, *_LIVE_PAIRS[0])
    return _merge_fact(
        duplicate="people/danaexamplecom", survivor="people/dana-scott",
        merge_block="  merge: people/danaexamplecom\n",
    )


def _refuse_merge_unknown_key(paths: VaultPaths) -> str:
    _seed_pair(paths, *_LIVE_PAIRS[0])
    return _merge_fact(
        duplicate="people/danaexamplecom", survivor="people/dana-scott",
        merge_block=(
            "  merge:\n    duplicat: people/danaexamplecom\n"
            "    survivor: people/dana-scott\n"
        ),
    )


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (_refuse_survivor_missing, "resolves to no note"),
        (_refuse_survivor_superseded, "is itself superseded"),
        (_refuse_same_node, "as both duplicate and survivor"),
        (_refuse_duplicate_not_a_node, "duplicate 'concepts/graphs' is not an entity node"),
        (_refuse_survivor_not_a_node, "survivor 'concepts/graphs' is not an entity node"),
        (_refuse_survivor_is_not_the_target, "is not promote.target"),
        (_refuse_duplicate_note_missing, "duplicate 'people/ghostexamplecom' resolves to no note"),
        (_refuse_merge_not_a_mapping, "promote.merge is a str, not a mapping"),
        (_refuse_merge_unknown_key, "unknown key(s) duplicat"),
    ],
    ids=[
        "survivor-missing", "survivor-superseded", "duplicate-is-survivor",
        "duplicate-not-a-node", "survivor-not-a-node", "survivor-is-not-target",
        "duplicate-note-missing", "merge-not-a-mapping", "merge-unknown-key",
    ],
)
def test_merge_refusal_leaves_everything_untouched(
    setup: Callable[[VaultPaths], str], expected: str, tmp_path: Path
) -> None:
    paths = _vault(tmp_path)
    _write(paths, f"{_INBOX}/merge-dana.md", setup(paths))
    _approve(paths, "merge-dana.md")
    before = _snapshot(paths.root)

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 0
    assert stats.unresolved == 1
    assert _snapshot(paths.root) == before
    assert any(expected in problem for problem in stats.problems), stats.problems
