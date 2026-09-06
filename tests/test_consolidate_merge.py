"""Merge-path tests for ingest_lib.consolidate — the ``promote.merge``
branches ``tests/test_consolidate.py`` has no room left for (that file is at
the 1000-line ceiling AGENTS.md sets, so these live in a sibling module
rather than pushing it over).

Everything here is about the refusals and the partial-failure paths: a merge
that cannot be performed must leave both entity notes and the fact note
byte-for-byte as it found them, and one that is performed halfway must still
report what it wrote. Same discipline as the parent module — no LLM, no
wall-clock, ``as_of`` injected, assertions on file bytes and stats.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

import ingest_lib.consolidate  # noqa: F401  (registers the module in sys.modules)
from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.consolidate import consolidate
from ingest_lib.notes import _atomic_write, _split_frontmatter

# ``ingest_lib/__init__.py`` re-exports the consolidate FUNCTION under the
# package attribute of that name, so ``from ingest_lib import consolidate``
# (and monkeypatch's dotted-string form) hands back the function, not the
# module. Take the module from sys.modules to patch a name inside it.
_CONSOLIDATE_MOD = sys.modules["ingest_lib.consolidate"]

_LOG = logging.getLogger("test")
AS_OF = date(2026, 6, 12)

_INBOX = "knowledge/assistant/inbox"
_MEETING = "meetings/2026/2026-07-02-offsite"


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    for sub in (_INBOX, "knowledge/assistant/archive", "knowledge/people"):
        (paths.root / sub).mkdir(parents=True, exist_ok=True)
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _read(paths: VaultPaths, rel: str) -> str:
    return (paths.root / rel).read_text(encoding="utf-8")


def _approve(paths: VaultPaths, *names: str) -> None:
    for name in names:
        note = paths.root / _INBOX / name
        text = note.read_text(encoding="utf-8")
        note.write_text(
            text.replace("approved: false", "approved: true", 1), encoding="utf-8"
        )


def _snapshot(root: Path) -> dict[str, bytes | None]:
    out: dict[str, bytes | None] = {}
    for p in sorted(root.rglob("*")):
        out[p.relative_to(root).as_posix()] = p.read_bytes() if p.is_file() else None
    return out


def _named_note(name: str) -> str:
    return (
        f'---\ntitle: "{name}"\ntype: person\naliases: []\ntopics: []\n'
        "relations: []\n---\n\n## Log\n"
    )


def _email_note(email: str) -> str:
    return (
        f"---\ntitle: {email}\ntype: person\naliases: []\ntopics: []\n"
        f"relations:\n- rel: attended\n  target: {_MEETING}\n"
        f"  valid_from: '2026-07-02'\n  source: knowledge/{_MEETING}\n"
        "---\n\n## Log\n"
    )


def _merge_fact(
    *,
    duplicate: str,
    survivor: str,
    target: str | None = None,
    merge_block: str | None = None,
    fact: str = "",
) -> str:
    """A memory fact carrying ``promote.merge`` — the shape
    ``ingest_lib.duplicates`` proposes. ``merge_block`` overrides the mapping
    verbatim so a malformed one can be tested."""
    block = merge_block if merge_block is not None else (
        f"  merge:\n    duplicate: {duplicate}\n    survivor: {survivor}\n"
    )
    return (
        "---\n"
        'title: "merge candidate"\n'
        "type: memory_fact\n"
        "created: '2026-06-10'\n"
        "memory_status: unconsolidated\n"
        "confirmations: 0\n"
        "approved: false\n"
        "promote:\n"
        + block
        + f"  target: {target if target is not None else survivor}\n"
        + "  relations: []\n"
        + f'  fact: "{fact}"\n'
        + f"  source: knowledge/{duplicate}\n"
        + "---\n\nProposed by the deterministic duplicate-entity pass.\n"
    )


def _seed_pair(paths: VaultPaths) -> None:
    _write(paths, "knowledge/people/dana-scott.md", _named_note("Dana Scott"))
    _write(paths, "knowledge/people/danaexamplecom.md", _email_note("dana@example.com"))


# ---------------------------------------------------------------------------
# MAJOR: a duplicate already superseded by a DIFFERENT node
# ---------------------------------------------------------------------------

def test_merge_refused_when_duplicate_is_superseded_by_another_node(
    tmp_path: Path,
) -> None:
    """Two approved facts merge one duplicate into two different survivors.
    The first performs the merge; the second must be REFUSED, not counted as
    a successful merge — otherwise the second survivor's Log gains a line
    saying a merge happened while nothing about it did: no alias, no
    relation, and the duplicate still superseded by the first survivor."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    _write(paths, "knowledge/people/dana-smith.md", _named_note("Dana Smith"))
    _write(
        paths, f"{_INBOX}/a-merge.md",
        _merge_fact(
            duplicate="people/danaexamplecom", survivor="people/dana-scott",
            fact="merged dana@example.com into Dana Scott",
        ),
    )
    _write(
        paths, f"{_INBOX}/b-merge.md",
        _merge_fact(
            duplicate="people/danaexamplecom", survivor="people/dana-smith",
            fact="merged dana@example.com into Dana Smith",
        ),
    )
    _approve(paths, "a-merge.md", "b-merge.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 1
    assert stats.unresolved == 1
    assert any(
        "is already superseded by knowledge/people/dana-scott, not "
        "people/dana-smith — merge into that survivor instead" in problem
        for problem in stats.problems
    ), stats.problems
    # The refused half wrote nothing at all, and its fact waits in the inbox.
    assert _read(paths, "knowledge/people/dana-smith.md") == _named_note("Dana Smith")
    assert (paths.root / _INBOX / "b-merge.md").is_file()
    assert stats.touched_entity_paths == (
        "knowledge/people/dana-scott.md", "knowledge/people/danaexamplecom.md",
    )
    dup_fm, _ = _split_frontmatter(_read(paths, "knowledge/people/danaexamplecom.md"))
    assert dup_fm["superseded_by"] == "knowledge/people/dana-scott"


def test_merge_into_the_same_survivor_again_stays_an_idempotent_noop(
    tmp_path: Path,
) -> None:
    """The other half of the branch above: when the duplicate is already
    superseded by THIS survivor the merge is a real no-op — nothing written,
    no problem reported, and the fact still archives."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    _write(
        paths, f"{_INBOX}/a-merge.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "a-merge.md")
    consolidate(paths, logger=_LOG, as_of=AS_OF)
    after_first = {
        rel: _read(paths, rel)
        for rel in (
            "knowledge/people/dana-scott.md", "knowledge/people/danaexamplecom.md",
        )
    }

    _write(
        paths, f"{_INBOX}/b-merge.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "b-merge.md")
    stats = consolidate(paths, logger=_LOG, as_of=date(2026, 6, 20))

    assert (stats.promoted, stats.unresolved, stats.problems) == (1, 0, ())
    assert stats.touched_entity_paths == ()
    assert {rel: _read(paths, rel) for rel in after_first} == after_first


# ---------------------------------------------------------------------------
# half-applied merge: the survivor is written, the duplicate write fails
# ---------------------------------------------------------------------------

def test_half_applied_merge_still_reports_the_survivor_as_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The survivor is written first on purpose. When the duplicate write
    then fails, the survivor IS on disk with the copied relation, the alias
    and the fact line — so it must appear in ``touched_entity_paths`` or
    ``scripts/consolidate.py`` never reindexes a note this run rewrote."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(
            duplicate="people/danaexamplecom", survivor="people/dana-scott",
            fact="merged dana@example.com",
        ),
    )
    _approve(paths, "merge-dana.md")

    duplicate_path = paths.root / "knowledge/people/danaexamplecom.md"

    def _failing_write(path: Path, text: str) -> None:
        if path == duplicate_path:
            raise OSError("disk full")
        _atomic_write(path, text)

    # The name as consolidate.py bound it: ingest_lib.notes._atomic_write
    # itself is left alone.
    monkeypatch.setattr(_CONSOLIDATE_MOD, "_atomic_write", _failing_write)

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert stats.promoted == 0
    assert stats.unresolved == 1
    assert any("HALF-APPLIED" in problem for problem in stats.problems), stats.problems
    assert stats.touched_entity_paths == ("knowledge/people/dana-scott.md",)
    survivor_fm, survivor_body = _split_frontmatter(
        _read(paths, "knowledge/people/dana-scott.md")
    )
    assert survivor_fm["aliases"] == ["dana@example.com"]
    assert "merged dana@example.com" in survivor_body
    # The fact stays in the inbox so the next run finishes the merge.
    assert (paths.root / _INBOX / "merge-dana.md").is_file()
    dup_fm, _ = _split_frontmatter(_read(paths, "knowledge/people/danaexamplecom.md"))
    assert "superseded_by" not in dup_fm


# ---------------------------------------------------------------------------
# aliases: the key is matched quote-tolerantly
# ---------------------------------------------------------------------------

def test_quoted_aliases_key_is_replaced_not_duplicated(tmp_path: Path) -> None:
    """PyYAML round-trips ``"aliases": [Dan]`` as an ordinary aliases key. A
    bare startswith() match would miss it and append a SECOND aliases key to
    the same note."""
    paths = _vault(tmp_path)
    _write(
        paths, "knowledge/people/dana-scott.md",
        '---\ntitle: Dana Scott\ntype: person\n"aliases": [Dan]\ntopics: []\n'
        "---\n\n## Log\n",
    )
    _write(paths, "knowledge/people/danaexamplecom.md", _email_note("dana@example.com"))
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "merge-dana.md")

    consolidate(paths, logger=_LOG, as_of=AS_OF)

    text = _read(paths, "knowledge/people/dana-scott.md")
    fm, _body = _split_frontmatter(text)
    assert fm["aliases"] == ["Dan", "dana@example.com"]
    assert sum(1 for line in text.split("\n") if "aliases" in line) == 1


# ---------------------------------------------------------------------------
# refusals — each setup seeds a vault and returns the fact note's text; the
# assertions are shared, because a refused merge must leave the WHOLE tree
# byte-for-byte as it found it.
# ---------------------------------------------------------------------------

def _refuse_duplicate_superseded_elsewhere(paths: VaultPaths) -> str:
    _seed_pair(paths)
    _write(
        paths, "knowledge/people/danaexamplecom.md",
        _email_note("dana@example.com").replace(
            "---\n\n## Log\n",
            "superseded_by: knowledge/people/dana-smith\n---\n\n## Log\n",
        ),
    )
    return _merge_fact(
        duplicate="people/danaexamplecom", survivor="people/dana-scott"
    )


def _refuse_inverted_direction_by_title(paths: VaultPaths) -> str:
    _seed_pair(paths)
    return _merge_fact(
        duplicate="people/dana-scott", survivor="people/danaexamplecom"
    )


def _refuse_inverted_direction_by_slug(paths: VaultPaths) -> str:
    """The email node's title has since been edited to something friendlier,
    but its node id is still the slug of the address it keeps as an alias."""
    _write(paths, "knowledge/people/dana-scott.md", _named_note("Dana Scott"))
    _write(
        paths, "knowledge/people/danaexamplecom.md",
        "---\ntitle: Dana (calendar)\ntype: person\n"
        "aliases: [dana@example.com]\n---\n\n## Log\n",
    )
    return _merge_fact(
        duplicate="people/dana-scott", survivor="people/danaexamplecom"
    )


def _refuse_merge_null(paths: VaultPaths) -> str:
    _seed_pair(paths)
    return _merge_fact(
        duplicate="people/danaexamplecom", survivor="people/dana-scott",
        merge_block="  merge:\n",
    )


def _refuse_unparseable_survivor(paths: VaultPaths) -> str:
    _seed_pair(paths)
    _write(
        paths, "knowledge/people/dana-scott.md",
        '---\ntitle: fps — "Quarterly Review: a fever dream"\ntype: person\n'
        "---\n\n## Log\n",
    )
    return _merge_fact(
        duplicate="people/danaexamplecom", survivor="people/dana-scott"
    )


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (
            _refuse_duplicate_superseded_elsewhere,
            "already superseded by knowledge/people/dana-smith",
        ),
        (
            _refuse_inverted_direction_by_title,
            "survivor people/danaexamplecom is the email-slugged node for "
            "'dana@example.com' — the merge is the other way round: write "
            "duplicate: people/danaexamplecom, survivor: people/dana-scott",
        ),
        (
            _refuse_inverted_direction_by_slug,
            "is the email-slugged node for 'dana@example.com'",
        ),
        (_refuse_merge_null, "promote.merge is a NoneType, not a mapping"),
        (
            _refuse_unparseable_survivor,
            "promote.merge survivor note has unparseable frontmatter",
        ),
    ],
    ids=[
        "duplicate-superseded-elsewhere", "inverted-by-title",
        "inverted-by-slug", "merge-null", "unparseable-survivor",
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


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads unreadable files anyway")
def test_merge_refused_when_the_duplicate_note_is_unreadable(tmp_path: Path) -> None:
    """A duplicate note that exists but cannot be read: refuse, and leave
    the survivor and the fact alone rather than merging half a note."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    duplicate = paths.root / "knowledge/people/danaexamplecom.md"
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(duplicate="people/danaexamplecom", survivor="people/dana-scott"),
    )
    _approve(paths, "merge-dana.md")
    survivor_before = _read(paths, "knowledge/people/dana-scott.md")
    fact_before = _read(paths, f"{_INBOX}/merge-dana.md")
    duplicate.chmod(0o000)
    try:
        stats = consolidate(paths, logger=_LOG, as_of=AS_OF)
    finally:
        duplicate.chmod(0o644)

    assert stats.promoted == 0
    assert stats.unresolved == 1
    assert any(
        "unreadable" in problem and "danaexamplecom" in problem
        for problem in stats.problems
    ), stats.problems
    assert _read(paths, "knowledge/people/dana-scott.md") == survivor_before
    assert _read(paths, f"{_INBOX}/merge-dana.md") == fact_before


# ---------------------------------------------------------------------------
# a merge-shaped fact with NO promote.merge
# ---------------------------------------------------------------------------

def test_fact_describing_a_merge_without_promote_merge_merges_nothing(
    tmp_path: Path,
) -> None:
    """The shape the first version of the duplicate pass wrote into the live
    inbox: prose describing a completed merge, no ``promote.merge`` to
    execute. Approving it appends the Log line and NOTHING else — no alias,
    no relation copied, no superseded_by — which is exactly why an existing
    proposal has to be reconciled rather than left beside a newer one."""
    paths = _vault(tmp_path)
    _seed_pair(paths)
    duplicate_before = _read(paths, "knowledge/people/danaexamplecom.md")
    _write(
        paths, f"{_INBOX}/merge-dana.md",
        _merge_fact(
            duplicate="people/danaexamplecom", survivor="people/dana-scott",
            merge_block="", fact="merged dana@example.com into Dana Scott",
        ),
    )
    _approve(paths, "merge-dana.md")

    stats = consolidate(paths, logger=_LOG, as_of=AS_OF)

    assert (stats.promoted, stats.unresolved, stats.problems) == (1, 0, ())
    survivor_fm, survivor_body = _split_frontmatter(
        _read(paths, "knowledge/people/dana-scott.md")
    )
    assert survivor_fm["aliases"] == []
    assert survivor_fm["relations"] == []
    assert "merged dana@example.com into Dana Scott" in survivor_body
    assert _read(paths, "knowledge/people/danaexamplecom.md") == duplicate_before
