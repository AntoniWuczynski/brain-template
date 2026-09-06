"""Tests for ingest_lib.duplicates (merge proposals for sweep's
entity-duplicate pairs). No LLM, no wall-clock quirks — mirrors
test_propose.py's wikilinks section for shape."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.duplicates import propose_duplicate_merges
from ingest_lib.notes import _split_frontmatter

_INBOX = "knowledge/assistant/inbox"


class _PromoteBlock(TypedDict):
    """The ``promote:`` mapping ``consolidate.py`` reads back off a proposal."""

    target: str
    relations: list[dict[str, str]]
    fact: str
    source: str


def _promote(frontmatter: Mapping[str, object]) -> _PromoteBlock:
    """The ``promote:`` block, shape-checked once so every assertion below
    reads a typed field instead of indexing YAML's untyped mapping."""
    block = frontmatter["promote"]
    assert isinstance(block, dict)
    assert isinstance(block.get("target"), str)
    assert isinstance(block.get("fact"), str)
    assert isinstance(block.get("source"), str)
    assert isinstance(block.get("relations"), list)
    return cast(_PromoteBlock, block)


EMAIL_PERSON = """---
title: dana@example.com
type: person
aliases: []
---

# dana@example.com
"""

NAMED_PERSON = """---
title: Dana Scott
type: person
aliases: []
---

# Dana Scott
"""


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> Path:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _inbox_notes(paths: VaultPaths) -> list[Path]:
    d = paths.root / _INBOX
    return sorted(d.glob("*.md")) if d.is_dir() else []


def _seed_pair(paths: VaultPaths) -> None:
    _write(paths, "knowledge/people/danaexamplecom.md", EMAIL_PERSON)
    _write(paths, "knowledge/people/dana-scott.md", NAMED_PERSON)


def test_propose_duplicate_merges_writes_a_candidate(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)

    results = propose_duplicate_merges(paths)
    assert [r.action for r in results] == ["proposed"]

    notes = _inbox_notes(paths)
    assert len(notes) == 1
    fm, body = _split_frontmatter(notes[0].read_text(encoding="utf-8"))
    promote = _promote(fm)
    assert promote["target"] == "people/dana-scott"
    assert promote["source"] == "people/danaexamplecom"
    assert promote["relations"] == []
    assert "people/dana-scott" in promote["fact"]
    assert "superseded_by" in promote["fact"]
    assert "consolidate.py cannot execute" in body


def test_propose_duplicate_merges_rerun_is_a_noop(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)

    propose_duplicate_merges(paths)
    before = {p.name: p.read_bytes() for p in _inbox_notes(paths)}

    second = propose_duplicate_merges(paths)
    assert [r.action for r in second] == ["exists"]
    after = {p.name: p.read_bytes() for p in _inbox_notes(paths)}
    assert after == before


def test_propose_duplicate_merges_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_pair(paths)

    results = propose_duplicate_merges(paths, dry_run=True)
    assert [r.action for r in results] == ["proposed"]
    assert not (paths.root / _INBOX).exists()


def test_propose_duplicate_merges_stops_once_pair_is_merged(tmp_path: Path) -> None:
    """Same suppression sweep uses: once the address is an alias on the
    named node, no proposal is written."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/danaexamplecom.md", EMAIL_PERSON)
    _write(
        paths, "knowledge/people/dana-scott.md",
        "---\ntitle: Dana Scott\ntype: person\naliases: [dana@example.com]\n"
        "---\n\n# Dana Scott\n",
    )

    results = propose_duplicate_merges(paths)
    assert results == []
    assert not (paths.root / _INBOX).exists()


def test_propose_duplicate_merges_no_pairs_is_empty(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/dana-scott.md", NAMED_PERSON)

    assert propose_duplicate_merges(paths) == []


def test_propose_duplicate_merges_skips_ambiguous_pairs(tmp_path: Path) -> None:
    """Two named nodes ('Andrew' and 'Andrew Basley') both name the local
    part of one email-slugged node: there is no single survivor to name as
    promote.target, so this pass proposes nothing and leaves it for a human
    (the sweep report still flags the pair)."""
    paths = _vault(tmp_path)
    _write(
        paths, "knowledge/people/andrewexamplecom.md",
        "---\ntitle: andrew@example.com\ntype: person\naliases: []\n---\n\n"
        "# andrew@example.com\n",
    )
    _write(
        paths, "knowledge/people/andrew.md",
        "---\ntitle: Andrew\ntype: person\naliases: []\n---\n\n# Andrew\n",
    )
    _write(
        paths, "knowledge/people/andrew-basley.md",
        "---\ntitle: Andrew Basley\ntype: person\naliases: []\n---\n\n"
        "# Andrew Basley\n",
    )

    assert propose_duplicate_merges(paths) == []
    assert not (paths.root / _INBOX).exists()
