"""Frontmatter serialization: YAML-safe, byte-stable for the common case.

Pins the fix that replaced repr()/f-string interpolation (which mangled
dates and could emit invalid YAML that the next rebuild silently dropped)
with the shared fm_scalar/fm_list helpers.
"""
from __future__ import annotations

import datetime
import os
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from ingest_lib.notes import fm_list, fm_scalar


def test_fm_scalar_leaves_plain_strings_unquoted() -> None:
    # Byte-stable: plain-safe strings render exactly as the old f-string did.
    for s in ["Beta", "Requirements Engineering", "Behaviour-Driven Development"]:
        assert fm_scalar(s) == s


def test_fm_scalar_quotes_yaml_hostile_strings_validly() -> None:
    for s in ["TCP: Congestion Control", "[bracket", "@handle", "a #b"]:
        rendered = fm_scalar(s)
        assert yaml.safe_load(f"title: {rendered}")["title"] == s


def test_fm_scalar_renders_dates_as_iso_not_repr() -> None:
    d = datetime.date(2026, 6, 9)
    assert fm_scalar(d) == "2026-06-09"
    assert "datetime.date" not in fm_scalar(d)
    assert yaml.safe_load(f"reviewed: {fm_scalar(d)}")["reviewed"] == d


def test_fm_list_empty_and_populated_and_coercion() -> None:
    assert fm_list([]) == "[]"                       # byte-stable with prior "[]"
    assert yaml.safe_load(f"aliases: {fm_list(['a', 'b'])}")["aliases"] == ["a", "b"]
    # A bare string coerces to a one-element list rather than being dropped.
    assert yaml.safe_load(f"aliases: {fm_list('solo')}")["aliases"] == ["solo"]
    # A non-list/str becomes [].
    assert fm_list(None) == "[]"


def test_bom_before_the_fence_still_parses_as_frontmatter() -> None:
    """AUD-045: an editor that writes a UTF-8 BOM would otherwise make the
    whole YAML block invisible, so the note loses its topics and reads as
    frontmatter-less to the provenance stamper."""
    from ingest_lib.notes import _split_frontmatter

    text = "﻿---\ntitle: x\ntopics: [alpha]\n---\n\nbody\n"
    fm, body = _split_frontmatter(text)
    assert fm["title"] == "x"
    assert fm["topics"] == ["alpha"]
    assert body.strip() == "body"


def test_frontmatter_state_distinguishes_absent_from_unparseable() -> None:
    """AUD-078: `_split_frontmatter` collapses "no fence" and "fence present
    but broken" to the same `({}, text)`. `frontmatter_state` must tell them
    apart so a caller can treat a broken note as a problem, not as bare."""
    from ingest_lib.notes import _split_frontmatter, frontmatter_state

    no_fence = "# just a heading\n\nbody\n"
    bad_yaml = "---\ntitle: [unterminated\n---\n\nbody\n"
    not_a_mapping = "---\n- just\n- a\n- list\n---\n\nbody\n"
    good = "---\ntitle: x\n---\n\nbody\n"

    assert frontmatter_state(no_fence) == "absent"
    assert frontmatter_state(bad_yaml) == "unparseable"
    assert frontmatter_state(not_a_mapping) == "unparseable"
    assert frontmatter_state(good) == "ok"

    # `_split_frontmatter` itself is unchanged: both broken cases still
    # collapse to the same lenient ({}, text) result its many read-only
    # callers rely on.
    assert _split_frontmatter(no_fence) == ({}, no_fence)
    assert _split_frontmatter(bad_yaml) == ({}, bad_yaml)
    assert _split_frontmatter(not_a_mapping) == ({}, not_a_mapping)


# ---------------------------------------------------------------------------
# AUD-079 / AUD-108: the shared write + rendering helpers in ingest_lib.atomic
# and ingest_lib.notes. Kept here because both are note-serialization
# helpers and every writer in the vault now routes through them.
# ---------------------------------------------------------------------------

def test_interrupted_atomic_write_leaves_no_temp_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each writer used to clean up in `except Exception`, which does not
    catch KeyboardInterrupt: a Ctrl-C between mkstemp and os.replace left a
    `.note-*.md` / `.index-*.jsonl` / `.snap-*` / `.<connector>-*.json` in
    the vault for Obsidian to index and `git status` to show."""
    from ingest_lib import metadata
    from ingest_lib.config import paths_for_root
    from ingest_lib.connectors import runner, state
    from ingest_lib.notes import _atomic_write

    paths = paths_for_root(tmp_path)
    record = metadata.IndexRecord(
        relative_path="a.md", source_hash="h", size_bytes=1, extension=".md",
        extractor="e", status="processed", raw_path="archive/raw/a.md",
        processed_path=None, index_note_path=None,
    )
    writers: dict[str, Callable[[], None]] = {
        "note": lambda: _atomic_write(tmp_path / "k" / "n.md", "x"),
        "index": lambda: metadata.append_record(tmp_path / "metadata" / "index.jsonl", record),
        "connector-state": lambda: state.save_state(paths, state.ConnectorState(name="zot")),
        "snapshot": lambda: runner._atomic_write_bytes(tmp_path / "inbox" / "s.json", b"x"),
    }

    def interrupt(src: object, dst: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)
    for label, write in writers.items():
        with pytest.raises(KeyboardInterrupt):
            write()
        strays = [p.name for p in tmp_path.rglob(".*") if p.is_file()]
        assert strays == [], f"{label} leaked {strays}"


def test_atomic_write_modes_are_unchanged_per_caller(tmp_path: Path) -> None:
    """Consolidating the seven copies must not change any caller's mode:
    notes keep mkstemp's 0600, connector snapshots/state keep the umask's."""
    from ingest_lib.config import paths_for_root
    from ingest_lib.connectors import runner, state
    from ingest_lib.notes import _atomic_write

    umask = os.umask(0o022)
    os.umask(umask)
    expected = 0o666 & ~umask

    _atomic_write(tmp_path / "n.md", "x")
    runner._atomic_write_bytes(tmp_path / "inbox" / "s.json", b"x")
    state.save_state(paths_for_root(tmp_path), state.ConnectorState(name="zot"))

    assert (tmp_path / "n.md").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "inbox" / "s.json").stat().st_mode & 0o777 == expected
    assert (
        tmp_path / "metadata" / "connectors" / "zot.json"
    ).stat().st_mode & 0o777 == expected


def test_md_cell_is_the_single_table_cell_escaper() -> None:
    """status and dashboards each had their own `_cell`; the status copy
    also wrapped in `str()`. One helper now, and it must still collapse
    newlines/tabs (which would break the row) and escape pipes."""
    from ingest_lib.notes import md_cell
    from ingest_lib.status import _table

    assert md_cell("a\nb\tc   d") == "a b c d"
    assert md_cell("a | b") == "a \\| b"
    # The status renderer routes every cell through it.
    assert _table(("h",), [("a | b\nc",)])[2] == "| a \\| b c |"


def test_vault_relative_fallback_dir_is_the_callers_own(tmp_path: Path) -> None:
    """concepts and dashboards shared a copy that differed only in the
    fallback literal; the parameter keeps each writer's own answer for a
    hand-built VaultPaths whose knowledge dir sits outside root."""
    from ingest_lib.concepts import _vault_relative
    from ingest_lib.config import paths_for_root

    paths = paths_for_root(tmp_path)
    outside = tmp_path.parent / "elsewhere" / "Alpha.md"

    assert _vault_relative(
        outside, paths, fallback_dir="knowledge/concepts"
    ) == "knowledge/concepts/Alpha.md"
    assert _vault_relative(
        outside, paths, fallback_dir="knowledge/index/entities"
    ) == "knowledge/index/entities/Alpha.md"
    inside = tmp_path / "knowledge" / "concepts" / "Beta.md"
    assert _vault_relative(
        inside, paths, fallback_dir="knowledge/concepts"
    ) == "knowledge/concepts/Beta.md"
