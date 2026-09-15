"""Vault linter: one fixture vault exhibiting every finding category
exactly once, plus the graceful degradations (no embeddings index ->
drift checks skipped) and the deterministic report rendering."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.hashing import sha256_of
from ingest_lib.metadata import IndexRecord, append_record
from ingest_lib.sweep import (
    SweepReport,
    find_duplicate_entity_pairs,
    render_report,
    run_sweep,
)

_LOG = logging.getLogger("test")
AS_OF = date(2026, 6, 12)

EXPECTED_CATEGORIES = (
    "alias-collision",
    "archive-orphan-file",
    "archive-orphan-record",
    "concept-fragmentation",
    "entity-duplicate",
    "dangling-wikilink",
    "dream-stalled",
    "index-drift-missing",
    "index-drift-stale",
    "index-drift-unindexed",
    "memory-unconsolidatable",
    "missing-artifact",
    "relation-bad-date",
    "relation-dangling-target",
    "relation-inverted-interval",
    "relation-overlap",
    "relation-problem",
    "stale-unconsolidated",
    "wikilink-ambiguous",
    "wikilink-not-canonical",
)

# Body wikilinks: one dangling, one self-link with anchor (resolves), one
# external URL (skipped), one embed of an existing asset (resolves).
# Topics rng + rngs on ONE note -> the fragmentation pair shares a source.
LINKS_NOTE = """---
title: Links
type: note
topics: [rng, rngs]
---

# Links

See [[knowledge/people/missing-person|Missing]], [[knowledge/notes/links#top]],
[[https://example.com]], and ![[archive/processed/asset.png]].
"""

# Short (Obsidian-style) links: `unique-stem` names exactly one note in
# the vault -> not canonical but resolvable; `twinned` names two ->
# ambiguous. Both are body links, so no relation/topic side effects.
SHORT_LINKS_NOTE = """---
title: Shortcuts
type: note
---

# Shortcuts

See [[unique-stem]] and [[twinned]].
"""

# Unconsolidated but undated, and outside consolidate's flat inbox: the
# staleness check cannot age it and consolidate never sees it (F13).
UNDATED_SESSION = """---
title: Session
type: note
memory_status: unconsolidated
---

A session note nothing can act on.
"""

# One relation entry per relation-finding category. Every dated entry
# targets organisations/acme (which exists) so only `ghost` dangles.
PERSON_NOTE = """---
title: Anna
type: person
relations:
  - rel: works_at
    target: organisations/acme
  - rel: employed_by
    target: organisations/acme
  - rel: member_of
    target: organisations/ghost
  - rel: collaborator_on
    target: organisations/acme
    valid_from: "March 2025"
  - rel: met_at
    target: organisations/acme
    valid_from: "2026-02-01"
    valid_until: "2026-01-01"
  - rel: attended
    target: organisations/acme
    valid_from: "2026-01-01"
    valid_until: "2026-03-01"
  - rel: attended
    target: organisations/acme
    valid_from: "2026-02-01"
---

# Anna
"""

# Two live notes claiming one alias: every lookup for `twin` is ambiguous.
TWIN_NOTE = """---
title: Twinned
type: note
aliases: [twin]
---

# twin
"""

# An attendee whose calendar payload carried an address, not a name — the
# same person as `dana-scott`, split into a second node.
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

ORG_NOTE = """---
title: ACME
type: organisation
---

# ACME
"""

# Unconsolidated for 72 days as of AS_OF -> stale with the default 30.
STALE_FACT = """---
title: Old fact
type: memory_fact
created: '2026-04-01T00:00:00Z'
memory_status: unconsolidated
---

An unconsolidated assistant fact awaiting review.
"""


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _record(rel: str, src_hash: str, **overrides: object) -> IndexRecord:
    base: dict[str, object] = dict(
        relative_path=rel,
        source_hash=src_hash,
        size_bytes=1,
        extension=".txt",
        extractor="text",
        status="processed",
        raw_path=f"archive/raw/{rel}",
        processed_path=f"archive/processed/{Path(rel).stem}.md",
        index_note_path=f"knowledge/index/{Path(rel).stem}.md",
    )
    base.update(overrides)
    return IndexRecord(**base)  # type: ignore[arg-type]


def _meta_row(src: str, src_hash: str, origin: str) -> dict[str, object]:
    return {
        "source_relative_path": src,
        "source_hash": src_hash,
        "title": src,
        "chunk_idx": 0,
        "text": "chunk text",
        "origin": origin,
        "model": "test-model",
    }


def _seed(tmp_path: Path) -> VaultPaths:
    """A vault where every category fires exactly once."""
    paths = paths_for_root(tmp_path)
    paths.ensure()

    # archive-orphan-file: a raw file no record references.
    _write(tmp_path, "archive/raw/orphan.txt", "orphan")

    # archive-orphan-record: gone.txt's raw file is missing, but its
    # processed + index artifacts exist (so no missing-artifact for it).
    append_record(paths.metadata_index_jsonl, _record("gone.txt", "a" * 64))
    _write(tmp_path, "archive/processed/gone.md", "# gone\n")
    _write(tmp_path, "knowledge/index/gone.md", "# gone\n")

    # missing-artifact: noproc.txt's raw + index note exist, processed
    # markdown is gone.
    append_record(paths.metadata_index_jsonl, _record("noproc.txt", "b" * 64))
    _write(tmp_path, "archive/raw/noproc.txt", "noproc")
    _write(tmp_path, "knowledge/index/noproc.md", "# noproc\n")

    # dangling-wikilink + concept-fragmentation host.
    links = _write(tmp_path, "knowledge/notes/links.md", LINKS_NOTE)
    _write(tmp_path, "archive/processed/asset.png", "png")

    # wikilink-not-canonical + wikilink-ambiguous.
    shortcuts = _write(tmp_path, "knowledge/notes/shortcuts.md", SHORT_LINKS_NOTE)
    unique = _write(tmp_path, "knowledge/research/unique-stem.md", "# unique\n")
    twin_a = _write(tmp_path, "knowledge/people/twinned.md", TWIN_NOTE)
    twin_b = _write(tmp_path, "knowledge/organisations/twinned.md", TWIN_NOTE)

    # entity-duplicate: an email-titled node beside the named one.
    email_person = _write(tmp_path, "knowledge/people/danaexamplecom.md", EMAIL_PERSON)
    named_person = _write(tmp_path, "knowledge/people/dana-scott.md", NAMED_PERSON)

    # memory-unconsolidatable.
    session = _write(
        tmp_path, "knowledge/assistant/sessions/undated.md", UNDATED_SESSION
    )

    # relation-* findings, all on one person note.
    anna = _write(tmp_path, "knowledge/people/anna.md", PERSON_NOTE)
    _write(tmp_path, "knowledge/organisations/acme.md", ORG_NOTE)

    # stale-unconsolidated.
    fact = _write(tmp_path, "knowledge/assistant/inbox/fact-old.md", STALE_FACT)

    # dream-stalled: the gate fired 11 days before AS_OF and nothing completed
    (paths.metadata / "dream.pending").write_text(
        json.dumps({"since": "2026-06-01T05:00:00Z"}) + "\n", encoding="utf-8"
    )

    # Embeddings meta: every source indexed with its current hash EXCEPT
    # links.md (wrong hash -> stale), vanished.txt (no backing -> missing),
    # and acme.md (no rows -> unindexed).
    rows = [
        _meta_row("gone.txt", "a" * 64, "text"),
        _meta_row("noproc.txt", "b" * 64, "text"),
        _meta_row("vanished.txt", "c" * 64, "text"),
        _meta_row("knowledge/notes/links.md", "0" * 64, "knowledge-note"),
        _meta_row("knowledge/people/anna.md", sha256_of(anna), "knowledge-note"),
        _meta_row(
            "knowledge/assistant/inbox/fact-old.md", sha256_of(fact), "knowledge-note"
        ),
        _meta_row("knowledge/notes/shortcuts.md", sha256_of(shortcuts), "knowledge-note"),
        _meta_row("knowledge/research/unique-stem.md", sha256_of(unique), "knowledge-note"),
        _meta_row("knowledge/people/twinned.md", sha256_of(twin_a), "knowledge-note"),
        _meta_row(
            "knowledge/organisations/twinned.md", sha256_of(twin_b), "knowledge-note"
        ),
        _meta_row(
            "knowledge/assistant/sessions/undated.md", sha256_of(session),
            "knowledge-note",
        ),
        _meta_row(
            "knowledge/people/danaexamplecom.md", sha256_of(email_person),
            "knowledge-note",
        ),
        _meta_row(
            "knowledge/people/dana-scott.md", sha256_of(named_person), "knowledge-note"
        ),
    ]
    assert sha256_of(links) != "0" * 64
    (paths.metadata / "embeddings_meta.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return paths


def test_each_category_fires_exactly_once(tmp_path: Path) -> None:
    paths = _seed(tmp_path)

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)

    assert report.counts == {c: 1 for c in EXPECTED_CATEGORIES}
    # Deterministic ordering: (category, path, detail).
    assert list(report.findings) == sorted(
        report.findings, key=lambda f: (f.category, f.path, f.detail)
    )

    by_category = {f.category: f for f in report.findings}
    assert by_category["archive-orphan-file"].path == "archive/raw/orphan.txt"
    assert by_category["archive-orphan-record"].path == "gone.txt"
    assert "archive/raw/gone.txt" in by_category["archive-orphan-record"].detail
    assert by_category["missing-artifact"].path == "noproc.txt"
    assert "processed_path" in by_category["missing-artifact"].detail
    assert by_category["dangling-wikilink"].path == "knowledge/notes/links.md"
    assert "knowledge/people/missing-person" in by_category["dangling-wikilink"].detail
    assert "unknown rel 'employed_by'" in by_category["relation-problem"].detail
    assert "organisations/ghost" in by_category["relation-dangling-target"].detail
    assert "'March 2025'" in by_category["relation-bad-date"].detail
    assert "met_at" in by_category["relation-inverted-interval"].detail
    assert "attended" in by_category["relation-overlap"].detail
    assert by_category["concept-fragmentation"].path == "knowledge/concepts/rng.md"
    assert "'rngs'" in by_category["concept-fragmentation"].detail
    assert by_category["index-drift-stale"].path == "knowledge/notes/links.md"
    assert by_category["index-drift-missing"].path == "vanished.txt"
    assert by_category["index-drift-unindexed"].path == "knowledge/organisations/acme.md"
    assert by_category["stale-unconsolidated"].path == (
        "knowledge/assistant/inbox/fact-old.md"
    )
    assert "72 day(s)" in by_category["stale-unconsolidated"].detail
    assert by_category["wikilink-not-canonical"].path == "knowledge/notes/shortcuts.md"
    assert "knowledge/research/unique-stem.md" in (
        by_category["wikilink-not-canonical"].detail
    )
    assert by_category["wikilink-ambiguous"].path == "knowledge/notes/shortcuts.md"
    assert "knowledge/organisations/twinned.md, knowledge/people/twinned.md" in (
        by_category["wikilink-ambiguous"].detail
    )
    assert by_category["entity-duplicate"].path == "knowledge/people/danaexamplecom.md"
    assert "knowledge/people/dana-scott.md" in by_category["entity-duplicate"].detail
    assert by_category["alias-collision"].path == "knowledge/organisations/twinned.md"
    assert "'twin'" in by_category["alias-collision"].detail
    assert by_category["memory-unconsolidatable"].path == (
        "knowledge/assistant/sessions/undated.md"
    )
    assert "created:" in by_category["memory-unconsolidatable"].detail
    assert by_category["dream-stalled"].path == "metadata/dream.pending"
    assert "11 days ago" in by_category["dream-stalled"].detail


def test_extensionless_link_to_binary_is_not_dangling(tmp_path: Path) -> None:
    """Generated index notes link binary sources with the extension
    stripped ([[archive/raw/x/y]] for y.pdf) — the vault-wide convention
    (AGENTS.md mandates extensionless wikilinks). Such links resolve via
    the <stem>.* sibling; a stem with no file behind it still fires."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(tmp_path, "archive/raw/uni/report.pdf", "%PDF")
    _write(
        tmp_path,
        "knowledge/index/report.md",
        "# report\n\nSource: [[archive/raw/uni/report]],"
        " but [[archive/raw/uni/nothing]] dangles.\n",
    )

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)

    dangling = [f for f in report.findings if f.category == "dangling-wikilink"]
    assert [f.detail for f in dangling] == [
        "[[archive/raw/uni/nothing]] resolves to no note or file"
    ]


def test_short_link_prefers_the_linking_notes_own_folder(tmp_path: Path) -> None:
    """F8/AUD-027: Obsidian resolves [[name]] by name across the vault,
    preferring the linking note's own folder — so a same-folder match is
    resolvable (not ambiguous) even when the name is used twice."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(tmp_path, "knowledge/projects/agent/floors.md", "# floors\n")
    _write(tmp_path, "knowledge/projects/kern/floors.md", "# floors\n")
    _write(
        tmp_path, "knowledge/projects/agent/plan.md",
        "# plan\n\nSee [[floors]] and [[log/2026-06-01]] and [[nowhere]].\n",
    )
    _write(tmp_path, "knowledge/projects/agent/log/2026-06-01.md", "# log\n")

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)

    by_category = {f.category: f for f in report.findings}
    assert "wikilink-ambiguous" not in by_category
    assert by_category["dangling-wikilink"].detail == (
        "[[nowhere]] resolves to no note or file"
    )
    # Both the same-folder name and the relative subpath resolve.
    not_canonical = sorted(
        f.detail for f in report.findings if f.category == "wikilink-not-canonical"
    )
    assert not_canonical == [
        "[[floors]] is a short link — Obsidian resolves it to "
        "knowledge/projects/agent/floors.md; AGENTS.md asks for the "
        "full vault-relative path",
        "[[log/2026-06-01]] is a short link — Obsidian resolves it to "
        "knowledge/projects/agent/log/2026-06-01.md; AGENTS.md asks for the "
        "full vault-relative path",
    ]


def test_archived_assistant_notes_are_not_relation_linted(tmp_path: Path) -> None:
    """AUD-095: entity_notes excludes assistant archive/ and digests/, so
    the sweep must too — otherwise the linter reports a broken graph edge
    that the graph itself will never contain."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    archived = (
        "---\n"
        "title: Promoted fact\n"
        "relations:\n"
        "  - rel: works_at\n"
        "    target: organisations/ghost\n"
        "---\n"
    )
    _write(tmp_path, "knowledge/assistant/archive/2026-04/done.md", archived)
    _write(tmp_path, "knowledge/assistant/digests/2026-04.md", archived)
    _write(tmp_path, "knowledge/assistant/inbox/live.md", archived)

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)

    dangling = [
        f for f in report.findings if f.category == "relation-dangling-target"
    ]
    assert [f.path for f in dangling] == ["knowledge/assistant/inbox/live.md"]


def test_inbox_note_consolidate_skips_is_reported(tmp_path: Path) -> None:
    """AUD-031: a note in the inbox with neither a promote mapping nor a
    memory_status is skipped by consolidate on every run and is invisible
    to the staleness check — the sweep is the only thing that can say so."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(tmp_path, "knowledge/assistant/inbox/stray.md", "# just a note\n")
    _write(tmp_path, "knowledge/assistant/inbox/fact.md", STALE_FACT)

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)

    unconsolidatable = [
        f for f in report.findings if f.category == "memory-unconsolidatable"
    ]
    assert [f.path for f in unconsolidatable] == [
        "knowledge/assistant/inbox/stray.md"
    ]
    assert "consolidate skips it" in unconsolidatable[0].detail


def test_rerun_is_deterministic(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    first = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    second = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert first == second


def test_missing_embeddings_index_skips_drift_checks(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    (paths.metadata / "embeddings_meta.jsonl").unlink()

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)

    # A fresh clone has no index: no drift findings, no error — and every
    # other check still runs.
    assert not [f for f in report.findings if f.category.startswith("index-drift")]
    assert report.counts["relation-overlap"] == 1
    assert report.counts["archive-orphan-file"] == 1


def test_stale_days_and_as_of_control_staleness(tmp_path: Path) -> None:
    paths = _seed(tmp_path)

    relaxed = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=100)
    assert "stale-unconsolidated" not in relaxed.counts

    # 9 days after creation with a 5-day threshold: stale again.
    earlier = run_sweep(
        paths, logger=_LOG, as_of=date(2026, 4, 10), stale_days=5
    )
    assert earlier.counts["stale-unconsolidated"] == 1


def test_archived_and_digested_notes_are_not_stale(tmp_path: Path) -> None:
    """F8: consolidate moves digested/promoted notes into
    knowledge/assistant/archive/ (digested ones KEEP
    memory_status: unconsolidated) and writes digests under
    knowledge/assistant/digests/. The sweep must treat both as handled
    history, never a backlog — otherwise the two maintenance tools
    contradict each other. Only inbox/ is the actionable backlog."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    # The same stale, unconsolidated content in all three locations. Using
    # STALE_FACT everywhere proves the skip is path-based, not content-based:
    # archive/ and digests/ copies WOULD fire on content but must not.
    _write(tmp_path, "knowledge/assistant/inbox/live.md", STALE_FACT)
    _write(tmp_path, "knowledge/assistant/archive/2026-04/done.md", STALE_FACT)
    _write(tmp_path, "knowledge/assistant/digests/2026-04.md", STALE_FACT)

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)

    stale = [f for f in report.findings if f.category == "stale-unconsolidated"]
    assert [f.path for f in stale] == ["knowledge/assistant/inbox/live.md"]


def test_fresh_vault_is_clean(tmp_path: Path) -> None:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)
    assert report.findings == ()
    assert report.counts == {}


def test_render_report_groups_and_counts(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)

    text = render_report(report, as_of=AS_OF)

    assert text.startswith(
        "---\n"
        "title: Vault sweep report\n"
        "type: report\n"
        "updated: '2026-06-12T00:00:00Z'\n"
        "counts:\n"
    )
    assert "  relation-overlap: 1" in text
    assert "  stale-unconsolidated: 1" in text
    assert "## archive-orphan-file (1)" in text
    assert "- `archive/raw/orphan.txt` — no index.jsonl record references" in text
    assert "## dangling-wikilink (1)" in text
    # Same inputs, same bytes.
    assert render_report(report, as_of=AS_OF) == text


def test_render_report_clean_vault() -> None:
    text = render_report(SweepReport(findings=()), as_of=AS_OF)
    assert "counts: {}" in text
    assert "_(no findings)_" in text


def test_undated_supersede_flow_is_not_flagged_as_overlap(tmp_path: Path) -> None:
    # F057: the documented supersede pattern (undated open -> close with a
    # valid_until -> undated reopen) produces two open-start entries; the
    # overlap check must NOT flag them, since an open start can't be located
    # on the calendar to prove any overlap.
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    _write(paths.root, "knowledge/organisations/acme.md", "---\ntitle: Acme\n---\n")
    _write(
        paths.root, "knowledge/people/anna.md",
        "---\n"
        "title: Anna\n"
        "relations:\n"
        "  - rel: works_at\n"
        "    target: organisations/acme\n"
        "    valid_until: \"2026-01-01\"\n"   # closed span, undated start
        "  - rel: works_at\n"
        "    target: organisations/acme\n"     # reopened, undated
        "---\n",
    )
    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert report.counts.get("relation-overlap", 0) == 0


def test_dated_overlap_is_still_flagged(tmp_path: Path) -> None:
    # The counterpart: two entries with CONCRETE overlapping starts are real.
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    _write(paths.root, "knowledge/organisations/acme.md", "---\ntitle: Acme\n---\n")
    _write(
        paths.root, "knowledge/people/anna.md",
        "---\n"
        "title: Anna\n"
        "relations:\n"
        "  - rel: works_at\n"
        "    target: organisations/acme\n"
        "    valid_from: \"2026-01-01\"\n"
        "    valid_until: \"2026-03-01\"\n"
        "  - rel: works_at\n"
        "    target: organisations/acme\n"
        "    valid_from: \"2026-02-01\"\n"
        "---\n",
    )
    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert report.counts.get("relation-overlap", 0) == 1


def _integrity_vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    good = _write(tmp_path, "archive/raw/good.txt", "the real bytes")
    bad = _write(tmp_path, "archive/raw/bad.txt", "original bytes")
    append_record(paths.metadata_index_jsonl, _record("good.txt", sha256_of(good)))
    # Record bad.txt's ORIGINAL hash, then corrupt the file on disk.
    orig_hash = sha256_of(bad)
    append_record(paths.metadata_index_jsonl, _record("bad.txt", orig_hash))
    _write(tmp_path, "archive/raw/bad.txt", "CORRUPTED bytes")
    # Give both processed/index artifacts so no missing-artifact noise.
    for stem in ("good", "bad"):
        _write(tmp_path, f"archive/processed/{stem}.md", "# x\n")
        _write(tmp_path, f"knowledge/index/{stem}.md", "# x\n")
    return paths


def test_integrity_check_flags_corrupted_raw_only_when_enabled(tmp_path: Path) -> None:
    paths = _integrity_vault(tmp_path)

    # Off by default: no re-hash, no archive-corrupt finding.
    default_run = run_sweep(paths, logger=_LOG, as_of=AS_OF)
    assert not any(f.category == "archive-corrupt" for f in default_run.findings)

    # Opt-in: the tampered file is flagged, the intact one is not.
    checked = run_sweep(paths, logger=_LOG, as_of=AS_OF, check_integrity=True)
    corrupt = [f for f in checked.findings if f.category == "archive-corrupt"]
    assert [f.path for f in corrupt] == ["bad.txt"]
    assert "immutable" in corrupt[0].detail


def test_archive_processed_size_tripwire(tmp_path: Path) -> None:
    from ingest_lib.sweep import _check_archive_processed_size
    paths = paths_for_root(tmp_path)
    paths.ensure()
    (paths.archive_processed / "a.md").parent.mkdir(parents=True, exist_ok=True)
    (paths.archive_processed / "a.md").write_bytes(b"x" * 2048)

    # Below a high threshold: silent.
    assert _check_archive_processed_size(paths, threshold_bytes=10**9) == []
    # Above a tiny threshold: one finding.
    findings = _check_archive_processed_size(paths, threshold_bytes=1024)
    assert [f.category for f in findings] == ["archive-processed-large"]
    assert "git-lfs" in findings[0].detail


def test_run_sweep_wires_archive_processed_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin the wiring in run_sweep (not just the helper): a big processed tree
    # produces the finding through the full sweep.
    import ingest_lib.sweep as sweep_mod
    monkeypatch.setattr(sweep_mod, "_PROCESSED_SIZE_THRESHOLD_BYTES", 1024)
    paths = paths_for_root(tmp_path)
    paths.ensure()
    (paths.archive_processed / "big.md").parent.mkdir(parents=True, exist_ok=True)
    (paths.archive_processed / "big.md").write_bytes(b"x" * 4096)

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)
    cats = {f.category for f in report.findings}
    assert "archive-processed-large" in cats


def test_dream_pending_fresh_marker_not_flagged(tmp_path: Path) -> None:
    """A pending marker under 2 days old is a normal in-flight run."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    (paths.metadata / "dream.pending").write_text(
        json.dumps({"since": "2026-06-11T05:00:00Z"}) + "\n", encoding="utf-8"
    )
    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)
    assert "dream-stalled" not in report.counts


def test_placeholder_wikilink_is_not_dangling(tmp_path: Path) -> None:
    """`Note Template.md` documents the note format with
    `[[archive/raw/<rel/path>]]`. An angle-bracketed segment can never name a
    file, so it is a placeholder, not a broken link (AUD-082)."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(
        tmp_path, "knowledge/index/Note Template.md",
        "---\ntitle: Note Template\n---\n\n- Source: [[archive/raw/<rel/path>]]\n",
    )

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)

    assert [f for f in report.findings if f.category == "dangling-wikilink"] == []


def test_entity_duplicate_clears_once_the_pair_is_merged(tmp_path: Path) -> None:
    """The vault merges by superseding, never deleting (AGENTS.md): the
    address becomes an alias on the named note, the two are joined by
    `related_to`, and the email node is marked `superseded_by`. The check
    must go quiet on all three shapes and stay loud without them."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    email = _write(tmp_path, "knowledge/people/danaexamplecom.md", EMAIL_PERSON)
    _write(tmp_path, "knowledge/people/dana-scott.md", NAMED_PERSON)

    def categories() -> list[str]:
        report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
        return [f.category for f in report.findings if f.category == "entity-duplicate"]

    assert categories() == ["entity-duplicate"]

    email.write_text(
        "---\ntitle: dana@example.com\ntype: person\naliases: []\n"
        "superseded_by: knowledge/people/dana-scott\n---\n\n# dana@example.com\n",
        encoding="utf-8",
    )
    assert categories() == []


def test_entity_duplicate_accepts_an_alias_or_a_related_to_edge(tmp_path: Path) -> None:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(tmp_path, "knowledge/people/danaexamplecom.md", EMAIL_PERSON)
    named = _write(tmp_path, "knowledge/people/dana-scott.md", NAMED_PERSON)

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert [f.category for f in report.findings] == ["entity-duplicate"]

    named.write_text(
        "---\ntitle: Dana Scott\ntype: person\naliases: [dana@example.com]\n---\n\n"
        "# Dana Scott\n",
        encoding="utf-8",
    )
    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert [f for f in report.findings if f.category == "entity-duplicate"] == []

    named.write_text(
        "---\ntitle: Dana Scott\ntype: person\naliases: []\nrelations:\n"
        "  - rel: related_to\n    target: people/danaexamplecom\n---\n\n# Dana Scott\n",
        encoding="utf-8",
    )
    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert [f for f in report.findings if f.category == "entity-duplicate"] == []


def test_find_duplicate_entity_pairs_matches_the_sweep_finding(tmp_path: Path) -> None:
    """ingest_lib.duplicates builds proposals off this function directly, so
    it must report exactly the pair the entity-duplicate check flags."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(tmp_path, "knowledge/people/danaexamplecom.md", EMAIL_PERSON)
    _write(tmp_path, "knowledge/people/dana-scott.md", NAMED_PERSON)

    pairs = find_duplicate_entity_pairs(paths)
    assert len(pairs) == 1
    assert pairs[0].email_node.rel_path == "knowledge/people/danaexamplecom.md"
    assert [c.rel_path for c in pairs[0].named_candidates] == [
        "knowledge/people/dana-scott.md"
    ]

    named = paths.root / "knowledge/people/dana-scott.md"
    named.write_text(
        "---\ntitle: Dana Scott\ntype: person\naliases: [dana@example.com]\n"
        "---\n\n# Dana Scott\n",
        encoding="utf-8",
    )
    assert find_duplicate_entity_pairs(paths) == []


def test_alias_collision_ignores_notes_that_are_no_longer_live(tmp_path: Path) -> None:
    """Three project notes shared the alias `randeval` (AUD-087). Marking the
    two non-canonical ones complete/superseded is the fix, so the check must
    only count notes that are still live."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(
        tmp_path, "knowledge/projects/one/one.md",
        "---\ntitle: One\ntype: project\nstatus: active\naliases: [shared]\n---\n\n# One\n",
    )
    two = _write(
        tmp_path, "knowledge/projects/two/two.md",
        "---\ntitle: Two\ntype: project\nstatus: active\naliases: [Shared]\n---\n\n# Two\n",
    )

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    collisions = [f for f in report.findings if f.category == "alias-collision"]
    assert len(collisions) == 1
    assert collisions[0].path == "knowledge/projects/one/one.md"

    two.write_text(
        "---\ntitle: Two\ntype: project\nstatus: complete\naliases: [Shared]\n---\n\n"
        "# Two\n",
        encoding="utf-8",
    )
    report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
    assert [f for f in report.findings if f.category == "alias-collision"] == []


def test_fragmentation_skips_reviewed_and_version_numbered_pairs(tmp_path: Path) -> None:
    """Four of the five nightly `concept-fragmentation` findings were
    distinct concepts (AUD-086): version pairs are excluded by shape, and a
    reviewed pair is silenced by `distinct_from:` on either note."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(
        tmp_path, "knowledge/notes/topics.md",
        "---\ntitle: Topics\ntype: note\n"
        "topics: [android-16, android-17, csprng, prng]\n---\n\n# Topics\n",
    )

    def pairs() -> list[str]:
        report = run_sweep(paths, logger=_LOG, as_of=AS_OF, stale_days=30)
        return [
            f.detail for f in report.findings if f.category == "concept-fragmentation"
        ]

    # android-16/-17 differ only in a number -> never reported.
    assert len(pairs()) == 1
    assert "'csprng' and 'prng'" in pairs()[0]

    _write(
        tmp_path, "knowledge/concepts/csprng.md",
        "---\ntitle: CSPRNG\ntype: concept\ndistinct_from: [prng]\n---\n\n# CSPRNG\n",
    )
    assert pairs() == []


# `archive-source-local-only` needs a real repository to ask, so it lives
# here rather than in the one-of-everything fixture above.
_needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git_vault(root: Path, gitignore: str) -> VaultPaths:
    """A throwaway vault that is its own git repository (never the real one)."""
    paths = paths_for_root(root)
    paths.ensure()
    subprocess.run(
        ["git", "init", "-q", str(paths.root)], check=True, capture_output=True
    )
    _write(paths.root, ".gitignore", gitignore)
    return paths


def _absent_source(rel: str, src_hash: str) -> IndexRecord:
    """A record whose raw file is not on disk, and which owns no artifacts
    (so only the archive check can speak about it)."""
    return _record(rel, src_hash, processed_path=None, index_note_path=None)


@_needs_git
def test_gitignored_absent_source_is_its_own_category(tmp_path: Path) -> None:
    """IMP-020: six ELEC0031 slide decks exceed GitHub's per-file limit and
    .gitignore keeps them local, so every other checkout is missing them by
    design. That must read as its own category, not as six lost sources —
    while a source nothing excludes stays an orphan record."""
    paths = _git_vault(tmp_path, "/archive/raw/big deck.pdf\n")
    append_record(paths.metadata_index_jsonl, _absent_source("big deck.pdf", "a" * 64))
    append_record(paths.metadata_index_jsonl, _absent_source("lost.txt", "b" * 64))

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)

    assert report.counts == {
        "archive-orphan-record": 1,
        "archive-source-local-only": 1,
    }
    by_category = {f.category: f for f in report.findings}
    local_only = by_category["archive-source-local-only"]
    assert local_only.path == "big deck.pdf"
    # The space in the filename survives the NUL-delimited git exchange.
    assert "archive/raw/big deck.pdf" in local_only.detail
    assert "git-ignored" in local_only.detail
    assert by_category["archive-orphan-record"].path == "lost.txt"


@_needs_git
def test_tracked_absent_source_is_still_an_orphan_record(tmp_path: Path) -> None:
    """A file git tracks cannot be excused by an ignore rule that also
    matches it: git answers from the index first, so a committed source
    that vanished is still a real loss."""
    paths = _git_vault(tmp_path, "/archive/raw/tracked.pdf\n")
    raw = _write(paths.root, "archive/raw/tracked.pdf", "slides")
    subprocess.run(
        ["git", "-C", str(paths.root), "add", "-f", "archive/raw/tracked.pdf"],
        check=True, capture_output=True,
    )
    raw.unlink()
    append_record(paths.metadata_index_jsonl, _absent_source("tracked.pdf", "a" * 64))

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)

    assert report.counts == {"archive-orphan-record": 1}


def test_absent_source_outside_a_git_repo_is_still_reported(tmp_path: Path) -> None:
    """No repository to ask (an exported tree, a plain directory): the
    ignore rules cannot be trusted, so the absent source stays a finding."""
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(paths.root, ".gitignore", "/archive/raw/lost.txt\n")
    append_record(paths.metadata_index_jsonl, _absent_source("lost.txt", "a" * 64))

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)

    assert report.counts == {"archive-orphan-record": 1}


@_needs_git
def test_absent_git_binary_leaves_absent_sources_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git not installed: the sweep neither crashes nor excuses anything."""
    paths = _git_vault(tmp_path, "/archive/raw/lost.txt\n")
    append_record(paths.metadata_index_jsonl, _absent_source("lost.txt", "a" * 64))
    monkeypatch.setenv("PATH", "")

    report = run_sweep(paths, logger=_LOG, as_of=AS_OF)

    assert report.counts == {"archive-orphan-record": 1}


def test_gitignore_inbox_rule_is_anchored() -> None:
    """The drop-zone rule must ignore only the top-level inbox/. An unanchored
    ``inbox/`` also swallowed knowledge/assistant/inbox/, the fact inbox every
    proposer and agent writes into, so new proposals landed untracked."""
    root = Path(__file__).resolve().parents[1]
    rules = [ln.strip() for ln in (root / ".gitignore").read_text().splitlines()]
    assert "/inbox/" in rules
    assert "inbox/" not in rules
