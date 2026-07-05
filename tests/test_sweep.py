"""Vault linter: one fixture vault exhibiting every finding category
exactly once, plus the graceful degradations (no embeddings index ->
drift checks skipped) and the deterministic report rendering."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.hashing import sha256_of
from ingest_lib.metadata import IndexRecord, append_record
from ingest_lib.sweep import SweepReport, render_report, run_sweep

_LOG = logging.getLogger("test")
AS_OF = date(2026, 6, 12)

EXPECTED_CATEGORIES = (
    "archive-orphan-file",
    "archive-orphan-record",
    "concept-fragmentation",
    "dangling-wikilink",
    "index-drift-missing",
    "index-drift-stale",
    "index-drift-unindexed",
    "missing-artifact",
    "relation-bad-date",
    "relation-dangling-target",
    "relation-inverted-interval",
    "relation-overlap",
    "relation-problem",
    "stale-unconsolidated",
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

    # relation-* findings, all on one person note.
    anna = _write(tmp_path, "knowledge/people/anna.md", PERSON_NOTE)
    _write(tmp_path, "knowledge/organisations/acme.md", ORG_NOTE)

    # stale-unconsolidated.
    fact = _write(tmp_path, "knowledge/assistant/inbox/fact-old.md", STALE_FACT)

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
