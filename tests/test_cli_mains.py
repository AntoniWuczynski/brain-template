"""F6: ``retry_partial``, ``backfill_summaries``, and the CLI ``main()``
entry points of scripts/sweep.py, consolidate.py, dream_gate.py and
rotate_logs.py — none of which had ever been called by the suite (only
``rotate_logs.main`` had partial coverage, in tests/test_rotate_logs.py).

``scripts/`` has no ``__init__.py`` and isn't installed as a package (only
``scripts/ingest_lib`` is), so the four CLI modules are pinned onto
``sys.path`` the same way tests/test_rotate_logs.py does. Every CLI reads
its vault root through ``ingest_lib.config.default_paths()`` at module
scope, so it is monkeypatched per test — mirroring
tests/test_rotate_logs.py::test_cli_rotates_oversized_targets_under_vault_root.

No real embedding model is loaded anywhere: ``retry_partial`` stubs the
extractor dispatch, and ``backfill_summaries`` stubs the summariser.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import consolidate  # noqa: E402
import dream_gate  # noqa: E402
import rotate_logs  # noqa: E402
import sweep  # noqa: E402

from ingest_lib import pipeline as pipeline_mod  # noqa: E402
from ingest_lib import status as status_mod  # noqa: E402
from ingest_lib.config import paths_for_root  # noqa: E402
from ingest_lib.extractors.base import ExtractionResult  # noqa: E402
from ingest_lib.metadata import IndexRecord, append_record, latest_records_by_path  # noqa: E402
from ingest_lib.summarize import SummaryResult  # noqa: E402

_LOG = logging.getLogger("test")


# --------------------------------------------------------------- retry_partial

def test_retry_partial_reprocesses_only_records_with_an_existing_raw_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()

    present_raw = paths.archive_raw / "uni" / "lecture.txt"
    present_raw.parent.mkdir(parents=True, exist_ok=True)
    present_raw.write_text("raw source text", encoding="utf-8")

    jsonl = paths.metadata_index_jsonl
    append_record(jsonl, IndexRecord(
        relative_path="uni/lecture.txt", source_hash="h-present", size_bytes=16,
        extension=".txt", extractor="pdf-pypdf-fallback", status="partial",
        raw_path="archive/raw/uni/lecture.txt",
        processed_path="archive/processed/uni/lecture.md", index_note_path=None,
    ))
    append_record(jsonl, IndexRecord(
        relative_path="uni/gone.txt", source_hash="h-gone", size_bytes=16,
        extension=".txt", extractor="pdf-pypdf-fallback", status="partial",
        raw_path="archive/raw/uni/gone.txt",  # never written to disk
        processed_path="archive/processed/uni/gone.md", index_note_path=None,
    ))
    lines_before = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines_before) == 2

    calls: list[str] = []

    def fake_dispatch(src: Path, *, relative_path: str) -> object:
        def fake_extractor(_src: Path, _assets_dir: Path) -> ExtractionResult:
            calls.append(relative_path)
            return ExtractionResult(
                status="processed", extractor="fake", markdown="# recovered\n\nbody text\n",
            )
        return fake_extractor

    monkeypatch.setattr(pipeline_mod, "dispatch_extractor", fake_dispatch)
    # Never load the real embedding model: run_ingest's post-write refresh
    # calls this unconditionally once anything was processed.
    monkeypatch.setattr(pipeline_mod, "_build_search_index", lambda _paths, *, logger: 0)

    stats = status_mod.retry_partial(paths, logger=_LOG, dry_run=False)

    assert stats.processed == 1
    assert calls == ["uni/lecture.txt"]

    latest = latest_records_by_path(jsonl)
    assert latest["uni/lecture.txt"].status == "processed"
    assert latest["uni/gone.txt"].status == "partial"  # untouched: no raw file
    # Exactly one new line appended (the reprocessed record); the untouched
    # record's original line is still there too.
    lines_after = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines_after) == 3


def test_retry_partial_is_a_noop_when_no_partial_record_has_a_raw_file(
    tmp_path: Path,
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path="uni/gone.txt", source_hash="h", size_bytes=1,
        extension=".txt", extractor="pdf-pypdf-fallback", status="partial",
        raw_path="archive/raw/uni/gone.txt", processed_path=None, index_note_path=None,
    ))

    stats = status_mod.retry_partial(paths, logger=_LOG, dry_run=False)

    assert stats == pipeline_mod.IngestStats()


# ----------------------------------------------------------- backfill_summaries

def test_backfill_summaries_fills_in_a_missing_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()

    processed_rel = "archive/processed/uni/lecture.md"
    processed_path = paths.root / processed_rel
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_text(
        "# Lecture\n\n"
        "> Source: `archive/raw/uni/lecture.pdf`  \n"
        "> Hash: `h123`  \n"
        "> Extractor: `pdf-mineru`  \n"
        "> Status: `processed`\n\n"
        "---\n\nBody content about TCP congestion control.\n\n---\n\n"
        "## Processing notes\n- n/a\n",
        encoding="utf-8",
    )
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path="archive/raw/uni/lecture.pdf", source_hash="h123", size_bytes=100,
        extension=".pdf", extractor="pdf-mineru", status="processed",
        raw_path="archive/raw/uni/lecture.pdf",
        processed_path=processed_rel, index_note_path=None,
        summary="", key_points=[], topics=[],
    ))

    fake_summary = SummaryResult(
        summary="A faithful summary of the TCP lecture.",
        key_points=["Slow start", "Congestion avoidance"],
        topics=["TCP Congestion Control"],
        notes=["summary: anthropic/claude-haiku-4-5"],
    )
    monkeypatch.setattr(pipeline_mod, "_summary_enabled", lambda: True)
    monkeypatch.setattr(pipeline_mod, "_summarize", lambda *a, **kw: fake_summary)

    stats = pipeline_mod.backfill_summaries(paths, logger=_LOG)

    assert stats.processed == 1
    assert stats.skipped == 0

    updated = latest_records_by_path(paths.metadata_index_jsonl)["archive/raw/uni/lecture.pdf"]
    assert updated.summary == fake_summary.summary
    assert updated.key_points == fake_summary.key_points
    assert updated.topics == fake_summary.topics


def test_backfill_summaries_disabled_reports_zero_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    monkeypatch.setattr(pipeline_mod, "_summary_enabled", lambda: False)

    stats = pipeline_mod.backfill_summaries(paths, logger=_LOG)

    assert stats == pipeline_mod.IngestStats()
    assert not paths.metadata_index_jsonl.exists()


# --------------------------------------------------------------------- sweep

def test_sweep_main_runs_clean_over_an_empty_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    monkeypatch.setattr(sweep, "default_paths", lambda: paths)

    rc = sweep.main([])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Sweep summary" in out


# --------------------------------------------------------------- consolidate

def test_consolidate_main_dry_run_over_an_empty_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    monkeypatch.setattr(consolidate, "default_paths", lambda: paths)

    rc = consolidate.main(["--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Consolidation summary (dry-run" in out
    assert "promoted   : 0" in out


# --------------------------------------------------------------- dream_gate

def test_dream_gate_main_dreams_on_first_run_with_a_committed_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    root = paths.root
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    note = root / "knowledge" / "notes" / "a.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# A\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "init"], check=True)

    monkeypatch.setattr(dream_gate, "default_paths", lambda: paths)

    rc = dream_gate.main([])

    assert rc == 0
    out = capsys.readouterr().out
    assert '"should_dream": true' in out


# --------------------------------------------------------------- rotate_logs

def test_rotate_logs_main_dry_run_reports_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    big = b'{"ts": "x"}\n' * 90_000  # ~1.08 MB, over a 1 MiB threshold
    audit_log = paths.logs / "mcp-audit.jsonl"
    audit_log.write_bytes(big)
    monkeypatch.setattr(rotate_logs, "default_paths", lambda: paths)

    rc = rotate_logs.main(["--max-mb", "1", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "would rotate" in out
    assert audit_log.read_bytes() == big  # dry-run: untouched
    assert not any(paths.logs.glob("*.gz"))
