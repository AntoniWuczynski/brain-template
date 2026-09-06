"""Tests for scripts/rotate_logs.py — size-triggered rotation of the MCP
telemetry streams and age-triggered compression of the dated per-run logs.

Constraints under test: an oversized target rotates to exactly one .gz
segment whose gunzipped bytes match the original (and the active path is
left absent), an under-threshold file is untouched, a missing file is a
no-op, dry-run writes nothing, and the --max-mb default honours
BRAIN_LOG_ROTATE_MB. For the run logs: an old one is gzipped in place and
kept (never deleted), a recent one is left alone, and the undated stdout
streams launchd holds open are never renamed.
"""
from __future__ import annotations

import gzip
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# scripts/ has no __init__.py and isn't installed as a package (only
# scripts/ingest_lib is); pin it onto sys.path so this file also runs
# standalone, mirroring tests/test_mcp_audit.py's treatment of mcp_server.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import rotate_logs  # noqa: E402
from ingest_lib.config import paths_for_root  # noqa: E402
from rotate_logs import _build_parser, rotate_if_large  # noqa: E402

_ONE_KB = 1024


def test_oversized_file_rotates_to_one_gz_with_matching_bytes(tmp_path: Path) -> None:
    target = tmp_path / "mcp-audit.jsonl"
    original = b'{"ts": "x"}\n' * 200  # comfortably over a 1 KiB threshold
    target.write_bytes(original)

    result = rotate_if_large(target, _ONE_KB)

    assert result is not None
    assert result.suffix == ".gz"
    assert not target.exists()  # active path left absent

    siblings = list(tmp_path.iterdir())
    assert siblings == [result]  # exactly one segment, the renamed .jsonl is gone

    with gzip.open(result, "rb") as fh:
        assert fh.read() == original


def test_under_threshold_file_untouched(tmp_path: Path) -> None:
    target = tmp_path / "mcp-audit.jsonl"
    content = b"small\n"
    target.write_bytes(content)

    result = rotate_if_large(target, _ONE_KB)

    assert result is None
    assert target.read_bytes() == content
    assert list(tmp_path.iterdir()) == [target]


def test_missing_file_is_noop(tmp_path: Path) -> None:
    target = tmp_path / "mcp-audit.jsonl"

    result = rotate_if_large(target, _ONE_KB)

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_dry_run_rotates_nothing(tmp_path: Path) -> None:
    target = tmp_path / "mcp-audit.jsonl"
    original = b'{"ts": "x"}\n' * 200
    target.write_bytes(original)

    result = rotate_if_large(target, _ONE_KB, dry_run=True)

    assert result is not None
    assert result.suffix == ".gz"
    assert not result.exists()  # nothing written to disk
    assert target.exists()
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_exactly_at_threshold_is_untouched(tmp_path: Path) -> None:
    target = tmp_path / "mcp-audit.jsonl"
    target.write_bytes(b"x" * _ONE_KB)

    result = rotate_if_large(target, _ONE_KB)

    assert result is None
    assert target.exists()


def test_max_mb_default_reads_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_LOG_ROTATE_MB", "7")
    args = _build_parser().parse_args([])
    assert args.max_mb == 7


def test_max_mb_default_without_env_is_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_LOG_ROTATE_MB", raising=False)
    args = _build_parser().parse_args([])
    assert args.max_mb == 10


def test_max_mb_flag_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_LOG_ROTATE_MB", "7")
    args = _build_parser().parse_args(["--max-mb", "3"])
    assert args.max_mb == 3


def test_cli_rotates_oversized_targets_under_vault_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # default_paths() resolves the real repo root regardless of cwd, so the
    # CLI is exercised in-process against a fixture vault via monkeypatch
    # (mirroring tests/test_evalmine.py's test_mine_log_cli_writes_candidates)
    # rather than as a subprocess, which would rotate the real repo's logs/.
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    row = b'{"ts": "x"}\n'
    big = row * 90_000  # ~1.08 MB, over a 1 MiB threshold
    (paths.logs / "mcp-audit.jsonl").write_bytes(big)
    (paths.logs / "mcp-access.jsonl").write_bytes(b"small\n")  # stays under 1 MiB
    monkeypatch.setattr(rotate_logs, "default_paths", lambda: paths)

    rc = rotate_logs.main(["--max-mb", "1"])

    assert rc == 0
    assert not (paths.logs / "mcp-audit.jsonl").exists()
    assert (paths.logs / "mcp-access.jsonl").read_bytes() == b"small\n"
    gz_files = sorted(paths.logs.glob("mcp-audit-*.jsonl.gz"))
    assert len(gz_files) == 1
    with gzip.open(gz_files[0], "rb") as fh:
        assert fh.read() == big


def test_orphaned_segment_is_finished_on_next_run(tmp_path: Path) -> None:
    # A crash between rename and gzip leaves the uncompressed segment behind.
    seg = tmp_path / "mcp-access-20260101T000000Z.jsonl"
    original = b'{"ts": "x"}\n' * 50
    seg.write_bytes(original)

    finished = rotate_logs.finish_orphaned_segments(tmp_path)

    gz = tmp_path / "mcp-access-20260101T000000Z.jsonl.gz"
    assert finished == [gz]
    assert not seg.exists()
    assert list(tmp_path.iterdir()) == [gz]  # no .tmp left over either
    with gzip.open(gz, "rb") as fh:
        assert fh.read() == original


def test_orphan_with_completed_gz_only_unlinks(tmp_path: Path) -> None:
    # A crash between the atomic .gz rename and the segment unlink: the .gz
    # is already whole, so recovery must only drop the uncompressed copy.
    seg = tmp_path / "mcp-audit-20260101T000000Z.jsonl"
    seg.write_bytes(b"raw\n")
    gz = tmp_path / "mcp-audit-20260101T000000Z.jsonl.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(b"raw\n")
    gz_bytes = gz.read_bytes()

    finished = rotate_logs.finish_orphaned_segments(tmp_path)

    assert finished == [gz]
    assert not seg.exists()
    assert gz.read_bytes() == gz_bytes  # completed segment byte-for-byte untouched


def test_orphan_recovery_dry_run_touches_nothing(tmp_path: Path) -> None:
    seg = tmp_path / "mcp-access-20260101T000000Z.jsonl"
    seg.write_bytes(b"raw\n")

    finished = rotate_logs.finish_orphaned_segments(tmp_path, dry_run=True)

    assert finished == [tmp_path / "mcp-access-20260101T000000Z.jsonl.gz"]
    assert list(tmp_path.iterdir()) == [seg]
    assert seg.read_bytes() == b"raw\n"


def test_active_streams_are_not_treated_as_orphans(tmp_path: Path) -> None:
    for name in ("mcp-access.jsonl", "mcp-audit.jsonl"):
        (tmp_path / name).write_bytes(b"active\n")

    assert rotate_logs.finish_orphaned_segments(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "mcp-access.jsonl", "mcp-audit.jsonl",
    ]


def test_same_second_collision_skips_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "mcp-audit.jsonl"
    original = b'{"ts": "x"}\n' * 200
    target.write_bytes(original)
    monkeypatch.setattr(rotate_logs, "_now_stamp", lambda: "20260101T000000Z")
    existing_gz = tmp_path / "mcp-audit-20260101T000000Z.jsonl.gz"
    existing_gz.write_bytes(b"do not clobber")

    result = rotate_if_large(target, _ONE_KB)

    assert result is None  # skipped, not overwritten; a later run gets a new ts
    assert target.read_bytes() == original
    assert existing_gz.read_bytes() == b"do not clobber"


def test_gzip_segment_is_durable_before_the_source_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The .gz becomes the only copy the moment the segment is unlinked, so
    its bytes (and the rename that publishes them) must reach the disk
    first: fsync the gzip, rename, fsync the directory, then unlink."""
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = Path.replace
    real_unlink = Path.unlink

    def spy_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def spy_replace(self: Path, target: str | Path) -> Path:
        events.append("replace")
        return real_replace(self, target)

    def spy_unlink(self: Path, missing_ok: bool = False) -> None:
        events.append("unlink")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(Path, "replace", spy_replace)
    monkeypatch.setattr(Path, "unlink", spy_unlink)

    seg = tmp_path / "mcp-audit-20260101T000000Z.jsonl"
    original = b'{"ts": "x"}\n' * 50
    seg.write_bytes(original)
    gz = tmp_path / "mcp-audit-20260101T000000Z.jsonl.gz"

    rotate_logs._gzip_segment(seg, gz)

    assert events == ["fsync", "replace", "fsync", "unlink"]
    assert not seg.exists()
    with gzip.open(gz, "rb") as fh:
        assert fh.read() == original


# ------------------------------------------- dated run logs (AUD-088)

_T0 = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def test_old_run_logs_are_compressed_in_place_and_kept(tmp_path: Path) -> None:
    old = tmp_path / "ingest-20260101T010203Z.log"
    body = b"INFO ingest: 3 files\n" * 20
    old.write_bytes(body)

    produced = rotate_logs.compress_run_logs(tmp_path, older_than_days=14, now=_T0)

    gz = tmp_path / "ingest-20260101T010203Z.log.gz"
    assert produced == [gz]
    assert not old.exists()
    assert list(tmp_path.iterdir()) == [gz]  # compressed, not deleted, no .tmp
    with gzip.open(gz, "rb") as fh:
        assert fh.read() == body


def test_recent_run_logs_are_left_alone(tmp_path: Path) -> None:
    recent = tmp_path / "sweep-20260903T020000Z.log"  # a day before _T0
    recent.write_bytes(b"findings: 0\n")

    assert rotate_logs.compress_run_logs(tmp_path, older_than_days=14, now=_T0) == []
    assert recent.read_bytes() == b"findings: 0\n"


def test_undated_stdout_streams_are_never_touched(tmp_path: Path) -> None:
    # launchd holds these open: renaming one strands the running writer on an
    # unlinked inode, so the stamp match must exclude them.
    for name in ("brain-mcp.launchd.log", "maintain.out.log", "dream.err.log"):
        (tmp_path / name).write_bytes(b"live\n")

    assert rotate_logs.compress_run_logs(tmp_path, older_than_days=0, now=_T0) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "brain-mcp.launchd.log", "dream.err.log", "maintain.out.log",
    ]


def test_suffixed_run_logs_compress_too(tmp_path: Path) -> None:
    # Real names in logs/: consolidate-<stamp>-dryrun.log, ingest-raw-<stamp>.log
    for name in ("consolidate-20260612T220229Z-dryrun.log", "ingest-raw-20260701T000000Z.log"):
        (tmp_path / name).write_bytes(b"x\n")

    produced = rotate_logs.compress_run_logs(tmp_path, older_than_days=14, now=_T0)

    assert [p.name for p in produced] == [
        "consolidate-20260612T220229Z-dryrun.log.gz",
        "ingest-raw-20260701T000000Z.log.gz",
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == [p.name for p in produced]


def test_run_log_compression_dry_run_writes_nothing(tmp_path: Path) -> None:
    old = tmp_path / "dream-20260101T010203Z.log"
    old.write_bytes(b"dreamt\n")

    produced = rotate_logs.compress_run_logs(
        tmp_path, older_than_days=14, now=_T0, dry_run=True
    )

    assert produced == [tmp_path / "dream-20260101T010203Z.log.gz"]
    assert list(tmp_path.iterdir()) == [old]
    assert old.read_bytes() == b"dreamt\n"


def test_run_log_with_completed_gz_only_drops_the_plain_copy(tmp_path: Path) -> None:
    old = tmp_path / "sweep-20260101T010203Z.log"
    old.write_bytes(b"raw\n")
    gz = tmp_path / "sweep-20260101T010203Z.log.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(b"raw\n")
    gz_bytes = gz.read_bytes()

    produced = rotate_logs.compress_run_logs(tmp_path, older_than_days=14, now=_T0)

    assert produced == [gz]
    assert not old.exists()
    assert gz.read_bytes() == gz_bytes


def test_impossible_stamp_is_skipped_not_crashed(tmp_path: Path) -> None:
    odd = tmp_path / "ingest-99991301T000000Z.log"
    odd.write_bytes(b"x\n")

    assert rotate_logs.compress_run_logs(tmp_path, older_than_days=14, now=_T0) == []
    assert odd.exists()


def test_compress_after_days_default_reads_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_LOG_COMPRESS_DAYS", "3")
    assert _build_parser().parse_args([]).compress_after_days == 3


def test_compress_after_days_default_without_env_is_fourteen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BRAIN_LOG_COMPRESS_DAYS", raising=False)
    assert _build_parser().parse_args([]).compress_after_days == 14


def test_cli_compresses_old_run_logs_under_vault_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    (paths.logs / "ingest-20260101T010203Z.log").write_bytes(b"old\n")
    (paths.logs / "brain-mcp.launchd.log").write_bytes(b"live\n")
    monkeypatch.setattr(rotate_logs, "default_paths", lambda: paths)

    rc = rotate_logs.main(["--compress-after-days", "1"])

    assert rc == 0
    assert not (paths.logs / "ingest-20260101T010203Z.log").exists()
    assert (paths.logs / "ingest-20260101T010203Z.log.gz").is_file()
    assert (paths.logs / "brain-mcp.launchd.log").read_bytes() == b"live\n"
