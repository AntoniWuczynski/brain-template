"""Tests for the MinerU availability banner (AUD-009).

``scripts/ingest.py`` should tell the operator up front, and in the run
summary, whether MinerU is on PATH — otherwise every PDF silently falls
back to pypdf and lands as ``status: partial`` with no warning.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

# scripts/ has no __init__.py and isn't installed as a package (only
# scripts/ingest_lib is); pin it onto sys.path so this file also runs
# standalone, mirroring tests/test_rotate_logs.py's treatment of scripts/.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import ingest  # noqa: E402
from ingest_lib.config import VaultPaths, paths_for_root  # noqa: E402
from ingest_lib.extractors import pdf as pdf_extractor  # noqa: E402


def test_mineru_available_true_when_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # pdf_extractor does `import shutil` and calls shutil.which(...) on the
    # same module object, so patching the real stdlib module here has the
    # identical effect while keeping shutil.which's real, typed signature.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/mineru")
    assert pdf_extractor.mineru_available() is True


def test_mineru_available_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert pdf_extractor.mineru_available() is False


def test_status_line_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/mineru")
    assert ingest._mineru_status_line() == "available"


def test_status_line_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    line = ingest._mineru_status_line()
    assert line.startswith("unavailable")
    assert "pypdf" in line
    assert "status: partial" in line
    assert "CLAUDE.md" in line


# ------------------------------------------------- the banner from main()
# AUD-104: the four tests above stop at the helpers. The user-facing fix is
# the log line and the summary line, and both live inside ``main`` behind a
# target-flag gate — so drive ``main`` itself over a throwaway vault.

def _tmp_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VaultPaths:
    """A vault under tmp_path with one ingestable file in its inbox.

    The real vault is never touched: ``default_paths`` is redirected, and the
    run is always ``--dry-run``. A non-empty inbox is required — ``main``
    returns at "nothing to do" before the run summary otherwise.
    """
    paths = paths_for_root(tmp_path)
    paths.ensure()
    (paths.inbox / "note.txt").write_text("a short note\n", encoding="utf-8")
    monkeypatch.setattr(ingest, "default_paths", lambda: paths)
    return paths


def _run_log(paths: VaultPaths) -> str:
    logs = sorted(paths.logs.glob("ingest-*.log"))
    assert logs, "the run wrote no log file"
    return logs[-1].read_text(encoding="utf-8")


def test_main_warns_and_summarises_when_mineru_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _tmp_vault(tmp_path, monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert ingest.main(["--dry-run", "--inbox"]) == 0

    log = _run_log(paths)
    assert "WARNING MinerU: unavailable" in log
    assert "pypdf" in log and "status: partial" in log
    # ...and again in the run summary the operator actually reads.
    assert "  MinerU        : unavailable" in capsys.readouterr().out


def test_main_logs_mineru_available_without_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _tmp_vault(tmp_path, monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/mineru")

    assert ingest.main(["--dry-run", "--inbox"]) == 0

    log = _run_log(paths)
    assert "INFO    MinerU: available" in log
    assert "MinerU: unavailable" not in log
    assert "  MinerU        : available" in capsys.readouterr().out
