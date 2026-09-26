"""Tests for the report-only ``scripts/secret_scan.py`` CLI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from secret_scan import scan  # noqa: E402


def _vault(tmp_path: Path) -> Path:
    for d in ("knowledge", "archive/processed", "metadata", "archive/raw"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_finds_a_real_shaped_aws_key(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/leak.md").write_text("AKIA1234567890ABCDEF\n", encoding="utf-8")
    hits = scan(root, include_raw=False)
    assert hits == ["knowledge/leak.md:1: aws-access-key"]


def test_catches_the_same_key_inside_quotes(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/leak.md").write_text('key = "AKIA1234567890ABCDEF"\n', encoding="utf-8")
    hits = scan(root, include_raw=False)
    assert len(hits) == 1
    assert "aws-access-key" in hits[0]


def test_ignores_an_already_redacted_placeholder(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/ok.md").write_text("[REDACTED:aws-access-key]\n", encoding="utf-8")
    assert scan(root, include_raw=False) == []


def test_ignores_a_git_sha(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/ok.md").write_text(
        "e83c5163316f89bfbde7d9ab23ca2e25604af290\n", encoding="utf-8"
    )
    assert scan(root, include_raw=False) == []


def test_ignores_a_github_url(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/ok.md").write_text(
        "https://github.com/AntoniWuczynski/brain-vs-memor\n", encoding="utf-8"
    )
    assert scan(root, include_raw=False) == []


def test_ignores_ordinary_prose(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/ok.md").write_text("Basic photolithography process\n", encoding="utf-8")
    assert scan(root, include_raw=False) == []


def test_denylist_catches_an_odd_shaped_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAIN_SECRET_DENYLIST", "not-a-normal-shape-but-secret")
    root = _vault(tmp_path)
    (root / "knowledge/ok.md").write_text(
        "value: not-a-normal-shape-but-secret\n", encoding="utf-8"
    )
    hits = scan(root, include_raw=False)
    assert len(hits) == 1
    assert "known-secret" in hits[0]


def test_never_writes_the_secret_value(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/leak.md").write_text("AKIA1234567890ABCDEF\n", encoding="utf-8")
    hits = scan(root, include_raw=False)
    joined = "\n".join(hits)
    assert "AKIA1234567890ABCDEF" not in joined


def test_include_raw_skips_a_binary_file_without_crashing(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "archive/raw/blob.bin").write_bytes(b"\xff\xfe\x00\x01AKIA1234567890ABCDEF\xfe")
    hits = scan(root, include_raw=True)
    assert hits == []


def test_include_raw_scans_a_text_file(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "archive/raw/note.txt").write_text("AKIA1234567890ABCDEF\n", encoding="utf-8")
    hits = scan(root, include_raw=True)
    assert hits == ["archive/raw/note.txt:1: aws-access-key"]


def test_without_include_raw_ignores_archive_raw(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "archive/raw/note.txt").write_text("AKIA1234567890ABCDEF\n", encoding="utf-8")
    assert scan(root, include_raw=False) == []
