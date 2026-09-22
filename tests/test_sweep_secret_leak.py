"""The `secret-leak` sweep check.

Redaction at the write boundaries stops new credentials entering the vault.
It says nothing about what is already there, and nothing about a note typed
straight into Obsidian, which never passes a writer at all. This check reads
what is on disk and reports.

It reports the file, the line and the kind. Never the value: a sweep report
that quotes the secret has simply moved it somewhere else, and
`knowledge/index/sweep-report.md` is written on every maintenance run.
"""
from __future__ import annotations

from pathlib import Path

from ingest_lib.sweep_checks_secrets import _check_secret_leak


def _vault(tmp_path: Path) -> Path:
    for d in ("knowledge", "archive/processed", "metadata"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_finds_a_credential_in_a_knowledge_note(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/leak.md").write_text("key AKIA1234567890ABCDEF\n", encoding="utf-8")
    findings = _check_secret_leak(root)
    assert len(findings) == 1
    assert findings[0].category == "secret-leak"
    assert findings[0].path == "knowledge/leak.md"


def test_never_quotes_the_secret_itself(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/leak.md").write_text("key AKIA1234567890ABCDEF\n", encoding="utf-8")
    detail = _check_secret_leak(root)[0].detail
    assert "AKIA1234567890ABCDEF" not in detail
    assert "aws-access-key" in detail
    assert "line 1" in detail


def test_ignores_an_already_redacted_placeholder(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/ok.md").write_text("key [REDACTED:aws-access-key]\n", encoding="utf-8")
    assert _check_secret_leak(root) == []


def test_ignores_a_git_sha_and_a_url(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/ok.md").write_text(
        "fixed in e83c5163316f89bfbde7d9ab23ca2e25604af290\n"
        "see https://github.com/AntoniWuczynski/brain-vs-memor\n",
        encoding="utf-8",
    )
    assert _check_secret_leak(root) == []


def test_covers_processed_and_metadata_too(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "archive/processed/a.md").write_text("sk-ant-api03-" + "a" * 24, encoding="utf-8")
    (root / "metadata/index.jsonl").write_text(
        '{"summary": "ghp_' + "b" * 36 + '"}\n', encoding="utf-8"
    )
    paths = {f.path for f in _check_secret_leak(root)}
    assert paths == {"archive/processed/a.md", "metadata/index.jsonl"}


def test_reports_each_line_once_not_each_pattern(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    (root / "knowledge/two.md").write_text(
        "AKIA1234567890ABCDEF\nsk-ant-api03-" + "c" * 24 + "\n", encoding="utf-8"
    )
    assert len(_check_secret_leak(root)) == 2
