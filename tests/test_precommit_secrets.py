"""Tests for the ``scripts/precommit_secrets.py`` commit tripwire."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from precommit_secrets import hits_in_diff  # noqa: E402


def test_added_line_with_a_credential_is_caught() -> None:
    diff = (
        "diff --git a/knowledge/leak.md b/knowledge/leak.md\n"
        "index 0000000..1111111 100644\n"
        "--- a/knowledge/leak.md\n"
        "+++ b/knowledge/leak.md\n"
        "@@ -0,0 +1 @@\n"
        "+AKIA1234567890ABCDEF\n"
    )
    hits = hits_in_diff(diff, [])
    assert hits == [("knowledge/leak.md", ["aws-access-key"])]


def test_credential_inside_quotes_is_caught() -> None:
    diff = (
        "--- a/knowledge/leak.md\n"
        "+++ b/knowledge/leak.md\n"
        '+key = "AKIA1234567890ABCDEF"\n'
    )
    hits = hits_in_diff(diff, [])
    assert hits == [("knowledge/leak.md", ["aws-access-key"])]


def test_already_redacted_placeholder_is_not_caught() -> None:
    diff = (
        "--- a/knowledge/ok.md\n"
        "+++ b/knowledge/ok.md\n"
        "+[REDACTED:aws-access-key]\n"
    )
    assert hits_in_diff(diff, []) == []


def test_git_sha_is_not_caught() -> None:
    diff = (
        "--- a/knowledge/ok.md\n"
        "+++ b/knowledge/ok.md\n"
        "+fixed in e83c5163316f89bfbde7d9ab23ca2e25604af290\n"
    )
    assert hits_in_diff(diff, []) == []


def test_github_url_is_not_caught() -> None:
    diff = (
        "--- a/knowledge/ok.md\n"
        "+++ b/knowledge/ok.md\n"
        "+see https://github.com/AntoniWuczynski/brain-vs-memor\n"
    )
    assert hits_in_diff(diff, []) == []


def test_prose_is_not_caught() -> None:
    diff = (
        "--- a/knowledge/ok.md\n"
        "+++ b/knowledge/ok.md\n"
        "+Basic photolithography process\n"
    )
    assert hits_in_diff(diff, []) == []


def test_removed_line_is_not_caught() -> None:
    diff = (
        "--- a/knowledge/ok.md\n"
        "+++ b/knowledge/ok.md\n"
        "-AKIA1234567890ABCDEF\n"
    )
    assert hits_in_diff(diff, []) == []


def test_file_header_itself_is_not_treated_as_content() -> None:
    diff = "+++ b/knowledge/leak.md\n"
    assert hits_in_diff(diff, []) == []


def test_hit_under_tests_dir_is_skipped() -> None:
    diff = (
        "--- a/tests/fixture.md\n"
        "+++ b/tests/fixture.md\n"
        "+AKIA1234567890ABCDEF\n"
    )
    assert hits_in_diff(diff, []) == []


def test_denylist_value_via_env_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_SECRET_DENYLIST", "not-a-normal-shape-but-secret")
    diff = (
        "--- a/knowledge/ok.md\n"
        "+++ b/knowledge/ok.md\n"
        "+value: not-a-normal-shape-but-secret\n"
    )
    hits = hits_in_diff(diff, ["not-a-normal-shape-but-secret"])
    assert hits == [("knowledge/ok.md", ["known-secret"])]


def test_added_line_whose_content_starts_with_plus_is_still_scanned() -> None:
    """A Markdown `+ item` bullet renders as `++ item` in a diff. Skipping
    every `++` line to dodge the `+++` header let a secret on one through."""
    diff = "+++ b/knowledge/list.md\n@@ -0,0 +1 @@\n++ key AKIA1234567890ABCDEF\n"
    assert hits_in_diff(diff, []) == [("knowledge/list.md", ["aws-access-key"])]
