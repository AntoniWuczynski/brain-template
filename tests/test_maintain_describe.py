"""maintain.sh's describe and open-loops steps, run against a stub
interpreter that records each call (the real passes are tested elsewhere),
plus ingest.py's ``--env-file`` loader that the describe step relies on."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = str(_REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import ingest  # noqa: E402
from ingest_lib.config import VaultPaths, paths_for_root  # noqa: E402
from ingest_lib.describe import DEFAULT_MIN_SOURCES, DescribeStats  # noqa: E402

_DESCRIBE_ARGS = "--describe-concepts --min-sources 3 --limit 50"
_GIT = shutil.which("git")

# A systemd EnvironmentFile (the same .env.ingest feeds brain-pull.service):
# an unquoted multi-word value, a secret holding `$`, and a line that would
# run a command if the file were ever evaluated as shell.
_SYSTEMD_ENV = (
    "BRAIN_DESCRIBE_PROVIDER=anthropic\n"
    "BRAIN_PULL_CONNECTORS=granola notion\n"
    "ANTHROPIC_API_KEY=sk-ab$cd\n"
    "PWN=$(touch pwned)\n"
)

# The stub interpreter: logs its argv; with --describe-concepts it writes a
# concept note (and exits $DESCRIBE_EXIT); the open-loops pass rewrites the
# Open Loops note. Writes go through the relative cwd, which maintain.sh sets
# to the repo root.
_STUB = """#!/bin/bash
echo "$*" >> "{calls}"
case "$*" in
    *--describe-concepts*)
        mkdir -p knowledge/concepts
        echo synthesis > knowledge/concepts/widgets.md
        exit "${{DESCRIBE_EXIT:-0}}" ;;
    "-m ingest_lib.open_loops")
        mkdir -p knowledge/index
        echo loops > "knowledge/index/Open Loops.md" ;;
esac
exit 0
"""


def _git(root: Path, *args: str) -> str:
    assert _GIT is not None
    return subprocess.run(
        [_GIT, "-C", str(root), *args], capture_output=True, text=True, check=True,
    ).stdout


def _run(
    tmp_path: Path, env_ingest: str | None, *args: str,
    describe_exit: int = 0, git: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    (tmp_path / "scripts").mkdir()
    shutil.copy(_REPO_ROOT / "scripts/maintain.sh", tmp_path / "scripts/maintain.sh")
    venv_bin = tmp_path / ".venv/bin"
    venv_bin.mkdir(parents=True)
    calls = tmp_path / "calls.log"
    python = venv_bin / "python"
    python.write_text(_STUB.format(calls=calls), encoding="utf-8")
    python.chmod(0o755)
    if env_ingest is not None:
        (tmp_path / ".env.ingest").write_text(env_ingest, encoding="utf-8")
    path = "/usr/bin:/bin"
    if git:
        assert _GIT is not None
        path = f"{Path(_GIT).parent}:{path}"
        _git(tmp_path, "init", "-q", "-b", "main")
        _git(tmp_path, "config", "user.email", "t@example.com")
        _git(tmp_path, "config", "user.name", "t")
        (tmp_path / ".gitignore").write_text(
            ".venv/\ncalls.log\n.env.ingest\n", encoding="utf-8",
        )
        (tmp_path / "knowledge").mkdir()
        (tmp_path / "knowledge/human.md").write_text("v1\n", encoding="utf-8")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", "init")
        # A human's pending edit, dirty before the run: must stay uncommitted.
        (tmp_path / "knowledge/human.md").write_text("v2\n", encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(tmp_path / "scripts/maintain.sh"), *args],
        capture_output=True, text=True, check=True, cwd=tmp_path,
        env={"PATH": path, "HOME": str(tmp_path), "DESCRIBE_EXIT": str(describe_exit)},
    )
    lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    return proc, lines


def _describe_call(tmp_path: Path) -> str:
    return f"scripts/ingest.py --env-file {tmp_path}/.env.ingest {_DESCRIBE_ARGS}"


def test_describe_runs_with_the_ingest_env_then_open_loops(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, "BRAIN_DESCRIBE_PROVIDER=anthropic\n")
    describe = [c for c in calls if "--describe-concepts" in c]
    assert describe == [_describe_call(tmp_path)]
    index = calls.index(describe[0])
    assert calls[index - 1].startswith("scripts/ingest.py --rebuild-search-index")
    assert calls[index + 1] == "-m ingest_lib.open_loops"
    assert "describe-concepts failed" not in proc.stdout


def test_env_ingest_is_never_evaluated_as_shell(tmp_path: Path) -> None:
    # M1: `. .env.ingest` ran `notion` as a command and executed $(...).
    proc, calls = _run(tmp_path, _SYSTEMD_ENV)
    assert "command not found" not in proc.stderr
    assert not (tmp_path / "pwned").exists()
    assert [c for c in calls if "--describe-concepts" in c] == [_describe_call(tmp_path)]


@pytest.mark.parametrize(
    ("env_ingest", "message"),
    [(None, "no .env.ingest — describe skipped"),
     ("ANTHROPIC_API_KEY=x\n", "no BRAIN_DESCRIBE_PROVIDER in .env.ingest — describe skipped"),
     ('BRAIN_DESCRIBE_PROVIDER=""\n', "no BRAIN_DESCRIBE_PROVIDER in .env.ingest — describe skipped"),
     ("# BRAIN_DESCRIBE_PROVIDER=anthropic\n",
      "no BRAIN_DESCRIBE_PROVIDER in .env.ingest — describe skipped")],
)
def test_describe_is_skipped_without_its_configuration(
    tmp_path: Path, env_ingest: str | None, message: str,
) -> None:
    proc, calls = _run(tmp_path, env_ingest)
    assert message in proc.stdout
    assert not [c for c in calls if "--describe-concepts" in c]
    assert "-m ingest_lib.open_loops" in calls


def test_dry_run_plans_describe_and_dry_runs_open_loops(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, "BRAIN_DESCRIBE_PROVIDER=anthropic\n", "--dry-run")
    assert "would describe stale concepts (skipped: --dry-run)" in proc.stdout
    assert not [c for c in calls if "--describe-concepts" in c]
    assert "-m ingest_lib.open_loops --dry-run" in calls


def test_describe_failure_does_not_abort_the_later_steps(tmp_path: Path) -> None:
    # M6: the `|| echo` guard and the absence of `set -e` keep the run going.
    proc, calls = _run(tmp_path, "BRAIN_DESCRIBE_PROVIDER=anthropic\n", describe_exit=3)
    assert "describe-concepts failed (non-fatal)" in proc.stdout
    after = calls[calls.index(_describe_call(tmp_path)) + 1:]
    assert after == [
        "-m ingest_lib.open_loops",
        "scripts/sweep.py --write-report",
        "scripts/rotate_logs.py",
    ]
    assert proc.stdout.rstrip().endswith("done")


@pytest.mark.skipif(_GIT is None, reason="git not installed")
@pytest.mark.parametrize("describe_exit", [0, 3])
def test_run_output_is_committed_and_a_prior_edit_is_not(
    tmp_path: Path, describe_exit: int,
) -> None:
    # M6: a real repo, so commit_output gets past `git rev-parse`.
    proc, _ = _run(
        tmp_path, "BRAIN_DESCRIBE_PROVIDER=anthropic\n",
        describe_exit=describe_exit, git=True,
    )
    assert "nothing committed" not in proc.stdout
    committed = _git(tmp_path, "show", "--name-only", "--format=%s", "HEAD").splitlines()
    assert committed[0].startswith("maintenance:")
    assert sorted(p for p in committed[1:] if p) == [
        "knowledge/concepts/widgets.md", "knowledge/index/Open Loops.md",
    ]
    assert _git(tmp_path, "status", "--porcelain", "--", "knowledge") == " M knowledge/human.md\n"


# ---------------------------------------------------------- ingest --env-file


def test_env_file_loads_systemd_values_literally_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env.ingest"
    env_file.write_text(_SYSTEMD_ENV, encoding="utf-8")
    monkeypatch.setenv("BRAIN_DESCRIBE_PROVIDER", "openai")
    for key in ("BRAIN_PULL_CONNECTORS", "ANTHROPIC_API_KEY", "PWN"):
        monkeypatch.setenv(key, "")
    paths = paths_for_root(tmp_path)
    monkeypatch.setattr(ingest, "default_paths", lambda: paths)
    seen: dict[str, str] = {}

    def fake_describe(
        _paths: VaultPaths, *, logger: logging.Logger, limit: int | None = None,
        force: bool = False, min_sources: int = DEFAULT_MIN_SOURCES,
    ) -> DescribeStats:
        for key in ("BRAIN_DESCRIBE_PROVIDER", "BRAIN_PULL_CONNECTORS", "ANTHROPIC_API_KEY", "PWN"):
            seen[key] = os.environ[key]
        return DescribeStats()

    monkeypatch.setattr(ingest, "rebuild_descriptions", fake_describe)
    assert ingest.main(["--env-file", str(env_file), "--describe-concepts"]) == 0
    assert seen == {
        "BRAIN_DESCRIBE_PROVIDER": "anthropic",
        "BRAIN_PULL_CONNECTORS": "granola notion",
        "ANTHROPIC_API_KEY": "sk-ab$cd",
        "PWN": "$(touch pwned)",
    }
    assert not (tmp_path / "pwned").exists()
