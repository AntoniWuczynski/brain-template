"""scripts/vault_sync.sh --pull-only: restart brain-mcp when server code changes.

Runs the real script against a throwaway origin + clone, with a stub
``systemctl`` in ``$HOME/.local/bin`` (the script puts that first on PATH)
that logs every call instead of touching systemd.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "vault_sync.sh"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


class _Env:
    def __init__(self, tmp: Path, *, active: bool = True) -> None:
        self.home = tmp / "home"
        (self.home / ".local" / "bin").mkdir(parents=True)
        (self.home / ".gitconfig").write_text(
            "[user]\n\temail = t@example.com\n\tname = t\n[init]\n\tdefaultBranch = main\n"
        )
        self.calls = tmp / "systemctl.log"
        stub = self.home / ".local" / "bin" / "systemctl"
        stub.write_text(
            "#!/bin/sh\n"
            f'echo "$*" >> "{self.calls}"\n'
            f'case "$*" in *is-active*) exit {0 if active else 3};; esac\n'
            "exit 0\n"
        )
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.env = {**os.environ, "HOME": str(self.home)}
        self.env.pop("BRAIN_MCP_GIT_BRANCH", None)
        self.env.pop("BRAIN_MCP_GIT_REMOTE", None)

        self.origin = tmp / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)],
                       check=True, env=self.env)
        self.work = tmp / "work"      # another machine pushing to origin
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.work)],
                       check=True, env=self.env)
        self.commit("README.md", "hi\n")
        self.server = tmp / "server"
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.server)],
                       check=True, env=self.env)
        (self.server / "scripts").mkdir()
        # The script resolves the repo root from its own location.
        (self.server / "scripts" / "vault_sync.sh").write_text(_SCRIPT.read_text())

    def commit(self, rel: str, text: str) -> None:
        f = self.work / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        subprocess.run(["git", "-C", str(self.work), "add", "-A"], check=True, env=self.env)
        subprocess.run(["git", "-C", str(self.work), "commit", "-qm", rel], check=True, env=self.env)
        subprocess.run(["git", "-C", str(self.work), "push", "-q", "origin", "main"],
                       check=True, env=self.env)

    def sync(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.server / "scripts" / "vault_sync.sh"), "--pull-only"],
            capture_output=True, text=True, env=self.env,
        )

    def restarts(self) -> int:
        if not self.calls.exists():
            return 0
        return sum("restart" in line for line in self.calls.read_text().splitlines())


@pytest.fixture
def env(tmp_path: Path) -> _Env:
    return _Env(tmp_path)


def test_first_run_stamps_without_restarting(env: _Env) -> None:
    assert env.sync().returncode == 0
    assert env.restarts() == 0
    stamp = env.server / ".git" / "brain-mcp-code-rev"
    assert stamp.read_text().strip() == _git(env.server, "rev-parse", "HEAD")


def test_note_change_does_not_restart(env: _Env) -> None:
    env.sync()
    env.commit("knowledge/notes/a.md", "x\n")
    assert env.sync().returncode == 0
    assert env.restarts() == 0


def test_server_code_change_restarts_once(env: _Env) -> None:
    env.sync()
    env.commit("mcp_server/tools.py", "x = 1\n")
    out = env.sync()
    assert out.returncode == 0
    assert "restarted brain-mcp" in out.stdout
    assert env.restarts() == 1
    env.sync()
    assert env.restarts() == 1


def test_code_arriving_outside_the_sync_still_restarts(env: _Env) -> None:
    env.sync()
    env.commit("scripts/ingest_lib/outline.py", "x = 1\n")
    # The MCP server's pull-before-write can fast-forward before the timer runs.
    _git(env.server, "pull", "-q", "--ff-only")
    env.sync()
    assert env.restarts() == 1


def test_dependency_change_fails_without_restarting(env: _Env) -> None:
    env.sync()
    env.commit("uv.lock", "lock\n")
    env.commit("mcp_server/tools.py", "x = 1\n")
    out = env.sync()
    assert out.returncode == 1
    assert "dependencies changed" in out.stdout
    assert env.restarts() == 0


def test_no_active_unit_is_a_no_op(tmp_path: Path) -> None:
    e = _Env(tmp_path, active=False)
    e.commit("mcp_server/tools.py", "x = 1\n")
    assert e.sync().returncode == 0
    assert e.restarts() == 0
    assert not (e.server / ".git" / "brain-mcp-code-rev").exists()
