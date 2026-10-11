"""scripts/dream_lock.sh: one dream pass at a time, scheduled or manual."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts/dream_lock.sh"


def _run(args: list[str], env: dict[str, str], script: Path = _SCRIPT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args], env=env, capture_output=True, text=True, check=False
    )


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_DREAM_LOCK")}
    env["BRAIN_DREAM_LOCK_HOST"] = ""  # local: never read the developer's .env
    env["BRAIN_DREAM_LOCK_DIR"] = str(tmp_path / "run.lock")
    env.update(extra)
    return env


def test_acquire_release_round_trip(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert _run(["acquire", "manual"], env).returncode == 0
    assert (tmp_path / "run.lock" / "owner").read_text().split()[0] == "manual"
    assert _run(["release", "manual"], env).returncode == 0
    assert not (tmp_path / "run.lock").exists()


def test_a_held_lock_refuses_a_second_pass(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert _run(["acquire", "scheduled"], env).returncode == 0
    busy = _run(["acquire", "manual"], env)
    assert busy.returncode == 1
    assert "already running: scheduled" in busy.stdout


def test_release_leaves_someone_elses_lock_alone(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _run(["acquire", "scheduled"], env)
    wrong = _run(["release", "manual"], env)
    assert wrong.returncode == 1
    assert (tmp_path / "run.lock").is_dir()


def test_release_without_a_lock_is_a_noop(tmp_path: Path) -> None:
    assert _run(["release", "manual"], _env(tmp_path)).returncode == 0


def test_a_stale_lock_is_broken(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    lock.mkdir()
    (lock / "owner").write_text(f"manual {int(time.time()) - 10_000}\n")
    out = _run(["acquire", "scheduled"], _env(tmp_path))
    assert out.returncode == 0
    assert "breaking a stale lock held by manual" in out.stdout
    assert (lock / "owner").read_text().split()[0] == "scheduled"


def test_a_recent_lock_is_not_stale(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    lock.mkdir()
    (lock / "owner").write_text(f"manual {int(time.time()) - 60}\n")
    assert _run(["acquire", "scheduled"], _env(tmp_path)).returncode == 1


def test_a_lock_without_its_owner_file_yet_is_not_stale(tmp_path: Path) -> None:
    """The instant between another acquire's mkdir and its owner write: the
    lock is dated by the directory, not read as held since the epoch."""
    (tmp_path / "run.lock").mkdir()
    busy = _run(["acquire", "scheduled"], _env(tmp_path))
    assert busy.returncode == 1
    assert "already running: unknown" in busy.stdout
    assert not (tmp_path / "run.lock" / "owner").exists()


def test_usage_errors_exit_2(tmp_path: Path) -> None:
    env = _env(tmp_path)
    for args in ([], ["acquire"], ["grab", "manual"], ["acquire", "bob; touch pwned"]):
        assert _run(args, env).returncode == 2, args


def test_lock_host_runs_the_script_over_ssh(tmp_path: Path) -> None:
    """With a lock host set, the script ships itself to that host over ssh.
    A fake ssh runs the remote command locally, so the lock lands in the
    test's directory exactly as it would on the host."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "ssh-calls"
    fake = bin_dir / "ssh"
    fake.write_text(f'#!/bin/sh\necho "$1" >> "{calls}"\nshift\nexec sh -c "$*"\n')
    fake.chmod(0o755)
    env = _env(tmp_path, BRAIN_DREAM_LOCK_HOST="homeserver", PATH=f"{bin_dir}:{os.environ['PATH']}")
    assert _run(["acquire", "manual"], env).returncode == 0
    assert (tmp_path / "run.lock" / "owner").read_text().split()[0] == "manual"
    assert _run(["release", "manual"], env).returncode == 0
    assert calls.read_text().split() == ["homeserver", "homeserver"]


def test_lock_host_is_read_from_the_repo_env_file(tmp_path: Path) -> None:
    """Unset in the environment, the host comes from the repo's .env, so a
    manual run on the Mac locks on the server without extra setup."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    script = repo / "scripts/dream_lock.sh"
    script.write_text(_SCRIPT.read_text())
    (repo / ".env").write_text('OTHER=1\nBRAIN_DREAM_LOCK_HOST="homeserver"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "ssh-calls"
    fake = bin_dir / "ssh"
    fake.write_text(f'#!/bin/sh\necho "$1" >> "{calls}"\nshift\nexec sh -c "$*"\n')
    fake.chmod(0o755)
    env = _env(tmp_path, PATH=f"{bin_dir}:{os.environ['PATH']}")
    del env["BRAIN_DREAM_LOCK_HOST"]
    assert _run(["acquire", "manual"], env, script=script).returncode == 0
    assert calls.read_text().split() == ["homeserver"]


def test_dream_sh_and_the_skill_share_the_lock() -> None:
    script = (_REPO_ROOT / "scripts/dream.sh").read_text(encoding="utf-8")
    assert '"$ROOT/scripts/dream_lock.sh" acquire scheduled' in script
    assert "release scheduled" in script and "trap " in script
    assert 'run_capped claude -p "/dream-pass --scheduled"' in script
    for rel in (".claude", "_template/.claude", ".agents"):
        text = (_REPO_ROOT / rel / "skills/dream-pass/SKILL.md").read_text(encoding="utf-8")
        assert "scripts/dream_lock.sh acquire manual" in text, rel
        assert "scripts/dream_lock.sh release manual" in text, rel
        assert "--scheduled" in text, rel
