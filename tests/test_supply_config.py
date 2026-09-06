"""Supply-chain / ops configuration guards.

These pin the parts of the repo that are configuration rather than code, and
that nothing else executes in CI: the workflow's install command, the
Dependabot feeds, .gitignore's coverage of agent and tool state, the
devcontainer's reproducibility, the launchd/scheduler entry points, and the
two template-sync scripts.

The one behavioural test here drives ``scripts/pull_from_upstream.sh`` against
a pair of throwaway repositories under tmp_path.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# CI install step
# --------------------------------------------------------------------------

def test_ci_install_asserts_the_lockfile_is_current() -> None:
    """A bare `uv sync` re-resolves a stale lock on the runner, so CI would go
    green against packages nobody committed. --locked forbids that."""
    commands = [
        line for line in _read(".github/workflows/ci.yml").splitlines()
        if "uv sync" in line and not line.lstrip().startswith("#")
    ]
    assert commands
    for line in commands:
        assert "uv sync --locked" in line, line


def test_ci_excludes_every_torch_only_cuda_package() -> None:
    """`--no-install-package torch` drops torch alone, never its dependency
    closure, so the linux-only CUDA wheels would still install on the runner.
    Every nvidia-*/cuda-*/triton package in the lock must be named."""
    lock = _read("uv.lock")
    locked = {m.group(1) for m in re.finditer(r'^name = "([^"]+)"$', lock, re.M)}
    cuda = {n for n in locked if n.startswith(("nvidia-", "cuda-")) or n == "triton"}
    assert cuda, "expected CUDA packages in uv.lock"

    ci = _read(".github/workflows/ci.yml")
    excluded = set(re.findall(r"--no-install-package\s+(\S+)", ci))
    assert "torch" in excluded
    assert cuda <= excluded, f"not excluded in CI: {sorted(cuda - excluded)}"


# --------------------------------------------------------------------------
# Dependabot
# --------------------------------------------------------------------------

def test_dependabot_feeds_python_dependencies_too() -> None:
    """Actions-only coverage leaves uv.lock with no update or vulnerability
    feed, and every pyproject bound is an open-ended lower bound."""
    cfg = _read(".github/dependabot.yml")
    ecosystems = set(re.findall(r"package-ecosystem:\s*\"?([\w-]+)", cfg))
    assert ecosystems == {"github-actions", "uv"}


# --------------------------------------------------------------------------
# .gitignore
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".claude/settings.local.json",
        ".claude/worktrees/",
    ],
)
def test_gitignore_covers_local_tool_and_agent_state(path: str) -> None:
    """These were ignored only by the machine's global excludes (or not at
    all), so `git add -A` here staged them and template users inherited no
    rule — .claude/worktrees/ in particular holds nested checkouts that stage
    as gitlinks."""
    assert path in _read(".gitignore").splitlines()


def test_gitignore_does_not_hide_the_tracked_agent_configs() -> None:
    """.agents/ and .codex/ are deliberately tracked (the Codex-side mirror of
    the Claude harness), so they must NOT be ignored."""
    lines = _read(".gitignore").splitlines()
    assert ".agents/" not in lines
    assert ".codex/" not in lines


# --------------------------------------------------------------------------
# Scheduler entry points
# --------------------------------------------------------------------------

@pytest.mark.parametrize("script", ["scripts/maintain.sh", "scripts/dream.sh"])
def test_scheduler_scripts_guard_the_repo_root_cd(script: str) -> None:
    """Neither runs under `set -e`. An unguarded `cd "$ROOT"` after a failed
    subshell leaves ROOT empty, and `cd ""` returns 0 without moving, so the
    run continues against the wrong tree behind non-fatal echoes."""
    body = _read(script)
    assert re.search(r'^ROOT=\$\(cd .* && pwd\) \|\| \{', body, re.M)
    assert re.search(r'^cd "\$ROOT" \|\| \{', body, re.M)


def test_maintain_sh_repairs_launchd_minimal_path() -> None:
    """launchd hands the job PATH=/usr/bin:/bin:/usr/sbin:/sbin, where the
    `uv` fallback in run_py is unfindable. dream.sh already does this."""
    assert 'export PATH="$HOME/.local/bin:' in _read("scripts/maintain.sh")


# --------------------------------------------------------------------------
# launchd unit for the server
# --------------------------------------------------------------------------

def test_mcp_launchd_template_restarts_only_on_failure() -> None:
    """An unconditional <KeepAlive><true/> plus a short throttle turns any
    config error into a permanent respawn loop, and a combined log file makes
    it illegible. The unit must also never carry the bearer token."""
    plist = _read("mcp_server/launchd/com.brain.mcp.plist")
    assert "<key>KeepAlive</key>\n    <dict>" in plist
    assert "<key>Crashed</key>" in plist
    assert re.search(r"<key>ThrottleInterval</key>\s*<integer>(\d+)</integer>", plist)
    throttle = re.search(r"<key>ThrottleInterval</key>\s*<integer>(\d+)</integer>", plist)
    assert throttle is not None and int(throttle.group(1)) >= 30
    assert "brain-mcp.out.log" in plist and "brain-mcp.err.log" in plist
    assert "BRAIN_MCP_BEARER_TOKEN" not in plist
    assert "setenv" not in plist.split("-->")[1]


# --------------------------------------------------------------------------
# Local permission grants
# --------------------------------------------------------------------------

def test_local_permissions_do_not_pre_approve_arbitrary_commands() -> None:
    """`Bash(uv run *)` approves `uv run bash -c ...` and `uv run --exact`
    (which prunes the env) as readily as the project's own scripts. The file
    is machine-local, so skip where it doesn't exist."""
    path = _ROOT / ".claude/settings.local.json"
    if not path.exists():
        pytest.skip("no machine-local settings file")
    parsed: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    permissions = parsed.get("permissions")
    assert isinstance(permissions, dict)
    allow = permissions.get("allow", [])
    assert isinstance(allow, list)
    for rule in allow:
        assert isinstance(rule, str)
        if rule.startswith("Bash(uv run"):
            assert rule.startswith("Bash(uv run --no-sync "), rule


# --------------------------------------------------------------------------
# Devcontainer
# --------------------------------------------------------------------------

def test_devcontainer_pins_its_inputs() -> None:
    """A floating base tag and an unpinned installer are two "latest" pulls in
    the one environment allowed to touch the lockfile."""
    dockerfile = _read(".devcontainer/Dockerfile")
    assert re.search(r"^FROM \S+@sha256:[0-9a-f]{64}$", dockerfile, re.M)
    installer = [line for line in dockerfile.splitlines() if "astral.sh/uv" in line and line.startswith("RUN ")]
    assert len(installer) == 1
    assert re.search(r"https://astral\.sh/uv/\d+\.\d+\.\d+/install\.sh", installer[0])


def test_devcontainer_postcreate_cannot_rewrite_the_lockfile() -> None:
    """The old `uv sync --frozen || uv sync` fallback re-resolved and rewrote
    uv.lock inside the container whenever the lock was stale."""
    parsed: object = json.loads(_read(".devcontainer/devcontainer.json"))
    assert isinstance(parsed, dict)
    post = parsed.get("postCreateCommand")
    assert post == "uv sync --locked"


# --------------------------------------------------------------------------
# Template sync: push side
# --------------------------------------------------------------------------

def _framework_paths() -> list[str]:
    body = _read("scripts/push_to_upstream.sh")
    block = re.search(r"^FRAMEWORK_PATHS=\((.*?)^\)$", body, re.M | re.S)
    assert block is not None
    return re.findall(r'^\s*"([^"]+)"', block.group(1), re.M)


def test_push_script_prints_the_pull_request_flow() -> None:
    """brain-template's main is branch-protected, so the printed
    `push upstream template:main` simply fails."""
    body = _read("scripts/push_to_upstream.sh")
    assert "push upstream template:main" not in body
    assert "framework-sync-$SYNC_DATE" in body
    assert "gh pr create" in body


def test_template_reminder_hook_covers_every_framework_path() -> None:
    """The hook is the only nudge that a change here needs publishing, so its
    path list must not drift from what push_to_upstream.sh actually syncs."""
    hook = _read(".claude/hooks/remind-template-sync.py")
    prefixes = tuple(re.findall(r'^\s+"([^"]+/)",$', hook, re.M))
    files = set(re.findall(r'^\s+"([^"]+[^/])",$', hook, re.M))
    assert prefixes and files

    # .gitkeep markers seed empty directories on the template branch; they are
    # not content anyone edits, so the hook has nothing to remind about.
    exempt = {".claude/hooks/.gitkeep", ".claude/memory/.gitkeep", ".claude/patterns/.gitkeep"}
    missing = [
        p for p in _framework_paths()
        if p not in exempt
        and p not in files
        and not p.startswith(prefixes)
        and f"{p}/" not in prefixes
    ]
    assert not missing, f"push_to_upstream.sh syncs these but the hook ignores them: {missing}"


# --------------------------------------------------------------------------
# Template sync: pull side (behavioural)
# --------------------------------------------------------------------------

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def _git(repo: Path, *args: str) -> str:
    env = dict(os.environ)
    env.update(_GIT_ENV)
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True, env=env,
    ).stdout


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_pull_from_upstream_keeps_the_private_root_files(tmp_path: Path) -> None:
    """upstream's README/TODO/Home.md are generated from _template/ by the
    push script, and LICENSE/CONTRIBUTING exist only on the public branch. A
    plain `git merge upstream/main` conflicts on the first group and silently
    adds the second; the pull must keep ours and still absorb real framework
    changes."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _write(upstream / "README.md", "public template readme\n")
    _write(upstream / "LICENSE", "MIT\n")
    _write(upstream / "knowledge/index/Home.md", "public home\n")
    _write(upstream / "scripts/framework_fix.py", "print('fixed')\n")
    _git(upstream, "add", "-A")
    _git(upstream, "commit", "-q", "-m", "template")

    private = tmp_path / "private"
    private.mkdir()
    _git(private, "init", "-q", "-b", "main")
    (private / "scripts").mkdir()
    shutil.copy2(_ROOT / "scripts/pull_from_upstream.sh", private / "scripts/pull_from_upstream.sh")
    _write(private / "README.md", "private vault readme\n")
    _write(private / "knowledge/index/Home.md", "private home\n")
    _git(private, "add", "-A")
    _git(private, "commit", "-q", "-m", "private")
    _git(private, "remote", "add", "upstream", str(upstream))

    env = dict(os.environ)
    env.update(_GIT_ENV)
    proc = subprocess.run(
        ["bash", "scripts/pull_from_upstream.sh", "--first-run"],
        cwd=private, input="y\n", capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert (private / "README.md").read_text(encoding="utf-8") == "private vault readme\n"
    assert (private / "knowledge/index/Home.md").read_text(encoding="utf-8") == "private home\n"
    assert not (private / "LICENSE").exists()
    assert (private / "scripts/framework_fix.py").exists()
    assert "LICENSE" not in _git(private, "ls-files")
    # The merge was recorded, not left half-applied.
    assert len(_git(private, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3
