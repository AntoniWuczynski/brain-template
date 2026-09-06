"""Shared pytest bootstrap and MCP test harness for the suite.

``ingest_lib`` is an installed package (see pyproject's hatch config) but
``mcp_server`` is not — it lives at the repo root and is only importable
when the root is on ``sys.path``. Stage-A test files carry their own
per-file shims so they run standalone; this conftest makes the same
guarantee for the whole suite regardless of collection order.

The MCP fixtures below (AUD-070) are the harness every tool-level test
needs: a throwaway git vault, a ``ServerConfig`` pointed at it, and a
``Runtime`` whose background workers are off. They used to be copied
verbatim into each test module, so a new required field on either
dataclass meant seven edits and the copies had already drifted apart.
"""
from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mcp_server import tools as _tools_mod  # noqa: E402
from mcp_server import tools_read as _tools_read_mod  # noqa: E402
from mcp_server.audit import AuditLog  # noqa: E402
from mcp_server.config import ServerConfig  # noqa: E402
from mcp_server.identity import AGENT_VAR  # noqa: E402
from mcp_server.push_queue import PushWorker  # noqa: E402
from mcp_server.reindex import IndexRefresher  # noqa: E402
from mcp_server.runtime import Runtime  # noqa: E402

# The two tokens every MCP test vault is configured with. Only app.py maps
# tokens -> agent names; the tool-level tests set AGENT_VAR directly, so
# these exist to make the config valid, not to authenticate anything.
TOKEN_A = "x" * 24
TOKEN_B = "y" * 24

McpEnv = tuple[Path, ServerConfig, Runtime]


class GitRunner(Protocol):
    def __call__(self, cwd: Path, /, *args: str) -> str: ...


class CfgFactory(Protocol):
    def __call__(self, root: Path, /, *, profile_max_bytes: int = ...) -> ServerConfig: ...


class WaitFor(Protocol):
    def __call__(
        self,
        cond: Callable[[], bool],
        /,
        timeout: float = ...,
        interval: float = ...,
    ) -> bool: ...


class RuntimeFactory(Protocol):
    def __call__(
        self,
        root: Path,
        /,
        *,
        refresher: IndexRefresher | None = ...,
    ) -> Runtime: ...


@pytest.fixture()
def run_git() -> GitRunner:
    """Run one git command in ``cwd`` and return its stdout, stripped.
    Raises ``CalledProcessError`` on failure — test setup must not fail
    silently."""
    def _run(cwd: Path, /, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    return _run


@pytest.fixture()
def mcp_vault(tmp_path: Path, run_git: GitRunner) -> Path:
    """A throwaway git repo standing in for the vault: branch ``main``,
    committer identity set so commits work with no global git config."""
    root = tmp_path.resolve() / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    run_git(root, "config", "user.email", "test@example.com")
    run_git(root, "config", "user.name", "test")
    return root


@pytest.fixture()
def make_cfg() -> CfgFactory:
    def _cfg(root: Path, /, *, profile_max_bytes: int = 4096) -> ServerConfig:
        return ServerConfig(
            vault_root=root,
            tokens=((TOKEN_A, "agent-a"), (TOKEN_B, "agent-b")),
            bind_host="127.0.0.1",
            bind_port=0,
            git_push_on_write=False,
            git_remote="origin",
            git_branch="main",
            log_level="warning",
            allowed_hosts=(),
            profile_max_bytes=profile_max_bytes,
        )

    return _cfg


@pytest.fixture()
def make_runtime() -> RuntimeFactory:
    """A Runtime with both background workers off by default. Pass
    ``refresher=`` to drive the real reindex thread."""
    def _runtime(root: Path, /, *, refresher: IndexRefresher | None = None) -> Runtime:
        audit = AuditLog(root)
        return Runtime(
            audit=audit,
            push_worker=PushWorker(root, remote="origin", branch="main", enabled=False),
            refresher=refresher or IndexRefresher(root, audit=audit, enabled=False),
        )

    return _runtime


@pytest.fixture()
def roomy_rate_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh, roomy write/search/read buckets for the duration of a test.

    The real buckets are module globals shared by the whole pytest
    process, so without this a fast full-suite run would eat the budget
    across files and make tests order-dependent. Tests that are ABOUT
    rate limiting install their own bucket instead of using this.
    """
    monkeypatch.setattr(_tools_mod, "_write_bucket", _tools_mod._RateBucket(10_000))
    monkeypatch.setattr(
        _tools_read_mod, "_search_bucket", _tools_mod._RateBucket(10_000)
    )
    monkeypatch.setattr(
        _tools_read_mod, "_read_bucket", _tools_mod._RateBucket(10_000)
    )


@pytest.fixture()
def mcp_env(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> Iterator[McpEnv]:
    """``(vault_root, cfg, runtime)`` with AGENT_VAR pinned to agent-a."""
    token = AGENT_VAR.set("agent-a")
    try:
        yield mcp_vault, make_cfg(mcp_vault), make_runtime(mcp_vault)
    finally:
        AGENT_VAR.reset(token)


@pytest.fixture()
def wait_for() -> WaitFor:
    """Poll ``cond`` until it is true or the timeout expires."""
    def _wait_for(
        cond: Callable[[], bool], /, timeout: float = 8.0, interval: float = 0.02
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(interval)
        return False

    return _wait_for
