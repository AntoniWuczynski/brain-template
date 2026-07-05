"""load_config auth-critical branches.

These enforce the provenance guarantee (each token -> exactly one agent)
and were previously untested — only the lower-level parse_token_spec had
coverage.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mcp_server.config import load_config


_ENV = [
    "BRAIN_MCP_VAULT_ROOT", "BRAIN_MCP_TOKENS", "BRAIN_MCP_BEARER_TOKEN",
    "BRAIN_MCP_BIND_HOST", "BRAIN_MCP_BIND_PORT", "BRAIN_MCP_GIT_PUSH_ON_WRITE",
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)


def _git_vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_missing_vault_root_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="BRAIN_MCP_VAULT_ROOT"):
        load_config()


def test_non_git_vault_raises(monkeypatch, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("BRAIN_MCP_VAULT_ROOT", str(plain))
    monkeypatch.setenv("BRAIN_MCP_BEARER_TOKEN", "x" * 40)
    with pytest.raises(RuntimeError, match="not a git repository"):
        load_config()


def test_no_tokens_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_MCP_VAULT_ROOT", str(_git_vault(tmp_path)))
    with pytest.raises(RuntimeError, match="set BRAIN_MCP_TOKENS"):
        load_config()


def test_short_bearer_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_MCP_VAULT_ROOT", str(_git_vault(tmp_path)))
    monkeypatch.setenv("BRAIN_MCP_BEARER_TOKEN", "tooshort")
    with pytest.raises(RuntimeError, match="at least 24 characters"):
        load_config()


def test_bearer_duplicating_a_named_token_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_MCP_VAULT_ROOT", str(_git_vault(tmp_path)))
    shared = "a" * 40
    monkeypatch.setenv("BRAIN_MCP_TOKENS", f"claude={shared}")
    monkeypatch.setenv("BRAIN_MCP_BEARER_TOKEN", shared)
    with pytest.raises(RuntimeError, match="duplicates a token"):
        load_config()


def test_default_agent_name_clash_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_MCP_VAULT_ROOT", str(_git_vault(tmp_path)))
    monkeypatch.setenv("BRAIN_MCP_TOKENS", f"default={'a' * 40}")
    monkeypatch.setenv("BRAIN_MCP_BEARER_TOKEN", "b" * 40)
    with pytest.raises(RuntimeError, match="clashes"):
        load_config()


def test_happy_path_merges_named_and_bearer(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_MCP_VAULT_ROOT", str(_git_vault(tmp_path)))
    monkeypatch.setenv("BRAIN_MCP_TOKENS", f"claude-code={'a' * 40}")
    monkeypatch.setenv("BRAIN_MCP_BEARER_TOKEN", "b" * 40)
    cfg = load_config()
    agents = {agent for _tok, agent in cfg.tokens}
    assert agents == {"claude-code", "default"}
    assert len(cfg.tokens) == 2
