"""Smoke test for the MCP server.

Spawns the server bound to 127.0.0.1, runs every tool through the
official ``mcp`` client SDK (no third-party deps), and exercises the
security boundaries. Cleans up any test writes via ``git reset`` so
the working tree ends up unchanged.

Run with::

    uv run python -m mcp_server.manual_test

Exits 0 if every check passes; non-zero with a diagnostic on the first
failure. Captures the server's stdout/stderr to a temp file you can
read when debugging.
"""
from __future__ import annotations

import asyncio
import base64
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_health(port: int, timeout: float = 15.0) -> None:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server didn't come up on :{port} within {timeout}s")


class TestRunner:
    # This is a live-server smoke harness, not a pytest test class. Tell
    # pytest not to collect it (its __init__ otherwise triggers a
    # PytestCollectionWarning on every run).
    __test__ = False

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def expect(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passes.append(label)
            print(f"  PASS  {label}")
        else:
            self.failures.append(f"{label}: {detail}")
            print(f"  FAIL  {label}  ({detail})")

    async def expect_error(self, label: str, coro, expect_in_msg: str = "") -> None:
        try:
            result = await coro
        except Exception as exc:  # noqa: BLE001 - manual test, we want to see anything
            msg = str(exc).lower()
            if expect_in_msg and expect_in_msg.lower() not in msg:
                self.expect(label, False, f"raised but message lacks {expect_in_msg!r}: {exc}")
            else:
                self.passes.append(label)
                print(f"  PASS  {label}  (refused: {str(exc)[:80]})")
            return
        # The mcp client wraps tool errors in CallToolResult.isError rather
        # than raising — check for that too.
        is_error = getattr(result, "isError", False)
        content = getattr(result, "content", None)
        if is_error:
            self.passes.append(label)
            print(f"  PASS  {label}  (tool error returned)")
            return
        self.expect(label, False, f"expected an error but got: {content!r}"[:120])


async def _call(session: ClientSession, name: str, **kwargs):
    return await session.call_tool(name, kwargs)


def _ok(result) -> bool:
    return not getattr(result, "isError", False)


@asynccontextmanager
async def _client(url: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def main() -> int:
    runner = TestRunner()

    port = _free_port()
    token = secrets.token_urlsafe(32)
    log_file = Path(tempfile.mkstemp(prefix="mcp-smoke-", suffix=".log")[1])
    print(f"\nbearer token (test-only): {token[:8]}…")
    print(f"server port:               {port}")
    print(f"server log:                {log_file}")
    print(f"vault root:                {_REPO_ROOT}\n")

    # --- path policy (no server needed) ---
    # The read deny-list must hold even when a case-insensitive filesystem
    # (macOS APFS, Windows) lets a case-variant basename open the real file.
    # Regression guard for the case-fold fix in safety.resolve_read.
    from mcp_server.safety import SafetyError as _SafetyError
    from mcp_server.safety import resolve_read as _resolve_read
    print("[path policy]")
    for _variant in (
        "metadata/EMBEDDINGS_META.jsonl",
        "metadata/Embeddings_Meta.JSONL",
        "metadata/EMBEDDINGS.NPY",
        "LOGS/x.log",
    ):
        try:
            _resolve_read(_REPO_ROOT, _variant)
            runner.expect(f"deny-list rejects case-variant {_variant}", False,
                          "resolve_read returned a path (deny-list bypass)")
        except _SafetyError:
            runner.expect(f"deny-list rejects case-variant {_variant}", True)
    print()

    # Subprocess env: pin everything explicitly so the test is reproducible.
    # The legacy single-token env (-> agent "default") is still supported
    # alongside BRAIN_MCP_TOKENS; this smoke test exercises that path.
    env = {
        **os.environ,
        "BRAIN_MCP_VAULT_ROOT": str(_REPO_ROOT),
        "BRAIN_MCP_BEARER_TOKEN": token,
        "BRAIN_MCP_BIND_HOST": "127.0.0.1",
        "BRAIN_MCP_BIND_PORT": str(port),
        "BRAIN_MCP_GIT_PUSH_ON_WRITE": "0",  # never push during tests
        "BRAIN_MCP_LOG_LEVEL": "warning",
        "BRAIN_PROFILE_MAX_BYTES": "4096",   # the budget the smoke checks assume
    }

    # Refuse to run on a dirty tree: teardown does `git reset --hard`,
    # which would destroy any uncommitted work.
    dirty = subprocess.check_output(
        ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        print("ABORT: working tree is dirty; commit or stash before running "
              "(teardown does git reset --hard).")
        print(dirty)
        return 2

    # Note the HEAD before any test writes so we can rewind after.
    head_before = subprocess.check_output(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    server = subprocess.Popen(
        [sys.executable, "-m", "mcp_server"],
        cwd=_REPO_ROOT,
        env=env,
        stdout=open(log_file, "wb"),
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_for_health(port)
        url = f"http://127.0.0.1:{port}/mcp"

        # --- AUTH ---
        print("\n[auth]")
        # request with wrong token must fail
        try:
            async with _client(url, token="wrong-" + token) as s:
                await _call(s, "vault_list", path="")
            runner.expect("auth: wrong token rejected", False, "got past initialize()")
        except Exception:
            runner.expect("auth: wrong token rejected", True)

        # request with empty bearer must fail
        try:
            async with _client(url, token="") as s:
                await _call(s, "vault_list", path="")
            runner.expect("auth: empty token rejected", False, "got past initialize()")
        except Exception:
            runner.expect("auth: empty token rejected", True)

        # /health is publicly reachable (no auth)
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r:
            runner.expect("auth: /health reachable without token", r.status == 200)

        # --- AUTHENTICATED FLOW ---
        async with _client(url, token=token) as s:
            tools = await s.list_tools()
            names = sorted(t.name for t in tools.tools)
            expected = {
                "vault_search", "vault_read", "vault_list", "vault_metadata_query",
                "vault_related",
                "vault_create_note", "vault_replace_note", "vault_append_to_note",
                "vault_update_concept_user_section", "vault_drop_inbox_file",
                "entity_upsert_relation", "entity_append_fact", "meeting_create",
                "memory_search", "profile_update",
            }
            runner.expect(
                "tools: all 15 registered",
                expected.issubset(set(names)),
                f"missing {expected - set(names)}",
            )

            # --- READS ---
            print("\n[reads]")
            r = await _call(s, "vault_list", path="")
            runner.expect("vault_list: root succeeds", _ok(r))

            # vault_related rejects an unknown concept (graph state-agnostic:
            # errors whether the concept is absent or the graph isn't built).
            await runner.expect_error(
                "vault_related: rejects unknown concept",
                _call(s, "vault_related", concept="__definitely_not_a_real_topic__"),
            )

            r = await _call(s, "vault_read", path="README.md")
            runner.expect("vault_read: README succeeds", _ok(r))

            # path traversal
            await runner.expect_error(
                "vault_read: rejects ../ escape",
                _call(s, "vault_read", path="../../etc/passwd"),
                expect_in_msg="escapes",
            )
            await runner.expect_error(
                "vault_read: rejects .env",
                _call(s, "vault_read", path=".env"),
                expect_in_msg="denied",
            )
            await runner.expect_error(
                "vault_read: rejects logs/",
                _call(s, "vault_read", path="logs/nope.log"),
            )
            # allowlist: framework code & root config are NOT readable
            await runner.expect_error(
                "vault_read: rejects scripts/ (not in read allowlist)",
                _call(s, "vault_read", path="scripts/ingest.py"),
            )
            await runner.expect_error(
                "vault_read: rejects pyproject.toml (not an allowed root doc)",
                _call(s, "vault_read", path="pyproject.toml"),
            )
            await runner.expect_error(
                "vault_read: rejects mcp_server/ (server's own code)",
                _call(s, "vault_read", path="mcp_server/config.py"),
            )
            # allowlist: vault content IS readable
            r = await _call(s, "vault_read", path="metadata/index.jsonl")
            runner.expect("vault_read: metadata/index.jsonl allowed", _ok(r))

            # metadata query (use 'all' to avoid depending on what's processed)
            r = await _call(s, "vault_metadata_query", by="all", limit=5)
            runner.expect("vault_metadata_query: by=all succeeds", _ok(r))

            await runner.expect_error(
                "vault_metadata_query: rejects unknown 'by'",
                _call(s, "vault_metadata_query", by="garbage"),
            )

            # --- WRITES ---
            print("\n[writes — committed locally, will be rewound]")
            test_note_path = "knowledge/notes/_mcp_smoke_test_note.md"
            r = await _call(s, "vault_create_note", path=test_note_path,
                            content="# smoke test\n\nhello from the MCP server smoke test.\n")
            runner.expect("vault_create_note: valid path succeeds", _ok(r))

            r = await _call(s, "vault_append_to_note", path=test_note_path,
                            content="appended line.\n")
            runner.expect("vault_append_to_note: succeeds", _ok(r))

            # Replace: full rewrite of the existing note.
            r = await _call(s, "vault_replace_note", path=test_note_path,
                            content="# replaced\n\nfull rewrite via the replace tool.\n")
            runner.expect("vault_replace_note: replaces existing note", _ok(r))
            await runner.expect_error(
                "vault_replace_note: refuses non-existent note",
                _call(s, "vault_replace_note",
                      path="knowledge/notes/_does_not_exist.md", content="x"),
                expect_in_msg="does not exist",
            )
            await runner.expect_error(
                "vault_replace_note: rejects path under archive/processed/",
                _call(s, "vault_replace_note",
                      path="archive/processed/x.md", content="nope"),
                expect_in_msg="denied",
            )
            await runner.expect_error(
                "vault_replace_note: rejects inbox/ (drop_inbox_file only)",
                _call(s, "vault_replace_note",
                      path="inbox/pending.md", content="nope"),
                expect_in_msg="denied",
            )
            await runner.expect_error(
                "vault_create_note: rejects inbox/ (drop_inbox_file only)",
                _call(s, "vault_create_note",
                      path="inbox/new_note.md", content="nope"),
                expect_in_msg="denied",
            )

            # Write rejections
            await runner.expect_error(
                "vault_create_note: rejects path under archive/processed/",
                _call(s, "vault_create_note",
                      path="archive/processed/x.md", content="nope"),
                expect_in_msg="denied",
            )
            await runner.expect_error(
                "vault_create_note: rejects path under archive/raw/",
                _call(s, "vault_create_note",
                      path="archive/raw/x.md", content="nope"),
                expect_in_msg="denied",
            )
            await runner.expect_error(
                "vault_create_note: rejects metadata/",
                _call(s, "vault_create_note",
                      path="metadata/x.md", content="nope"),
                expect_in_msg="denied",
            )
            await runner.expect_error(
                "vault_create_note: rejects ../ escape",
                _call(s, "vault_create_note",
                      path="../escape.md", content="nope"),
                expect_in_msg="escapes",
            )
            await runner.expect_error(
                "vault_create_note: rejects re-create over existing",
                _call(s, "vault_create_note",
                      path=test_note_path, content="dup"),
                expect_in_msg="exists",
            )
            await runner.expect_error(
                "vault_create_note: rejects oversize content",
                _call(s, "vault_create_note",
                      path="knowledge/notes/_huge.md", content="x" * (6 * 1024 * 1024)),
                expect_in_msg="max",
            )

            # Inbox upload
            test_inbox_path = "_mcp_smoke_test_inbox.txt"
            small_bytes = b"hello from the inbox upload smoke test"
            b64 = base64.b64encode(small_bytes).decode()
            r = await _call(s, "vault_drop_inbox_file",
                            path=test_inbox_path, content_base64=b64)
            runner.expect("vault_drop_inbox_file: small upload succeeds", _ok(r))
            await runner.expect_error(
                "vault_drop_inbox_file: rejects duplicate path",
                _call(s, "vault_drop_inbox_file",
                      path=test_inbox_path, content_base64=b64),
                expect_in_msg="exists",
            )
            await runner.expect_error(
                "vault_drop_inbox_file: rejects invalid base64",
                _call(s, "vault_drop_inbox_file",
                      path="_inv.bin", content_base64="not-base64!!"),
                expect_in_msg="base64",
            )
            # sandbox: a traversal path must not escape inbox/ into a
            # ground-truth layer (archive/processed, archive/raw, metadata)
            import base64 as _b64
            _payload = _b64.b64encode(b"x").decode()
            await runner.expect_error(
                "vault_drop_inbox_file: rejects ../archive escape",
                _call(s, "vault_drop_inbox_file",
                      path="../archive/processed/evil.md", content_base64=_payload),
            )

            # --- ENTITY-MEMORY TOOLS (refusal paths only) ---
            # Happy-path writes are deliberately not smoke-tested here:
            # entity/meeting writes flag graph_changed, so on the live vault
            # the background refresher would kick off a full connections/
            # concepts/dashboards rebuild that races the teardown's git
            # reset. TODO: cover the happy paths against a throwaway vault
            # (the test_replace_note.py pattern) instead.
            print("\n[entity-memory tools — refusal paths]")
            await runner.expect_error(
                "entity_upsert_relation: rejects rel outside the vocabulary",
                _call(s, "entity_upsert_relation",
                      entity_path="knowledge/people/_mcp_smoke_absent.md",
                      rel="employed_by", target="organisations/acme"),
                expect_in_msg="unknown rel",
            )
            await runner.expect_error(
                "entity_upsert_relation: refuses missing entity note",
                _call(s, "entity_upsert_relation",
                      entity_path="knowledge/people/_mcp_smoke_absent.md",
                      rel="works_at", target="organisations/acme"),
                expect_in_msg="does not exist",
            )
            await runner.expect_error(
                "entity_append_fact: rejects multi-line text",
                _call(s, "entity_append_fact",
                      entity_path="knowledge/people/_mcp_smoke_absent.md",
                      text="line one\nline two", source="knowledge/notes/x"),
                expect_in_msg="single line",
            )
            await runner.expect_error(
                "meeting_create: rejects non-canonical date",
                _call(s, "meeting_create", date="2026-1-1", title="Smoke",
                      attendees=["people/_mcp_smoke_nobody"]),
                expect_in_msg="yyyy-mm-dd",
            )
            await runner.expect_error(
                "memory_search: rejects out-of-range top_k",
                _call(s, "memory_search", query="anything", top_k=0),
                expect_in_msg="top_k",
            )
            await runner.expect_error(
                "profile_update: rejects content over the byte budget",
                _call(s, "profile_update", content="x" * 8192),
                expect_in_msg="cap",
            )

    finally:
        # Tear down server
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

        # Rewind every test commit so the branch is back where we found it.
        head_after = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        if head_after != head_before:
            print(f"\nrewinding test commits: {head_after[:8]} -> {head_before[:8]}")
            subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "reset", "--hard", head_before],
                check=True, capture_output=True,
            )
        # Also clean any untracked test artefacts that might have slipped through
        for stray in (
            _REPO_ROOT / "knowledge/notes/_mcp_smoke_test_note.md",
            _REPO_ROOT / "inbox/_mcp_smoke_test_inbox.txt",
        ):
            stray.unlink(missing_ok=True)

    print(f"\nresults: {len(runner.passes)} passed, {len(runner.failures)} failed")
    if runner.failures:
        print("\nfailures:")
        for f in runner.failures:
            print(f"  - {f}")
        print(f"\nserver log: {log_file}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
