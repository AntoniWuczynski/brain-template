"""End-to-end pytest coverage for the MCP write tools' happy paths (AUD-014).

Until now these paths — create a note, upsert a relation, query relations,
read a note back, append a fact — were only exercised manually via
mcp_server/manual_test.py, never under pytest. The throwaway-vault +
ServerConfig + Runtime harness lives in tests/conftest.py (mcp_vault /
make_cfg / make_runtime / mcp_env): a real git repo, background workers
(push, reindex) disabled unless a test asks for one, no model loads,
AGENT_VAR pinned to a fake agent identity. Assertions run against real
on-disk content and real git history, not mocks.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

# Type-only import of the shared harness contracts. At runtime the module
# is pytest's ``conftest``; ``tests.conftest`` is the name mypy resolves
# (explicit_package_bases), and importing it for real would collide with an
# unrelated installed ``tests`` package. Annotations are postponed, so
# nothing here is evaluated at import time.
if TYPE_CHECKING:
    from tests.conftest import CfgFactory, GitRunner, McpEnv, RuntimeFactory, WaitFor

from mcp_server.audit import AuditLog
from mcp_server.entity_tools import (
    tool_entity_append_fact,
    tool_entity_upsert_relation,
    tool_relations_query,
)
from mcp_server.config import ServerConfig
from mcp_server.identity import AGENT_VAR
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.tools import tool_create_note
from mcp_server.tools_read import tool_read

pytestmark = pytest.mark.usefixtures("roomy_rate_buckets")


def _minimal_note(title: str, type_: str) -> str:
    return (
        "---\n"
        f"title: '{title}'\n"
        f"type: {type_}\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        "## Notes\n"
        "\n"
        "_(empty)_\n"
        "\n"
        "## Log\n"
        "\n"
        "_(empty)_\n"
    )


ANNA = "knowledge/people/anna-kowalska.md"
ACME = "knowledge/organisations/acme.md"
ACME_SOURCE = "knowledge/organisations/acme"  # no-extension wikilink form


def _create_anna_and_acme(cfg: ServerConfig, runtime: Runtime) -> None:
    tool_create_note(cfg, runtime, path=ANNA, content=_minimal_note("Anna Kowalska", "person"))
    tool_create_note(cfg, runtime, path=ACME, content=_minimal_note("Acme", "organisation"))


# --------------------------------------------------------------- (1) create


def test_create_note_commits_and_stamps_provenance(
    mcp_env: McpEnv, run_git: GitRunner
) -> None:
    root, cfg, runtime = mcp_env
    res_anna = tool_create_note(cfg, runtime, path=ANNA, content=_minimal_note("Anna Kowalska", "person"))
    res_acme = tool_create_note(cfg, runtime, path=ACME, content=_minimal_note("Acme", "organisation"))

    assert res_anna.committed and res_anna.commit_sha
    assert res_acme.committed and res_acme.commit_sha
    assert res_anna.commit_sha != res_acme.commit_sha

    log_lines = run_git(root, "log", "--oneline").splitlines()
    assert len(log_lines) == 2
    assert all("mcp(agent-a):" in line for line in log_lines)
    assert any("create note knowledge/people/anna-kowalska.md" in line for line in log_lines)
    assert any("create note knowledge/organisations/acme.md" in line for line in log_lines)

    anna_on_disk = (root / ANNA).read_text(encoding="utf-8")
    acme_on_disk = (root / ACME).read_text(encoding="utf-8")
    for on_disk in (anna_on_disk, acme_on_disk):
        assert "author: 'agent:agent-a'" in on_disk
        assert "written_via: mcp" in on_disk


# --------------------------------------------------- (2) relation + query


def test_upsert_relation_read_back_and_query(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    _create_anna_and_acme(cfg, runtime)

    res = tool_entity_upsert_relation(
        cfg, runtime, entity_path=ANNA,
        rel="works_at", target="organisations/acme", valid_from="2026-01-01",
    )
    assert res.action == "added"
    assert res.committed and res.commit_sha

    read_back = tool_read(cfg, runtime, path=ANNA)
    assert "rel: works_at" in read_back.content
    assert "target: organisations/acme" in read_back.content
    assert "valid_from: '2026-01-01'" in read_back.content

    out = tool_relations_query(cfg, runtime, rel="works_at")
    assert len(out.relations) == 1
    hit = out.relations[0]
    assert hit.entity == "people/anna-kowalska"
    assert hit.rel == "works_at"
    assert hit.target == "organisations/acme"
    assert hit.valid_from == "2026-01-01"


# ------------------------------------------------------------- (3) fact


def test_append_fact_appends_newest_last(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    _create_anna_and_acme(cfg, runtime)

    first = tool_entity_append_fact(
        cfg, runtime, entity_path=ANNA,
        text="Prefers async standups", source=ACME_SOURCE, date="2026-06-01",
    )
    assert first.committed and first.commit_sha

    second = tool_entity_append_fact(
        cfg, runtime, entity_path=ANNA,
        text="Joined the platform guild", source=ACME_SOURCE, date="2026-06-02",
    )
    assert second.committed and second.commit_sha

    on_disk = (root / ANNA).read_text(encoding="utf-8")
    first_bullet = f"- 2026-06-01 — Prefers async standups ([[{ACME_SOURCE}]])"
    second_bullet = f"- 2026-06-02 — Joined the platform guild ([[{ACME_SOURCE}]])"
    assert first_bullet in on_disk
    assert second_bullet in on_disk
    # Newest fact lands after the first one, both inside ## Log.
    assert on_disk.index("## Log") < on_disk.index(first_bullet) < on_disk.index(second_bullet)


# ------------------------------------------------------- (4) concurrency


def test_two_agents_sequential_provenance(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory,
    run_git: GitRunner,
) -> None:
    # Sequential by construction (one thread, one agent at a time): what it
    # pins is that agent identity is per-call state, not process state — the
    # second call must not inherit the first agent's attribution. The real
    # overlap case is test_concurrent_creates_from_two_threads below.
    root = mcp_vault
    cfg = make_cfg(root)
    runtime = make_runtime(root)

    token_a = AGENT_VAR.set("agent-a")
    try:
        res_a = tool_create_note(
            cfg, runtime, path=ANNA,
            content=_minimal_note("Anna Kowalska", "person"),
        )
    finally:
        AGENT_VAR.reset(token_a)

    token_b = AGENT_VAR.set("agent-b")
    try:
        res_b = tool_create_note(
            cfg, runtime, path=ACME,
            content=_minimal_note("Acme", "organisation"),
        )
    finally:
        AGENT_VAR.reset(token_b)

    assert res_a.committed and res_a.commit_sha
    assert res_b.committed and res_b.commit_sha
    assert res_a.commit_sha != res_b.commit_sha

    log_lines = run_git(root, "log", "--format=%H %s").splitlines()
    assert len(log_lines) == 2
    assert any(line.startswith(f"{res_a.commit_sha} mcp(agent-a):") for line in log_lines)
    assert any(line.startswith(f"{res_b.commit_sha} mcp(agent-b):") for line in log_lines)

    anna_on_disk = (root / ANNA).read_text(encoding="utf-8")
    acme_on_disk = (root / ACME).read_text(encoding="utf-8")
    assert "author: 'agent:agent-a'" in anna_on_disk
    assert "author: 'agent:agent-b'" in acme_on_disk


_THREADS = 6


def test_concurrent_creates_from_two_threads(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory,
    run_git: GitRunner,
) -> None:
    # Real overlap: N threads enter tool_create_note at once, so _write_lock
    # and git_ops._GIT_LOCK are actually contended. Every write must land as
    # its own commit attributed to the calling thread's agent — a lost
    # update, a shared index.lock race or identity bleeding across threads
    # all show up as a missing commit, a missing note or a wrong author.
    root = mcp_vault
    cfg = make_cfg(root)
    runtime = make_runtime(root)
    start = threading.Barrier(_THREADS)
    results: dict[int, str] = {}
    failures: list[str] = []
    guard = threading.Lock()

    def worker(i: int) -> None:
        agent = "agent-a" if i % 2 == 0 else "agent-b"
        AGENT_VAR.set(agent)   # fresh context per thread; nothing to reset
        start.wait(timeout=10.0)
        try:
            res = tool_create_note(
                cfg, runtime, path=f"knowledge/notes/n{i}.md",
                content=f"---\ntitle: 'Note {i}'\n---\n\n# Note {i}\n",
            )
        except Exception as exc:   # noqa: BLE001 — reported, not swallowed
            with guard:
                failures.append(f"{type(exc).__name__}: {exc}")
            return
        with guard:
            assert res.commit_sha is not None
            results[i] = res.commit_sha

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    assert not failures, failures
    assert not any(t.is_alive() for t in threads)

    assert len(results) == _THREADS
    assert len(set(results.values())) == _THREADS   # one commit each, no collisions
    log_lines = run_git(root, "log", "--format=%H %s").splitlines()
    assert len(log_lines) == _THREADS
    by_sha = {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in log_lines}
    for i, sha in results.items():
        agent = "agent-a" if i % 2 == 0 else "agent-b"
        assert by_sha[sha] == f"mcp({agent}): create note knowledge/notes/n{i}.md"
        note = (root / f"knowledge/notes/n{i}.md").read_text(encoding="utf-8")
        assert f"author: 'agent:{agent}'" in note
    # Nothing left staged or half-written: the index survived the contention.
    # (Untracked files excluded: the audit log is never committed.)
    assert run_git(root, "status", "--porcelain", "--untracked-files=no") == ""


# ----------------------------------------------------------- (5) reindex

_DIM = 8


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Deterministic per-text unit vectors — the real embedder never loads."""
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:_DIM], dtype=np.uint8).astype(np.float32) + 1.0
        rows.append(vec / np.linalg.norm(vec))
    return np.vstack(rows)


def _seed_index(root: Path) -> None:
    """A one-row index so semantic.upsert_notes patches it instead of
    falling back to a full build (which would load the real model)."""
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    text = "An archived source paragraph that must survive the refresh."
    np.save(meta_dir / "embeddings.npy", _fake_encode([text]))
    (meta_dir / "embeddings_meta.jsonl").write_text(
        json.dumps({
            "source_relative_path": "uni/lecture.md",
            "source_hash": "h-uni/lecture.md",
            "title": "lecture",
            "chunk_idx": 0,
            "text": text,
            "origin": "text",
            "model": "BAAI/bge-small-en-v1.5",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_create_note_triggers_the_derived_refresh_commit(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory,
    run_git: GitRunner, wait_for: WaitFor,
) -> None:
    # The last leg of the happy path: create -> commit -> reindex -> graph.
    # Every other test here disables the refresher, so the write's
    # index_refresh contract and the background derived-notes commit were
    # only ever exercised by hand (mcp_server/manual_test.py).
    root = mcp_vault
    _seed_index(root)
    refresher = IndexRefresher(
        root, audit=AuditLog(root), debounce_seconds=0.05, encode=_fake_encode,
    )
    cfg = make_cfg(root)
    runtime = make_runtime(root, refresher=refresher)
    token = AGENT_VAR.set("agent-a")
    try:
        res = tool_create_note(
            cfg, runtime, path="knowledge/notes/alpha.md",
            content=(
                "---\ntitle: 'Alpha note'\ntopics: [Alpha]\n---\n\n"
                "# Alpha note\n\nA paragraph about alpha long enough to clear the "
                "eighty-character minimum chunk size and produce one embedding row.\n"
            ),
        )
        assert res.committed
        assert res.index_refresh == "queued"    # not "off"/"skipped" this time
        assert wait_for(
            lambda: (root / "knowledge" / "concepts" / "alpha.md").exists()
            and len(run_git(root, "log", "--format=%s").splitlines()) > 1
        ), "the derived rebuild never committed"
    finally:
        AGENT_VAR.reset(token)
        refresher.stop(flush_seconds=5.0)

    subjects = run_git(root, "log", "--format=%s").splitlines()
    # Newest first: the refresh commit lands AFTER the write it followed.
    assert subjects[0] == "mcp: refresh derived notes"
    assert subjects[1] == "mcp(agent-a): create note knowledge/notes/alpha.md"
    committed = set(
        run_git(root, "show", "--name-only", "--format=", "HEAD").splitlines()
    )
    assert "knowledge/concepts/alpha.md" in committed
    # Only derived notes ride in the refresh commit — never the source note.
    assert all(
        p.startswith(("knowledge/concepts/", "knowledge/index/")) for p in committed
    ), committed
    # The note itself was indexed, and the pre-existing row survived.
    sources = {
        json.loads(ln)["source_relative_path"]
        for ln in (root / "metadata" / "embeddings_meta.jsonl")
        .read_text(encoding="utf-8").splitlines() if ln.strip()
    }
    assert sources == {"uni/lecture.md", "knowledge/notes/alpha.md"}
