"""Tests for mcp_server.reindex.IndexRefresher — the background
derived-state refresher behind the MCP write tools.

The real embedding model is never loaded: every test seeds the index
files by hand with a deterministic hash encoder (the fixture approach
from test_semantic_upsert) and injects the same encoder into the
refresher. Graph rebuilds and the derived-notes commit run against a
throwaway git repo, fully offline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from ingest_lib import semantic
from ingest_lib.config import VaultPaths, paths_for_root

from mcp_server.audit import AuditLog
from mcp_server.reindex import IndexRefresher

_LOG = logging.getLogger("test")
_DIM = 8


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Deterministic per-text unit vectors derived from sha256."""
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:_DIM], dtype=np.uint8).astype(np.float32) + 1.0
        rows.append(vec / np.linalg.norm(vec))
    return np.vstack(rows)


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _git_vault(tmp_path: Path) -> VaultPaths:
    paths = _vault(tmp_path)
    subprocess.run(["git", "init", "-q", "-b", "main", str(paths.root)], check=True)
    for k, v in (("user.email", "test@example.com"), ("user.name", "test")):
        subprocess.run(["git", "-C", str(paths.root), "config", k, v], check=True)
    return paths


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _seed_index(paths: VaultPaths, rows: list[tuple[str, int, str]]) -> None:
    """Hand-build the index files so build_index (and the model) never runs."""
    vecs = _fake_encode([text for _, _, text in rows])
    np.save(paths.metadata / "embeddings.npy", vecs)
    with (paths.metadata / "embeddings_meta.jsonl").open("w", encoding="utf-8") as fh:
        for rel, idx, text in rows:
            fh.write(json.dumps({
                "source_relative_path": rel,
                "source_hash": "h-" + rel,
                "title": Path(rel).stem,
                "chunk_idx": idx,
                "text": text,
                "origin": "knowledge-note" if rel.startswith("knowledge/") else "text",
                "model": "BAAI/bge-small-en-v1.5",
            }, ensure_ascii=False) + "\n")


def _meta_sources(paths: VaultPaths) -> set[str]:
    with (paths.metadata / "embeddings_meta.jsonl").open("r", encoding="utf-8") as fh:
        return {json.loads(ln)["source_relative_path"] for ln in fh if ln.strip()}


def _write_note(paths: VaultPaths, rel: str, title: str, topics: str = "[]") -> None:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"---\ntitle: {title}\ntopics: {topics}\n---\n\n# {title}\n\n"
        f"A paragraph about {title} long enough to clear the eighty-character "
        f"minimum chunk size and produce one embedding row.\n",
        encoding="utf-8",
    )


_KEPT = ("An unrelated archived source paragraph that must survive every "
         "refresh completely untouched, byte for byte.")


def _wait_for(cond, timeout: float = 8.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


# ----------------------------------------------------------------- tests

def test_burst_of_enqueues_coalesces_into_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT)])
    _write_note(paths, "knowledge/notes/a.md", "alpha note")
    _write_note(paths, "knowledge/people/b.md", "bravo person")

    calls: list[list[str]] = []
    real_upsert = semantic.upsert_notes

    def recording_upsert(p, rels, **kwargs):
        calls.append(list(rels))
        return real_upsert(p, rels, **kwargs)

    monkeypatch.setattr(semantic, "upsert_notes", recording_upsert)
    audit = AuditLog(paths.root)
    # 0.2s debounce: long enough that the two enqueues below can't be
    # split into separate batches by scheduler jitter, short enough to
    # keep the test fast.
    refresher = IndexRefresher(
        paths.root, audit=audit, debounce_seconds=0.2, encode=_fake_encode
    )
    try:
        assert refresher.enqueue("knowledge/notes/a.md", graph_changed=False) == "queued"
        assert refresher.enqueue("knowledge/people/b.md", graph_changed=False) == "queued"
        assert _wait_for(
            lambda: {"knowledge/notes/a.md", "knowledge/people/b.md"}
            <= _meta_sources(paths)
        ), "upsert never reflected both notes in the meta jsonl"
    finally:
        refresher.stop(flush_seconds=2.0)

    # The 0.2s debounce coalesced both enqueues into exactly one batch.
    assert calls == [["knowledge/notes/a.md", "knowledge/people/b.md"]]
    assert "uni/lecture.md" in _meta_sources(paths)  # untouched row survives
    # No graph_changed entry: no rebuild ran, so no concept notes appeared.
    assert not list((paths.knowledge / "concepts").glob("*.md"))


def test_disabled_refresher_reports_off_and_does_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    refresher = IndexRefresher(paths.root, audit=AuditLog(paths.root), enabled=False)
    assert refresher.enqueue("knowledge/notes/a.md", graph_changed=True) == "off"
    assert refresher.pending() == 0
    assert refresher._thread is None  # lazy start never happened


def test_graph_change_rebuilds_and_commits_only_derived_paths(tmp_path: Path) -> None:
    paths = _git_vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT)])
    _write_note(paths, "knowledge/notes/alpha.md", "alpha", topics="[Alpha]")

    pushes: list[str] = []

    def fake_push() -> str:
        pushes.append("requested")
        return "queued"

    refresher = IndexRefresher(
        paths.root,
        audit=AuditLog(paths.root),
        debounce_seconds=0.05,
        encode=_fake_encode,
        request_push=fake_push,
    )
    try:
        assert refresher.enqueue("knowledge/notes/alpha.md", graph_changed=True) == "queued"
        assert _wait_for(
            lambda: (paths.knowledge / "concepts" / "alpha.md").exists()
            and _git(paths.root, "log", "--oneline", "--all").strip() != ""
        ), "derived rebuild never committed"
    finally:
        refresher.stop(flush_seconds=2.0)

    assert _git(paths.root, "log", "-1", "--format=%s") == "mcp: refresh derived notes"
    committed = set(_git(paths.root, "show", "--name-only", "--format=", "HEAD").splitlines())
    # ONLY derived notes are in the commit — not the source note, not the
    # (untracked) embeddings/connections artefacts. Concept notes always;
    # entity dashboards ride along if the dashboards module covers the group.
    assert "knowledge/concepts/alpha.md" in committed
    assert all(
        p.startswith(("knowledge/concepts/", "knowledge/index/")) for p in committed
    ), committed
    assert pushes == ["requested"]
    # The connection graph was rebuilt as part of the pass.
    assert (paths.metadata / "connections.jsonl").exists()


def test_poisoned_batch_keeps_thread_alive_and_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT)])
    _write_note(paths, "knowledge/notes/ok.md", "recovers fine")

    def boom(*args, **kwargs):
        raise RuntimeError("poisoned batch")

    monkeypatch.setattr(semantic, "upsert_notes", boom)
    audit_path = paths.root / "logs" / "mcp-audit.jsonl"
    refresher = IndexRefresher(
        paths.root, audit=AuditLog(paths.root), debounce_seconds=0.05,
        encode=_fake_encode,
    )
    try:
        refresher.enqueue("knowledge/notes/ok.md", graph_changed=False)
        assert _wait_for(audit_path.exists), "failure was never audited"
        row = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["agent"] == "system"
        assert row["tool"] == "reindex"
        assert row["outcome"] == "failed"
        assert "poisoned batch" in row["detail"]

        # The worker thread survived: heal the upsert and enqueue again.
        monkeypatch.undo()
        refresher.enqueue("knowledge/notes/ok.md", graph_changed=False)
        assert _wait_for(
            lambda: "knowledge/notes/ok.md" in _meta_sources(paths)
        ), "thread died after the poisoned batch"
    finally:
        refresher.stop(flush_seconds=2.0)


def test_stop_drains_pending_work(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _seed_index(paths, [("uni/lecture.md", 0, _KEPT)])
    _write_note(paths, "knowledge/notes/late.md", "last-second write")

    # Debounce far longer than the test: the batch can only be processed
    # by stop()'s best-effort drain, never by the worker loop.
    refresher = IndexRefresher(
        paths.root, audit=AuditLog(paths.root), debounce_seconds=60.0,
        encode=_fake_encode,
    )
    assert refresher.enqueue("knowledge/notes/late.md", graph_changed=False) == "queued"
    refresher.stop(flush_seconds=5.0)
    assert "knowledge/notes/late.md" in _meta_sources(paths)
    assert refresher.pending() == 0
