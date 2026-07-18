"""Dream-pass gate: state IO, pending marker, gate arithmetic, packet
determinism. All vaults are tmp_path fixtures with a real git repo so the
gate's git arithmetic runs against genuine history; commit dates are
pinned via GIT_AUTHOR_DATE/GIT_COMMITTER_DATE for determinism."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.dream import (
    DreamState,
    build_packet,
    evaluate_gate,
    load_pending_since,
    load_state,
    mark_done,
    record_pending,
)
from ingest_lib.hashing import sha256_of
from ingest_lib.metadata import IndexRecord, append_record

_LOG = logging.getLogger("test")
AS_OF = datetime(2026, 6, 12, 5, 0, tzinfo=UTC)


def _git(root: Path, *args: str, when: str | None = None) -> str:
    env = dict(os.environ)
    if when is not None:
        env["GIT_AUTHOR_DATE"] = when
        env["GIT_COMMITTER_DATE"] = when
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=True, env=env,
    )
    return result.stdout


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / ".keep").write_text("", encoding="utf-8")
    _commit(paths, "init", when="2026-05-01T12:00:00+00:00")
    return paths


def _commit(paths: VaultPaths, msg: str, *, when: str) -> str:
    _git(paths.root, "add", "-A")
    # --allow-empty: some tests commit purely to mint a base sha
    _git(paths.root, "commit", "-q", "--allow-empty", "-m", msg, when=when)
    return _git(paths.root, "rev-parse", "HEAD").strip()


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    target = paths.root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_load_state_missing_returns_none(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    assert load_state(paths) is None


def test_load_state_malformed_returns_none(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    (paths.metadata / "dream.json").write_text("not json", encoding="utf-8")
    assert load_state(paths) is None
    (paths.metadata / "dream.json").write_text('{"last_run": 3}', encoding="utf-8")
    assert load_state(paths) is None


def test_mark_done_writes_state_and_clears_pending(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    record_pending(paths, now=AS_OF)
    assert (paths.metadata / "dream.pending").is_file()

    state = mark_done(paths, now=AS_OF, logger=_LOG)

    head = _git(paths.root, "rev-parse", "HEAD").strip()
    assert state == DreamState(last_run="2026-06-12T05:00:00Z", last_commit=head)
    assert load_state(paths) == state
    assert not (paths.metadata / "dream.pending").exists()


def test_record_pending_first_detection_wins(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    record_pending(paths, now=AS_OF)
    later = datetime(2026, 6, 14, 5, 0, tzinfo=UTC)
    record_pending(paths, now=later)  # must NOT overwrite the first stamp
    since = load_pending_since(paths)
    assert since == AS_OF


def test_load_pending_since_absent_or_malformed(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    assert load_pending_since(paths) is None
    (paths.metadata / "dream.pending").write_text("garbage", encoding="utf-8")
    assert load_pending_since(paths) is None


def _note(i: int) -> str:
    return f"knowledge/notes/n{i}.md"


def _add_notes(paths: VaultPaths, count: int, *, when: str, start: int = 0) -> str:
    for i in range(start, start + count):
        _write(paths, _note(i), f"# n{i}\nbody\n")
    return _commit(paths, f"add {count} notes", when=when)


def _add_source_record(paths: VaultPaths, rel: str, created_at: str) -> None:
    raw = paths.archive_raw / rel
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("source", encoding="utf-8")
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path=rel,
        source_hash=sha256_of(raw),
        size_bytes=6,
        extension=".txt",
        extractor="text",
        status="processed",
        raw_path=f"archive/raw/{rel}",
        processed_path=None,
        index_note_path=None,
        created_at=created_at,
    ))


def test_gate_first_run_nothing_recent_skips(tmp_path: Path) -> None:
    paths = _vault(tmp_path)  # only the 2026-05-01 init commit, > 7 days old
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    assert not verdict.should_dream
    assert verdict.days_since_last is None
    assert verdict.changed_notes == ()


def test_gate_first_run_with_recent_change_dreams(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _add_notes(paths, 1, when="2026-06-10T12:00:00+00:00")  # inside the 7-day window
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    assert verdict.should_dream
    assert verdict.changed_notes == (_note(0),)


def test_gate_below_threshold_skips(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    _add_notes(paths, 4, when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert not verdict.should_dream
    assert len(verdict.changed_notes) == 4
    assert verdict.days_since_last == 1


def test_gate_at_threshold_dreams(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    _add_notes(paths, 5, when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert verdict.should_dream
    assert verdict.base_commit == base


def test_gate_stale_override_dreams_on_one_change(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-01T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-01T12:00:00Z"}), encoding="utf-8")
    _add_notes(paths, 1, when="2026-06-02T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5, stale_days=7)
    assert verdict.should_dream
    assert "stale" in verdict.reason


def test_gate_counts_new_index_sources(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    base = _commit(paths, "base", when="2026-06-10T12:00:00+00:00")
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": base, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    for i in range(5):
        _add_source_record(paths, f"inbox/s{i}.txt", "2026-06-11T09:00:00Z")
    _add_source_record(paths, "inbox/old.txt", "2026-06-01T09:00:00Z")  # before last_run
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF, threshold=5)
    assert verdict.should_dream
    assert len(verdict.new_sources) == 5
    assert "inbox/old.txt" not in verdict.new_sources


def test_gate_missing_base_commit_falls_back(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    (paths.metadata / "dream.json").write_text(json.dumps(
        {"last_commit": "0" * 40, "last_run": "2026-06-10T12:00:00Z"}), encoding="utf-8")
    _add_notes(paths, 1, when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)  # must not raise
    assert verdict.changed_notes == (_note(0),)


def _write_connections(paths: VaultPaths, lines: list[dict[str, object]]) -> None:
    text = "\n".join(json.dumps(ln, sort_keys=True) for ln in lines) + "\n"
    (paths.metadata / "connections.jsonl").write_text(text, encoding="utf-8")


def test_packet_candidate_pairs_semantic_without_cooccurrence(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _add_notes(paths, 1, when="2026-06-10T12:00:00+00:00")
    _write_connections(paths, [
        {"a": "alpha", "b": "beta", "kind": "semantic", "weight": 0.9, "sources": []},
        {"a": "alpha", "b": "beta", "kind": "cooccurrence", "weight": 2.0, "sources": ["x"]},
        {"a": "gamma", "b": "delta", "kind": "semantic", "weight": 0.7, "sources": []},
        {"a": "gamma", "b": "epsilon", "kind": "semantic", "weight": 0.8, "sources": []},
    ])
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    packet = build_packet(paths, verdict=verdict, top_k=10, logger=_LOG)
    # (alpha, beta) is already linked by cooccurrence -> excluded;
    # the rest ranked by weight descending.
    assert packet["candidate_pairs"] == [
        {"a": "gamma", "b": "epsilon", "weight": 0.8},
        {"a": "gamma", "b": "delta", "weight": 0.7},
    ]


def test_packet_top_k_caps_pairs(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _add_notes(paths, 1, when="2026-06-10T12:00:00+00:00")
    _write_connections(paths, [
        {"a": f"c{i}", "b": f"d{i}", "kind": "semantic", "weight": 0.5 + i / 100, "sources": []}
        for i in range(15)
    ])
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    packet = build_packet(paths, verdict=verdict, top_k=3, logger=_LOG)
    assert len(packet["candidate_pairs"]) == 3
    assert packet["candidate_pairs"][0]["a"] == "c14"


def test_packet_active_entities_and_dream_inventory(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, "knowledge/people/anna.md", "# anna\n")
    _write(paths, "knowledge/projects/brain/log/2026-06-11.md", "# log\n")
    _write(paths, "knowledge/notes/dreams/digests/brain-week.md", "# digest\n")
    _commit(paths, "entities", when="2026-06-11T12:00:00+00:00")
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    packet = build_packet(paths, verdict=verdict, top_k=10, logger=_LOG)
    assert packet["active_entities"] == [
        "knowledge/people/anna",
        "knowledge/projects/brain",
    ]
    assert packet["existing_dream_notes"] == [
        "knowledge/notes/dreams/digests/brain-week.md",
    ]


def test_packet_deterministic(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _add_notes(paths, 2, when="2026-06-10T12:00:00+00:00")
    _write_connections(paths, [
        {"a": "x", "b": "y", "kind": "semantic", "weight": 0.6, "sources": []},
    ])
    verdict = evaluate_gate(paths, logger=_LOG, as_of=AS_OF)
    one = json.dumps(build_packet(paths, verdict=verdict, top_k=5, logger=_LOG), sort_keys=True)
    two = json.dumps(build_packet(paths, verdict=verdict, top_k=5, logger=_LOG), sort_keys=True)
    assert one == two
