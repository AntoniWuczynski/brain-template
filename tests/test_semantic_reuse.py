"""`build_index` reuses stored vectors for chunks whose embedded text is unchanged.

Every ingest ends with a full index build, and so does the nightly
maintenance pass. Re-encoding all of it each time is a millisecond a chunk on
a Mac's GPU and seventeen minutes for 16,000 chunks on the CPU-only server
(measured 2026-09-20, after ONE new meeting). Same model, same input string,
same vector — so only text the index has not seen needs the model at all.

The property that matters: an incremental build writes exactly what a
from-scratch build would.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from ingest_lib import semantic
from ingest_lib.config import VaultPaths, paths_for_root

_LOG = logging.getLogger("test")
_DIM = 8


def _vec(text: str) -> np.ndarray:
    raw = hashlib.sha256(text.encode("utf-8")).digest()[:_DIM]
    v = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) + 1.0
    return v / np.linalg.norm(v)


class _CountingEmbedder:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        self.encoded.extend(sentences)
        return np.stack([_vec(s) for s in sentences]) if sentences else np.zeros((0, _DIM), np.float32)


def _vault(tmp_path: Path, notes: dict[str, str]) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    for name, body in notes.items():
        p = tmp_path / "knowledge/notes" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return paths


def _use(monkeypatch: pytest.MonkeyPatch, embedder: _CountingEmbedder) -> None:
    monkeypatch.setattr(semantic, "_load_embedder", lambda: (embedder, "cpu"))
    monkeypatch.setattr(semantic, "_INDEX_CACHE", None)


def _index(paths: VaultPaths) -> tuple[np.ndarray, list[dict[str, object]]]:
    vecs = np.load(paths.metadata / "embeddings.npy")
    with (paths.metadata / "embeddings_meta.jsonl").open(encoding="utf-8") as fh:
        rows: list[dict[str, object]] = [json.loads(ln) for ln in fh if ln.strip()]
    return vecs, rows


def _comparable(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rows minus the two fields that legitimately differ between vaults: the
    generation stamp and the hash of the note file."""
    return [{k: v for k, v in r.items() if k not in ("gen", "source_hash")} for r in rows]


_NOTES = {
    "alpha.md": "# Alpha\n\nThe first note talks about photolithography at length.\n",
    "beta.md": "# Beta\n\nThe second note is about pull-before-write and rebasing.\n",
    "gamma.md": "# Gamma\n\nThe third note covers tailnets and firewalls.\n",
}


def test_an_unchanged_vault_encodes_nothing_and_never_loads_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _vault(tmp_path, _NOTES)
    first = _CountingEmbedder()
    _use(monkeypatch, first)
    n = semantic.build_index(paths, logger=_LOG)
    assert n == len(first.encoded) > 0
    before_vecs, before_rows = _index(paths)

    def _must_not_load() -> tuple[_CountingEmbedder, str]:
        raise AssertionError("the model was loaded for a build with nothing to encode")

    monkeypatch.setattr(semantic, "_load_embedder", _must_not_load)
    monkeypatch.setattr(semantic, "_INDEX_CACHE", None)
    assert semantic.build_index(paths, logger=_LOG) == n

    after_vecs, after_rows = _index(paths)
    assert np.array_equal(before_vecs, after_vecs)
    assert [r["text"] for r in before_rows] == [r["text"] for r in after_rows]


def test_only_changed_text_is_encoded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _vault(tmp_path, _NOTES)
    _use(monkeypatch, _CountingEmbedder())
    semantic.build_index(paths, logger=_LOG)

    (tmp_path / "knowledge/notes/beta.md").write_text(
        "# Beta\n\nThe second note now says something else entirely.\n", encoding="utf-8"
    )
    second = _CountingEmbedder()
    _use(monkeypatch, second)
    semantic.build_index(paths, logger=_LOG)

    assert len(second.encoded) >= 1
    assert all("something else entirely" in s or "Beta" in s for s in second.encoded)
    assert not any("photolithography" in s or "tailnets" in s for s in second.encoded)


def test_incremental_build_equals_a_from_scratch_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Add, edit and delete between builds, then compare against a fresh vault
    holding the same final notes."""
    paths = _vault(tmp_path / "inc", _NOTES)
    _use(monkeypatch, _CountingEmbedder())
    semantic.build_index(paths, logger=_LOG)

    final = dict(_NOTES)
    final["beta.md"] = "# Beta\n\nRewritten to be about something different.\n"
    final["delta.md"] = "# Delta\n\nA brand new note about nightly pulls.\n"
    del final["gamma.md"]
    (tmp_path / "inc/knowledge/notes/gamma.md").unlink()
    for name in ("beta.md", "delta.md"):
        (tmp_path / "inc/knowledge/notes" / name).write_text(final[name], encoding="utf-8")
    _use(monkeypatch, _CountingEmbedder())
    semantic.build_index(paths, logger=_LOG)

    fresh = _vault(tmp_path / "fresh", final)
    _use(monkeypatch, _CountingEmbedder())
    semantic.build_index(fresh, logger=_LOG)

    inc_vecs, inc_rows = _index(paths)
    fresh_vecs, fresh_rows = _index(fresh)
    assert np.array_equal(inc_vecs, fresh_vecs)
    assert _comparable(inc_rows) == _comparable(fresh_rows)


def test_a_different_vector_space_reuses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`BRAIN_EMBED_HEADING_CONTEXT` changes what is embedded without changing
    the model. Rows tagged for the other space must not be reused."""
    paths = _vault(tmp_path, _NOTES)
    first = _CountingEmbedder()
    _use(monkeypatch, first)
    semantic.build_index(paths, logger=_LOG)

    monkeypatch.setenv("BRAIN_EMBED_HEADING_CONTEXT", "1")
    second = _CountingEmbedder()
    _use(monkeypatch, second)
    semantic.build_index(paths, logger=_LOG)
    assert len(second.encoded) == len(first.encoded)


def test_a_torn_index_is_rebuilt_in_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _vault(tmp_path, _NOTES)
    first = _CountingEmbedder()
    _use(monkeypatch, first)
    semantic.build_index(paths, logger=_LOG)

    meta = paths.metadata / "embeddings_meta.jsonl"
    meta.write_text("\n".join(meta.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")  # one row short
    second = _CountingEmbedder()
    _use(monkeypatch, second)
    semantic.build_index(paths, logger=_LOG)
    assert len(second.encoded) == len(first.encoded)


def test_identical_text_in_two_notes_shares_one_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = "# Same\n\nTwo notes holding exactly the same paragraph of text.\n"
    paths = _vault(tmp_path, {"one.md": same, "two.md": same, "three.md": _NOTES["alpha.md"]})
    emb = _CountingEmbedder()
    _use(monkeypatch, emb)
    n = semantic.build_index(paths, logger=_LOG)
    assert len(emb.encoded) < n
    vecs, rows = _index(paths)
    for i, row in enumerate(rows):
        assert np.allclose(vecs[i], _vec(str(row["text"])))
