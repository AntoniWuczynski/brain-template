"""Unit tests for the deterministic concept-description logic.

The LLM call and embedding retrieval are exercised by running the real
pipeline; here we cover the pure pieces: cache-key hashing, AI-zone
rendering, and the surgical insert/replace that must preserve the user's
hand-written notes.
"""
from __future__ import annotations

import json

import numpy as np

from ingest_lib.config import paths_for_root
from ingest_lib.describe import (
    ConceptDescription,
    KeyDefinition,
    _retrieve,
    existing_description_hash,
    render_ai_zone_body,
    source_set_hash,
    upsert_ai_zone,
)

_NOTE = (
    "---\n"
    "title: Beta\n"
    "type: concept\n"
    "---\n\n"
    "<!-- AUTO-GENERATED-START -->\n"
    "# Beta\n\n"
    "## Sources (1)\n"
    "- [[knowledge/index/a]]\n"
    "<!-- AUTO-GENERATED-END -->\n\n"
    "# Notes\n\n"
    "my own thoughts\n"
)


def test_source_set_hash_is_order_independent_and_sensitive():
    h = source_set_hash(["b.md", "a.md"], "anthropic/claude-haiku-4-5")
    assert h == source_set_hash(["a.md", "b.md"], "anthropic/claude-haiku-4-5")
    assert h != source_set_hash(["a.md"], "anthropic/claude-haiku-4-5")
    assert h != source_set_hash(["a.md", "b.md"], "openai/gpt-5-mini")


def test_content_keyed_sources_invalidate_on_hash_change():
    # D3: rebuild_descriptions keys on `path@source_hash`, so a revised source
    # (same path, new content hash) produces a different cache key -> regenerate.
    same_path_v1 = source_set_hash(["uni/x.pdf@hash1"], "m")
    same_path_v2 = source_set_hash(["uni/x.pdf@hash2"], "m")
    assert same_path_v1 != same_path_v2


def test_render_ai_zone_body_embeds_hash_and_sections():
    desc = ConceptDescription(
        short_summary="A short take.",
        detailed_explanation="## Overview\nLong text.",
        key_definitions=[KeyDefinition(term="Foo", definition="A foo.")],
    )
    body = render_ai_zone_body(desc, source_hash="abc123", model="anthropic/x")
    assert "ai-hash: abc123" in body
    assert "A short take." in body
    assert "Long text." in body
    assert "**Foo**" in body and "A foo." in body


def test_upsert_inserts_ai_zone_after_auto_marker_preserving_user_notes():
    out = upsert_ai_zone(_NOTE, "<!-- ai-hash: abc -->\n## Description\nhello")
    assert "<!-- AI-GENERATED-START -->" in out
    # Ordering: auto-zone, then AI-zone, then the user's Notes.
    assert out.index("AUTO-GENERATED-END") < out.index("AI-GENERATED-START")
    assert out.index("AI-GENERATED-END") < out.index("# Notes")
    assert "my own thoughts" in out


def _seed_index(tmp_path, rows):
    """Write a tiny embeddings index: identity-ish vectors so query==row text."""
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    # One orthonormal basis vector per row; query encodes to the basis vector
    # matching its text (see the fake embedder below).
    dim = len(rows)
    vecs = np.eye(dim, dtype=np.float32)
    np.save(paths.metadata / "embeddings.npy", vecs)
    with (paths.metadata / "embeddings_meta.jsonl").open("w") as fh:
        for src, text in rows:
            fh.write(json.dumps({
                "source_relative_path": src, "text": text, "chunk_idx": 0,
                "title": src, "origin": "", "source_hash": "h",
            }) + "\n")
    return paths, vecs


def test_retrieve_masks_to_concepts_own_sources(tmp_path, monkeypatch):
    # F031: retrieval must return ONLY chunks from the concept's own sources,
    # never borrow semantically-close text from unrelated documents.
    rows = [
        ("uni/own.pdf", "chunk from the concept's own source"),
        ("uni/other.pdf", "chunk from an unrelated document"),
    ]
    paths, vecs = _seed_index(tmp_path, rows)

    class _FakeModel:
        def encode(self, texts, **kw):
            # Query 0 ("concept") points at row 0; but we allow only row 1's
            # source, so masking must yield empty rather than row 0's text.
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("ingest_lib.semantic._load_embedder", lambda: (_FakeModel(), "cpu"))

    # Allowed = the OTHER source only; the top cosine hit (row 0) is masked out.
    got = _retrieve(paths, [("concept", {"uni/other.pdf"})], top_k=5)
    assert got == [["chunk from an unrelated document"]]

    # Allowed = a source with no chunks -> honest empty (no borrowing).
    got2 = _retrieve(paths, [("concept", {"uni/missing.pdf"})], top_k=5)
    assert got2 == [[]]


def test_upsert_replaces_existing_ai_zone_in_place_keeping_user_notes():
    once = upsert_ai_zone(_NOTE, "<!-- ai-hash: old -->\nold description")
    twice = upsert_ai_zone(once, "<!-- ai-hash: new -->\nnew description")
    assert "old description" not in twice
    assert "new description" in twice
    assert twice.count("<!-- AI-GENERATED-START -->") == 1
    assert "my own thoughts" in twice


def test_existing_description_hash_reads_ai_zone_comment():
    note = upsert_ai_zone(_NOTE, "<!-- ai-hash: deadbeef -->\n## Description\nx")
    assert existing_description_hash(note) == "deadbeef"
    assert existing_description_hash(_NOTE) is None
