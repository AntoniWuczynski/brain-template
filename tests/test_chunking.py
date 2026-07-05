"""chunk_markdown must not embed YAML frontmatter as content. Knowledge
notes (unlike processed notes) start with a ``---`` fenced YAML block."""
from __future__ import annotations

from ingest_lib.semantic import chunk_markdown

NOTE = (
    "---\n"
    'title: "brain"\n'
    "type: project\n"
    "source_repo: \"git@github.com:acme/brain.git\"\n"
    "topics: [knowledge-management, mcp]\n"
    "---\n"
    "\n"
    "# brain\n"
    "\n"
    "## Overview\n"
    "A personal knowledge vault with an MCP server in front of it, "
    "letting agents search, read, and write curated notes from anywhere.\n"
)


def test_frontmatter_is_not_embedded() -> None:
    chunks = chunk_markdown(NOTE)
    joined = "\n".join(chunks)
    assert "source_repo" not in joined
    assert "type: project" not in joined
    assert "personal knowledge vault" in joined


def test_plain_text_unaffected() -> None:
    text = "word " * 40  # one paragraph, > _MIN_CHARS
    assert chunk_markdown(text) == [text.strip()]


def test_processing_notes_footer_not_embedded() -> None:
    # D5a: the trailing '## Processing notes' footer (incl. verbatim MinerU
    # errors) must not be embedded — it would surface for metadata queries.
    processed = (
        "# Lecture\n\n"
        "> Source: `x.pdf`  \n"
        "> Status: `partial`\n\n"
        "---\n\n"
        "The real lecture body about congestion control, long enough to "
        "clear the minimum chunk size threshold easily.\n\n"
        "---\n\n"
        "## Processing notes\n\n"
        "- mineru-error: CUDA out of memory on device 0\n"
    )
    joined = "\n".join(chunk_markdown(processed))
    assert "congestion control" in joined
    assert "CUDA out of memory" not in joined
    assert "Processing notes" not in joined


def test_short_chunk_survives_with_lower_floor() -> None:
    # D4: a one-line memory fact is below the default floor but must index
    # when the curated-note floor (1) is passed.
    fact = "Prefers uv over pip."
    assert chunk_markdown(fact) == []                 # dropped at default floor
    assert chunk_markdown(fact, min_chars=1) == [fact]  # kept for curated notes


def test_horizontal_rule_mid_document_not_treated_as_frontmatter() -> None:
    text = (
        "Intro paragraph that is long enough to clear the minimum chunk "
        "size threshold easily.\n\n---\n\nSecond paragraph also long enough "
        "to clear the minimum chunk size threshold easily.\n"
    )
    joined = "\n".join(chunk_markdown(text))
    assert "Intro paragraph" in joined
    assert "Second paragraph" in joined


def test_trailing_whitespace_fence_still_stripped_without_eating_body() -> None:
    # Hand-edited fences aren't byte-exact. The chunker must agree with
    # notes._split_frontmatter (which accepts '--- ') instead of scanning
    # past the real fence and swallowing body up to a later '---' rule.
    text = (
        "---\n"
        "topics: [rng]\n"
        "--- \n"
        "\n"
        "Opening paragraph that is long enough to clear the minimum chunk "
        "size threshold easily.\n\n---\n\nClosing paragraph also long enough "
        "to clear the minimum chunk size threshold easily.\n"
    )
    joined = "\n".join(chunk_markdown(text))
    assert "Opening paragraph" in joined
    assert "Closing paragraph" in joined
    assert "topics" not in joined


def test_leading_horizontal_rule_without_frontmatter_keeps_body() -> None:
    text = (
        "---\n"
        "\n"
        "Opening paragraph that is long enough to clear the minimum chunk "
        "size threshold easily.\n\n---\n\nClosing paragraph also long enough "
        "to clear the minimum chunk size threshold easily.\n"
    )
    joined = "\n".join(chunk_markdown(text))
    assert "Opening paragraph" in joined
    assert "Closing paragraph" in joined


# ------------------------------------------------ hybrid search modes (P4)

def test_search_modes(tmp_path, monkeypatch):
    import json
    import numpy as np
    from ingest_lib.config import paths_for_root
    from ingest_lib import semantic, lexical

    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    # Two chunks. Dense vectors: row0 aligns with the query vector; row1 does
    # not. But only row1's TEXT contains the exact token 'comp0141'.
    rows = [
        {"source_relative_path": "a.pdf", "text": "an introduction to graphs",
         "chunk_idx": 0, "title": "a", "origin": "", "source_hash": "h"},
        {"source_relative_path": "b.pdf", "text": "COMP0141 error handling lecture",
         "chunk_idx": 0, "title": "b", "origin": "", "source_hash": "h"},
    ]
    np.save(paths.metadata / "embeddings.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    with (paths.metadata / "embeddings_meta.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lexical._CACHE = None  # avoid cross-test cache

    class _FakeModel:
        def encode(self, texts, **kw):
            return np.array([[1.0, 0.0]], dtype=np.float32)  # points at row0
    monkeypatch.setattr(semantic, "_load_embedder", lambda: (_FakeModel(), "cpu"))

    q = "COMP0141"
    dense = semantic.search(paths, q, top_k=2, mode="dense")
    lex = semantic.search(paths, q, top_k=2, mode="lexical")
    hyb = semantic.search(paths, q, top_k=2, mode="hybrid")

    # Dense ranks row0 first (its vector aligns), ignoring the exact token.
    assert dense[0].source_relative_path == "a.pdf"
    # Lexical ranks the exact-token doc first (and only it matches).
    assert [h.source_relative_path for h in lex] == ["b.pdf"]
    # Hybrid surfaces the exact-token doc at the top via RRF.
    assert hyb[0].source_relative_path == "b.pdf"


def test_lexical_mode_needs_no_embedder(tmp_path, monkeypatch):
    import json
    import numpy as np
    from ingest_lib.config import paths_for_root
    from ingest_lib import semantic, lexical

    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    np.save(paths.metadata / "embeddings.npy", np.array([[1.0]], dtype=np.float32))
    (paths.metadata / "embeddings_meta.jsonl").write_text(
        json.dumps({"source_relative_path": "x.pdf", "text": "unique-token here",
                    "chunk_idx": 0, "title": "x", "origin": "", "source_hash": "h"}) + "\n",
        encoding="utf-8")
    lexical._CACHE = None

    # If lexical mode touched the embedder this would raise.
    def _boom():
        raise AssertionError("lexical mode must not load the embedding model")
    monkeypatch.setattr(semantic, "_load_embedder", _boom)

    hits = semantic.search(paths, "unique-token", top_k=1, mode="lexical")
    assert [h.source_relative_path for h in hits] == ["x.pdf"]
