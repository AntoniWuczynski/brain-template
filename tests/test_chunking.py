"""chunk_markdown must not embed YAML frontmatter as content. Knowledge
notes (unlike processed notes) start with a ``---`` fenced YAML block."""
from __future__ import annotations

from pathlib import Path

import pytest

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


def test_knowledge_note_body_before_quote_and_rule_is_not_eaten() -> None:
    # The header strip must anchor on the generated '> Source:' block, not
    # any '# heading ... > quote ... ---' shape. A curated note whose body
    # precedes a blockquote and a '---' separator kept all its body invisible
    # to search under the old DOTALL regex.
    note = (
        "# Kern call\n\n"
        "We discussed the retrieval eval harness and agreed to expand the "
        "golden query set before shipping the ranking change.\n\n"
        "> A memorable aside worth keeping in the record.\n\n"
        "---\n\n"
        "Action items captured after the call, also long enough to index.\n"
    )
    joined = "\n".join(chunk_markdown(note))
    assert "retrieval eval harness" in joined
    assert "memorable aside" in joined
    assert "Action items" in joined


def test_generated_processed_header_still_stripped() -> None:
    # The real write_processed_note header (title + '> Source/Hash/Extractor/
    # Status' block + '---') must still be stripped, not embedded.
    processed = (
        "# Lecture 3\n\n"
        "> Source: `university/COMP0005/lec3.pdf`  \n"
        "> Hash: `abc123`  \n"
        "> Extractor: `mineru`  \n"
        "> Status: `processed`\n\n"
        "---\n\n"
        "The genuine lecture body about amortised analysis, long enough to "
        "clear the minimum chunk size threshold comfortably.\n"
    )
    joined = "\n".join(chunk_markdown(processed))
    assert "amortised analysis" in joined
    assert "Source:" not in joined
    assert "Extractor:" not in joined


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

def test_search_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import numpy as np
    from ingest_lib.config import paths_for_root
    from ingest_lib import semantic

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

    class _FakeModel:
        def encode(
            self,
            sentences: list[str],
            *,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
            batch_size: int = 32,
        ) -> np.ndarray:
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


def test_lexical_mode_needs_no_embedder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import numpy as np
    from ingest_lib.config import paths_for_root
    from ingest_lib import semantic

    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    np.save(paths.metadata / "embeddings.npy", np.array([[1.0]], dtype=np.float32))
    (paths.metadata / "embeddings_meta.jsonl").write_text(
        json.dumps({"source_relative_path": "x.pdf", "text": "unique-token here",
                    "chunk_idx": 0, "title": "x", "origin": "", "source_hash": "h"}) + "\n",
        encoding="utf-8")

    # If lexical mode touched the embedder this would raise.
    def _boom() -> tuple[semantic.Embedder, str]:
        raise AssertionError("lexical mode must not load the embedding model")
    monkeypatch.setattr(semantic, "_load_embedder", _boom)

    hits = semantic.search(paths, "unique-token", top_k=1, mode="lexical")
    assert [h.source_relative_path for h in hits] == ["x.pdf"]


def test_search_honors_top_k_above_default_candidate_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A top_k larger than the 100-candidate pool must widen the pool, not be
    # silently truncated — recency.memory_search's filtered path fetches 500.
    import json
    import numpy as np
    from ingest_lib.config import paths_for_root
    from ingest_lib import semantic

    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    n = 150
    # Distinct unit vectors in a 150-d space so every row is a candidate.
    vecs = np.eye(n, dtype=np.float32)
    np.save(paths.metadata / "embeddings.npy", vecs)
    with (paths.metadata / "embeddings_meta.jsonl").open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "source_relative_path": f"doc{i}.md", "text": f"chunk number {i}",
                "chunk_idx": 0, "title": f"t{i}", "origin": "", "source_hash": "h",
            }) + "\n")

    class _FakeModel:
        def encode(
            self,
            sentences: list[str],
            *,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
            batch_size: int = 32,
        ) -> np.ndarray:
            return np.ones((1, n), dtype=np.float32)  # ties across all rows
    monkeypatch.setattr(semantic, "_load_embedder", lambda: (_FakeModel(), "cpu"))

    hits = semantic.search(paths, "anything", top_k=150, mode="dense")
    assert len(hits) == 150


def test_query_instruction_prepended_and_toggleable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BGE v1.5's documented retrieval usage: the instruction is prepended to
    # the QUERY only (passages stay raw), so no reindex is needed.
    import json
    import numpy as np
    from ingest_lib.config import paths_for_root
    from ingest_lib import semantic

    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    np.save(paths.metadata / "embeddings.npy", np.array([[1.0]], dtype=np.float32))
    (paths.metadata / "embeddings_meta.jsonl").write_text(
        json.dumps({"source_relative_path": "x.md", "text": "t", "chunk_idx": 0,
                    "title": "x", "origin": "", "source_hash": "h"}) + "\n",
        encoding="utf-8")

    seen: list[str] = []

    class _FakeModel:
        def encode(
            self,
            sentences: list[str],
            *,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
            batch_size: int = 32,
        ) -> np.ndarray:
            seen.extend(sentences)
            return np.array([[1.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "_load_embedder", lambda: (_FakeModel(), "cpu"))

    monkeypatch.delenv("BRAIN_QUERY_INSTRUCTION", raising=False)
    semantic.search(paths, "graphs", top_k=1, mode="dense")
    assert seen[-1] == "Represent this sentence for searching relevant passages: graphs"

    # Toggle off for A/B measurement: the raw query is encoded.
    monkeypatch.setenv("BRAIN_QUERY_INSTRUCTION", "0")
    semantic.search(paths, "graphs", top_k=1, mode="dense")
    assert seen[-1] == "graphs"


def test_sqlite_vec_tripwire_warns_past_threshold(caplog: pytest.LogCaptureFixture) -> None:
    # 3.b: warn as the chunk count approaches the ~50k sqlite-vec migration
    # point, and stay quiet below it.
    import logging
    from ingest_lib.semantic import _warn_if_index_large, _SQLITE_VEC_HINT_THRESHOLD

    log = logging.getLogger("test.tripwire")
    with caplog.at_level(logging.WARNING, logger="test.tripwire"):
        _warn_if_index_large(_SQLITE_VEC_HINT_THRESHOLD - 1, log)
    assert not caplog.records                       # below: silent

    with caplog.at_level(logging.WARNING, logger="test.tripwire"):
        _warn_if_index_large(_SQLITE_VEC_HINT_THRESHOLD, log)
    assert any("sqlite-vec" in r.message or "sqlite-vec" in r.getMessage() for r in caplog.records)


def test_heading_path_tracked_per_chunk() -> None:
    # 9.a: each chunk carries the heading path in force at its start.
    from ingest_lib.semantic import _pack_blocks_with_headings
    p1 = "Directed content. " * 130   # > _TARGET_CHARS -> own chunk
    p2 = "Undirected content. " * 130
    doc = (
        "# Graph Theory\n\n"
        "## Directed Graphs\n\n" + p1 + "\n\n"
        "## Undirected Graphs\n\n" + p2 + "\n"
    )
    pairs = _pack_blocks_with_headings(doc, min_chars=80)
    kept = {("directed" if "Directed content" in t else "undirected"): h
            for t, h in pairs if "content" in t}
    assert kept["directed"] == "Graph Theory > Directed Graphs"
    assert kept["undirected"] == "Graph Theory > Undirected Graphs"


def test_embed_text_flag_off_is_raw_on_prepends_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from ingest_lib.semantic import Chunk, _embed_text
    c = Chunk(source_relative_path="uni/g.pdf", source_hash="h", title="COMP0005",
              chunk_idx=3, text="a directed graph", origin="pdf-mineru",
              heading_path="4 Graphs > 4.2 Directed")

    monkeypatch.delenv("BRAIN_EMBED_HEADING_CONTEXT", raising=False)
    assert _embed_text(c) == "a directed graph"          # default: raw

    monkeypatch.setenv("BRAIN_EMBED_HEADING_CONTEXT", "1")
    assert _embed_text(c) == "COMP0005 > 4 Graphs > 4.2 Directed\na directed graph"


def test_embed_text_flag_on_without_heading_uses_title_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from ingest_lib.semantic import Chunk, _embed_text
    monkeypatch.setenv("BRAIN_EMBED_HEADING_CONTEXT", "1")
    c = Chunk(source_relative_path="x", source_hash="h", title="Title",
              chunk_idx=0, text="body", heading_path="")
    assert _embed_text(c) == "Title\nbody"


def test_heading_stack_ignores_fenced_code_comments() -> None:
    # 9.a fence-awareness: a '# comment' inside a ``` code fence must NOT be
    # treated as a section heading.
    from ingest_lib.semantic import _pack_blocks_with_headings
    body = "content " * 20
    doc = (
        "# Real Section\n\n"
        "```python\n\n"
        "# not a heading — a code comment\n\n"
        "```\n\n"
        + body + "\n"
    )
    pairs = _pack_blocks_with_headings(doc, min_chars=80)
    heads = {h for t, h in pairs if "content" in t}
    assert heads == {"Real Section"}                 # not "Real Section > not a heading..."


def test_chunk_markdown_byte_identical_with_fence_tracking() -> None:
    # The fence guard must not change chunk TEXT (only heading_path values).
    from ingest_lib.semantic import chunk_markdown, _pack_blocks_with_headings
    doc = "# H\n\n```\n# x\n```\n\n" + ("word " * 60) + "\n"
    assert chunk_markdown(doc) == [t for t, _h in _pack_blocks_with_headings(doc, 80)]


def test_lone_heading_is_carried_into_the_next_chunk() -> None:
    # AUD-042: a buffer holding only a heading must not be flushed as its own
    # chunk — that leaves a "## Log" row of pure noise AND strips the section
    # head off the paragraph that follows.
    from ingest_lib.semantic import _pack_blocks_with_headings

    pairs = _pack_blocks_with_headings("## Log\n\n" + "x" * 1600, min_chars=1)
    assert [t for t, _h in pairs if t.strip() == "## Log"] == []
    assert pairs[0][0].startswith("## Log\n\nxxx")
    assert pairs[0][1] == "Log"


def test_oversize_block_is_split_on_line_boundaries() -> None:
    # AUD-043: a block past the budget used to be embedded whole, so
    # everything beyond the model's window never reached the vector.
    from ingest_lib.semantic import _pack_blocks_with_headings, _TARGET_CHARS

    block = "\n".join(f"[{i}] Author, A. A short bibliography line." for i in range(400))
    pairs = _pack_blocks_with_headings("# Refs\n\n" + block, min_chars=1)
    assert len(pairs) > 1
    assert max(len(t) for t, _h in pairs) <= _TARGET_CHARS + len("# Refs") + 2
    # split on line boundaries: no bibliography entry is cut in half
    rejoined = "\n".join(t for t, _h in pairs).replace("# Refs\n\n", "")
    assert rejoined == block
    assert all(h == "Refs" for _t, h in pairs)


def test_single_line_longer_than_the_budget_is_still_cut() -> None:
    # A blank-line-free, newline-free blob has to be cut mid-line or it
    # cannot be embedded at all.
    from ingest_lib.semantic import _split_oversize_block, _TARGET_CHARS

    pieces = _split_oversize_block("y" * (_TARGET_CHARS * 3 + 7))
    assert max(len(p) for p in pieces) <= _TARGET_CHARS
    assert "".join(pieces) == "y" * (_TARGET_CHARS * 3 + 7)


def test_bom_before_frontmatter_does_not_embed_the_yaml() -> None:
    # AUD-045: a UTF-8 BOM (Windows editors) made the fence unrecognisable,
    # so the whole YAML block got embedded as body text.
    chunks = chunk_markdown("﻿" + NOTE, min_chars=1)
    joined = "\n".join(chunks)
    assert "source_repo" not in joined
    assert "﻿" not in joined
    assert "A personal knowledge vault" in joined


def test_heading_context_flag_is_recorded_in_the_row_model_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # AUD-046: the flag changes the passage vectors without changing the
    # model or the dimension, so it has to travel in the per-row model tag
    # for upsert_notes' mismatch guard to see it.
    from ingest_lib.semantic import Chunk, _MODEL_NAME, _meta_row

    c = Chunk(source_relative_path="knowledge/notes/a.md", source_hash="h",
              title="A", chunk_idx=0, text="body", origin="knowledge-note",
              heading_path="H")
    monkeypatch.delenv("BRAIN_EMBED_HEADING_CONTEXT", raising=False)
    assert _meta_row(c)["model"] == _MODEL_NAME
    monkeypatch.setenv("BRAIN_EMBED_HEADING_CONTEXT", "1")
    assert _meta_row(c)["model"] == f"{_MODEL_NAME}+heading"


def test_hybrid_score_is_not_documented_as_a_cosine() -> None:
    # AUD-037: SearchHit.score is mode-dependent; the default (hybrid) is an
    # RRF rank score bounded by 2/(k+1), not a similarity.
    from ingest_lib.semantic import SearchHit, _RRF_K

    doc = SearchHit.__doc__ or ""
    assert "reciprocal-rank-fusion" in doc
    assert "2/(_RRF_K + 1)" in doc
    assert _RRF_K == 60
