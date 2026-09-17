"""The ``notes/notion/`` extractor (``extractors/notion_page.py``)."""
from __future__ import annotations

import json
from pathlib import Path

from ingest_lib.extractors import notion_page as notion_page_ex


def _snapshot(**kw: object) -> bytes:
    return json.dumps(kw).encode("utf-8")


def test_notion_page_extractor_renders_note(tmp_path: Path) -> None:
    src = tmp_path / "p.json"
    src.write_bytes(_snapshot(
        connector="notion", id="p1", title="Project Notes",
        url="https://notion.so/p1", created_time="2026-09-01T00:00:00.000Z",
        markdown="# Project Notes\n\nSome content.",
        truncated=False, unknown_blocks=0,
    ))
    res = notion_page_ex.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "# Project Notes" in res.markdown
    assert "**Notion URL:** https://notion.so/p1" in res.markdown
    assert "**Created:** 2026-09-01T00:00:00.000Z" in res.markdown
    assert "**Last edited:**" not in res.markdown  # F4: dropped from the schema
    assert "Some content." in res.markdown
    assert "## Content" in res.markdown


def test_notion_page_extractor_sanitises_url_and_created_time(tmp_path: Path) -> None:
    """F5: url/created_time are connector-supplied and interpolated into a
    Markdown bullet raw; a newline-bearing value must not open a new
    heading/wikilink the way the title-injection defence already stops."""
    src = tmp_path / "p.json"
    malicious_url = "https://notion.so/p1\n\n## Links\n\n- [[knowledge/people/ceo]]"
    malicious_created = "2026-09-01\n\n[[knowledge/people/ceo]]"
    src.write_bytes(_snapshot(
        connector="notion", id="p1", title="Notes",
        url=malicious_url, created_time=malicious_created,
        markdown="body", truncated=False, unknown_blocks=0,
    ))
    res = notion_page_ex.extract(src, tmp_path / "a")
    assert "\n\n## Links" not in res.markdown
    assert "[[" not in res.markdown
    for line in res.markdown.splitlines():
        assert not line.startswith("## Links")


def test_notion_page_empty_markdown_is_partial(tmp_path: Path) -> None:
    src = tmp_path / "p.json"
    src.write_bytes(_snapshot(id="p2", title="Empty", markdown="   ", truncated=False, unknown_blocks=0))
    res = notion_page_ex.extract(src, tmp_path / "a")
    assert res.status == "partial"
    assert any("no content in snapshot" in n for n in res.notes)


def test_notion_page_truncated_is_partial_with_note(tmp_path: Path) -> None:
    src = tmp_path / "p.json"
    src.write_bytes(_snapshot(id="p3", title="Big", markdown="content", truncated=True, unknown_blocks=2))
    res = notion_page_ex.extract(src, tmp_path / "a")
    assert res.status == "partial"
    assert any("truncated" in n for n in res.notes)
    assert any("unknown block" in n for n in res.notes)


def test_notion_page_bad_json_is_manual_review(tmp_path: Path) -> None:
    src = tmp_path / "p.json"
    src.write_bytes(b"{ not json")
    res = notion_page_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"


def test_notion_page_non_object_is_manual_review(tmp_path: Path) -> None:
    src = tmp_path / "p.json"
    src.write_bytes(b"[1, 2, 3]")
    res = notion_page_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"


def test_notion_page_snapshot_routes_by_source_class() -> None:
    from ingest_lib.extractors import dispatch_extractor
    routed = dispatch_extractor(
        Path("x.json"), relative_path="notes/notion/project-notes-abcd1234-20260910t100000.json"
    )
    assert routed is notion_page_ex.extract
