"""Unit tests for the deterministic figure-captioning logic.

The vision call and file walking are verified by running; here we cover the
pure pieces: finding asset image refs, hashing, and the idempotent inline
caption upsert.
"""
from __future__ import annotations

from ingest_lib.caption import image_refs, image_sha256, upsert_caption

_MD = (
    "# Doc\n\nIntro text.\n\n"
    "![](doc_assets/abc.jpg)\n\n"
    "Middle text.\n\n"
    "![](doc_assets/def.png)\n\n"
    "End text.\n"
)


def test_image_refs_finds_asset_images_only():
    refs = image_refs(_MD)
    assert refs == ["doc_assets/abc.jpg", "doc_assets/def.png"]


def test_image_refs_ignores_external_and_non_asset_images():
    md = "![](https://example.com/logo.png)\n![](doc_assets/x.jpg)\n![](plain.png)"
    assert image_refs(md) == ["doc_assets/x.jpg"]


def test_image_sha256_is_stable_lowercase_hex():
    h = image_sha256(b"hello")
    assert h == image_sha256(b"hello")
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_upsert_caption_inserts_after_its_image_and_is_idempotent():
    once = upsert_caption(_MD, "doc_assets/abc.jpg", "abcd1234", "A bar chart of latency.")
    assert "<!-- caption: abcd1234 -->" in once
    assert "_Figure: A bar chart of latency._" in once
    # The caption sits between its own image and the next one.
    assert once.index("abc.jpg") < once.index("A bar chart") < once.index("def.png")
    assert "Middle text." in once and "End text." in once
    # Running again changes nothing.
    assert upsert_caption(once, "doc_assets/abc.jpg", "abcd1234", "A bar chart of latency.") == once


def test_upsert_caption_collapses_multiline_caption_to_one_line():
    out = upsert_caption(_MD, "doc_assets/abc.jpg", "h1", "Line one.\nLine two.")
    assert "_Figure: Line one. Line two._" in out


def test_upsert_caption_replaces_caption_for_same_hash():
    once = upsert_caption(_MD, "doc_assets/abc.jpg", "h1", "Old caption.")
    twice = upsert_caption(once, "doc_assets/abc.jpg", "h1", "New caption.")
    assert "Old caption." not in twice
    assert "New caption." in twice
    assert twice.count("<!-- caption: h1 -->") == 1
