"""Unit tests for the deterministic figure-captioning logic.

The vision call and file walking are verified by running; here we cover the
pure pieces: finding asset image refs, hashing, and the idempotent inline
caption upsert.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from ingest_lib import claude_cli
from ingest_lib.caption import (
    _CAPTION_PROMPT, _CAPTION_SYSTEM, _VISION_PROVIDERS, _append_caption,
    caption_image, image_refs, image_sha256, upsert_caption,
)
from ingest_lib.config import VaultPaths

_MD = (
    "# Doc\n\nIntro text.\n\n"
    "![](doc_assets/abc.jpg)\n\n"
    "Middle text.\n\n"
    "![](doc_assets/def.png)\n\n"
    "End text.\n"
)


def test_image_refs_finds_asset_images_only() -> None:
    refs = image_refs(_MD)
    assert refs == ["doc_assets/abc.jpg", "doc_assets/def.png"]


def test_image_refs_ignores_external_and_non_asset_images() -> None:
    md = "![](https://example.com/logo.png)\n![](doc_assets/x.jpg)\n![](plain.png)"
    assert image_refs(md) == ["doc_assets/x.jpg"]


def test_image_sha256_is_stable_lowercase_hex() -> None:
    h = image_sha256(b"hello")
    assert h == image_sha256(b"hello")
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_upsert_caption_inserts_after_its_image_and_is_idempotent() -> None:
    once = upsert_caption(_MD, "doc_assets/abc.jpg", "abcd1234", "A bar chart of latency.")
    assert "<!-- caption: abcd1234 -->" in once
    assert "_Figure: A bar chart of latency._" in once
    # The caption sits between its own image and the next one.
    assert once.index("abc.jpg") < once.index("A bar chart") < once.index("def.png")
    assert "Middle text." in once and "End text." in once
    # Running again changes nothing.
    assert upsert_caption(once, "doc_assets/abc.jpg", "abcd1234", "A bar chart of latency.") == once


def test_upsert_caption_collapses_multiline_caption_to_one_line() -> None:
    out = upsert_caption(_MD, "doc_assets/abc.jpg", "h1", "Line one.\nLine two.")
    assert "_Figure: Line one. Line two._" in out


def test_upsert_caption_replaces_caption_for_same_hash() -> None:
    once = upsert_caption(_MD, "doc_assets/abc.jpg", "h1", "Old caption.")
    twice = upsert_caption(once, "doc_assets/abc.jpg", "h1", "New caption.")
    assert "Old caption." not in twice
    assert "New caption." in twice
    assert twice.count("<!-- caption: h1 -->") == 1


# ---------------------------------------------- _append_caption torn-tail heal

def _vault(root: Path) -> VaultPaths:
    return VaultPaths(
        root=root,
        inbox=root / "inbox",
        archive_raw=root / "archive" / "raw",
        archive_processed=root / "archive" / "processed",
        archive_failed=root / "archive" / "failed",
        knowledge=root / "knowledge",
        knowledge_index=root / "knowledge" / "index",
        metadata=root / "metadata",
        metadata_index_jsonl=root / "metadata" / "index.jsonl",
        logs=root / "logs",
    )


def test_append_caption_heals_torn_tail(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    (tmp_path / "metadata").mkdir()
    cap = tmp_path / "metadata" / "captions.jsonl"
    # A torn last line: a previous write was cut off with no trailing newline.
    cap.write_text('{"hash": "aaa", "caption": "first", "model": "m"}', encoding="utf-8")

    _append_caption(paths, "bbb", "second", "m")

    rows = [json.loads(ln) for ln in cap.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 2
    assert {r["hash"] for r in rows} == {"aaa", "bbb"}


def test_append_caption_torn_multibyte_tail_does_not_crash(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    (tmp_path / "metadata").mkdir()
    cap = tmp_path / "metadata" / "captions.jsonl"
    # Tail torn mid-multibyte char (é = 0xC3 0xA9, keep only 0xC3): a text-mode
    # last-byte probe would raise UnicodeDecodeError; the binary probe must not.
    cap.write_bytes(b'{"hash": "aaa", "caption": "caf\xc3')

    _append_caption(paths, "bbb", "second", "m")  # must not raise

    good = [ln for ln in cap.read_bytes().split(b"\n") if ln.strip()]
    assert json.loads(good[-1].decode("utf-8"))["hash"] == "bbb"


def test_append_caption_interrupted_create_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create branch cleaned up in `except Exception`, which does not
    catch KeyboardInterrupt: a Ctrl-C between mkstemp and os.replace left a
    `.captions-*.jsonl` behind (gitignored, but still a stray sidecar the
    next run would have to step over)."""
    paths = _vault(tmp_path)
    (tmp_path / "metadata").mkdir()

    def interrupt(src: object, dst: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _append_caption(paths, "aaa", "first", "m")

    assert list((tmp_path / "metadata").iterdir()) == []


def test_claude_code_is_a_vision_provider() -> None:
    assert "claude-code" in _VISION_PROVIDERS


def test_caption_image_routes_to_claude_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG fake figure bytes")
    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "claude-code")
    monkeypatch.delenv("BRAIN_LLM_MODEL", raising=False)
    calls: list[tuple[bytes, str, str, str, str]] = []

    def fake_run_image(
        *, image: bytes, suffix: str, system: str, prompt: str, model: str,
        log: logging.Logger,
    ) -> claude_cli.CliText | None:
        calls.append((image, suffix, system, prompt, model))
        return claude_cli.CliText("A bar chart", False, "m")

    monkeypatch.setattr(claude_cli, "run_image", fake_run_image)

    assert caption_image(img) == "A bar chart"
    assert calls == [
        (img.read_bytes(), ".png", _CAPTION_SYSTEM, _CAPTION_PROMPT, "sonnet")
    ]


def test_rebuild_captions_skips_refs_outside_archive_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ingest_lib import caption as caption_mod

    paths = _vault(tmp_path)
    paths.archive_processed.mkdir(parents=True)
    outside = tmp_path / "private_assets"
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"\x89PNG" + b"x" * 5000)
    (paths.archive_processed / "doc.md").write_text(
        "![](../../private_assets/secret.png)\n", encoding="utf-8"
    )
    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "claude-code")
    calls: list[Path] = []

    def fake_caption_image(path: Path, *, logger: logging.Logger | None = None) -> str:
        calls.append(path)
        return "leaked"

    monkeypatch.setattr(caption_mod, "caption_image", fake_caption_image)

    stats = caption_mod.rebuild_captions(paths, logger=logging.getLogger("t"), min_bytes=1)

    assert calls == []
    assert stats.captioned == 0
    assert stats.skipped_missing == 1
