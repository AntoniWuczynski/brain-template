"""The shared transcript extractor and its connector/extractor registration.

Split out of ``tests/test_transcripts.py`` (that file's own 1000-line limit)
along its existing section boundary; see ``tests/test_transcripts.py`` and
its sibling ``test_transcripts_*`` modules for the rest of the split.

Fixtures are hand-built JSONL/JSON shaped like real exports rather than
recorded real sessions, so these run offline and deterministically without
embedding anyone's actual conversation content.
"""
from __future__ import annotations

import json
from pathlib import Path

from ingest_lib.connectors import CONNECTORS
from ingest_lib.extractors import dispatch_extractor
from ingest_lib.extractors import transcript as transcript_ex


def _snapshot(**kw: object) -> bytes:
    return json.dumps(kw).encode("utf-8")


def test_transcript_extractor_renders_note(tmp_path: Path) -> None:
    src = tmp_path / "t.json"
    src.write_bytes(_snapshot(
        connector="claude_code", id="s1", title="brain: fix the thing",
        date="2026-09-03", context="~/brain",
        turns=[
            {"role": "user", "text": "Can you fix the bug?"},
            {"role": "assistant", "text": "Done, here is what changed."},
        ],
        stats={"dropped_tool_uses": 3, "dropped_thinking_blocks": 1},
    ))
    res = transcript_ex.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "# brain: fix the thing" in res.markdown
    assert "**Source:** claude_code" in res.markdown
    assert "**Context:** ~/brain" in res.markdown
    assert "Can you fix the bug?" in res.markdown
    assert "Done, here is what changed." in res.markdown
    assert "3 tool calls dropped" in res.markdown
    assert "internal reasoning" in res.markdown


def test_transcript_extractor_no_turns_is_partial(tmp_path: Path) -> None:
    src = tmp_path / "t.json"
    src.write_bytes(_snapshot(connector="chatgpt", id="c1", title="Empty", turns=[]))
    res = transcript_ex.extract(src, tmp_path / "a")
    assert res.status == "partial"
    assert "no transcript content survived" in res.markdown


def test_transcript_extractor_bad_json_is_manual_review(tmp_path: Path) -> None:
    src = tmp_path / "t.json"
    src.write_bytes(b"{ not json")
    res = transcript_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"


def test_transcript_extractor_non_object_is_manual_review(tmp_path: Path) -> None:
    src = tmp_path / "t.json"
    src.write_bytes(b"[1, 2, 3]")
    res = transcript_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"


def test_transcript_snapshot_routes_by_source_class() -> None:
    routed = dispatch_extractor(
        Path("x.json"), relative_path="transcripts/claude_code/2026-09-03-x-abcd1234.json"
    )
    assert routed is transcript_ex.extract
    routed2 = dispatch_extractor(
        Path("x.json"), relative_path="transcripts/chat_export/2026-09-03-x-abcd1234.json"
    )
    assert routed2 is transcript_ex.extract


def test_transcripts_registered_in_connectors_and_extractors() -> None:
    assert "claude_code" in CONNECTORS
    assert "chat_export" in CONNECTORS
