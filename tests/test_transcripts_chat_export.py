"""The ``chat_export`` transcript connector (claude.ai / ChatGPT conversation
exports) and ``scripts/pull.py``'s ``--path`` flag.

Split out of ``tests/test_transcripts.py`` (that file's own 1000-line limit)
along its existing section boundaries; see ``tests/test_transcripts.py`` and
its sibling ``test_transcripts_*`` modules for the rest of the split.

Fixtures are hand-built JSONL/JSON shaped like real exports rather than
recorded real sessions, so these run offline and deterministically without
embedding anyone's actual conversation content.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.connectors import CONNECTORS, run_connector
from ingest_lib.connectors import _transcript_common as tc

_LOG = logging.getLogger("test")
_AT = "2026-09-03T00:00:00Z"


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def test_chat_export_connector_claude_ai_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {
            "uuid": "conv-1", "name": "Planning a trip",
            "created_at": "2026-09-01T10:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "Where should I go?"},
                {"sender": "assistant", "content": [
                    {"type": "text", "text": "How about Kyoto?"},
                    {"type": "image", "source": {}},
                ]},
                {"sender": "system", "text": "irrelevant bookkeeping"},
            ],
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["connector"] == "claude_ai"
    assert data["id"] == "conv-1"
    assert data["date"] == "2026-09-01"
    turns = data["turns"]
    assert turns == [
        {"role": "user", "text": "Where should I go?"},
        {"role": "assistant", "text": "How about Kyoto?"},
    ]
    assert data["stats"]["dropped_non_text_parts"] == 1
    assert data["stats"]["dropped_other_events"] == 1  # the "system" sender


def test_chat_export_connector_chatgpt_shape_orders_by_create_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"

    def node(author_role: str, text: str, create_time: float) -> dict[str, object]:
        return {
            "message": {
                "author": {"role": author_role},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": create_time,
            }
        }

    export.write_text(json.dumps([
        {
            "id": "conv-gpt-1", "title": "Debugging session",
            "create_time": 1735689600,
            "mapping": {
                "n2": node("assistant", "Second: here's the fix.", 200),
                "n1": node("user", "First: it crashes on start.", 100),
                "n3": node("system", "hidden", 50),
                "n4": {"message": {"author": {"role": "user"},
                                   "content": {"content_type": "code", "parts": ["x=1"]}}},
            },
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["connector"] == "chatgpt"
    assert data["turns"] == [
        {"role": "user", "text": "First: it crashes on start."},
        {"role": "assistant", "text": "Second: here's the fix."},
    ]
    assert data["stats"]["dropped_other_events"] == 1     # system author
    assert data["stats"]["dropped_non_text_parts"] == 1   # non-text content_type


def test_chat_export_connector_chatgpt_shape_includes_all_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documented behaviour (module docstring): every node's message is
    walked, ordered by create_time — NOT just the current_node path — so an
    edited-and-regenerated reply is interleaved into the transcript rather
    than silently dropped. This was previously untested (review m4); it is
    a deliberate choice, not a bug, so this test only pins the documented
    shape rather than changing it."""
    export = tmp_path / "conversations.json"

    def node(author_role: str, text: str, create_time: float) -> dict[str, object]:
        return {
            "message": {
                "author": {"role": author_role},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": create_time,
            }
        }

    export.write_text(json.dumps([
        {
            "id": "conv-fork-1", "title": "Edited prompt",
            "create_time": 1735689600,
            "mapping": {
                "n1": node("user", "How do I center a div?", 100),
                # Two ALTERNATIVE assistant replies to the same prompt —
                # one before the user edited/regenerated, one after.
                "n2a": node("assistant", "Use margin: auto.", 200),
                "n2b": node("assistant", "Use flexbox: display: flex.", 300),
            },
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    # Both alternative branches survive, ordered by create_time.
    assert data["turns"] == [
        {"role": "user", "text": "How do I center a div?"},
        {"role": "assistant", "text": "Use margin: auto."},
        {"role": "assistant", "text": "Use flexbox: display: flex."},
    ]


def test_chat_export_connector_missing_id_fallback_is_stable_across_reordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversation with none of uuid/conversation_id/id must get the SAME
    native id whether it's first or second in the export list — a
    position-based fallback breaks idempotency the moment an export is
    re-run with items reordered/prepended (review m5)."""
    conv = {"name": "No id here", "created_at": "2026-09-01T00:00:00Z",
            "chat_messages": [{"sender": "human", "text": "hi"},
                              {"sender": "assistant", "text": "yo"}]}
    other = {"uuid": "conv-other", "name": "Other", "created_at": "2026-08-01T00:00:00Z",
              "chat_messages": [{"sender": "human", "text": "hi"},
                               {"sender": "assistant", "text": "yo"}]}

    export_a = tmp_path / "a.json"
    export_a.write_text(json.dumps([conv, other]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export_a))
    paths_a = _vault(tmp_path)
    run_connector(CONNECTORS["chat_export"](), paths_a, pulled_at=_AT, logger=_LOG)
    ids_a = {json.loads(f.read_text())["id"]
             for f in (paths_a.root / "inbox/transcripts/chat_export").glob("*.json")}

    export_b = tmp_path / "b.json"
    export_b.write_text(json.dumps([other, conv]), encoding="utf-8")  # reordered
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export_b))
    paths_b = paths_for_root(tmp_path / "vault-b")
    paths_b.ensure()
    run_connector(CONNECTORS["chat_export"](), paths_b, pulled_at=_AT, logger=_LOG)
    ids_b = {json.loads(f.read_text())["id"]
             for f in (paths_b.root / "inbox/transcripts/chat_export").glob("*.json")}

    assert ids_a == ids_b


def test_chat_export_connector_unrecognized_shape_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([{"id": "x", "some_other_shape": True}]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_chat_export_connector_no_env_yields_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAIN_CHAT_EXPORT_PATH", raising=False)
    monkeypatch.delenv("BRAIN_PULL_PATH", raising=False)
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_chat_export_connector_falls_back_to_generic_pull_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {"uuid": "conv-1", "name": "T", "created_at": "2026-09-01T00:00:00Z",
         "chat_messages": [{"sender": "human", "text": "hi"},
                          {"sender": "assistant", "text": "hello"}]},
    ]), encoding="utf-8")
    monkeypatch.delenv("BRAIN_CHAT_EXPORT_PATH", raising=False)
    monkeypatch.setenv("BRAIN_PULL_PATH", str(export))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1


def test_chat_export_connector_explicit_path_flag_beats_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A .env-supplied BRAIN_CHAT_EXPORT_PATH must not silently win over an
    # explicit --path (BRAIN_PULL_PATH) — the more specific, more recently
    # expressed configuration (review M2).
    default_export = tmp_path / "env_default.json"
    default_export.write_text(json.dumps([
        {"uuid": "conv-env", "name": "From env", "created_at": "2026-01-01T00:00:00Z",
         "chat_messages": [{"sender": "human", "text": "hi"}, {"sender": "assistant", "text": "yo"}]},
    ]), encoding="utf-8")
    flag_export = tmp_path / "cli_flag.json"
    flag_export.write_text(json.dumps([
        {"uuid": "conv-flag", "name": "From flag", "created_at": "2026-01-02T00:00:00Z",
         "chat_messages": [{"sender": "human", "text": "hi"}, {"sender": "assistant", "text": "yo"}]},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(default_export))
    monkeypatch.setenv("BRAIN_PULL_PATH", str(flag_export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["id"] == "conv-flag"


def test_chat_export_connector_missing_configured_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A configured-but-missing path is a loud failure, not a silent
    # zero-item pull indistinguishable from "nothing new" (review M2).
    monkeypatch.delenv("BRAIN_CHAT_EXPORT_PATH", raising=False)
    monkeypatch.setenv("BRAIN_PULL_PATH", str(tmp_path / "typo.json"))
    paths = _vault(tmp_path)
    with pytest.raises(tc.TranscriptSourceError):
        run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)


def test_chat_export_connector_long_title_does_not_break_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_title = "Refactor the pipeline " * 20  # ~460 chars, user-controlled
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {"uuid": "conv-long", "name": long_title, "created_at": "2026-08-01T00:00:00Z",
         "chat_messages": [{"sender": "human", "text": "hi"}, {"sender": "assistant", "text": "yo"}]},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    assert len(snaps) == 1
    assert len(snaps[0].name) <= 200
    data = json.loads(snaps[0].read_text())
    assert data["title"] == long_title  # full title kept in the note itself


def test_chat_export_connector_caps_conversation_total_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    messages = []
    for i in range(10):
        messages.append({"sender": "human", "text": f"user turn {i} " + "x" * 500})
        messages.append({"sender": "assistant", "text": f"assistant turn {i} " + "y" * 500})
    export.write_text(json.dumps([
        {"uuid": "conv-big", "name": "Big", "created_at": "2026-09-01T00:00:00Z",
         "chat_messages": messages},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    monkeypatch.setenv("BRAIN_TRANSCRIPT_MAX_CHARS", "2000")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    total_chars = sum(len(t["text"]) for t in data["turns"])
    assert total_chars <= 2000
    assert data["stats"]["dropped_overflow_turns"] > 0
    assert any("assistant turn 9" in t["text"] for t in data["turns"])


def test_chat_export_connector_respects_max_and_orders_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    convs = []
    for i in range(3):
        convs.append({
            "uuid": f"conv-{i}", "name": f"Conversation {i}",
            "created_at": f"2026-09-0{i + 1}T00:00:00Z",
            "chat_messages": [{"sender": "human", "text": "hi"},
                              {"sender": "assistant", "text": "hello"}],
        })
    export.write_text(json.dumps(convs), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_MAX", "1")
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["id"] == "conv-2"   # newest created_at kept


def test_chat_export_connector_claude_ai_non_text_only_message_not_double_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A message whose only content is a non-text part is already counted
    # via dropped_non_text_parts — it must not ALSO be counted as a dropped
    # "wrapper turn" under a second, misleading label (review N4).
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {
            "uuid": "conv-imgonly", "name": "Image only reply",
            "created_at": "2026-09-01T00:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "hi"},
                {"sender": "assistant", "content": [{"type": "image", "source": {}}]},
                {"sender": "assistant", "text": "real reply"},
            ],
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["stats"]["dropped_non_text_parts"] == 1
    assert data["stats"]["dropped_wrapper_turns"] == 0


def test_chat_export_connector_chatgpt_non_string_part_only_not_double_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"

    def node(author_role: str, content: dict[str, object], create_time: float) -> dict[str, object]:
        return {"message": {"author": {"role": author_role}, "content": content,
                             "create_time": create_time}}

    export.write_text(json.dumps([
        {
            "id": "conv-gpt-nontext", "title": "Non-string part",
            "mapping": {
                "n1": node("user", {"content_type": "text", "parts": ["hi"]}, 100),
                "n2": node("assistant", {"content_type": "text", "parts": [{"weird": "struct"}]}, 150),
                "n3": node("assistant", {"content_type": "text", "parts": ["real reply"]}, 200),
            },
        },
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 1
    snaps = list((paths.root / "inbox/transcripts/chat_export").glob("*.json"))
    data = json.loads(snaps[0].read_text())
    assert data["stats"]["dropped_non_text_parts"] == 1
    assert data["stats"]["dropped_wrapper_turns"] == 0


def test_chat_export_connector_cap_never_leaves_zero_user_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {"uuid": "conv-lopsided", "name": "Lopsided", "created_at": "2026-09-01T00:00:00Z",
         "chat_messages": [
             {"sender": "assistant", "text": "assistant opening " + "a" * 50},
             {"sender": "human", "text": "user middle " + "u" * 5000},
             {"sender": "assistant", "text": "assistant closing " + "c" * 5000},
         ]},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    monkeypatch.setenv("BRAIN_TRANSCRIPT_MAX_CHARS", "200")

    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


def test_chat_export_connector_drops_conversation_with_no_kept_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([
        {"uuid": "conv-empty", "name": "Empty", "created_at": "2026-09-01T00:00:00Z",
         "chat_messages": [{"sender": "system", "text": "just bookkeeping"}]},
    ]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CHAT_EXPORT_PATH", str(export))
    paths = _vault(tmp_path)
    stats = run_connector(CONNECTORS["chat_export"](), paths, pulled_at=_AT, logger=_LOG)
    assert stats.written == 0


# ------------------------------------------------------------------ pull.py

def test_pull_path_flag_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    _scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import pull

    monkeypatch.delenv("BRAIN_PULL_PATH", raising=False)
    try:
        pull.main(["--list", "--path", "/tmp/some/export.json"])
        assert os.environ.get("BRAIN_PULL_PATH") == "/tmp/some/export.json"
    finally:
        # pull.main sets this directly via os.environ (not through
        # monkeypatch), so it survives this test's teardown and leaks into
        # every test that runs after it in the same process unless cleared
        # here explicitly — observed corrupting an unrelated later test's
        # BRAIN_CLAUDE_CODE_DIR resolution when the suite ran in full.
        os.environ.pop("BRAIN_PULL_PATH", None)
