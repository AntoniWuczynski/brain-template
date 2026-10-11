"""``ask.py --save``: the answer lands as a note under knowledge/notes/answers/
with provenance (frontmatter ``sources`` + a numbered ``## Sources`` list),
never cites an earlier saved answer, and is redacted before it hits disk.
``chat.ask`` is stubbed throughout, so no index or LLM is needed."""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import ask as ask_cli  # noqa: E402

from ingest_lib import chat  # noqa: E402
from ingest_lib.chat import AnswerResult, save_answer  # noqa: E402
from ingest_lib.config import VaultPaths, paths_for_root  # noqa: E402
from ingest_lib.metadata import IndexRecord, append_record  # noqa: E402
from ingest_lib.notes import _split_frontmatter  # noqa: E402
from ingest_lib.semantic import SearchHit  # noqa: E402

_TODAY = date(2026, 10, 11)


def _hit(rel: str, chunk: int = 0) -> SearchHit:
    return SearchHit(
        score=0.03, source_relative_path=rel, title=rel, chunk_idx=chunk,
        snippet="text", origin="knowledge-note" if rel.startswith("knowledge/") else "",
    )


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path="uni/tcp.pdf", source_hash="h", size_bytes=1, extension=".pdf",
        extractor="pdf-mineru", status="processed", raw_path="archive/raw/uni/tcp.pdf",
        processed_path="archive/processed/uni/tcp.pdf.md",
        index_note_path="knowledge/index/uni/tcp.pdf.md",
    ))
    return paths


def _result(answer: str, hits: list[SearchHit]) -> AnswerResult:
    return AnswerResult(answer=answer, sources=hits, provider="anthropic", model="claude-x")


def test_save_writes_note_with_frontmatter_sections_and_source_links(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    hits = [
        _hit("uni/tcp.pdf", 0),
        _hit("knowledge/notes/networking.md"),
        _hit("uni/tcp.pdf", 4),
        _hit("uni/unindexed.txt"),
    ]
    out = save_answer(paths, "How does TCP congestion control work?",
                      _result("Slow start doubles cwnd [1, 3]. See [2].", hits), today=_TODAY)

    rel = out.relative_to(paths.root).as_posix()
    assert rel == "knowledge/notes/answers/2026-10-11-how-does-tcp-congestion-control-work.md"
    text = out.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)
    assert fm == {
        "title": "How does TCP congestion control work?",
        "type": "answer",
        "generated_by": "ask",
        "author": "script:ask",
        "written_via": "script",
        "created": _TODAY,
        "provider": "anthropic",
        "model": "claude-x",
        "sources": [
            "archive/raw/uni/tcp.pdf",
            "knowledge/notes/networking.md",
            "archive/raw/uni/unindexed.txt",
        ],
    }
    assert body.lstrip("\n") == (
        "## Question\n\nHow does TCP congestion control work?\n\n"
        "## Answer\n\nSlow start doubles cwnd [1, 3]. See [2].\n\n"
        "## Sources\n\n"
        "1. [[knowledge/index/uni/tcp.pdf]]\n"
        "2. [[knowledge/notes/networking]]\n"
        "3. [[knowledge/index/uni/tcp.pdf]]\n"
        "4. `archive/raw/uni/unindexed.txt`\n"
    )


def test_save_suffixes_colliding_paths(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    result = _result("a [1]", [_hit("uni/tcp.pdf")])
    names = [save_answer(paths, "Same question", result, today=_TODAY).name for _ in range(3)]
    assert names == [
        "2026-10-11-same-question.md",
        "2026-10-11-same-question-2.md",
        "2026-10-11-same-question-3.md",
    ]


def test_save_slug_is_capped_and_never_empty(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    result = _result("a [1]", [_hit("uni/tcp.pdf")])
    long_name = save_answer(paths, "word " * 40, result, today=_TODAY).name
    assert long_name == "2026-10-11-" + ("word-" * 12)[:59] + ".md"
    assert save_answer(paths, "???", result, today=_TODAY).name == "2026-10-11-answer.md"


def test_save_redacts_question_and_answer(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    secret = "AKIAIOSFODNN7EXAMPLE"
    out = save_answer(paths, f"what is {secret}?",
                      _result(f"It is {secret} [1].", [_hit("uni/tcp.pdf")]), today=_TODAY)
    text = out.read_text(encoding="utf-8")
    assert secret not in text
    assert secret.lower() not in out.name
    assert text.count("[REDACTED:aws-access-key]") == 3  # title, question, answer


def test_cli_save_prints_path_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _vault(tmp_path)
    monkeypatch.setattr(ask_cli, "default_paths", lambda: paths)
    monkeypatch.setattr(ask_cli, "ask", lambda *a, **k: _result("x [1]", [_hit("uni/tcp.pdf")]))
    monkeypatch.setattr(ask_cli, "_today", lambda: _TODAY)

    assert ask_cli.main(["--save", "-q", "a question"]) == 0
    saved = paths.root / "knowledge/notes/answers/2026-10-11-a-question.md"
    assert saved.is_file()
    assert "Saved: knowledge/notes/answers/2026-10-11-a-question.md" in capsys.readouterr().out


def test_cli_without_save_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _vault(tmp_path)
    monkeypatch.setattr(ask_cli, "default_paths", lambda: paths)
    monkeypatch.setattr(ask_cli, "ask", lambda *a, **k: _result("x [1]", [_hit("uni/tcp.pdf")]))
    assert ask_cli.main(["-q", "a question"]) == 0
    assert not (paths.root / "knowledge/notes/answers").exists()


def test_cli_save_refuses_without_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _vault(tmp_path)
    monkeypatch.setattr(ask_cli, "default_paths", lambda: paths)
    monkeypatch.setattr(ask_cli, "ask", lambda *a, **k: _result("I don't know.", []))
    assert ask_cli.main(["--save", "a question"]) == 1
    assert not (paths.root / "knowledge/notes/answers").exists()
    assert "no sources" in capsys.readouterr().err


def test_ask_never_retrieves_saved_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No answer cites an earlier LLM answer: answer notes are dropped from
    the retrieved passages, and the top-k is refilled from the over-fetch."""
    paths = _vault(tmp_path)
    pool = [
        _hit("knowledge/notes/answers/2026-10-01-old.md"),
        _hit("uni/tcp.pdf", 0),
        _hit("knowledge/notes/answers/2026-10-02-older.md"),
        _hit("knowledge/notes/networking.md"),
        _hit("uni/tcp.pdf", 1),
    ]
    asked: list[int] = []
    prompts: list[str] = []

    def fake_search(_paths: VaultPaths, _q: str, *, top_k: int, mode: str,
                    logger: logging.Logger) -> list[SearchHit]:
        asked.append(top_k)
        return pool[:top_k]

    def fake_call(*, provider: str, model: str, system: str, user: str,
                  log: logging.Logger) -> str:
        prompts.append(user)
        return "answer [1]"

    monkeypatch.setattr(chat, "semantic_search", fake_search)
    monkeypatch.setattr(chat, "_call_chat", fake_call)
    result = chat.ask(paths, "q", top_k=2, provider_override="anthropic", model_override="m")

    assert result is not None
    assert asked[0] > 2
    assert [h.source_relative_path for h in result.sources] == [
        "uni/tcp.pdf", "knowledge/notes/networking.md",
    ]
    assert "knowledge/notes/answers/" not in prompts[0]


def test_ask_widens_fetch_past_many_saved_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-4: saved answers that repeat the question outrank every source in the
    first over-fetch; ask keeps widening until it has ``top_k`` real passages."""
    paths = _vault(tmp_path)
    pool = [_hit(f"knowledge/notes/answers/a-{i}.md") for i in range(24)]
    pool += [_hit(f"projects/real-{i}.pdf") for i in range(20)]

    def fake_search(_paths: VaultPaths, _q: str, *, top_k: int, mode: str,
                    logger: logging.Logger) -> list[SearchHit]:
        return pool[:top_k]

    def fake_call(*, provider: str, model: str, system: str, user: str,
                  log: logging.Logger) -> str:
        return "answer [1]"

    monkeypatch.setattr(chat, "semantic_search", fake_search)
    monkeypatch.setattr(chat, "_call_chat", fake_call)
    result = chat.ask(paths, "q", top_k=8, provider_override="anthropic", model_override="m")

    assert result is not None
    assert [h.source_relative_path for h in result.sources] == [
        f"projects/real-{i}.pdf" for i in range(8)
    ]


def test_save_classifies_hits_by_origin_not_path_prefix(tmp_path: Path) -> None:
    """F-5: a source dropped at inbox/knowledge/report.pdf is labelled
    ``knowledge/report.pdf`` but is an archived source (its origin is its
    extractor), so it cites archive/raw and its index note."""
    paths = _vault(tmp_path)
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path="knowledge/report.pdf", source_hash="h2", size_bytes=1,
        extension=".pdf", extractor="pdf-mineru", status="processed",
        raw_path="archive/raw/knowledge/report.pdf",
        processed_path="archive/processed/knowledge/report.pdf.md",
        index_note_path="knowledge/index/knowledge/report.pdf.md",
    ))
    hit = SearchHit(score=1.0, source_relative_path="knowledge/report.pdf", title="r",
                    chunk_idx=0, snippet="s", origin="pdf-mineru")
    out = save_answer(paths, "what is in the report?", _result("X [1].", [hit]), today=_TODAY)
    fm, body = _split_frontmatter(out.read_text(encoding="utf-8"))
    assert fm["sources"] == ["archive/raw/knowledge/report.pdf"]
    assert "1. [[knowledge/index/knowledge/report.pdf]]\n" in body


def test_ask_keeps_archive_source_under_answers_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-5: only vault notes (origin knowledge-note) under the answers dir are
    excluded, not an archived source whose label happens to share the prefix."""
    paths = _vault(tmp_path)
    source = SearchHit(score=1.0, source_relative_path="knowledge/notes/answers/x.pdf",
                       title="x", chunk_idx=0, snippet="s", origin="pdf-mineru")

    def fake_search(_paths: VaultPaths, _q: str, *, top_k: int, mode: str,
                    logger: logging.Logger) -> list[SearchHit]:
        return [_hit("knowledge/notes/answers/old.md"), source][:top_k]

    def fake_call(*, provider: str, model: str, system: str, user: str,
                  log: logging.Logger) -> str:
        return "answer [1]"

    monkeypatch.setattr(chat, "semantic_search", fake_search)
    monkeypatch.setattr(chat, "_call_chat", fake_call)
    result = chat.ask(paths, "q", top_k=2, provider_override="anthropic", model_override="m")
    assert result is not None and result.sources == [source]


def test_save_never_overwrites_a_racing_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-6: a concurrent save that also saw the name free must not replace
    the earlier answer; the name is claimed exclusively and the loser takes
    the next suffix. Simulated by hiding existing answer files from exists()."""
    paths = _vault(tmp_path)
    first = save_answer(paths, "Same question", _result("first [1]", [_hit("uni/tcp.pdf")]),
                        today=_TODAY)
    real_exists = Path.exists

    def racing_exists(self: Path, *, follow_symlinks: bool = True) -> bool:
        if "notes/answers" in self.as_posix():
            return False
        return real_exists(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "exists", racing_exists)
    second = save_answer(paths, "Same question", _result("second [1]", [_hit("uni/tcp.pdf")]),
                         today=_TODAY)
    monkeypatch.undo()

    assert second.name == "2026-10-11-same-question-2.md"
    assert "first [1]" in first.read_text(encoding="utf-8")
    assert "second [1]" in second.read_text(encoding="utf-8")
    assert sorted(p.name for p in first.parent.iterdir()) == [
        "2026-10-11-same-question-2.md", "2026-10-11-same-question.md",
    ]
