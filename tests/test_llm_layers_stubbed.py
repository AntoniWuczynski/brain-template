"""Drive the summariser / describe / caption LLM layers against a stubbed
provider SDK client (F5): before this file, ``summarize.summarize``,
``generate_structured``, and every ``_call_*`` provider function were never
entered by the suite, not even against a stub — only ``_select_provider``/
``is_enabled`` (pure env logic) had coverage.

The provider SDK classes (``anthropic.Anthropic``, ``openai.OpenAI``) are
constructed *inside* the ``_call_*`` functions with no injection seam, so
each test monkeypatches the class in its own module's namespace — the same
technique tests/test_extractors.py already uses for the VLM extractor's
provider calls (e.g. ``monkeypatch.setattr(anthropic, "Anthropic", FakeClient)``).
No network, no real API key.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest_lib import caption as caption_mod
from ingest_lib import describe as describe_mod
from ingest_lib import summarize as summarize_mod
from ingest_lib.summarize import DocSummary

_LOG = logging.getLogger("test")

_KEYS = [
    "BRAIN_LLM_PROVIDER", "BRAIN_LLM_MODEL", "BRAIN_SKIP_SUMMARY",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "BRAIN_LOCAL_URL", "BRAIN_LOCAL_MODEL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    summarize_mod._warned_invalid_provider.clear()


# --------------------------------------------------------------- anthropic

def _fake_anthropic_parse_client(parsed: object, *, cache_read: int = 0) -> type:
    class FakeMessages:
        def parse(self, **_kw: object) -> SimpleNamespace:
            return SimpleNamespace(
                parsed_output=parsed,
                usage=SimpleNamespace(cache_read_input_tokens=cache_read),
                stop_reason="end_turn",
            )

    class FakeClient:
        def __init__(self, **_kw: object) -> None:
            self.messages = FakeMessages()

    return FakeClient


def test_summarize_anthropic_provider_returns_docsummary_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anthropic

    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "anthropic")
    parsed = DocSummary(
        summary="A faithful two-sentence summary of the lecture.",
        key_points=["Point one", "Point two", "Point three"],
        topics=["Topic One", "Topic Two"],
    )
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_parse_client(parsed, cache_read=42))

    result = summarize_mod.summarize(
        "Real extracted markdown body, long enough to summarize.",
        title="Lecture 4", source_relative_path="uni/lecture4.pdf",
    )

    assert result is not None
    assert result.summary == parsed.summary
    assert result.key_points == parsed.key_points
    assert result.topics == parsed.topics
    assert "summary: anthropic/claude-haiku-4-5" in result.notes


def test_summarize_anthropic_malformed_reply_returns_none_not_invented(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    import anthropic

    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_parse_client(None))

    with caplog.at_level(logging.WARNING):
        result = summarize_mod.summarize(
            "Some body text.", title="T", source_relative_path="x.md",
        )

    assert result is None
    assert any("parse failed" in r.message for r in caplog.records)


# ------------------------------------------------------------------ openai

def _fake_openai_parse_client(parsed: object, *, refusal: str | None = None) -> type:
    class FakeMessage:
        def __init__(self) -> None:
            self.parsed = parsed
            self.refusal = refusal

    class FakeChoice:
        def __init__(self) -> None:
            self.message = FakeMessage()

    class FakeCompletions:
        def parse(self, **_kw: object) -> SimpleNamespace:
            return SimpleNamespace(choices=[FakeChoice()])

    class FakeClient:
        def __init__(self, **_kw: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    return FakeClient


def test_summarize_openai_provider_returns_docsummary_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "openai")
    parsed = DocSummary(
        summary="An openai-produced summary of the paper.",
        key_points=["Finding A", "Finding B"],
        topics=["Distributed Systems"],
    )
    monkeypatch.setattr(openai, "OpenAI", _fake_openai_parse_client(parsed))

    result = summarize_mod.summarize(
        "Real extracted markdown body for the openai path.",
        title="Paper", source_relative_path="research/paper.pdf",
    )

    assert result is not None
    assert result.summary == parsed.summary
    assert result.key_points == parsed.key_points
    assert result.topics == parsed.topics
    assert "summary: openai/gpt-5-mini" in result.notes


def test_summarize_openai_refusal_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "openai")
    monkeypatch.setattr(
        openai, "OpenAI", _fake_openai_parse_client(None, refusal="cannot help with that")
    )

    result = summarize_mod.summarize(
        "Some body text.", title="T", source_relative_path="x.md",
    )

    assert result is None


# ----------------------------------------------------------------- describe

def test_describe_concept_anthropic_provider_returns_concept_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anthropic

    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "anthropic")
    parsed = describe_mod.ConceptDescription(
        short_summary="TCP congestion control avoids overwhelming the network.",
        detailed_explanation="## Overview\nSlow start doubles the window each RTT.",
        key_definitions=[
            describe_mod.KeyDefinition(term="RTT", definition="Round-trip time."),
        ],
    )
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_parse_client(parsed))

    result = describe_mod.describe_concept(
        "TCP Congestion Control",
        ["Congestion control slows the sender when the network is loaded."],
        logger=_LOG,
    )

    assert result is not None
    assert result.short_summary == parsed.short_summary
    assert result.detailed_explanation == parsed.detailed_explanation
    assert result.key_definitions[0].term == "RTT"


def test_describe_concept_with_no_context_returns_none_without_calling_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "anthropic")
    assert describe_mod.describe_concept("X", ["", "   "], logger=_LOG) is None


# ------------------------------------------------------------------ caption

def test_caption_image_anthropic_provider_returns_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import anthropic

    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "anthropic")
    block = anthropic.types.TextBlock.model_construct(
        type="text", text="A line chart showing TCP throughput rising then plateauing."
    )
    resp = SimpleNamespace(content=[block])

    class FakeMessages:
        def create(self, **_kw: object) -> SimpleNamespace:
            return resp

    class FakeClient:
        def __init__(self, **_kw: object) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)

    img = tmp_path / "figure.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nnot a real png but bytes are enough")

    caption = caption_mod.caption_image(img, logger=_LOG)

    assert caption == "A line chart showing TCP throughput rising then plateauing."


def test_caption_image_no_provider_configured_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    img = tmp_path / "figure.png"
    img.write_bytes(b"irrelevant")
    assert caption_mod.caption_image(img, logger=_LOG) is None
