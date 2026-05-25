"""Chat-style Q&A over the vault.

Combines the existing semantic search index with whatever LLM provider
the summarizer is configured for, so you can ask natural-language
questions and get citation-backed answers. The provider router from
``summarize.py`` is reused, so the same env vars apply
(``BRAIN_LLM_PROVIDER``, ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
``GOOGLE_API_KEY``/``GEMINI_API_KEY``, ``BRAIN_LOCAL_URL``).

This module does NOT write anything to disk and does NOT re-embed the
corpus. It only embeds the question at query time using the same model
that built the index.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

from .config import VaultPaths
from .semantic import SearchHit, search as semantic_search
from .summarize import _select_model, _select_provider


_DEFAULT_TOP_K: Final[int] = 8
_MAX_TOKENS: Final[int] = 1500


_CHAT_SYSTEM: Final[str] = (
    "You are a research assistant answering questions about a personal "
    "knowledge vault. Below the user's question, several relevant "
    "passages from the vault are provided, each labelled with a "
    "bracketed number like [1] and its source path.\n\n"
    "Rules:\n"
    "- Answer using ONLY the provided passages. If they don't cover the "
    "question, say so plainly. Do not invent facts.\n"
    "- Cite sources inline using the bracketed numbers, e.g. [1] or "
    "[2, 4]. Cite every claim. Do not list sources at the end; the "
    "caller renders them separately.\n"
    "- Be concise but complete. Prefer 2-4 short paragraphs over a "
    "single dense one. Use bullets when listing distinct items.\n"
    "- If passages conflict or only partially answer the question, "
    "flag that explicitly."
)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[SearchHit]
    provider: str
    model: str


def ask(
    paths: VaultPaths,
    question: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    provider_override: str | None = None,
    model_override: str | None = None,
    logger: logging.Logger | None = None,
) -> AnswerResult | None:
    """Retrieve relevant chunks, ask the LLM, return the answer.

    Returns ``None`` if the semantic index doesn't exist, no chunks
    match, no provider is configured, or the LLM call fails.
    """
    log = logger or logging.getLogger(__name__)

    hits = semantic_search(paths, question, top_k=top_k, logger=log)
    if not hits:
        log.warning(
            "ask: no search results — has the index been built? "
            "Run scripts/ingest.py --rebuild-search-index."
        )
        return None

    provider = provider_override or _select_provider()
    if provider is None:
        log.warning(
            "ask: no LLM provider configured. Set ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY, GOOGLE_API_KEY/GEMINI_API_KEY, or "
            "BRAIN_LOCAL_URL — or set BRAIN_LLM_PROVIDER explicitly."
        )
        return None
    model = model_override or _select_model(provider)

    context = _format_context(hits)
    user_block = f"Vault passages:\n\n{context}\n\nQuestion: {question}"

    answer_text = _call_chat(
        provider=provider,
        model=model,
        system=_CHAT_SYSTEM,
        user=user_block,
        log=log,
    )
    if answer_text is None:
        return None

    return AnswerResult(
        answer=answer_text.strip(),
        sources=hits,
        provider=provider,
        model=model,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] (source: {h.source_relative_path}, chunk {h.chunk_idx})\n"
            f"{h.snippet}"
        )
    return "\n\n---\n\n".join(blocks)


def _call_chat(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    log: logging.Logger,
) -> str | None:
    if provider == "anthropic":
        return _call_anthropic(model=model, system=system, user=user, log=log)
    if provider == "openai":
        return _call_openai_compat(
            model=model, system=system, user=user, log=log,
            base_url=None, api_key=None, label="openai",
        )
    if provider == "local":
        base_url = os.environ.get("BRAIN_LOCAL_URL")
        if not base_url:
            log.warning("ask: local provider requires BRAIN_LOCAL_URL")
            return None
        api_key = os.environ.get("BRAIN_LOCAL_API_KEY") or "not-needed"
        return _call_openai_compat(
            model=model, system=system, user=user, log=log,
            base_url=base_url, api_key=api_key, label="local",
        )
    if provider == "gemini":
        return _call_gemini(model=model, system=system, user=user, log=log)
    log.warning("ask: unknown provider %r", provider)
    return None


def _call_anthropic(*, model: str, system: str, user: str, log: logging.Logger) -> str | None:
    try:
        import anthropic
    except ImportError as exc:
        log.warning("ask: anthropic SDK missing (%s)", exc)
        return None
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIError as exc:
        log.warning("ask: anthropic API error (%s)", exc)
        return None
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _call_openai_compat(
    *,
    model: str,
    system: str,
    user: str,
    log: logging.Logger,
    base_url: str | None,
    api_key: str | None,
    label: str,
) -> str | None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        log.warning("ask: openai SDK missing (%s) — needed for %s provider", exc, label)
        return None
    client_kwargs: dict[str, object] = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)
    try:
        completion = client.chat.completions.create(
            model=model,
            max_completion_tokens=_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:  # SDK exception types vary
        hint = ""
        if label == "local":
            hint = (
                " — verify BRAIN_LOCAL_URL is reachable and "
                "BRAIN_LOCAL_MODEL is pulled (ollama list)."
            )
        log.warning("ask: %s call failed (%r)%s", label, exc, hint)
        return None
    if not completion.choices:
        log.warning("ask: %s returned no choices", label)
        return None
    return completion.choices[0].message.content or ""


def _call_gemini(*, model: str, system: str, user: str, log: logging.Logger) -> str | None:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        log.warning("ask: google-genai SDK missing (%s)", exc)
        return None
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.warning("ask: gemini requires GOOGLE_API_KEY or GEMINI_API_KEY")
        return None
    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=_MAX_TOKENS,
            ),
        )
    except Exception as exc:
        log.warning("ask: gemini call failed (%r)", exc)
        return None
    return getattr(resp, "text", None)
