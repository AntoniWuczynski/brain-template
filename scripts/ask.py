#!/usr/bin/env python3
"""Ask the vault a question and get a citation-backed answer.

Combines the semantic-search index with whatever LLM provider is
configured (anthropic / openai / gemini / local) and returns a
concise answer with bracketed citations.

Examples:

    uv run python scripts/ask.py "what does my vault say about TCP congestion control?"
    uv run python scripts/ask.py --top-k 12 "how does my vault treat BDD?"
    uv run python scripts/ask.py --provider local --model gemma4:31b "..."

The index must already exist (built by every ingest, or by
``scripts/ingest.py --rebuild-search-index``). This script does not
write anything to disk.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``ingest_lib`` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env at the repo root before any ingest_lib import sees env vars.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    _repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass

from ingest_lib import ask, default_paths  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brain-ask",
        description="Ask the vault a question; get a citation-backed answer.",
    )
    p.add_argument("question", help="The question to ask, in natural language.")
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many vault chunks to retrieve as context (default: 8).",
    )
    p.add_argument(
        "--provider",
        choices=["anthropic", "openai", "gemini", "local"],
        default=None,
        help="Override BRAIN_LLM_PROVIDER for this call.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override BRAIN_LLM_MODEL for this call.",
    )
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Don't print the model / provider line at the end.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    paths = default_paths()
    result = ask(
        paths,
        args.question,
        top_k=args.top_k,
        provider_override=args.provider,
        model_override=args.model,
    )
    if result is None:
        return 1

    print(result.answer)
    print()
    print("Sources:")
    for i, h in enumerate(result.sources, start=1):
        print(
            f"  [{i}] {h.source_relative_path}  "
            f"(chunk {h.chunk_idx}, score={h.score:.3f})"
        )
    if not args.quiet:
        print()
        print(f"— {result.provider}/{result.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
