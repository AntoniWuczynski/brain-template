#!/usr/bin/env python3
"""Score section-level retrieval on long documents: global search, search
restricted to the right document, and an LLM navigating the document's
outline (what ``vault_outline`` gives an agent).

    uv run python scripts/eval_sections.py          # search strategies only
    uv run python scripts/eval_sections.py --llm    # + outline navigation

The golden set is ``scripts/eval/section_golden.jsonl``. The outline
strategy assumes the agent already knows the document (document-level
recall is what ``eval_retrieval.py`` measures). ``--llm`` calls the
configured summariser provider, so it costs a little and is not
deterministic. Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from pydantic import BaseModel  # noqa: E402

from ingest_lib import chunks_for_source, default_paths, semantic_search  # noqa: E402
from ingest_lib.evalsec import (  # noqa: E402
    SectionGolden,
    SectionResult,
    expected_chunks,
    first_hit_rank,
    hit_rate,
    render_outline,
    section_chunks,
)
from ingest_lib.outline import Section, build_outline  # noqa: E402
from ingest_lib.summarize import generate_structured, is_enabled  # noqa: E402

_GOLDEN = Path(__file__).resolve().parent / "eval" / "section_golden.jsonl"
_KS = (1, 3, 5)
_DOC_POOL = 1000  # chunks fetched before filtering to one document

_NAV_SYSTEM = (
    "You navigate a document by its table of contents. Given a question and "
    "the document's numbered sections (heading path | opening text), return "
    "the numbers of up to 3 sections most likely to answer it, best first."
)


class SectionPicks(BaseModel):
    sections: list[int]


def _load_golden(path: Path) -> list[SectionGolden]:
    out: list[SectionGolden] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row[k].strip()
            for k in ("query", "source", "expected_heading")
        ):
            raise ValueError(f"{path}:{lineno}: needs non-empty query/source/expected_heading")
        out.append(SectionGolden(
            query=row["query"], source=row["source"], expected_heading=row["expected_heading"],
        ))
    return out


def _llm_picks(query: str, title: str, sections: list[Section]) -> list[frozenset[int]] | None:
    user = f"Question: {query}\n\nDocument: {title}\n\nSections:\n{render_outline(sections)}"
    picks = generate_structured(system=_NAV_SYSTEM, user=user, schema=SectionPicks, max_tokens=200)
    if not isinstance(picks, SectionPicks):
        return None
    return [section_chunks(sections[n - 1]) for n in picks.sections if 1 <= n <= len(sections)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score section-level retrieval on long documents.")
    ap.add_argument("--golden", type=Path, default=_GOLDEN)
    ap.add_argument("--llm", action="store_true",
                    help="also score an LLM navigating the outline (calls the configured provider)")
    args = ap.parse_args(argv)
    if args.llm and not is_enabled():
        print("--llm needs a configured LLM provider (see scripts/README.md)", file=sys.stderr)
        return 2

    paths = default_paths()
    strategies = ["search", "search-in-doc"] + (["outline-llm"] if args.llm else [])
    results: list[SectionResult] = []
    for g in _load_golden(args.golden):
        rows = chunks_for_source(paths, g["source"])
        if not rows:
            print(f"skip (not indexed): {g['source']}", file=sys.stderr)
            continue
        sections = build_outline(rows)
        expected = expected_chunks(sections, g["expected_heading"])
        if not expected:
            print(f"skip (no section matches {g['expected_heading']!r}): {g['source']}",
                  file=sys.stderr)
            continue
        hits = semantic_search(paths, g["query"], top_k=_DOC_POOL)
        ranks: dict[str, int | None] = {
            "search": first_hit_rank(
                [frozenset({h.chunk_idx}) if h.source_relative_path == g["source"] else frozenset()
                 for h in hits[:max(_KS)]],
                expected,
            ),
            "search-in-doc": first_hit_rank(
                [frozenset({h.chunk_idx}) for h in hits if h.source_relative_path == g["source"]],
                expected,
            ),
        }
        if args.llm:
            picked = _llm_picks(g["query"], rows[0]["title"], sections)
            if picked is None:
                print(f"llm call failed: {g['query']!r}", file=sys.stderr)
            else:
                ranks["outline-llm"] = first_hit_rank(picked, expected)
        results.append(SectionResult(
            query=g["query"], source=g["source"], expected_heading=g["expected_heading"],
            ranks=ranks,
        ))

    def cell(rank: int | None) -> str:
        return "-" if rank is None else str(rank)

    print("| " + " | ".join(strategies) + " | query |")
    print("|" + " --- |" * (len(strategies) + 1))
    for r in results:
        print("| " + " | ".join(cell(r.ranks.get(s)) for s in strategies) + f" | {r.query} |")
    print()
    for s in strategies:
        scored = sum(1 for r in results if s in r.ranks)
        rates = " · ".join(f"hit@{k} = {hit_rate(results, s, k):.2f}" for k in _KS)
        print(f"{s:>14}: {rates if scored else 'n/a'}  ({scored} scored)")
    print(f"\n{len(results)} queries. A search candidate is one chunk; an outline "
          "candidate is a whole section, so outline hit@k reads more text per rank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
