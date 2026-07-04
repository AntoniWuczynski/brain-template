#!/usr/bin/env python3
"""Retrieval eval CLI: score the live search path against a golden query set.

    uv run python scripts/eval_retrieval.py                 # print the table
    uv run python scripts/eval_retrieval.py --write-report  # + knowledge/index/retrieval-eval.md
    uv run python scripts/eval_retrieval.py --golden path.jsonl --top-k 10

Reads ``scripts/eval/retrieval_golden.jsonl`` — lines of
``{"query", "expected": [source_paths], "note"?}`` — runs each query through
``semantic.search`` (the same path vault_search / --search use), and reports
recall@5, recall@10, MRR, and per-query hit/miss. The workflow for "search
failed me just now": paste the query and the source it should have found.

Deterministic given a fixed index and golden set; always exits 0 (a metric,
not a gate) — mirrors scripts/sweep.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ingest_lib.config import default_paths  # noqa: E402
from ingest_lib.evalret import EvalReport, evaluate  # noqa: E402
from ingest_lib.notes import _atomic_write  # noqa: E402
from ingest_lib.semantic import search as semantic_search  # noqa: E402

_DEFAULT_GOLDEN = _SCRIPTS_DIR / "eval" / "retrieval_golden.jsonl"
_REPORT = "knowledge/index/retrieval-eval.md"
_KS = (5, 10)


def _load_golden(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(json.loads(line))
    return out


def _retriever(paths, top_k: int):
    """query -> ordered, de-duplicated source paths from the live index."""
    def retrieve(query: str, n: int) -> list[str]:
        hits = semantic_search(paths, query, top_k=max(n, top_k))
        seen: set[str] = set()
        ordered: list[str] = []
        for h in hits:
            if h.source_relative_path not in seen:
                seen.add(h.source_relative_path)
                ordered.append(h.source_relative_path)
        return ordered
    return retrieve


def _render_table(report: EvalReport) -> list[str]:
    lines = [
        f"recall@5: {report.recall_at(5):.3f}   "
        f"recall@10: {report.recall_at(10):.3f}   "
        f"MRR: {report.mrr():.3f}   (n={len(report.results)})",
        "",
        f"{'rank':>4}  {'r@5':>4} {'r@10':>4}  query",
        "-" * 72,
    ]
    for r in report.results:
        rank = str(r.first_hit_rank) if r.first_hit_rank else "miss"
        lines.append(
            f"{rank:>4}  {r.recall.get(5,0):.2f} {r.recall.get(10,0):.2f}  {r.query[:52]}"
        )
    return lines


def _render_report(report: EvalReport, *, as_of: str) -> str:
    rows = ["| first-hit | recall@5 | recall@10 | query | expected |",
            "| --- | --- | --- | --- | --- |"]
    for r in report.results:
        rank = str(r.first_hit_rank) if r.first_hit_rank else "**miss**"
        exp = ", ".join(f"`{e}`" for e in r.expected)
        rows.append(
            f"| {rank} | {r.recall.get(5,0):.2f} | {r.recall.get(10,0):.2f} | "
            f"{r.query.replace('|', chr(92)+'|')} | {exp} |"
        )
    misses = report.misses()
    return (
        "---\n"
        "title: Retrieval eval\n"
        "type: dashboard\n"
        f"updated: '{as_of}'\n"
        "---\n\n"
        "<!-- AUTO-GENERATED-START -->\n\n"
        "# Retrieval eval\n\n"
        f"> _Scored against `scripts/eval/retrieval_golden.jsonl` "
        f"({len(report.results)} queries). Add a line there whenever search "
        f"fails you: the query and the source it should have found._\n\n"
        f"**recall@5 = {report.recall_at(5):.3f}** · "
        f"**recall@10 = {report.recall_at(10):.3f}** · "
        f"**MRR = {report.mrr():.3f}** · misses = {len(misses)}\n\n"
        + "\n".join(rows)
        + "\n\n<!-- AUTO-GENERATED-END -->\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score retrieval against a golden query set.")
    ap.add_argument("--golden", type=Path, default=_DEFAULT_GOLDEN)
    ap.add_argument("--top-k", type=int, default=10, help="results fetched per query")
    ap.add_argument("--write-report", action="store_true",
                    help=f"also write {_REPORT}")
    args = ap.parse_args(argv)

    if not args.golden.is_file():
        print(f"golden set not found: {args.golden}", file=sys.stderr)
        return 0
    golden = _load_golden(args.golden)
    if not golden:
        print("golden set is empty — add {query, expected} lines to it.")
        return 0

    paths = default_paths()
    report = evaluate(golden, _retriever(paths, args.top_k), ks=_KS, fetch=args.top_k)

    print("\n".join(_render_table(report)))

    if args.write_report:
        as_of = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        target = paths.root / _REPORT
        _atomic_write(target, _render_report(report, as_of=as_of))
        print(f"\nwrote {_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
