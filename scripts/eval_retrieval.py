#!/usr/bin/env python3
"""Retrieval eval CLI: score the live search path against a golden query set.

    uv run python scripts/eval_retrieval.py                 # print the table
    uv run python scripts/eval_retrieval.py --write-report  # + knowledge/index/retrieval-eval.md
    uv run python scripts/eval_retrieval.py --golden path.jsonl --top-k 10
    uv run python scripts/eval_retrieval.py --compare-modes # dense vs lexical vs hybrid
    uv run python scripts/eval_retrieval.py --mine-log      # real queries -> candidates + zero-hit report
    uv run python scripts/eval_retrieval.py \
        --compare mode=hybrid --compare mode=hybrid,BRAIN_QUERY_INSTRUCTION=0

Reads ``scripts/eval/retrieval_golden.jsonl`` — lines of
``{"query", "expected": [source_paths], "note"?}`` — runs each query through
``semantic.search`` (the same path vault_search / --search use), and reports
recall@5, recall@10, MRR, and per-query hit/miss. Every query that did not
retrieve all of its expected sources gets a breakdown naming the expected path
that was absent and what outranked it. The workflow for "search failed me just
now": paste the query and the source it should have found.

``--compare A --compare B`` is the A/B tool a ranking change has to win on: each
argument is a comma-separated config (``mode=dense|lexical|hybrid`` plus any
``ENV_VAR=value`` the ranker reads; write ``\\,`` for a literal comma in a
value), and the output is the per-query rank delta between the two — plus the
found/expected fraction for multi-source queries, so a change that lifts the
aggregate by sinking individual queries, or by dropping the *second* expected
source of a query whose first hit never moved, is visible rather than averaged
away.

Deterministic given a fixed index and golden set; always exits 0 (a metric,
not a gate) — mirrors scripts/sweep.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import cast

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ingest_lib.config import VaultPaths, default_paths  # noqa: E402
from ingest_lib.evalret import (  # noqa: E402
    EvalReport,
    GoldenQuery,
    QueryResult,
    evaluate,
)
from ingest_lib.notes import _atomic_write  # noqa: E402
from ingest_lib.semantic import search as semantic_search  # noqa: E402

_DEFAULT_GOLDEN = _SCRIPTS_DIR / "eval" / "retrieval_golden.jsonl"
_REPORT = "knowledge/index/retrieval-eval.md"
_KS = (5, 10)


def _load_golden(path: Path) -> list[GoldenQuery]:
    out: list[GoldenQuery] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"{path}:{lineno}: golden line missing a non-empty \"query\": {line!r}")
        expected = row.get("expected")
        if not isinstance(expected, list) or not expected or not all(
            isinstance(e, str) and e.strip() for e in expected
        ):
            raise ValueError(
                f"{path}:{lineno}: golden line needs a non-empty \"expected\" list "
                f"of source paths (mined candidates must be confirmed before "
                f"promotion): {line!r}"
            )
        golden: GoldenQuery = {
            "query": cast(str, row["query"]),
            "expected": cast("list[str]", expected),
        }
        note = row.get("note")
        if isinstance(note, str):
            golden["note"] = note
        out.append(golden)
    return out


_OVERFETCH_MAX_MULTIPLIER = 40  # cap the escalation below (~400x n at n=10)


def _retriever(
    paths: VaultPaths, top_k: int, mode: str = "hybrid"
) -> Callable[[str, int], list[str]]:
    """query -> ordered, de-duplicated source paths from the live index."""
    def retrieve(query: str, n: int) -> list[str]:
        # Over-fetch CHUNKS before de-duping to SOURCES: n chunks dominated by
        # one multi-chunk source would otherwise yield far fewer than n
        # distinct sources, so an expected source whose best chunk ranks just
        # outside n is scored a miss. Start at a 5x chunk pool and double it
        # until n distinct sources are found or the candidate pool is
        # exhausted (semantic.search returns fewer chunks than requested).
        base = max(n, top_k)
        multiplier = 5
        ordered: list[str] = []
        while True:
            fetch_k = base * multiplier
            hits = semantic_search(paths, query, top_k=fetch_k, mode=mode)
            seen: set[str] = set()
            ordered = []
            for h in hits:
                if h.source_relative_path not in seen:
                    seen.add(h.source_relative_path)
                    ordered.append(h.source_relative_path)
            if len(ordered) >= n or len(hits) < fetch_k or multiplier >= _OVERFETCH_MAX_MULTIPLIER:
                break
            multiplier *= 2
        return ordered[:n]
    return retrieve


_MODES = ("dense", "lexical", "hybrid")


def _run_mine_log(paths: VaultPaths) -> int:
    """Mine the MCP access log into golden candidates + a zero-hit report."""
    from ingest_lib.evalmine import candidate_lines, load_access_log

    log_path = paths.logs / "mcp-access.jsonl"
    mined = load_access_log(log_path)
    if not mined:
        print(f"no search queries mined from {log_path} "
              "(no access log yet, or no vault_search/memory_search calls).")
        return 0
    zero_hit = [m for m in mined if not m.ever_hit]
    out_path = _SCRIPTS_DIR / "eval" / "mined_candidates.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_path, "\n".join(candidate_lines(mined)) + "\n")

    print(f"mined {len(mined)} distinct quer{'y' if len(mined) == 1 else 'ies'} "
          f"from {log_path.name} -> {out_path.relative_to(paths.root)}")
    print("  (expected left EMPTY — confirm relevance before promoting to "
          "retrieval_golden.jsonl)")
    if zero_hit:
        print(f"\n{len(zero_hit)} query/queries that NEVER returned a hit "
              "(retrieval failures to investigate):")
        for m in zero_hit[:20]:
            print(f"  - ({m.occurrences}x) {m.query[:70]}")
    return 0


def _run_compare_modes(paths: VaultPaths, golden: list[GoldenQuery], top_k: int) -> int:
    """Score each retrieval mode over the golden set, side by side."""
    fetch = max(top_k, max(_KS))
    print(f"{'mode':>8}  {'recall@5':>9} {'recall@10':>10} {'MRR':>6}  (n={len(golden)})")
    print("-" * 48)
    for mode in _MODES:
        rep = evaluate(golden, _retriever(paths, top_k, mode), ks=_KS, fetch=fetch)
        print(f"{mode:>8}  {rep.recall_at(5):>9.3f} {rep.recall_at(10):>10.3f} "
              f"{rep.mrr():>6.3f}")
    return 0


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


_MISS_CONTEXT = 3  # how many outranking sources to name per missing expected path
_MISS_PROBE_DEPTH = 100  # diagnostic-only retrieval depth (never scored)


def _render_miss_breakdown(
    report: EvalReport,
    *,
    k: int = 10,
    probe: Callable[[str, int], list[str]] | None = None,
    probe_depth: int = _MISS_PROBE_DEPTH,
) -> list[str]:
    """Per-query breakdown of the queries that did not retrieve everything they
    should have: which expected path was absent from the top-k, and what
    occupied the ranks it should have held. This is the part a ranking change
    is meant to move, so it is printed, not buried in the aggregate.

    ``probe`` is a second, deeper retrieval used for the diagnostic ONLY: the
    scored ``retrieved`` list stops at the fetch depth the metrics are computed
    over, so without it an expected path sitting at rank 29 is indistinguishable
    from one the index never returns — and "rank 29, a boost could reach it" is
    exactly the number a ranking change needs. Scoring never sees the deeper
    list (feeding it to ``evaluate`` would inflate MRR instead).
    """
    lines: list[str] = []
    for r in report.results:
        top = r.retrieved[:k]
        missing = [e for e in r.expected if e not in top]
        if not missing:
            continue
        if not lines:
            lines = ["", f"misses (expected source absent from the top {k}):", "-" * 72]
        scanned = probe(r.query, probe_depth) if probe is not None else list(r.retrieved)
        lines.append(f"  query: {r.query}")
        for path in missing:
            where = scanned.index(path) + 1 if path in scanned else None
            if where:
                place = f"rank {where}"
            elif scanned:
                place = f"not retrieved within the top {len(scanned)} fetched"
            else:
                place = "nothing retrieved for this query"
            lines.append(f"    absent: {path}  ({place})")
        lines.append("    ranked instead: " + (
            ", ".join(f"{i}. {pth}" for i, pth in enumerate(top[:_MISS_CONTEXT], start=1))
            or "(nothing retrieved)"
        ))
        if r.note:
            lines.append(f"    label: {r.note}")
    return lines


@dataclass(frozen=True)
class _Config:
    """One side of an A/B: a retrieval mode plus environment overrides."""
    label: str
    mode: str
    env: dict[str, str]


def _parse_config(spec: str) -> _Config:
    """``"mode=dense,BRAIN_QUERY_INSTRUCTION=0"`` -> a config.

    Every ``key=value`` other than ``mode`` is an environment variable set for
    the duration of that side's run — the way a ranking change under test is
    expected to be gated so both sides can run in one process. A literal comma
    inside a value is written ``\\,`` (an unescaped comma always separates
    pairs), so list-valued knobs such as
    ``BRAIN_QUERY_EXPANSION_VARIANTS=a\\,b`` are expressible."""
    mode = "hybrid"
    env: dict[str, str] = {}
    for raw_part in re.split(r"(?<!\\),", spec):
        part = raw_part.strip().replace("\\,", ",")
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"--compare config needs key=value pairs, got {part!r}")
        key = key.strip()
        if key == "mode":
            if value not in _MODES:
                raise ValueError(f"--compare mode must be one of {_MODES}, got {value!r}")
            mode = value
        else:
            env[key] = value
    return _Config(label=spec, mode=mode, env=env)


@contextmanager
def _env_overrides(env: dict[str, str]) -> Iterator[None]:
    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _rank_label(rank: int | None) -> str:
    return str(rank) if rank else "miss"


def _found_at(result: QueryResult, k: int) -> int:
    """How many of this query's expected sources are in the top ``k``."""
    return len(set(result.expected) & set(result.retrieved[:k]))


def _delta_marker(before: QueryResult, after: QueryResult) -> str:
    """``+`` / ``-`` / ``" "`` for one query's A -> B move.

    Recall is compared FIRST and at every k: a variant that keeps the first hit
    where it was but drops the second expected source out of the top k is a
    regression, and classifying on ``reciprocal_rank`` alone reported it as
    unchanged. Only when recall is level at every k does the first-hit rank
    (a smaller rank is better; a miss is worse than any rank) decide.
    """
    down = any(after.recall.get(k, 0.0) < before.recall.get(k, 0.0) for k in before.recall)
    up = any(after.recall.get(k, 0.0) > before.recall.get(k, 0.0) for k in before.recall)
    if down:
        return "-"
    if up:
        return "+"
    if after.reciprocal_rank > before.reciprocal_rank:
        return "+"
    if after.reciprocal_rank < before.reciprocal_rank:
        return "-"
    return " "


def _env_notes(*configs: _Config) -> list[str]:
    """Warn where the spec labels are not the effective environment.

    An exported ``BRAIN_*`` applies to both sides, so an A/B whose only
    difference is a variable already set that way is a silent no-op that still
    prints "improves 0, regresses 0" — indistinguishable from a real null
    result.
    """
    overridden = {key for cfg in configs for key in cfg.env}
    ambient = {k: v for k, v in sorted(os.environ.items()) if k.startswith("BRAIN_")}
    lines = []
    inherited = {k: v for k, v in ambient.items() if k not in overridden}
    if inherited:
        lines.append("    ambient (applies to BOTH sides): "
                     + ", ".join(f"{k}={v}" for k, v in inherited.items()))
    for cfg in configs:
        no_ops = [k for k, v in cfg.env.items() if ambient.get(k) == v]
        if no_ops:
            lines.append(f"    warning: {cfg.label} sets {', '.join(no_ops)} to the value "
                         "already in the environment — that side is not a change")
    return lines


def _run_compare(
    paths: VaultPaths, golden: list[GoldenQuery], top_k: int, specs: list[str]
) -> int:
    """Score two configurations over the same golden set and print the delta."""
    if len(specs) != 2:
        print("--compare takes exactly two configurations "
              "(pass it twice, e.g. --compare mode=hybrid --compare mode=dense)",
              file=sys.stderr)
        return 0
    fetch = max(top_k, max(_KS))
    try:
        configs = [_parse_config(spec) for spec in specs]
    except ValueError as exc:
        print(f"--compare: {exc}", file=sys.stderr)
        return 0
    reports: list[EvalReport] = []
    for cfg in configs:
        with _env_overrides(cfg.env):
            reports.append(
                evaluate(golden, _retriever(paths, top_k, cfg.mode), ks=_KS, fetch=fetch)
            )
    a, b = reports
    ca, cb = configs
    print(f"A = {ca.label}")
    print(f"B = {cb.label}")
    for line in _env_notes(ca, cb):
        print(line)
    print()
    print(f"{'':>10}  {'recall@5':>9} {'recall@10':>10} {'MRR':>6}  (n={len(golden)})")
    print("-" * 52)
    for name, rep in (("A", a), ("B", b)):
        print(f"{name:>10}  {rep.recall_at(5):>9.3f} {rep.recall_at(10):>10.3f} "
              f"{rep.mrr():>6.3f}")
    print(f"{'delta':>10}  {b.recall_at(5) - a.recall_at(5):>+9.3f} "
          f"{b.recall_at(10) - a.recall_at(10):>+10.3f} {b.mrr() - a.mrr():>+6.3f}")

    print()
    heads = "".join(f"{f'found@{k}':>10}" for k in _KS)
    print(f"{'A':>4} {'B':>4}  {'':2}{heads}  query")
    print("-" * 78)
    improved = regressed = 0
    for ra, rb in zip(a.results, b.results, strict=True):
        marker = _delta_marker(ra, rb)
        if marker == "+":
            improved += 1
        elif marker == "-":
            regressed += 1
        # Multi-source queries carry their found/expected fraction at every k:
        # the first-hit columns alone cannot show a second expected source
        # dropping out, and the fraction is what explains the marker.
        found = "".join(
            f"{f'{_found_at(ra, k)}/{len(ra.expected)}->{_found_at(rb, k)}/{len(rb.expected)}':>10}"
            if len(ra.expected) > 1 else f"{'':>10}"
            for k in _KS
        )
        print(f"{_rank_label(ra.first_hit_rank):>4} {_rank_label(rb.first_hit_rank):>4}"
              f"  {marker:2}{found}  {ra.query[:48]}")
    print()
    print(f"B improves {improved} quer{'y' if improved == 1 else 'ies'}, "
          f"regresses {regressed}, leaves {len(golden) - improved - regressed} unchanged.")
    return 0


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
    ap.add_argument("--mine-log", action="store_true",
                    help="mine real queries from logs/mcp-access.jsonl into "
                         "scripts/eval/mined_candidates.jsonl + a zero-hit report")
    ap.add_argument("--compare-modes", action="store_true",
                    help="score dense/lexical/hybrid side by side over the golden set")
    ap.add_argument("--compare", action="append", default=None, metavar="CONFIG",
                    help="pass TWICE to A/B two configurations over the same golden "
                         "set and print the per-query rank delta. Each CONFIG is "
                         "comma-separated key=value: mode=dense|lexical|hybrid plus "
                         "any ENV_VAR=value the ranker reads, e.g. "
                         "--compare mode=hybrid --compare mode=hybrid,BRAIN_QUERY_INSTRUCTION=0"
                         " (write \\, for a literal comma inside a value)")
    ap.add_argument("--no-miss-breakdown", action="store_true",
                    help="suppress the per-query breakdown of what was missed")
    args = ap.parse_args(argv)

    paths = default_paths()

    if args.mine_log:
        return _run_mine_log(paths)

    if not args.golden.is_file():
        print(f"golden set not found: {args.golden}", file=sys.stderr)
        return 0
    golden = _load_golden(args.golden)
    if not golden:
        print("golden set is empty — add {query, expected} lines to it.")
        return 0

    if args.compare:
        if args.write_report:
            print(f"--write-report is ignored with --compare ({_REPORT} describes a "
                  "single run, not an A/B).", file=sys.stderr)
        return _run_compare(paths, golden, args.top_k, list(args.compare))

    if args.compare_modes:
        return _run_compare_modes(paths, golden, args.top_k)
    # fetch enough distinct sources to score the largest k honestly: at
    # --top-k 5, recall@10 must still see 10 sources, not silently equal r@5.
    fetch = max(args.top_k, max(_KS))
    retriever = _retriever(paths, args.top_k)
    report = evaluate(golden, retriever, ks=_KS, fetch=fetch)

    lines = _render_table(report)
    if not args.no_miss_breakdown:
        # The same retriever, asked for far more results: scoring stops at
        # `fetch`, so only a deeper probe can say a miss sits at rank 29
        # rather than nowhere.
        lines += _render_miss_breakdown(report, k=max(_KS), probe=retriever)
    print("\n".join(lines))

    if args.write_report:
        as_of = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        target = paths.root / _REPORT
        _atomic_write(target, _render_report(report, as_of=as_of))
        print(f"\nwrote {_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
