"""Mine real search queries from the MCP access log into eval candidates.

``logs/mcp-access.jsonl`` records every ``vault_search`` / ``memory_search``
call as ``{ts, agent, tool, paths, query}`` where ``paths`` are the GATED hits
the agent actually saw. That is the real query distribution — far more
representative than a small hand-written golden set — but nothing consumes it.

This module is a pure parser: it groups the log's queries, flags the ones that
never returned a hit (retrieval failures worth investigating), and emits
golden *candidates*. Candidates carry ``expected: []`` plus the paths the
query happened to retrieve as a *suggestion* — never an auto-accepted label
(the log has no relevance judgements). A human confirms the real ``expected``
before promoting a candidate into ``retrieval_golden.jsonl``.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_SEARCH_TOOLS = ("vault_search", "memory_search")


@dataclass(frozen=True)
class MinedQuery:
    query: str
    occurrences: int
    tools: tuple[str, ...]           # which search tools issued it
    suggested_paths: tuple[str, ...]  # union of paths it ever returned (a hint)
    ever_hit: bool                    # did any call return at least one path?


def mine_access_log(
    lines: Iterable[str], *, tools: tuple[str, ...] = _SEARCH_TOOLS
) -> list[MinedQuery]:
    """Group search queries from raw access-log JSONL lines.

    Malformed lines and non-search rows are skipped. Output is ordered by
    descending frequency then query text, so it is deterministic given the
    same log."""
    agg: dict[str, dict] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("tool") not in tools:
            continue
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        key = query.strip()
        paths = row.get("paths")
        paths = [p for p in paths if isinstance(p, str)] if isinstance(paths, list) else []
        entry = agg.setdefault(
            key, {"occurrences": 0, "tools": set(), "paths": set(), "ever_hit": False}
        )
        entry["occurrences"] += 1
        entry["tools"].add(str(row.get("tool")))
        entry["paths"].update(paths)
        if paths:
            entry["ever_hit"] = True

    out = [
        MinedQuery(
            query=q,
            occurrences=e["occurrences"],
            tools=tuple(sorted(e["tools"])),
            suggested_paths=tuple(sorted(e["paths"])),
            ever_hit=e["ever_hit"],
        )
        for q, e in agg.items()
    ]
    out.sort(key=lambda m: (-m.occurrences, m.query))
    return out


def load_access_log(path: Path) -> list[MinedQuery]:
    """Mine the on-disk access log; empty list if it is absent/unreadable."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return mine_access_log(fh)
    except OSError:
        return []


def candidate_lines(mined: list[MinedQuery]) -> list[str]:
    """Render mined queries as golden-candidate JSONL. ``expected`` is left
    EMPTY on purpose — a human fills it after confirming relevance; the
    retrieved paths ride along as ``suggested`` only."""
    lines: list[str] = []
    for m in mined:
        lines.append(json.dumps({
            "query": m.query,
            "expected": [],
            "suggested": list(m.suggested_paths),
            "note": "CONFIRM: set expected from suggested (or the real source) "
                    "before promoting to retrieval_golden.jsonl",
        }, ensure_ascii=False, sort_keys=True))
    return lines
