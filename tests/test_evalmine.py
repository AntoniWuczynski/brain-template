"""Mining real MCP queries from the access log into eval candidates."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from ingest_lib.evalmine import candidate_lines, load_access_log, mine_access_log


def _row(tool: str, query: str | None, paths: list[str]) -> str:
    return json.dumps({"ts": "t", "agent": "a", "tool": tool, "paths": paths, "query": query})


def test_groups_dedupes_and_counts() -> None:
    lines = [
        _row("vault_search", "graphs", ["archive/processed/uni/a.md"]),
        _row("vault_search", "graphs", ["archive/processed/uni/a.md", "b.md"]),
        _row("memory_search", "anna role", ["knowledge/people/anna.md"]),
    ]
    mined = mine_access_log(lines)
    by_q = {m.query: m for m in mined}
    assert by_q["graphs"].occurrences == 2
    assert by_q["graphs"].tools == ("vault_search",)
    # Suggested paths are the UNION across the query's runs.
    assert set(by_q["graphs"].suggested_paths) == {"archive/processed/uni/a.md", "b.md"}
    assert by_q["anna role"].tools == ("memory_search",)


def test_zero_hit_queries_flagged() -> None:
    lines = [
        _row("vault_search", "found", ["x.md"]),
        _row("vault_search", "nothing here", []),
    ]
    mined = {m.query: m for m in mine_access_log(lines)}
    assert mined["found"].ever_hit is True
    assert mined["nothing here"].ever_hit is False


def test_non_search_and_malformed_rows_skipped() -> None:
    lines = [
        _row("vault_read", "not a search", []),      # non-search tool
        _row("vault_search", None, []),               # no query
        _row("vault_search", "   ", []),              # blank query
        "{ not json",                                  # malformed
        _row("vault_search", "real", ["x.md"]),
    ]
    mined = mine_access_log(lines)
    assert [m.query for m in mined] == ["real"]


def test_ordered_by_frequency_then_query() -> None:
    lines = [
        _row("vault_search", "b", ["x"]),
        _row("vault_search", "a", ["x"]),
        _row("vault_search", "a", ["x"]),
    ]
    mined = mine_access_log(lines)
    assert [m.query for m in mined] == ["a", "b"]  # 'a' twice, then 'b'


def test_candidate_lines_leave_expected_empty() -> None:
    mined = mine_access_log([_row("vault_search", "graphs", ["uni/a.md"])])
    line = json.loads(candidate_lines(mined)[0])
    assert line["query"] == "graphs"
    assert line["expected"] == []                 # never auto-labelled
    assert line["suggested"] == ["uni/a.md"]
    assert "CONFIRM" in line["note"]


def test_load_access_log_includes_rotated_segments(tmp_path: Path) -> None:
    active = tmp_path / "mcp-access.jsonl"
    active.write_text(_row("vault_search", "active query", ["a.md"]) + "\n", encoding="utf-8")
    rotated = tmp_path / "mcp-access-20260101T000000Z.jsonl.gz"
    with gzip.open(rotated, mode="wt", encoding="utf-8") as fh:
        fh.write(_row("vault_search", "rotated query", ["b.md"]) + "\n")
    # A different stream in the same directory must not be pulled in.
    (tmp_path / "mcp-audit.jsonl").write_text(
        _row("vault_search", "audit stream query", ["c.md"]) + "\n", encoding="utf-8"
    )

    mined = {m.query for m in load_access_log(active)}
    assert mined == {"active query", "rotated query"}


def test_load_access_log_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_access_log(tmp_path / "mcp-access.jsonl") == []


def test_mine_log_cli_writes_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    sys.path.insert(0, "scripts")
    import eval_retrieval
    from ingest_lib.config import paths_for_root

    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    (paths.logs / "mcp-access.jsonl").write_text(
        _row("vault_search", "packet switching", ["archive/processed/uni/net.md"]) + "\n"
        + _row("vault_search", "no results query", []) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(eval_retrieval, "default_paths", lambda: paths)
    # Redirect the candidates output into the tmp vault's scripts/eval dir.
    monkeypatch.setattr(eval_retrieval, "_SCRIPTS_DIR", paths.root / "scripts")

    rc = eval_retrieval.main(["--mine-log"])
    assert rc == 0
    cand = paths.root / "scripts" / "eval" / "mined_candidates.jsonl"
    body = cand.read_text(encoding="utf-8")
    assert "packet switching" in body
    assert '"expected": []' in body
