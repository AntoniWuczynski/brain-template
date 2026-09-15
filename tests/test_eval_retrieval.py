"""The retrieval eval CLI's A/B tooling and the golden set's own integrity.

The golden set is the gate two approved ranking changes have to pass, so it
gets tested like code: honest shape, no duplicates, enough headroom to be a
gate at all, and nothing personal in it.

``scripts/push_to_upstream.sh`` does NOT copy this file to the public template:
its ``expected`` paths are the owner's real vault inventory, so the sync
replaces it with a two-row placeholder. The tests that assert on the real set's
content therefore skip on the template branch rather than failing there.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")

import eval_retrieval  # noqa: E402
from ingest_lib.config import VaultPaths, paths_for_root  # noqa: E402
from ingest_lib.evalret import EvalReport, GoldenQuery, evaluate  # noqa: E402
from ingest_lib.semantic import SearchHit  # noqa: E402

_GOLDEN = Path("scripts/eval/retrieval_golden.jsonl")


# --------------------------------------------------------------------------
# per-query miss breakdown
# --------------------------------------------------------------------------

def _report(golden: list[GoldenQuery], retrieved: dict[str, list[str]]) -> EvalReport:
    # fetch=10 is what the CLI scores at, so the stub cannot invent a state
    # (an expected path at rank 11 inside `retrieved`) that a real run has.
    return evaluate(golden, lambda q, n: retrieved.get(q, [])[:n], ks=(5, 10), fetch=10)


def test_miss_breakdown_names_the_absent_path_and_what_outranked_it() -> None:
    golden: list[GoldenQuery] = [
        {"query": "green sheet", "expected": ["ref.pdf"], "note": "the reference card"},
    ]
    rep = _report(golden, {"green sheet": ["lecture2.pdf", "lecture3.pdf"]})
    lines = eval_retrieval._render_miss_breakdown(rep, k=10)
    text = "\n".join(lines)

    assert "green sheet" in text
    assert "absent: ref.pdf" in text
    assert "not retrieved within the top 2 fetched" in text
    assert "lecture2.pdf" in text          # what ranked instead
    assert "the reference card" in text    # the label, so a mislabel is visible


def test_miss_breakdown_names_the_rank_a_deeper_probe_finds() -> None:
    """The CLI scores at fetch=10, so a path at rank 29 is absent from every
    scored list. Without the diagnostic probe it was printed as "not retrieved
    at all" — the opposite of "a boost could reach it"."""
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["deep.pdf"]}]
    scored = [f"noise{i}.pdf" for i in range(10)]
    deeper = [f"noise{i}.pdf" for i in range(28)] + ["deep.pdf"]
    rep = _report(golden, {"q": scored})

    text = "\n".join(eval_retrieval._render_miss_breakdown(
        rep, k=10, probe=lambda q, n: deeper[:n], probe_depth=100,
    ))

    assert "absent: deep.pdf" in text
    assert "rank 29" in text
    assert "not retrieved" not in text
    assert rep.mrr() == 0.0   # the probe is diagnostic only: scoring is untouched


def test_miss_breakdown_says_not_retrieved_only_when_the_probe_missed_it_too() -> None:
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["gone.pdf"]}]
    deeper = [f"noise{i}.pdf" for i in range(30)]
    rep = _report(golden, {"q": deeper[:10]})

    text = "\n".join(eval_retrieval._render_miss_breakdown(
        rep, k=10, probe=lambda q, n: deeper[:n], probe_depth=100,
    ))

    assert "absent: gone.pdf  (not retrieved within the top 30 fetched)" in text


def test_miss_breakdown_reports_a_path_that_ranked_below_k() -> None:
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["deep.pdf"]}]
    retrieved = [f"noise{i}.pdf" for i in range(10)] + ["deep.pdf"]
    rep = _report(golden, {"q": retrieved[:10]})
    text = "\n".join(eval_retrieval._render_miss_breakdown(
        rep, k=10, probe=lambda q, n: retrieved[:n],
    ))

    assert "absent: deep.pdf" in text
    assert "rank 11" in text


def test_miss_breakdown_is_empty_when_every_expected_path_was_found() -> None:
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["a.pdf", "b.pdf"]}]
    rep = _report(golden, {"q": ["a.pdf", "b.pdf", "c.pdf"]})
    assert eval_retrieval._render_miss_breakdown(rep, k=10) == []


def test_miss_breakdown_flags_a_partially_found_multi_source_query() -> None:
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["a.pdf", "b.pdf"]}]
    rep = _report(golden, {"q": ["a.pdf", "c.pdf"]})
    text = "\n".join(eval_retrieval._render_miss_breakdown(rep, k=10))

    assert "absent: b.pdf" in text
    assert "absent: a.pdf" not in text


# --------------------------------------------------------------------------
# --compare configuration parsing
# --------------------------------------------------------------------------

def test_parse_config_reads_mode_and_env_overrides() -> None:
    cfg = eval_retrieval._parse_config("mode=dense,BRAIN_QUERY_INSTRUCTION=0")
    assert cfg.mode == "dense"
    assert cfg.env == {"BRAIN_QUERY_INSTRUCTION": "0"}
    assert cfg.label == "mode=dense,BRAIN_QUERY_INSTRUCTION=0"


def test_parse_config_defaults_to_hybrid_with_no_env() -> None:
    cfg = eval_retrieval._parse_config("BRAIN_RANK_ALPHA=0.7")
    assert cfg.mode == "hybrid"
    assert cfg.env == {"BRAIN_RANK_ALPHA": "0.7"}


def test_parse_config_rejects_an_unknown_mode_and_a_bare_token() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        eval_retrieval._parse_config("mode=magic")
    with pytest.raises(ValueError, match="key=value"):
        eval_retrieval._parse_config("dense")


def test_env_overrides_restore_previous_values_and_unset_new_ones() -> None:
    os.environ["BRAIN_TEST_EXISTING"] = "before"
    os.environ.pop("BRAIN_TEST_NEW", None)
    try:
        with eval_retrieval._env_overrides(
            {"BRAIN_TEST_EXISTING": "during", "BRAIN_TEST_NEW": "during"}
        ):
            assert os.environ["BRAIN_TEST_EXISTING"] == "during"
            assert os.environ["BRAIN_TEST_NEW"] == "during"
        assert os.environ["BRAIN_TEST_EXISTING"] == "before"
        assert "BRAIN_TEST_NEW" not in os.environ
    finally:
        os.environ.pop("BRAIN_TEST_EXISTING", None)


# --------------------------------------------------------------------------
# --compare end to end
# --------------------------------------------------------------------------

def _stub_search_on(
    monkeypatch: pytest.MonkeyPatch, results: dict[tuple[str, str], list[str]]
) -> None:
    """Stub semantic_search so a run's results depend on (mode, env flag) —
    the shape a ranking change gated behind an env var actually has."""
    def fake_search(
        paths: VaultPaths, query: str, *, top_k: int, mode: str = "hybrid",
        logger: logging.Logger | None = None,
    ) -> list[SearchHit]:
        flag = os.environ.get("BRAIN_TEST_RANKER", "off")
        srcs = results.get((mode, flag), [])
        return [
            SearchHit(score=1.0, source_relative_path=s, title=s,
                      chunk_idx=0, snippet="s", origin="")
            for s in srcs
        ]
    monkeypatch.setattr(eval_retrieval, "semantic_search", fake_search)


def test_compare_prints_per_query_delta_and_improve_regress_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_search_on(monkeypatch, {
        ("hybrid", "off"): ["noise.pdf", "want.pdf"],   # expected at rank 2
        ("hybrid", "on"): ["want.pdf", "noise.pdf"],    # lifted to rank 1
    })
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["want.pdf"]}]

    rc = eval_retrieval._run_compare(
        paths_for_root(tmp_path), golden, 10,
        ["mode=hybrid", "mode=hybrid,BRAIN_TEST_RANKER=on"],
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "A = mode=hybrid" in out
    assert "B = mode=hybrid,BRAIN_TEST_RANKER=on" in out
    assert "+0.500" in out                     # MRR delta 0.5 -> 1.0
    assert "B improves 1 query, regresses 0" in out
    assert "BRAIN_TEST_RANKER" not in os.environ   # the override was undone


def test_compare_counts_a_regression_when_the_variant_sinks_a_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_search_on(monkeypatch, {
        ("hybrid", "off"): ["want.pdf"],
        ("hybrid", "on"): ["noise.pdf"],       # expected lost entirely
    })
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["want.pdf"]}]

    eval_retrieval._run_compare(
        paths_for_root(tmp_path), golden, 10,
        ["mode=hybrid", "mode=hybrid,BRAIN_TEST_RANKER=on"],
    )
    out = capsys.readouterr().out

    assert "regresses 1" in out
    assert "miss" in out                       # B's per-query rank column


def test_compare_needs_exactly_two_configurations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["a.pdf"]}]
    rc = eval_retrieval._run_compare(paths_for_root(tmp_path), golden, 10, ["mode=hybrid"])
    assert rc == 0
    assert "exactly two configurations" in capsys.readouterr().err


def test_compare_counts_a_multi_source_query_that_loses_its_second_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The first hit does not move, so reciprocal_rank is identical on both
    sides — classifying on it alone called this regression "unchanged"."""
    _stub_search_on(monkeypatch, {
        ("hybrid", "off"): ["a.pdf", "b.pdf"],
        ("hybrid", "on"): ["a.pdf", "noise.pdf"],      # b.pdf dropped entirely
    })
    golden: list[GoldenQuery] = [{"query": "multi", "expected": ["a.pdf", "b.pdf"]}]

    eval_retrieval._run_compare(
        paths_for_root(tmp_path), golden, 10,
        ["mode=hybrid", "mode=hybrid,BRAIN_TEST_RANKER=on"],
    )
    out = capsys.readouterr().out

    assert "regresses 1" in out
    assert "2/2->1/2" in out                   # the per-query found fraction
    assert "leaves 0 unchanged" in out


def test_compare_marks_a_recovered_second_source_as_an_improvement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_search_on(monkeypatch, {
        ("hybrid", "off"): ["a.pdf", "noise.pdf"],
        ("hybrid", "on"): ["a.pdf", "b.pdf"],
    })
    golden: list[GoldenQuery] = [{"query": "multi", "expected": ["a.pdf", "b.pdf"]}]

    eval_retrieval._run_compare(
        paths_for_root(tmp_path), golden, 10,
        ["mode=hybrid", "mode=hybrid,BRAIN_TEST_RANKER=on"],
    )
    out = capsys.readouterr().out

    assert "B improves 1 query, regresses 0" in out
    assert "1/2->2/2" in out


def test_compare_reports_a_bad_config_instead_of_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["a.pdf"]}]
    rc = eval_retrieval._run_compare(
        paths_for_root(tmp_path), golden, 10, ["mode=hybrid", "dense"]
    )
    assert rc == 0
    assert "--compare: " in capsys.readouterr().err


def test_compare_warns_when_an_override_matches_the_ambient_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """An exported BRAIN_* makes the A/B a no-op that still prints
    "improves 0, regresses 0" — indistinguishable from a real null result."""
    _stub_search_on(monkeypatch, {("hybrid", "on"): ["want.pdf"]})
    monkeypatch.setenv("BRAIN_TEST_RANKER", "on")
    monkeypatch.setenv("BRAIN_TEST_AMBIENT", "7")
    golden: list[GoldenQuery] = [{"query": "q", "expected": ["want.pdf"]}]

    eval_retrieval._run_compare(
        paths_for_root(tmp_path), golden, 10,
        ["mode=hybrid", "mode=hybrid,BRAIN_TEST_RANKER=on"],
    )
    out = capsys.readouterr().out

    # BRAIN_TEST_RANKER is already `on`, so B is not a change at all...
    assert "sets BRAIN_TEST_RANKER to the value already in the environment" in out
    # ...and BRAIN_TEST_AMBIENT, which neither side names, silently applies to both.
    assert "ambient (applies to BOTH sides): BRAIN_TEST_AMBIENT=7" in out
    assert "B improves 0 queries, regresses 0" in out


def test_compare_says_write_report_is_ignored_rather_than_dropping_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_search_on(monkeypatch, {("hybrid", "off"): ["want.pdf"]})
    golden_file = tmp_path / "golden.jsonl"
    golden_file.write_text(
        json.dumps({"query": "q", "expected": ["want.pdf"], "note": "n"}) + "\n",
        encoding="utf-8",
    )

    rc = eval_retrieval.main([
        "--golden", str(golden_file), "--write-report",
        "--compare", "mode=hybrid", "--compare", "mode=dense",
    ])
    err = capsys.readouterr().err

    assert rc == 0
    assert "--write-report is ignored with --compare" in err


def test_parse_config_accepts_an_escaped_comma_inside_a_value() -> None:
    cfg = eval_retrieval._parse_config(r"mode=dense,BRAIN_QUERY_EXPANSION_VARIANTS=a\,b")
    assert cfg.mode == "dense"
    assert cfg.env == {"BRAIN_QUERY_EXPANSION_VARIANTS": "a,b"}


# --------------------------------------------------------------------------
# the golden set itself
# --------------------------------------------------------------------------

def _golden_rows() -> list[GoldenQuery]:
    return eval_retrieval._load_golden(_GOLDEN)


def _real_golden_rows() -> list[GoldenQuery]:
    """The real set, or a clean skip on the public template branch.

    push_to_upstream.sh replaces this file with a two-row placeholder before
    publishing (the `expected` paths are the owner's vault inventory), and
    copies tests/ verbatim, so a content assertion about the real set would go
    red on the template for a reason nobody there can fix.
    """
    rows = _golden_rows()
    if rows and all(g.get("note", "").startswith("placeholder") for g in rows):
        pytest.skip(
            "private-repo check: scripts/eval/retrieval_golden.jsonl is the "
            "template placeholder set, not the owner's golden set"
        )
    return rows


def test_golden_set_is_large_enough_to_be_a_gate() -> None:
    # A saturated 19-query set could not decide an A/B; 60 is the floor agreed
    # for the ranking work that consumes it.
    assert len(_real_golden_rows()) >= 60


def test_every_golden_query_records_why_its_labels_are_right() -> None:
    missing = [g["query"] for g in _golden_rows() if not g.get("note", "").strip()]
    assert missing == []


def test_golden_queries_are_unique() -> None:
    queries = [g["query"] for g in _golden_rows()]
    assert len(queries) == len(set(queries))


def test_golden_expected_paths_look_like_vault_source_paths() -> None:
    bad = [
        path
        for g in _golden_rows()
        for path in g.get("expected", [])
        if path.startswith("/") or path.startswith("archive/") or "\\" in path
    ]
    assert bad == []


def test_golden_set_carries_no_email_addresses() -> None:
    # The file does NOT ship (push_to_upstream.sh substitutes a placeholder),
    # but it is committed to a repo that could be made public by mistake, and
    # its queries are pasted into reviews and commit messages: course codes and
    # topics are fine, contact details are not.
    text = _GOLDEN.read_text(encoding="utf-8")
    assert re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text) == []


def test_golden_set_covers_knowledge_notes_not_just_archive_sources() -> None:
    # Two of the ranking features are about entity/curated notes beating raw
    # sources, so the gate has to contain queries that should land there.
    knowledge = [
        g["query"] for g in _real_golden_rows()
        if any(p.startswith("knowledge/") for p in g.get("expected", []))
    ]
    assert len(knowledge) >= 10


# Queries whose answer exists in more than one source: a solutions/tutorial
# document that answers the query outright is as correct as the lecture or the
# paper, so labelling only one charges the ranker with a miss it did not
# commit. Each pair below was confirmed by reading both processed documents.
_CO_EQUAL: dict[str, str] = {
    "DiVincenzo criteria for building a quantum computer":
        "university/ELEC0035/Problem sheet_4_with_solutions.pdf",
    "COMP0141 access control models discretionary mandatory role based":
        "university/COMP0141/Tutorial_8_sol.pdf",
    "problem-based learning 6 scope analysis and scoping rules answers":
        "university/COMP0012/sols-ps6.pdf",
}


def test_golden_labels_keep_the_confirmed_co_equal_sources() -> None:
    by_query = {g["query"]: g.get("expected", []) for g in _real_golden_rows()}
    missing = [
        (query, path) for query, path in _CO_EQUAL.items()
        if query in by_query and path not in by_query[query]
    ]
    assert missing == []


def test_golden_file_is_one_json_object_per_line() -> None:
    for lineno, line in enumerate(_GOLDEN.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        assert isinstance(row, dict), f"line {lineno} is not a JSON object"
