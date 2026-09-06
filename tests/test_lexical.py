"""Lexical BM25 retrieval (P4) — pure, no model."""
from __future__ import annotations

from ingest_lib.lexical import build_lexical_index, ranking, score, tokenize


def test_tokenize_keeps_identifiers_whole() -> None:
    assert tokenize("COMP0141: Intro to X") == ["comp0141", "intro", "to", "x"]
    assert tokenize("TCP/IP and UDP") == ["tcp", "ip", "and", "udp"]


def test_bm25_ranks_the_matching_doc_first() -> None:
    docs = [
        "an introduction to graphs and trees",
        "error handling and exception design in COMP0141",
        "recursion in functional programming",
    ]
    idx = build_lexical_index(docs)
    order = ranking(idx, "COMP0141 error handling")
    assert order[0] == 1                       # the doc containing the exact terms
    # A doc with no query token is absent entirely.
    assert 2 not in order


def test_bm25_idf_favours_rare_terms() -> None:
    # 'the' is in every doc (idf ~ 0); 'quasar' is rare (high idf).
    docs = ["the cat", "the dog", "the quasar"]
    idx = build_lexical_index(docs)
    s = score(idx, "the quasar")
    # The quasar doc must outscore the others despite all sharing 'the'.
    assert s[2] == max(s.values())


def test_empty_index_scores_nothing() -> None:
    idx = build_lexical_index([])
    assert score(idx, "anything") == {}
    assert ranking(idx, "anything") == []


def test_tokenize_keeps_non_ascii_names_whole() -> None:
    # The ASCII-only pattern shredded these into fragments ('wuczy', 'ski')
    # that collide with every other -ski name (AUD-044); the lexical leg
    # exists precisely to match people names exactly.
    assert tokenize("Łukasz Wuczyński") == ["łukasz", "wuczyński"]
    assert tokenize("über café Kraków") == ["über", "café", "kraków"]
    assert tokenize("Kowalska") == ["kowalska"]


def test_tokenize_still_splits_snake_case_and_folds_compatibility_forms() -> None:
    # Underscore stays a separator (a bare \w+ would swallow the whole name).
    assert tokenize("memory_status: superseded") == ["memory", "status", "superseded"]
    # NFKC folds compatibility forms onto their plain spelling.
    assert tokenize("ﬁle ４２") == ["file", "42"]


def test_bm25_matches_a_non_ascii_name_without_matching_its_fragments() -> None:
    docs = ["a meeting with Anna Wuczyńska", "a meeting with Jan Kowalski"]
    idx = build_lexical_index(docs)
    assert ranking(idx, "Wuczyńska") == [0]
