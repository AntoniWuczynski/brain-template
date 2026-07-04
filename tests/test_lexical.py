"""Lexical BM25 retrieval (P4) — pure, no model."""
from __future__ import annotations

from ingest_lib.lexical import build_lexical_index, ranking, score, tokenize


def test_tokenize_keeps_identifiers_whole():
    assert tokenize("COMP0141: Intro to X") == ["comp0141", "intro", "to", "x"]
    assert tokenize("TCP/IP and UDP") == ["tcp", "ip", "and", "udp"]


def test_bm25_ranks_the_matching_doc_first():
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


def test_bm25_idf_favours_rare_terms():
    # 'the' is in every doc (idf ~ 0); 'quasar' is rare (high idf).
    docs = ["the cat", "the dog", "the quasar"]
    idx = build_lexical_index(docs)
    s = score(idx, "the quasar")
    # The quasar doc must outscore the others despite all sharing 'the'.
    assert s[2] == max(s.values())


def test_empty_index_scores_nothing():
    idx = build_lexical_index([])
    assert score(idx, "anything") == {}
    assert ranking(idx, "anything") == []
