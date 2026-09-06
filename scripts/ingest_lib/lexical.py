"""Lexical (BM25) retrieval over the chunk texts already in the semantic
index's metadata sidecar.

Dense retrieval on a small embedding model is systematically weak on exact
identifiers — course codes (COMP0157), project slugs (kern), people names,
error strings — which this vault is saturated with. BM25 over the same chunk
texts (``metadata/embeddings_meta.jsonl``) covers exactly that gap, and
fusing the two rankings (see ``semantic.search`` hybrid mode) beats either
alone. No new files on disk, no model, no network: a pure-Python inverted
index. ``semantic.search`` builds it from the already-loaded meta rows and
caches it under the same generation key as the vectors, so the two rankings
never come from different rebuilds.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

# Letters and digits in ANY script, underscore excluded so snake_case
# identifiers still split into their parts (``memory_status`` -> ``memory``,
# ``status``). ``\w`` alone would keep the underscore and swallow them whole.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """Case-folded alphanumeric runs in any script. Keeps identifiers whole
    (``COMP0157`` -> ``comp0157``) so exact-token queries can match them, and
    keeps non-ASCII names whole (``Wuczyński`` -> ``wuczyński``) — an
    ASCII-only pattern shredded them into fragments (``wuczy``, ``ski``) that
    collide with every other name, which defeats the exact-identifier job this
    leg exists for. NFKC first so compatibility forms (fullwidth digits,
    ligatures) fold onto the same token as their plain spelling."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


@dataclass(frozen=True)
class LexicalIndex:
    postings: dict[str, list[tuple[int, int]]]  # token -> [(row_idx, tf), ...]
    idf: dict[str, float]
    doc_len: list[int]
    avgdl: float
    n_docs: int


def build_lexical_index(texts: list[str]) -> LexicalIndex:
    """Build a BM25 inverted index over ``texts`` (row order preserved)."""
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_len: list[int] = []
    df: dict[str, int] = {}
    for idx, text in enumerate(texts):
        toks = tokenize(text)
        doc_len.append(len(toks))
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            postings.setdefault(t, []).append((idx, c))
            df[t] = df.get(t, 0) + 1
    n = len(texts)
    avgdl = (sum(doc_len) / n) if n else 0.0
    # BM25 idf with the +1 inside the log so it stays non-negative.
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    return LexicalIndex(postings=postings, idf=idf, doc_len=doc_len, avgdl=avgdl, n_docs=n)


def score(index: LexicalIndex, query: str) -> dict[int, float]:
    """BM25 score per document (row index) for ``query``. Only documents that
    contain at least one query token appear in the result."""
    scores: dict[int, float] = {}
    if index.avgdl <= 0:
        return scores
    seen_tokens: set[str] = set()
    for token in tokenize(query):
        if token in seen_tokens:
            continue  # a repeated query token adds nothing under this scoring
        seen_tokens.add(token)
        idf = index.idf.get(token)
        if idf is None:
            continue
        for row_idx, tf in index.postings[token]:
            dl = index.doc_len[row_idx]
            denom = tf + _K1 * (1 - _B + _B * dl / index.avgdl)
            scores[row_idx] = scores.get(row_idx, 0.0) + idf * (tf * (_K1 + 1)) / denom
    return scores


def ranking(index: LexicalIndex, query: str) -> list[int]:
    """Row indices ordered by descending BM25 score (ties broken by row index
    for determinism)."""
    scored = score(index, query)
    return [i for i, _s in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))]
