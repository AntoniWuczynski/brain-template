"""Vault paths and constants. The repo root is auto-detected from this file."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    # scripts/ingest_lib/config.py -> scripts/ingest_lib -> scripts -> <root>
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VaultPaths:
    """All vault directories. Use ``default_paths()`` to construct."""

    root: Path
    inbox: Path
    archive_raw: Path
    archive_processed: Path
    archive_failed: Path
    knowledge: Path
    knowledge_index: Path
    metadata: Path
    metadata_index_jsonl: Path
    logs: Path

    def ensure(self) -> None:
        for p in (
            self.inbox,
            self.archive_raw,
            self.archive_processed,
            self.archive_failed,
            self.knowledge,
            self.knowledge_index,
            self.metadata,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)


def paths_for_root(root: Path) -> VaultPaths:
    """Build VaultPaths for an explicit vault root."""
    root = Path(root).expanduser().resolve()
    return VaultPaths(
        root=root,
        inbox=root / "inbox",
        archive_raw=root / "archive" / "raw",
        archive_processed=root / "archive" / "processed",
        archive_failed=root / "archive" / "failed",
        knowledge=root / "knowledge",
        knowledge_index=root / "knowledge" / "index",
        metadata=root / "metadata",
        metadata_index_jsonl=root / "metadata" / "index.jsonl",
        logs=root / "logs",
    )


def default_paths() -> VaultPaths:
    return paths_for_root(_repo_root())


# ---------------------------------------------------------------------------
# Optional ranking layers (see rank.py)
# ---------------------------------------------------------------------------
#
# Four retrieval-polish layers sit on top of the hybrid fusion in
# ``semantic.search``. EVERY ONE IS OFF BY DEFAULT: with no environment set,
# ``ranking_config()`` returns all-false and the ranker behaves exactly as it
# did before they existed. Each is switched independently so the eval harness
# can A/B one at a time:
#
#     uv run python scripts/eval_retrieval.py \
#         --compare mode=hybrid --compare mode=hybrid,BRAIN_RANK_GRAPH=1
#
# Weights are expressed in units of the top RRF contribution (1/(k+1)), so a
# weight of 1.0 means "at most as much as being ranked first in one of the two
# fused rankings". They are read at call time, not import time, so a test or
# an A/B run can set them around a single call.

_TRUE = frozenset({"1", "true", "yes", "on"})


def _env_flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in _TRUE


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RankingConfig:
    """Which optional ranking layers are on, and how strongly.

    ``graph_boost``/``salience`` are bounded ADDITIVE terms on the RRF score.
    ``rerank`` reorders the top ``rerank_top_n`` fused hits with a local
    cross-encoder. ``query_expansion`` unions LLM paraphrases of the query
    into the fusion. All four apply to ``mode="hybrid"`` only — the other two
    modes return raw cosine / BM25 scores, which these layers would silently
    change the meaning of.
    """

    graph_boost: bool = False
    graph_weight: float = 0.5
    salience: bool = False
    salience_weight: float = 0.5
    rerank: bool = False
    rerank_top_n: int = 30
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    query_expansion: bool = False
    query_expansion_variants: int = 3
    query_expansion_weight: float = 0.5

    @property
    def any_enabled(self) -> bool:
        return (
            self.graph_boost or self.salience or self.rerank or self.query_expansion
        )


def ranking_config(env: Mapping[str, str] | None = None) -> RankingConfig:
    """Read the ranking layers from the environment (defaults: all off)."""
    e = os.environ if env is None else env
    return RankingConfig(
        graph_boost=_env_flag(e, "BRAIN_RANK_GRAPH"),
        graph_weight=_env_float(e, "BRAIN_RANK_GRAPH_WEIGHT", 0.5),
        salience=_env_flag(e, "BRAIN_RANK_SALIENCE"),
        salience_weight=_env_float(e, "BRAIN_RANK_SALIENCE_WEIGHT", 0.5),
        rerank=_env_flag(e, "BRAIN_RANK_RERANK"),
        rerank_top_n=max(2, _env_int(e, "BRAIN_RANK_RERANK_TOP_N", 30)),
        rerank_model=e.get("BRAIN_RANK_RERANK_MODEL")
        or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        query_expansion=_env_flag(e, "BRAIN_QUERY_EXPANSION"),
        query_expansion_variants=min(
            5, max(1, _env_int(e, "BRAIN_QUERY_EXPANSION_VARIANTS", 3))
        ),
        query_expansion_weight=_env_float(e, "BRAIN_QUERY_EXPANSION_WEIGHT", 0.5),
    )
