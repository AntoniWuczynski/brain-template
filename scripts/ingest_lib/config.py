"""Vault paths and constants. The repo root is auto-detected from this file."""
from __future__ import annotations

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
