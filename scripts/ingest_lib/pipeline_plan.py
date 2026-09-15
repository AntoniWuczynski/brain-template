"""Ingestion planning: candidate discovery and the plan/stat dataclasses.

Split out of :mod:`pipeline` (AUD-121) — see that module's docstring for
the split's re-export contract. These pieces carry no dependency on the
extractor/summarizer/search hooks that tests monkeypatch on ``pipeline``
itself, so they can live here without changing how that patching works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator

from .config import VaultPaths


# Asset extensions counted as "figures" for the index note's figures: list.
_FIGURE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"})


@dataclass(frozen=True)
class PlannedItem:
    """One file scheduled for ingestion."""

    src: Path                 # path to read content from (inbox or archive/raw)
    relative_path: str        # repo-root-relative under inbox/ or archive/raw/
    is_in_archive: bool       # True when scanning archive/raw directly


@dataclass
class IngestPlan:
    items: list[PlannedItem] = field(default_factory=list)
    skipped_already_processed: list[PlannedItem] = field(default_factory=list)
    skipped_unsupported: list[PlannedItem] = field(default_factory=list)


@dataclass
class IngestStats:
    processed: int = 0
    partial: int = 0
    manual_review: int = 0
    skipped: int = 0


# The extractor name a connector snapshot records; the meeting-promotion
# pass below is gated on one of these having been processed this run.
_MEETING_EXTRACTOR = "meeting"


def _iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # ".tmp-*": a raw copy torn apart by a kill mid-`_atomic_copy`. It is
        # not a source, and ingesting it would archive the truncated bytes.
        if (
            path.name == ".DS_Store"
            or path.name.startswith("._")
            or path.name.startswith(".tmp-")
        ):
            continue
        yield path


def _relative_to_logical_root(
    f: Path, paths: VaultPaths, *, from_archive: bool
) -> str | None:
    """Compute the path under ``inbox/`` or ``archive/raw/`` for a file.

    If the file is under neither (i.e. ``--path`` pointing outside the
    vault), return None and let the caller fall back to ``f.name``.
    """
    bases: list[Path] = (
        [paths.archive_raw] if from_archive else [paths.inbox, paths.archive_raw]
    )
    for base in bases:
        try:
            rel = f.relative_to(base)
        except ValueError:
            continue
        return rel.as_posix()
    return None
