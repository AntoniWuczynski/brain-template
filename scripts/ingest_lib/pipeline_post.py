"""Post-extraction helpers: raw copies, legacy-path migration, dedupe notes,
failed-file archiving, and backfill's frontmatter stripping.

Split out of :mod:`pipeline` (AUD-121) — see that module's docstring for
the split's re-export contract. These pieces carry no dependency on the
extractor/summarizer/search hooks that tests monkeypatch on ``pipeline``
itself, so they can live here without changing how that patching works.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, UTC
from pathlib import Path

from .config import VaultPaths
from .hashing import sha256_of
from .metadata import IndexRecord


def _atomic_copy(src: Path, dest: Path) -> None:
    """Copy a source file into the immutable tree atomically.

    A plain ``copy2`` that dies mid-write leaves a truncated file at
    ``dest`` — and since ``archive/raw`` is never overwritten, that torn
    file reads as a permanent ``archive-clash`` on every later run. Write
    to a temp name in the same directory, fsync, then rename.
    """
    tmp = dest.parent / (".tmp-" + dest.name)
    try:
        shutil.copy2(src, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _migrate_legacy_note_paths(
    paths: VaultPaths,
    prev: IndexRecord,
    *,
    processed_target: Path,
    index_note_target: Path,
    assets_dir: Path,
    logger: logging.Logger,
) -> None:
    """Move a record's notes onto today's derived names before re-ingesting.

    The derived name gained the source's own extension in 6e015a63
    (``report.pdf`` -> ``report.pdf.md``), but the targets are computed from
    the source path alone, so a re-ingest of a note written under the old
    scheme wrote a ``report.pdf.md`` twin beside ``report.md``: the old note
    kept its user frontmatter (rule 9's merge reads the NEW path), its
    assets dir was left behind, and both notes embedded into the search
    index. Renaming the recorded paths onto the current ones first makes the
    re-ingest refresh the note the user has been editing. Only ever renames
    onto a free name, and never touches ``archive/raw``.
    """
    moves: list[tuple[Path, Path]] = []
    if prev.processed_path:
        old_processed = paths.root / prev.processed_path
        moves.append((old_processed, processed_target))
        # The assets dir sits next to the processed note and carries its name
        # (minus ".md") — true under both the old and the current scheme.
        moves.append((
            old_processed.parent / (old_processed.name.removesuffix(".md") + "_assets"),
            assets_dir,
        ))
    if prev.index_note_path:
        moves.append((paths.root / prev.index_note_path, index_note_target))
    for old_path, new_path in moves:
        if old_path == new_path or new_path.exists() or not old_path.exists():
            continue
        new_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(old_path, new_path)
        logger.info(
            "  migrated legacy note path: %s -> %s",
            old_path.relative_to(paths.root),
            new_path.relative_to(paths.root),
        )


def _duplicate_source_notes(
    *, rel: str, src_hash: str, known: dict[str, IndexRecord]
) -> list[str]:
    """Cross-reference an identical file already ingested under another path.

    Same bytes under two paths are archived, extracted and indexed twice
    (only the summary is deduplicated), with nothing in either note saying
    so. Name the first one, alphabetically, so the note is deterministic.
    """
    others = sorted(
        r.relative_path
        for r in known.values()
        if r.source_hash == src_hash
        and r.relative_path != rel
        and r.status in ("processed", "partial")
    )
    if not others:
        return []
    return [f"duplicate of `{others[0]}` (identical content, same source_hash)"]


def _title_from_relpath(rel: str) -> str:
    stem = Path(rel).stem
    return stem.replace("_", " ").replace("-", " ").strip() or rel


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strip_frontmatter_header(text: str) -> str:
    """Recover just the extracted body from a processed note.

    ``write_processed_note`` wraps the body as::

        # Title
        > Source/Hash/Extractor/Status
        ---
        <body>
        ---
        ## Processing notes
        <notes>

    We must strip BOTH the leading title/meta block (up to the first
    ``---``) AND the trailing ``---`` + ``## Processing notes`` footer.
    Missing the footer meant ``backfill_summaries`` fed the old notes back
    to the summarizer and re-wrapped them, growing a duplicate
    ``## Processing notes`` section on every backfill (and embedding the
    stale metadata into the search index). The trailing strip loops so a
    note already carrying several duplicated footers heals to a single body.
    """
    lines = text.splitlines(keepends=True)
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip() == "---" and i > 0:
            start = i + 1
            break
    body = "".join(lines[start:]).lstrip()
    # Strip every trailing '---\n\n## Processing notes\n...' footer.
    while True:
        blines = body.splitlines(keepends=True)
        cut = None
        for j in range(len(blines) - 1, -1, -1):
            if blines[j].strip() == "## Processing notes":
                k = j - 1
                while k >= 0 and blines[k].strip() == "":
                    k -= 1
                if k >= 0 and blines[k].strip() == "---":
                    cut = k
                break
        if cut is None:
            break
        body = "".join(blines[:cut]).rstrip() + "\n"
    return body


def _collect_existing_topics(latest: dict[str, IndexRecord]) -> list[str]:
    """Return distinct canonical topics across all records."""
    seen: set[str] = set()
    out: list[str] = []
    for r in latest.values():
        for t in r.topics or []:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _move_to_failed(raw_target: Path, paths: VaultPaths) -> Path | None:
    """Move a failed raw file into ``archive/failed/``. Returns the FINAL
    destination path (which may carry a ``.N`` suffix if a prior failure
    already occupies the plain name), or None if there was nothing to move,
    so the caller can record the true location in metadata."""
    if not raw_target.exists():
        return None
    try:
        rel = raw_target.relative_to(paths.archive_raw)
    except ValueError:
        return None
    failed_target = paths.archive_failed / rel
    failed_target.parent.mkdir(parents=True, exist_ok=True)
    if failed_target.exists():
        # A prior failure already sits here. If it's byte-identical (the same
        # file failing again on a re-run — inbox sources are never deleted, so
        # this is the common case), the bytes are already preserved: drop the
        # duplicate rather than stacking x.docx.1, x.docx.2, ... on every run.
        raw_hash = sha256_of(raw_target)
        if sha256_of(failed_target) == raw_hash:
            raw_target.unlink()
            return failed_target
        # Different bytes: don't overwrite — find the next free suffix whose
        # existing occupant also differs (an identical .N copy is likewise a
        # no-op we reuse).
        i = 1
        while True:
            cand = failed_target.with_name(failed_target.name + f".{i}")
            if not cand.exists():
                failed_target = cand
                break
            if sha256_of(cand) == raw_hash:
                raw_target.unlink()
                return cand
            i += 1
    shutil.move(str(raw_target), str(failed_target))
    return failed_target
