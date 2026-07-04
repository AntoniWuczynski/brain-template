"""End-to-end ingestion pipeline.

Public entry points:
- :func:`plan_ingest` — collect candidate files and decide what would happen
  (used by ``--dry-run`` and as the first step of a real run).
- :func:`run_ingest` — execute a plan, writing outputs and metadata.

Both functions are deterministic given the same filesystem state.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from collections.abc import Iterator, Sequence

from .concepts import rebuild_concepts
from .config import VaultPaths
from .connections import rebuild_connections
from .dashboards import rebuild_dashboards
from .extractors import ExtractionResult, dispatch_extractor
from .hashing import sha256_of
from .metadata import IndexRecord, append_record, latest_records_by_path
from .notes import (
    NoteContent,
    derived_assets_dirname,
    derived_note_relpath,
    write_index_note,
    write_processed_note,
)
from .semantic import build_index as _build_search_index
from .summarize import is_enabled as _summary_enabled, summarize as _summarize


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


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_ingest(
    paths: VaultPaths,
    *,
    sources: Sequence[Path],
    from_archive: bool,
    logger: logging.Logger,
) -> IngestPlan:
    """Walk the source(s), filter by registered extensions, dedupe by hash."""
    paths.ensure()
    known_by_path = latest_records_by_path(paths.metadata_index_jsonl)

    plan = IngestPlan()
    seen_relative: set[str] = set()

    for source in sources:
        if not source.exists():
            logger.warning("source path does not exist: %s", source)
            continue
        for f in _iter_files(source):
            rel = _relative_to_logical_root(f, paths, from_archive=from_archive)
            if rel is None:
                # Outside both inbox and archive/raw — single-file mode.
                rel = f.name
            if rel in seen_relative:
                continue
            seen_relative.add(rel)

            extractor = dispatch_extractor(f)
            item = PlannedItem(src=f, relative_path=rel, is_in_archive=from_archive)
            if extractor is None:
                plan.skipped_unsupported.append(item)
                continue

            existing = known_by_path.get(rel)
            if existing and existing.status == "processed":
                # Cheap shortcut: if the size hasn't changed we trust the hash.
                try:
                    same_size = f.stat().st_size == existing.size_bytes
                except OSError:
                    same_size = False
                if same_size and sha256_of(f) == existing.source_hash:
                    plan.skipped_already_processed.append(item)
                    continue
            plan.items.append(item)

    plan.items.sort(key=lambda i: i.relative_path)
    plan.skipped_already_processed.sort(key=lambda i: i.relative_path)
    plan.skipped_unsupported.sort(key=lambda i: i.relative_path)
    return plan


def _iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".DS_Store" or path.name.startswith("._"):
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


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_ingest(
    paths: VaultPaths,
    plan: IngestPlan,
    *,
    dry_run: bool,
    logger: logging.Logger,
) -> IngestStats:
    paths.ensure()
    stats = IngestStats(skipped=len(plan.skipped_already_processed) + len(plan.skipped_unsupported))

    for item in plan.skipped_already_processed:
        logger.info("skip (already processed): %s", item.relative_path)
    for item in plan.skipped_unsupported:
        logger.info("skip (unsupported extension): %s", item.relative_path)

    # Load the record index ONCE and update it in memory after each append
    # (the same pattern backfill_summaries uses). Reading it per file made
    # ingest O(files x JSONL size) — a 200-file drop re-parsed the whole
    # multi-MB index 200 times.
    known = {} if dry_run else latest_records_by_path(paths.metadata_index_jsonl)

    for item in plan.items:
        outcome = _process_one(paths, item, dry_run=dry_run, logger=logger, known=known)
        if outcome == "processed":
            stats.processed += 1
        elif outcome == "partial":
            stats.partial += 1
        elif outcome == "manual_review":
            stats.manual_review += 1
        else:
            # 'skipped' (stat failure, or a file that vanished between
            # planning and execution): count it so the run summary adds up
            # to the number of planned items instead of silently dropping one.
            stats.skipped += 1

    # Refresh concept notes whenever we actually wrote new content (and
    # not on dry-runs). Cheap: just walks the JSONL, no LLM calls.
    if not dry_run and (stats.processed or stats.partial):
        # Build the semantic index first: concept centroids (and thus the
        # connection graph's semantic edges) read fresh vectors from it.
        # Cheap (~1 chunk/ms on MPS); failure is non-fatal — search just
        # stays stale until the next rebuild.
        try:
            n = _build_search_index(paths, logger=logger)
            logger.info("semantic: indexed %d chunks", n)
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic: index build failed (%r) — skipping", exc)
        # Discover concept relationships (deterministic; uses the index above
        # when present, co-occurrence alone when not). Non-fatal.
        related = None
        try:
            conn = rebuild_connections(paths, logger=logger)
            related = conn.related
        except Exception as exc:  # noqa: BLE001
            logger.warning("connections: rebuild failed (%r) — skipping", exc)
        # Refresh concept notes last, linking each to its related concepts.
        # Wrapped like the other derived-view rebuilds: a bad concept note
        # must not crash a run whose files were already processed & recorded.
        try:
            cs = rebuild_concepts(paths, logger=logger, related=related)
            logger.info(
                "concepts: written=%d unchanged=%d skipped=%d removed=%d",
                cs.written, cs.unchanged, cs.skipped, cs.removed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("concepts: rebuild failed (%r) — skipping", exc)
        # Entity dashboards are derived views over the same notes; a
        # failure must not fail the ingest run (same pattern as the
        # connections rebuild above).
        try:
            db = rebuild_dashboards(paths, logger=logger)
            logger.info(
                "dashboards: written=%d unchanged=%d", db.written, db.unchanged
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("dashboards: rebuild failed (%r) — skipping", exc)
        # Processing Dashboard + Manual Review: derived from index.jsonl + the
        # filesystem, non-fatal like the other derived views.
        try:
            from .status import rebuild_status
            rebuild_status(paths, logger=logger)
        except Exception as exc:  # noqa: BLE001
            logger.warning("status: rebuild failed (%r) — skipping", exc)
        # Opt-in: generate AI concept descriptions inline (costs LLM calls, so
        # off by default). Only stale concepts regenerate. Non-fatal.
        if os.environ.get("BRAIN_AUTO_DESCRIBE") == "1":
            try:
                from .describe import rebuild_descriptions
                ds = rebuild_descriptions(paths, logger=logger)
                logger.info("describe: generated=%d skipped=%d",
                            ds.generated, ds.skipped_uptodate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("describe: failed (%r) — skipping", exc)
    return stats


def _process_one(
    paths: VaultPaths,
    item: PlannedItem,
    *,
    dry_run: bool,
    logger: logging.Logger,
    known: dict[str, IndexRecord],
) -> str:
    rel = item.relative_path
    src = item.src
    extractor = dispatch_extractor(src)
    if extractor is None:
        # Defensive — planning already filtered these.
        logger.warning("unexpected unsupported file in plan: %s", rel)
        return "skipped"

    logger.info("processing: %s", rel)

    try:
        size = src.stat().st_size
    except OSError as exc:
        logger.error("stat failed for %s: %s", rel, exc)
        return "skipped"

    src_hash = sha256_of(src)
    raw_target = paths.archive_raw / rel
    # Keep the source's extension in the derived names so report.pdf and
    # report.docx in one folder don't clobber each other at report.md.
    derived_rel = derived_note_relpath(rel)
    processed_target = paths.archive_processed / derived_rel
    index_note_target = paths.knowledge_index / derived_rel
    assets_dir = processed_target.parent / derived_assets_dirname(rel)

    if dry_run:
        logger.info(
            "  would copy %s -> %s (size=%d, hash=%s, extractor=%s)",
            src,
            raw_target,
            size,
            src_hash[:12],
            extractor.__module__.rsplit(".", 1)[-1],
        )
        logger.info("  would write processed -> %s", processed_target)
        logger.info("  would write index note -> %s", index_note_target)
        return "processed"  # best-effort label for stats; not actually written

    # 1. Copy raw if needed.
    if not item.is_in_archive:
        if raw_target.exists():
            existing_hash = sha256_of(raw_target)
            if existing_hash != src_hash:
                logger.error(
                    "raw file already exists with a different hash: %s "
                    "(existing=%s, incoming=%s) — refusing to overwrite",
                    raw_target,
                    existing_hash[:12],
                    src_hash[:12],
                )
                # Treat as manual review.
                _record_failure(
                    paths,
                    rel=rel,
                    src_hash=src_hash,
                    size=size,
                    extension=src.suffix.lower(),
                    error="raw archive already has a different file at this path",
                    raw_path=str(raw_target.relative_to(paths.root)),
                    extractor_name="archive-clash",
                )
                return "manual_review"
        else:
            raw_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, raw_target)

    # 2. Run the extractor. Clear any assets from a PREVIOUS extraction of
    #    this source first: on a re-ingest (changed hash) the old
    #    content-hash-named images would otherwise linger unreferenced and
    #    accumulate forever. archive/processed is regenerable by contract.
    if assets_dir.exists():
        shutil.rmtree(assets_dir, ignore_errors=True)
    try:
        result = extractor(src, assets_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("extractor crashed for %s", rel)
        result = ExtractionResult(
            status="manual_review",
            extractor=extractor.__module__.rsplit(".", 1)[-1],
            markdown="",
            error=f"extractor crashed: {exc!r}",
        )

    # 3. On manual_review move file to archive/failed and update metadata.
    if result.status == "manual_review":
        moved = _move_to_failed(raw_target, paths)
        # Record the ACTUAL destination (which may be a .N suffix), not the
        # plain path — otherwise the metadata source-of-truth points at the
        # wrong bytes when a path fails more than once.
        failed_rel = (
            str(moved.relative_to(paths.root)) if moved is not None
            else str((paths.archive_failed / rel).relative_to(paths.root))
        )
        _record_failure(
            paths,
            rel=rel,
            src_hash=src_hash,
            size=size,
            extension=src.suffix.lower(),
            error=result.error or "unknown extractor error",
            raw_path=failed_rel,
            extractor_name=result.extractor,
        )
        logger.warning("  manual_review: %s — %s", rel, result.error)
        return "manual_review"

    # 4. Optionally summarize. Reuse a cached summary keyed by source_hash
    #    when the same content was summarized in a previous run.
    title = _title_from_relpath(rel)
    summary_text, key_points, topics, summary_notes = _maybe_summarize(
        rel=rel,
        src_hash=src_hash,
        result=result,
        title=title,
        paths=paths,
        logger=logger,
        known=known,
    )
    full_notes = list(result.notes) + summary_notes

    # 5. Write the processed Markdown.
    note_payload = NoteContent(
        title=title,
        source_relative_path=rel,
        source_hash=src_hash,
        status=result.status,
        extracted_markdown=result.markdown,
        processing_notes=full_notes,
        extractor=result.extractor,
        summary=summary_text,
        key_points=tuple(key_points),
        topics=tuple(topics),
    )
    write_processed_note(target=processed_target, content=note_payload)

    # 6. Write/refresh the index note.
    write_index_note(target=index_note_target, content=note_payload)

    # 7. Append metadata record.
    now = _utc_now_iso()
    record = IndexRecord(
        relative_path=rel,
        source_hash=src_hash,
        size_bytes=size,
        extension=src.suffix.lower(),
        extractor=result.extractor,
        status=result.status,  # type: ignore[arg-type]
        raw_path=str(raw_target.relative_to(paths.root)),
        processed_path=str(processed_target.relative_to(paths.root)),
        index_note_path=str(index_note_target.relative_to(paths.root)),
        assets=[
            str(p.relative_to(paths.root))
            for p in result.assets
        ],
        created_at=now,
        updated_at=now,
        error=result.error,
        notes=full_notes,
        summary=summary_text,
        key_points=list(key_points),
        topics=list(topics),
    )
    append_record(paths.metadata_index_jsonl, record)
    # Reflect the append in the in-memory index so later files in this run
    # reuse the same summary cache / canonical-topic list without re-reading.
    known[record.relative_path] = record
    logger.info(
        "  %s: %s (extractor=%s, %d asset(s))",
        result.status,
        rel,
        result.extractor,
        len(result.assets),
    )
    return result.status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _title_from_relpath(rel: str) -> str:
    stem = Path(rel).stem
    return stem.replace("_", " ").replace("-", " ").strip() or rel


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def backfill_summaries(
    paths: VaultPaths,
    *,
    logger: logging.Logger,
) -> IngestStats:
    """Add summaries to processed records that lack them.

    Walks ``metadata/index.jsonl``, reads each record's ``processed_path``
    from disk, calls the summarizer, rewrites the index note, and appends
    an updated record. Does **not** re-extract — much faster than ``--raw``
    and only touches LLM credit, not MinerU.
    """
    paths.ensure()
    stats = IngestStats()

    if not _summary_enabled():
        logger.error(
            "summarization is disabled — configure an LLM provider "
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / "
            "BRAIN_LOCAL_URL; see scripts/README.md) and unset "
            "BRAIN_SKIP_SUMMARY — nothing to do"
        )
        return stats

    latest = latest_records_by_path(paths.metadata_index_jsonl)
    candidates = [
        rec for rec in latest.values()
        if rec.status == "processed"
        and not (rec.summary and rec.key_points and rec.topics)
        and rec.processed_path
    ]
    logger.info("backfill plan: %d record(s) need summaries", len(candidates))

    for rec in candidates:
        if rec.processed_path is None:
            # Filtered out of `candidates` already; re-check narrows the type
            # (str | None -> str) so the path join below is well-typed.
            continue
        processed_full = paths.root / rec.processed_path
        if not processed_full.is_file():
            logger.warning("processed file missing for %s — skipping", rec.relative_path)
            stats.skipped += 1
            continue

        body = _strip_frontmatter_header(processed_full.read_text(encoding="utf-8"))
        title = _title_from_relpath(rec.relative_path)
        existing_topics = _collect_existing_topics(latest)
        out = _summarize(
            body,
            title=title,
            source_relative_path=rec.relative_path,
            existing_topics=existing_topics,
            logger=logger,
        )
        if out is None:
            logger.warning("  summary skipped: %s", rec.relative_path)
            stats.skipped += 1
            continue

        # Write the index note + processed note with the new summary, and
        # append a fresh record so the JSONL reflects the change.
        index_target = paths.root / rec.index_note_path if rec.index_note_path else None
        processed_target = processed_full
        new_notes = list(rec.notes) + list(out.notes)
        payload = NoteContent(
            title=title,
            source_relative_path=rec.relative_path,
            source_hash=rec.source_hash,
            status=rec.status,
            extracted_markdown=body,
            processing_notes=new_notes,
            extractor=rec.extractor,
            summary=out.summary,
            key_points=tuple(out.key_points),
            topics=tuple(out.topics),
        )
        write_processed_note(target=processed_target, content=payload)
        if index_target is not None:
            write_index_note(target=index_target, content=payload)

        now = _utc_now_iso()
        new_record = IndexRecord(
            relative_path=rec.relative_path,
            source_hash=rec.source_hash,
            size_bytes=rec.size_bytes,
            extension=rec.extension,
            extractor=rec.extractor,
            status=rec.status,  # type: ignore[arg-type]
            raw_path=rec.raw_path,
            processed_path=rec.processed_path,
            index_note_path=rec.index_note_path,
            assets=list(rec.assets),
            created_at=rec.created_at or now,
            updated_at=now,
            error=rec.error,
            notes=new_notes,
            summary=out.summary,
            key_points=list(out.key_points),
            topics=list(out.topics),
        )
        append_record(paths.metadata_index_jsonl, new_record)
        # Make this record visible to subsequent iterations so the
        # canonical-topic list stays consistent within one backfill run.
        latest[new_record.relative_path] = new_record
        stats.processed += 1
        logger.info("  summarized: %s (topics=%d)", rec.relative_path, len(out.topics))

    # After backfilling summaries, refresh the connection graph then the
    # concept notes so cross-source links and related-concept links land in
    # one shot.
    related = None
    try:
        conn = rebuild_connections(paths, logger=logger)
        related = conn.related
    except Exception as exc:  # noqa: BLE001
        logger.warning("connections: rebuild failed (%r) — skipping", exc)
    cs = rebuild_concepts(paths, logger=logger, related=related)
    logger.info(
        "concepts: written=%d unchanged=%d skipped=%d removed=%d",
        cs.written, cs.unchanged, cs.skipped, cs.removed,
    )
    return stats


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


def _maybe_summarize(
    *,
    rel: str,
    src_hash: str,
    result: ExtractionResult,
    title: str,
    paths: VaultPaths,
    logger: logging.Logger,
    known: dict[str, IndexRecord],
) -> tuple[str, list[str], list[str], list[str]]:
    """Return ``(summary, key_points, topics, extra_notes)``.

    Reuses a cached summary from the latest record whose ``source_hash``
    matches; otherwise calls the LLM. Skipped when the result is not
    ``processed``, when the body is empty, or when summarization is
    disabled (no ``ANTHROPIC_API_KEY`` / ``BRAIN_SKIP_SUMMARY=1``).
    ``known`` is the caller's in-memory record index (updated as the run
    progresses) — read from it instead of re-parsing the JSONL per file.
    """
    if result.status != "processed":
        return "", [], [], []
    if not (result.markdown or "").strip():
        return "", [], [], []

    latest = known

    # Hash-keyed cache: any prior record with the same source_hash carries
    # a summary we can reuse — but only if it has the new ``topics``
    # field. Older records pre-dating topics get re-summarized.
    for prev in latest.values():
        if (
            prev.source_hash == src_hash
            and prev.summary
            and prev.key_points
            and prev.topics
        ):
            return (
                prev.summary,
                list(prev.key_points),
                list(prev.topics),
                ["summary: reused from previous run (same source_hash)"],
            )

    if not _summary_enabled():
        return "", [], [], []

    existing_topics = _collect_existing_topics(latest)
    out = _summarize(
        result.markdown,
        title=title,
        source_relative_path=rel,
        existing_topics=existing_topics,
        logger=logger,
    )
    if out is None:
        return "", [], [], ["summary: skipped (see warnings in log)"]
    return out.summary, list(out.key_points), list(out.topics), list(out.notes)


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
        # Don't overwrite — append a numeric suffix so we never lose data.
        i = 1
        while True:
            cand = failed_target.with_name(failed_target.name + f".{i}")
            if not cand.exists():
                failed_target = cand
                break
            i += 1
    shutil.move(str(raw_target), str(failed_target))
    return failed_target


def _record_failure(
    paths: VaultPaths,
    *,
    rel: str,
    src_hash: str,
    size: int,
    extension: str,
    error: str,
    raw_path: str,
    extractor_name: str,
) -> None:
    now = _utc_now_iso()
    record = IndexRecord(
        relative_path=rel,
        source_hash=src_hash,
        size_bytes=size,
        extension=extension,
        extractor=extractor_name,
        status="manual_review",
        raw_path=raw_path,
        processed_path=None,
        index_note_path=None,
        assets=[],
        created_at=now,
        updated_at=now,
        error=error,
        notes=[],
    )
    append_record(paths.metadata_index_jsonl, record)
