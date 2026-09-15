"""Sweep checks over the embeddings index, assistant memory, and the dream
gate — split out of ``sweep.py`` (AUD-121). Pure compute, same "never
raises, always a finding" contract as the rest of the sweep.

Finding categories produced here: ``index-drift-stale``,
``index-drift-missing``, ``index-drift-unindexed``, ``stale-unconsolidated``,
``memory-unconsolidatable``, ``dream-stalled``."""
from __future__ import annotations

import json
from datetime import date, datetime

from .config import VaultPaths
from .dream import load_pending_since
from .hashing import sha256_of
from .knowledge import KNOWLEDGE_EXTRACTOR
from .metadata import IndexRecord
from .notes import _split_frontmatter  # private helper, but module-internal
from .sweep_checks_graph import _parse_day
from .sweep_common import Finding, _read_text


def _check_index_drift(
    paths: VaultPaths,
    latest: dict[str, IndexRecord],
    knowledge_recs: list[IndexRecord],
) -> list[Finding]:
    """index-drift-stale / index-drift-missing / index-drift-unindexed.

    Knowledge-note rows (origin ``knowledge-note``) embed the note itself,
    so staleness compares against the note's current hash; archive rows
    compare against the latest index.jsonl record. No embeddings index on
    disk (fresh clone) -> all three checks are skipped gracefully.
    """
    meta_path = paths.metadata / "embeddings_meta.jsonl"
    if not meta_path.is_file():
        return []
    meta_text = _read_text(meta_path)
    if meta_text is None:
        return []

    # All rows of one source share its hash/origin — the first one speaks
    # for the source. Malformed lines are skipped, never fatal.
    first_by_source: dict[str, dict[str, object]] = {}
    for line in meta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        src = row.get("source_relative_path")
        if isinstance(src, str) and src and src not in first_by_source:
            first_by_source[src] = row

    findings: list[Finding] = []
    for src in sorted(first_by_source):
        row = first_by_source[src]
        indexed_hash = str(row.get("source_hash") or "")
        if str(row.get("origin") or "") == KNOWLEDGE_EXTRACTOR:
            note = paths.root / src
            if not note.is_file():
                findings.append(Finding(
                    "index-drift-missing", src,
                    "indexed knowledge note no longer exists on disk",
                ))
            else:
                # A note that vanishes/loses read permission between is_file()
                # and the hash must not crash the whole linter (its contract is
                # 'never raises'); every other read here goes through _read_text.
                try:
                    current_hash = sha256_of(note)
                except OSError:
                    findings.append(Finding(
                        "index-drift-missing", src,
                        "indexed knowledge note is unreadable",
                    ))
                    continue
                if current_hash != indexed_hash:
                    findings.append(Finding(
                        "index-drift-stale", src,
                        "note content changed since indexing — rebuild the search index",
                    ))
        else:
            rec = latest.get(src)
            if rec is None:
                findings.append(Finding(
                    "index-drift-missing", src,
                    "no index.jsonl record for this indexed source",
                ))
            elif rec.source_hash != indexed_hash:
                findings.append(Finding(
                    "index-drift-stale", src,
                    "source_hash changed since indexing — rebuild the search index",
                ))

    indexed = set(first_by_source)
    for rel, rec in sorted(latest.items()):
        # Only sources the index build would actually embed (processed,
        # with a processed_path) can meaningfully be "unindexed".
        if rec.status == "processed" and rec.processed_path and rel not in indexed:
            findings.append(Finding(
                "index-drift-unindexed", rel,
                "processed record has no rows in the embeddings index",
            ))
    for rec in knowledge_recs:
        if rec.relative_path not in indexed:
            findings.append(Finding(
                "index-drift-unindexed", rec.relative_path,
                "knowledge note has no rows in the embeddings index",
            ))
    return findings


def _coerce_day(raw: object) -> date | None:
    """Frontmatter ``created:`` -> date. Quoted strings, bare YAML dates,
    and full ISO timestamps all collapse to the calendar day; anything
    unparseable is None (a note without a clock can't be aged)."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and len(raw.strip()) >= 10:
        return _parse_day(raw.strip()[:10])
    return None


def _check_stale_memory(
    paths: VaultPaths, *, as_of: date, stale_days: int
) -> list[Finding]:
    """stale-unconsolidated: assistant memory left unconsolidated longer
    than ``stale_days`` should be promoted or digested, not forgotten —
    plus memory-unconsolidatable for the notes whose age is unknowable."""
    findings: list[Finding] = []
    base = paths.knowledge / "assistant"
    if not base.is_dir():
        return findings
    # F8: archive/ and digests/ are consolidate's HANDLED history, not a
    # backlog. Digested notes keep ``memory_status: unconsolidated`` after
    # being moved into archive/, so without this skip the sweep would flag
    # them stale forever and contradict consolidate. Only inbox/ holds the
    # actionable backlog.
    archive_dir = base / "archive"
    digests_dir = base / "digests"
    for md in sorted(base.rglob("*.md")):
        if not md.is_file():
            continue
        if archive_dir in md.parents or digests_dir in md.parents:
            continue
        text = _read_text(md)
        if text is None or not text.strip():
            continue
        frontmatter, _body = _split_frontmatter(text)
        raw_status = frontmatter.get("memory_status")
        status = raw_status.strip() if isinstance(raw_status, str) else ""
        if status != "unconsolidated":
            continue
        created = _coerce_day(frontmatter.get("created"))
        if created is None:
            # F13: no clock, so this note can never age out — and if it also
            # sits outside consolidate's flat inbox, nothing else will ever
            # look at it either. Report it instead of skipping in silence.
            findings.append(Finding(
                "memory-unconsolidatable",
                md.relative_to(paths.root).as_posix(),
                "memory_status: unconsolidated but created: is missing or "
                "unparseable — the staleness check can never fire",
            ))
            continue
        age = (as_of - created).days
        if age > stale_days:
            findings.append(Finding(
                "stale-unconsolidated",
                md.relative_to(paths.root).as_posix(),
                f"unconsolidated for {age} day(s) "
                f"(created {created.isoformat()}, threshold {stale_days})",
            ))
    return findings


def _check_inbox_unprocessable(paths: VaultPaths) -> list[Finding]:
    """memory-unconsolidatable: a note in consolidate's inbox that
    consolidate itself classifies as "not a memory fact note" (no
    ``promote`` mapping and no ``memory_status``) — it is skipped on every
    run, never promoted, never digested, and never stale (F13)."""
    findings: list[Finding] = []
    inbox = paths.knowledge / "assistant" / "inbox"
    if not inbox.is_dir():
        return findings
    for md in sorted(inbox.glob("*.md")):  # flat, exactly like consolidate
        if not md.is_file():
            continue
        text = _read_text(md)
        if text is None or not text.strip():
            continue
        frontmatter, _body = _split_frontmatter(text)
        raw_status = frontmatter.get("memory_status")
        status = raw_status.strip() if isinstance(raw_status, str) else ""
        if isinstance(frontmatter.get("promote"), dict) or status:
            continue
        findings.append(Finding(
            "memory-unconsolidatable",
            md.relative_to(paths.root).as_posix(),
            "not a memory fact note (no promote mapping, no memory_status) — "
            "consolidate skips it on every run",
        ))
    return findings


def _check_dream_stalled(paths: VaultPaths, *, as_of: date) -> list[Finding]:
    """The dream gate records ``metadata/dream.pending`` when it fires; a
    completed run clears it via ``dream_gate.py --mark-done``. A marker
    two or more days old means dream runs keep failing or never launch."""
    since = load_pending_since(paths)
    if since is None:
        return []
    age_days = (as_of - since.date()).days
    if age_days < 2:
        return []
    return [
        Finding(
            "dream-stalled",
            "metadata/dream.pending",
            f"dream gate fired {age_days} days ago but no run completed "
            "(dream_gate.py --mark-done never ran) — check logs/dream-*.log",
        )
    ]
