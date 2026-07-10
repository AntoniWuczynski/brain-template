"""Read/write ``metadata/index.jsonl``. Append-mostly with atomic writes."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from collections.abc import Iterator

Status = Literal["processed", "partial", "manual_review", "skipped"]


@dataclass(frozen=True)
class IndexRecord:
    """One line in ``metadata/index.jsonl``.

    ``relative_path`` is repo-root-relative. ``source_hash`` keys the
    record against re-ingestion: identical hash means same content.
    """

    relative_path: str
    source_hash: str
    size_bytes: int
    extension: str
    extractor: str
    status: Status
    raw_path: str
    processed_path: str | None
    index_note_path: str | None
    assets: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    # LLM-generated, faithful to the extracted content. Empty string / list
    # when no summary was produced (no API key, opted out, or call failed).
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    # Canonical topic tags this document covers. Used by the concept-note
    # generator to build cross-source links under ``knowledge/concepts/``.
    topics: list[str] = field(default_factory=list)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


_RECORD_FIELDS: frozenset[str] = frozenset(IndexRecord.__dataclass_fields__)


def iter_records(jsonl_path: Path) -> Iterator[IndexRecord]:
    if not jsonl_path.exists():
        return
    # errors="replace": a tail torn mid-UTF-8 (see append_record) must not
    # crash the read — the mojibaked line fails json.loads below and is
    # skipped like any other malformed line, keeping the valid records.
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines but don't lose the others.
                continue
            if not isinstance(data, dict):
                continue
            # Drop unknown keys rather than crashing: the schema has grown
            # twice already (summary/key_points, then topics), and a single
            # record written by a newer/older tool version — or hand-edited
            # with a stray key — must not abort the whole run with a
            # TypeError. Missing keys still fall back to dataclass defaults.
            known = {k: v for k, v in data.items() if k in _RECORD_FIELDS}
            try:
                yield IndexRecord(**known)
            except TypeError:
                # e.g. a required field is absent: skip this line, keep the rest.
                continue


def latest_records_by_path(jsonl_path: Path) -> dict[str, IndexRecord]:
    """Return the *latest* record per relative_path. Later lines win."""
    out: dict[str, IndexRecord] = {}
    for rec in iter_records(jsonl_path):
        out[rec.relative_path] = rec
    return out


def append_record(jsonl_path: Path, record: IndexRecord) -> None:
    """Append a record to the JSONL. Creates the file if missing.

    Uses a tempfile + os.replace dance only when the file doesn't yet
    exist; subsequent writes append with fsync. Records routinely exceed
    the PIPE_BUF (4 KiB) single-write atomicity window — asset-heavy
    MinerU records reach hundreds of KB — so before appending we ensure
    the file ends in a newline: if a previous write was torn (crash /
    concurrent run left a partial line), this starts the new record on its
    own line instead of concatenating onto the stub, so at most one record
    is lost to a torn tail rather than two silently merging.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line = record.to_json_line() + "\n"
    if not jsonl_path.exists():
        # Atomic create: write to temp and rename.
        fd, tmp = tempfile.mkstemp(prefix=".index-", suffix=".jsonl", dir=str(jsonl_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, jsonl_path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
    else:
        # Self-heal a torn tail: if the file doesn't end in '\n', add one
        # before appending so a partial prior line can't swallow this one.
        # Probe the last byte in BINARY — a text-mode read of a tail cut
        # mid-UTF-8 (index.jsonl uses ensure_ascii=False) would raise
        # UnicodeDecodeError before any write, crashing every retry.
        with jsonl_path.open("rb") as probe:
            probe.seek(0, os.SEEK_END)
            needs_nl = probe.tell() > 0
            if needs_nl:
                probe.seek(-1, os.SEEK_END)
                needs_nl = probe.read(1) != b"\n"
        with jsonl_path.open("a", encoding="utf-8") as fh:
            if needs_nl:
                fh.write("\n")
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
