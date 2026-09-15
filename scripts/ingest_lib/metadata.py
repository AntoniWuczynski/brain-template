"""Read/write ``metadata/index.jsonl``. Append-mostly with atomic writes."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from collections.abc import Iterator

from .atomic import append_jsonl_line

_LOGGER = logging.getLogger(__name__)

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
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                # Skip malformed lines but don't lose the others — and say so:
                # a silently dropped record makes its source look un-ingested,
                # which costs a re-extraction, an LLM call and a duplicate
                # record with no trace anywhere.
                _LOGGER.warning(
                    "%s line %d: unparseable JSON (%s) — record skipped",
                    jsonl_path, line_no, exc,
                )
                continue
            if not isinstance(data, dict):
                _LOGGER.warning(
                    "%s line %d: not a JSON object — record skipped", jsonl_path, line_no
                )
                continue
            # Drop unknown keys rather than crashing: the schema has grown
            # twice already (summary/key_points, then topics), and a single
            # record written by a newer/older tool version — or hand-edited
            # with a stray key — must not abort the whole run with a
            # TypeError. Missing keys still fall back to dataclass defaults.
            known = {k: v for k, v in data.items() if k in _RECORD_FIELDS}
            try:
                yield IndexRecord(**known)
            except TypeError as exc:
                # e.g. a required field is absent: skip this line, keep the rest.
                _LOGGER.warning(
                    "%s line %d: does not match the record schema (%s) — record skipped",
                    jsonl_path, line_no, exc,
                )
                continue


def latest_records_by_path(jsonl_path: Path) -> dict[str, IndexRecord]:
    """Return the *latest* record per relative_path. Later lines win."""
    out: dict[str, IndexRecord] = {}
    for rec in iter_records(jsonl_path):
        out[rec.relative_path] = rec
    return out


def append_record(jsonl_path: Path, record: IndexRecord) -> None:
    """Append a record to the JSONL. Creates the file if missing.

    Delegates to ``atomic.append_jsonl_line``: atomic create when the file
    doesn't exist, otherwise an fsync'd append that first self-heals a torn
    tail. Records routinely exceed the PIPE_BUF (4 KiB) single-write
    atomicity window — asset-heavy MinerU records reach hundreds of KB — so
    a previous write can have been cut short.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl_line(jsonl_path, record.to_json_line() + "\n", prefix=".index-")
