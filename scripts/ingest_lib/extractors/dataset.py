"""Dataset extractors. We never dump rows — only schemas + a tiny preview."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .base import ExtractionResult

_PREVIEW_ROWS = 5


def extract(src: Path, _assets_dir: Path) -> ExtractionResult:
    """Handles CSV, TSV, JSONL by extension."""
    ext = src.suffix.lower()
    if ext in {".csv", ".tsv"}:
        return _extract_dsv(src, delimiter="," if ext == ".csv" else "\t")
    if ext == ".jsonl":
        return _extract_jsonl(src)
    return ExtractionResult(
        status="manual_review",
        extractor="dataset",
        markdown="",
        error=f"unrecognised dataset extension: {ext}",
    )


def extract_parquet_stub(src: Path, _assets_dir: Path) -> ExtractionResult:
    """Parquet support is intentionally a stub. Add pyarrow + implement when needed."""
    try:
        size = src.stat().st_size
    except OSError as exc:
        size = -1
        notes = [f"stat failed: {exc}"]
    else:
        notes = [f"file size: {size} bytes"]
    return ExtractionResult(
        status="manual_review",
        extractor="dataset-parquet",
        markdown="_(parquet support not implemented yet — install pyarrow and add an extractor)_\n",
        error="parquet extractor not implemented",
        notes=notes,
    )


def _extract_dsv(src: Path, *, delimiter: str) -> ExtractionResult:
    try:
        with src.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                return ExtractionResult(
                    status="processed",
                    extractor="dataset-dsv",
                    markdown="_(empty file)_\n",
                )
            preview: list[list[str]] = []
            row_count = 0
            for row in reader:
                row_count += 1
                if len(preview) < _PREVIEW_ROWS:
                    preview.append(row)
    except OSError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="dataset-dsv",
            markdown="",
            error=f"read failed: {exc}",
        )
    except csv.Error as exc:
        # e.g. a single field exceeding csv.field_size_limit (128 KiB) —
        # common in scraped/LLM datasets. Surface it as a clean review
        # rather than letting it escape as an 'extractor crashed'.
        return ExtractionResult(
            status="manual_review",
            extractor="dataset-dsv",
            markdown="",
            error=f"csv parse failed: {exc}",
        )

    md = [
        f"**Rows (excluding header):** {row_count}",
        f"**Columns:** {len(header)}",
        "",
        "## Schema",
        "",
        "| # | column |",
        "| --- | --- |",
    ]
    md.extend(f"| {i+1} | `{_clip_code(c)}` |" for i, c in enumerate(header))
    if preview:
        md.append("")
        md.append("## Preview (first 5 rows)")
        md.append("")
        md.append("| " + " | ".join(_clip(c) for c in header) + " |")
        md.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in preview:
            row = row + [""] * (len(header) - len(row))
            md.append("| " + " | ".join(_clip(c) for c in row[: len(header)]) + " |")
    return ExtractionResult(
        status="processed",
        extractor="dataset-dsv",
        markdown="\n".join(md) + "\n",
    )


def _extract_jsonl(src: Path) -> ExtractionResult:
    keys: dict[str, int] = {}
    row_count = 0
    skipped = 0
    sample: list[dict[str, object]] = []
    try:
        with src.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                row_count += 1
                if isinstance(obj, dict):
                    for k in obj:
                        keys[k] = keys.get(k, 0) + 1
                if len(sample) < _PREVIEW_ROWS and isinstance(obj, dict):
                    sample.append(obj)
    except OSError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="dataset-jsonl",
            markdown="",
            error=f"read failed: {exc}",
        )

    md = [
        f"**Records:** {row_count}",
        f"**Distinct top-level keys:** {len(keys)}",
        "",
        "## Schema (key → record-count)",
        "",
        "| key | count |",
        "| --- | --- |",
    ]
    for k in sorted(keys):
        md.append(f"| `{_clip_code(k)}` | {keys[k]} |")
    if sample:
        md.append("")
        md.append("## Preview (first 5 records)")
        md.append("")
        for obj in sample:
            md.append("```json")
            md.append(json.dumps(obj, ensure_ascii=False, sort_keys=True)[:500])
            md.append("```")
    # Honesty: unparseable lines are dropped from the schema. Say so, and
    # mark the note partial rather than silently claiming full extraction.
    notes: list[str] = []
    status = "processed"
    if skipped:
        notes.append(f"skipped {skipped} unparseable JSON line(s)")
        status = "partial"
    return ExtractionResult(
        status=status,
        extractor="dataset-jsonl",
        markdown="\n".join(md) + "\n",
        notes=notes,
    )


def _clip(s: str, n: int = 80) -> str:
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _clip_code(s: str, n: int = 80) -> str:
    """Clip a value destined for a Markdown ``code`` span in a table cell:
    strip backticks (which would break the span) and pipes/newlines (which
    would break the table)."""
    s = s.replace("`", "").replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
