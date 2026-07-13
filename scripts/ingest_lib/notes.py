"""Generate processed Markdown + index notes; merge user-edited frontmatter."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, UTC
from pathlib import Path

import yaml


@dataclass(frozen=True)
class NoteContent:
    """Fields the note generator needs to populate frontmatter and body."""

    title: str
    source_relative_path: str    # path under archive/raw, used for source_file + Source: link
    source_hash: str
    status: str                  # "processed" | "partial" | "manual_review"
    extracted_markdown: str      # body content (may be empty if manual_review)
    processing_notes: list[str]  # bullet points for the "Processing notes" section
    extractor: str               # informational; recorded in processing notes
    # Optional LLM-generated, faithful summary + key points + topics.
    summary: str = ""
    key_points: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    # Vault-relative paths to extracted figure images (for the index note's
    # `figures:` frontmatter — fast visual review in Obsidian).
    figures: tuple[str, ...] = ()


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".note-", suffix=".md", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def fm_scalar(value: object) -> str:
    """YAML-safe serialization of one frontmatter scalar for line assembly.

    Plain-safe strings stay UNQUOTED — byte-identical to the old
    ``f"{value}"`` interpolation for the common case, so notes that don't
    trigger an edge case are not rewritten (skip-unchanged holds). A string
    that needs quoting (contains ``: ``, leading ``[``/``@``/``{``…) gets a
    valid quoted form instead of unparseable YAML; a date/datetime renders
    as its ISO string instead of ``datetime.date(...)``.
    """
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    dumped = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    if dumped.endswith("..."):          # safe_dump appends a doc-end marker to bare scalars
        dumped = dumped[:-3].strip()
    return dumped


def fm_list(value: object) -> str:
    """YAML-safe inline list. A bare string coerces to a one-element list
    (rather than being dropped); a non-list/str becomes ``[]``."""
    if isinstance(value, list):
        seq: list[object] = value
    elif isinstance(value, str) and value.strip():
        seq = [value]
    else:
        seq = []
    return yaml.safe_dump(seq, default_flow_style=True, allow_unicode=True).strip()


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Return (frontmatter_dict, body). Empty dict if no frontmatter."""
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return {}, text
    # Find the closing fence.
    lines = text.splitlines(keepends=True)
    if not lines:
        return {}, text
    # First line is "---"; scan from line 1 for the next "---".
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text
    yaml_block = "".join(lines[1:end])
    body = "".join(lines[end + 1 :])
    try:
        loaded = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(loaded, dict):
        return {}, text
    return loaded, body


def _merge_frontmatter(
    existing: dict[str, object],
    *,
    generated: dict[str, object],
) -> dict[str, object]:
    """Generated keys are refreshed; everything else from `existing` is kept.

    `created` is preserved from `existing` if present (immutable once set).
    """
    merged: dict[str, object] = dict(existing)
    for k, v in generated.items():
        if k == "created" and existing.get("created"):
            continue
        merged[k] = v
    # Ensure required keys exist even if neither side provided them.
    required_lists: tuple[tuple[str, list[str]], ...] = (
        ("topics", []),
        ("aliases", []),
    )
    for k, default in required_lists:
        merged.setdefault(k, default)
    return merged


def _frontmatter_to_yaml(fm: dict[str, object]) -> str:
    # Stable key ordering for determinism: required keys first, then alphabetical.
    required = [
        "title",
        "type",
        "source_file",
        "source_hash",
        "created",
        "updated",
        "status",
        "topics",
        "aliases",
    ]
    ordered: dict[str, object] = {}
    for k in required:
        if k in fm:
            ordered[k] = fm[k]
    for k in sorted(fm):
        if k not in ordered:
            ordered[k] = fm[k]
    return yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, default_flow_style=False)


def write_processed_note(
    *,
    target: Path,
    content: NoteContent,
) -> None:
    """Write the long-form processed Markdown to ``archive/processed/...``.

    This file is regenerable; we don't try to merge user edits here.
    """
    body = content.extracted_markdown or "_(no content extracted)_\n"
    notes_block = "\n".join(f"- {n}" for n in content.processing_notes) or "- _(no notes)_"
    rendered = (
        f"# {content.title}\n\n"
        f"> Source: `{content.source_relative_path}`  \n"
        f"> Hash: `{content.source_hash}`  \n"
        f"> Extractor: `{content.extractor}`  \n"
        f"> Status: `{content.status}`\n\n"
        "---\n\n"
        f"{body}\n\n"
        "---\n\n"
        "## Processing notes\n\n"
        f"{notes_block}\n"
    )
    _atomic_write(target, rendered)


def write_index_note(
    *,
    target: Path,
    content: NoteContent,
) -> None:
    """Write the Obsidian-friendly index note. Preserves user frontmatter on update."""
    existing_fm: dict[str, object] = {}
    if target.exists():
        # errors="replace": a hand-corrupted index note (invalid UTF-8) must
        # not abort the whole batch. The read only feeds _split_frontmatter
        # (already YAML-error tolerant) and the file is fully rewritten below.
        existing_text = target.read_text(encoding="utf-8", errors="replace")
        existing_fm, _ = _split_frontmatter(existing_text)

    now_iso = _utc_now_iso()
    generated_fm: dict[str, object] = {
        "title": content.title,
        "type": "source_note",
        "source_file": content.source_relative_path,
        "source_hash": content.source_hash,
        "created": existing_fm.get("created") or now_iso,
        "updated": now_iso,
        "status": content.status,
    }
    # figures is a managed key: always set (refreshed each ingest) so a
    # re-extraction with different figures doesn't leave a stale list.
    if content.figures:
        generated_fm["figures"] = list(content.figures)
    merged_fm = _merge_frontmatter(existing_fm, generated=generated_fm)
    # Drop a stale figures list if this extraction produced none.
    if not content.figures:
        merged_fm.pop("figures", None)
    # Topics merge: take the union of auto-extracted and user-edited.
    if content.topics:
        raw_topics = merged_fm.get("topics")
        # Frontmatter is heterogeneous; a well-formed topics value is a list.
        existing_topics = (
            [str(t) for t in raw_topics] if isinstance(raw_topics, list) else []
        )
        merged_topics: list[str] = []
        seen: set[str] = set()
        for t in list(content.topics) + existing_topics:
            t = t.strip()
            if t and t not in seen:
                merged_topics.append(t)
                seen.add(t)
        merged_fm["topics"] = merged_topics

    yaml_block = _frontmatter_to_yaml(merged_fm)

    summary_block = _summary_block(content)
    key_points_block = _key_points_block(content)
    processed_link = _processed_link_path(content.source_relative_path)
    body = (
        "# Summary\n\n"
        f"{summary_block}\n\n"
        "# Key points\n\n"
        f"{key_points_block}\n\n"
        "# Extracted content\n\n"
        f"![[archive/processed/{processed_link}]]\n\n"
        "# Links\n\n"
        f"- Source: [[archive/raw/{_strip_extension(content.source_relative_path)}]]\n"
        f"- Processed Markdown: [[archive/processed/{processed_link}]]\n\n"
        "# Processing notes\n\n"
        f"{_processing_notes_block(content)}\n"
    )
    _atomic_write(target, f"---\n{yaml_block}---\n\n{body}")


def _summary_block(content: NoteContent) -> str:
    if content.summary:
        return content.summary
    if content.status == "manual_review":
        return "_(empty — extraction failed; see Processing notes)_"
    if content.status == "partial":
        return "_(extraction was incomplete; see Processing notes)_"
    return (
        "_(no auto-summary — configure an LLM provider (ANTHROPIC_API_KEY / "
        "OPENAI_API_KEY / GOOGLE_API_KEY / BRAIN_LOCAL_URL) to enable, or "
        "write one here.)_"
    )


def _key_points_block(content: NoteContent) -> str:
    if content.key_points:
        return "\n".join(f"- {kp}" for kp in content.key_points)
    return "- _(empty)_"


def _processing_notes_block(content: NoteContent) -> str:
    if not content.processing_notes:
        return f"- Extractor: `{content.extractor}`"
    bullets = "\n".join(f"- {n}" for n in content.processing_notes)
    return f"- Extractor: `{content.extractor}`\n{bullets}"


def derived_note_relpath(source_relative_path: str) -> str:
    """Repo-relative path (under ``archive/processed`` or ``knowledge/index``)
    of a source's generated Markdown note.

    Keeps the source's OWN extension and appends ``.md``, so ``report.pdf`` ->
    ``report.pdf.md`` and ``report.docx`` -> ``report.docx.md`` never collide
    at ``report.md`` (which silently clobbered one source's note, index note
    and assets dir). Extensionless sources are unchanged (``README`` ->
    ``README.md``). The processed-note wikilink references this path WITH the
    ``.md`` so Obsidian still resolves the embed."""
    return source_relative_path.replace(os.sep, "/") + ".md"


def derived_assets_dirname(source_relative_path: str) -> str:
    """Assets-dir name next to the processed note: keeps the extension too
    (``report.pdf`` -> ``report.pdf_assets``) so two same-stem sources don't
    share one assets dir."""
    return Path(source_relative_path).name + "_assets"


def _processed_link_path(source_relative_path: str) -> str:
    # Reference the processed note by its full name (incl. the trailing .md)
    # so an Obsidian embed of report.pdf.md resolves — a bare [[report.pdf]]
    # would be read as a literal .pdf file reference, not the .md twin.
    return derived_note_relpath(source_relative_path)


def _strip_extension(path_str: str) -> str:
    p = Path(path_str)
    return str(p.with_suffix("")).replace(os.sep, "/")
