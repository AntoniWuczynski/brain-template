"""Generate processed Markdown + index notes; merge user-edited frontmatter."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Literal

import yaml

from .atomic import atomic_write_text

_LOGGER = logging.getLogger(__name__)

# Keys write_index_note regenerates or actively merges every run. Any other
# top-level key is the user's alone (AGENTS.md rule 9) and its exact source
# text is preserved verbatim (see `_top_level_raw_blocks`) rather than
# round-tripped through yaml.safe_load/safe_dump.
_PIPELINE_OWNED_KEYS = frozenset(
    {"title", "type", "source_file", "source_hash", "created", "updated", "status", "figures", "topics"}
)


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
    """Every note writer in the vault goes through here (see ``atomic``)."""
    atomic_write_text(path, content, prefix=".note-", suffix=".md")


def _yaml_dump_str(data: object, **kwargs: bool) -> str:
    """``yaml.safe_dump`` ships no type stubs, so mypy sees its return as
    ``Any``. Every call site here omits ``stream=``, so it always returns
    ``str`` at runtime (PyYAML docs) — narrow that once here instead of
    leaking Any through each caller."""
    dumped = yaml.safe_dump(data, **kwargs)
    if not isinstance(dumped, str):
        raise TypeError(f"yaml.safe_dump did not return str: {type(dumped)!r}")
    return dumped


def md_cell(text: str) -> str:
    """Markdown-table-safe cell text: a stray pipe or newline would break
    the row, so collapse whitespace and escape pipes."""
    return " ".join(text.split()).replace("|", "\\|")


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
    dumped = _yaml_dump_str(value, default_flow_style=True, allow_unicode=True).strip()
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
    return _yaml_dump_str(seq, default_flow_style=True, allow_unicode=True).strip()


def _split_frontmatter_raw(text: str) -> tuple[str, str] | None:
    """Return (yaml_block_text, body) if `text` opens with a fenced
    frontmatter block, regardless of whether the YAML inside parses.
    None if there is no such fence.

    A leading UTF-8 BOM is ignored: an editor that writes one would
    otherwise make the whole YAML block invisible here, so the note would
    silently lose its topics and read as frontmatter-less to the
    provenance stamper (audit AUD-045)."""
    text = text.lstrip("\ufeff")
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return None
    lines = text.splitlines(keepends=True)
    if not lines:
        return None
    # First line is "---"; scan from line 1 for the next "---".
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return None
    yaml_block = "".join(lines[1:end])
    body = "".join(lines[end + 1 :])
    return yaml_block, body


def _try_parse_frontmatter_block(yaml_block: str) -> dict[str, object] | None:
    """Parse one already-fenced-out frontmatter block as a YAML mapping.
    None means the block is present but doesn't parse as one (invalid YAML,
    or valid YAML that isn't a mapping) — the shared parse step behind both
    `_split_frontmatter` and `_parse_existing_index_frontmatter`."""
    try:
        loaded = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


FrontmatterState = Literal["absent", "unparseable", "ok"]


def frontmatter_state(text: str) -> FrontmatterState:
    """Classify `text`'s frontmatter without collapsing "no fence at all"
    and "fence present but unparseable" into the same outcome the way
    `_split_frontmatter` does (AUD-078): "absent", "unparseable", or "ok".

    `_split_frontmatter` stays lenient — its many read-only callers
    (concepts/dashboards/relations/sweep/dream/...) only ever look up known
    keys and are fine treating unreadable frontmatter as absent, and that
    behaviour is relied on elsewhere, so it is unchanged. A caller that must
    not silently read a broken note as frontmatter-less (e.g. a sweep
    finding, or a decision to skip a rebuild) should check this first.
    """
    raw = _split_frontmatter_raw(text)
    if raw is None:
        return "absent"
    yaml_block, _body = raw
    return "ok" if _try_parse_frontmatter_block(yaml_block) is not None else "unparseable"


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Return (frontmatter_dict, body). Empty dict if no frontmatter, or if
    the block fails to parse as a YAML mapping — lenient by design, for the
    many read-only callers (concepts/dashboards/relations/sweep/dream/...)
    that only ever look up specific known keys and are fine treating
    unreadable frontmatter as absent. `write_index_note`, which rewrites the
    file, needs a stricter distinction and uses `_parse_existing_index_frontmatter`
    instead; a caller that needs to tell "absent" from "unparseable" without
    that stricter dict-or-None shape can use `frontmatter_state`."""
    raw = _split_frontmatter_raw(text)
    if raw is None:
        return {}, text
    yaml_block, body = raw
    parsed = _try_parse_frontmatter_block(yaml_block)
    if parsed is None:
        return {}, text
    return parsed, body


_TOP_LEVEL_KEY_RE = re.compile(r"^(?:\"([^\"]*)\"|'([^']*)'|([^:\s][^:]*)):(?:\s|$)")


def _top_level_raw_blocks(yaml_block: str) -> dict[str, str]:
    """Map each top-level mapping key to its raw source text (its key line
    through the last line of its value, verbatim newlines included), so a
    value's exact spelling can be spliced back in unchanged instead of being
    round-tripped through yaml.safe_load/safe_dump — which is YAML 1.1 and
    silently rewrites e.g. ``yes`` -> ``true``, ``010`` -> ``8``."""
    lines = yaml_block.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not line.strip() or line[0] in " \t#":
            continue  # blank, a comment, or an indented continuation
        m = _TOP_LEVEL_KEY_RE.match(line)
        if not m:
            continue
        key = next(g for g in m.groups() if g is not None)
        starts.append((i, key))
    blocks: dict[str, str] = {}
    for idx, (line_no, key) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        blocks[key] = "".join(lines[line_no:end])
    return blocks


def _parse_existing_index_frontmatter(
    text: str,
) -> tuple[dict[str, object] | None, dict[str, str]]:
    """Parse an existing index note's frontmatter strictly, for the merge in
    `write_index_note`. Returns (frontmatter_dict, raw_top_level_blocks);
    frontmatter_dict is None if a frontmatter block is present but does not
    parse as a YAML mapping — distinct from "no block at all" (`{}`), which
    the caller must not treat as "no user keys to preserve"."""
    raw = _split_frontmatter_raw(text)
    if raw is None:
        return {}, {}
    yaml_block, _body = raw
    parsed = _try_parse_frontmatter_block(yaml_block)
    if parsed is None:
        return None, {}
    return parsed, _top_level_raw_blocks(yaml_block)


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


def _frontmatter_to_yaml(fm: dict[str, object], *, raw_blocks: dict[str, str] | None = None) -> str:
    """Serialize frontmatter, preserving key order (required keys first,
    then alphabetical). Any key present in `raw_blocks` is emitted using its
    original source text verbatim instead of being re-dumped from the
    parsed Python value — callers pass only user-owned keys here (see
    `_PIPELINE_OWNED_KEYS`), so a value the pipeline never touches keeps its
    exact user-written spelling (F6)."""
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
    ordered_keys: list[str] = [k for k in required if k in fm]
    ordered_keys += sorted(k for k in fm if k not in ordered_keys)
    raw_blocks = raw_blocks or {}
    parts: list[str] = []
    for k in ordered_keys:
        if k in raw_blocks:
            parts.append(raw_blocks[k])
            continue
        parts.append(
            _yaml_dump_str({k: fm[k]}, sort_keys=False, allow_unicode=True, default_flow_style=False)
        )
    return "".join(parts)


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
    """Write the Obsidian-friendly index note. Preserves user frontmatter on update.

    If the existing note's frontmatter block does not parse as a YAML
    mapping, the note is left untouched (logged, not rewritten): AGENTS.md
    rule 9 requires user-added keys survive re-ingest, and treating an
    unparseable block as empty (the previous behaviour) silently discarded
    them, which is the exact "invent/guess over marking for review" failure
    rule 6 forbids — so an unreadable block is a manual-review condition on
    the note, not a license to regenerate it from scratch.
    """
    existing_fm: dict[str, object] = {}
    existing_raw_blocks: dict[str, str] = {}
    if target.exists():
        # errors="replace": a hand-corrupted index note (invalid UTF-8) must
        # not abort the whole batch. The read only feeds
        # _parse_existing_index_frontmatter and the file is (conditionally)
        # fully rewritten below.
        existing_text = target.read_text(encoding="utf-8", errors="replace")
        parsed, existing_raw_blocks = _parse_existing_index_frontmatter(existing_text)
        if parsed is None:
            _LOGGER.warning(
                "index note has unparseable frontmatter; leaving it untouched: %s", target
            )
            return
        existing_fm = parsed

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

    raw_blocks = {
        k: v for k, v in existing_raw_blocks.items() if k in merged_fm and k not in _PIPELINE_OWNED_KEYS
    }
    yaml_block = _frontmatter_to_yaml(merged_fm, raw_blocks=raw_blocks)

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
        f"- Source: [[archive/raw/{_source_link_path(content.source_relative_path)}]]\n"
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


def sanitise_derived_name(name: str) -> str:
    """A source filename made safe to use as a DERIVED note/dir name.

    Strips surrounding whitespace from the stem and the extension. A derived
    name ending in a space can never be linked to: every wikilink parser
    (Obsidian, and this repo's own sweep) strips whitespace inside
    ``[[...]]``, so the link names a file that does not exist. Trailing-space
    stems are also illegal on Windows/NTFS, which makes the derived tree
    unportable.

    Interior whitespace is deliberately left alone — a double space inside a
    name breaks nothing, and collapsing it would rewrite names (and the links
    that reproduce them) for no defect. ``archive/raw`` is immutable, so this
    only ever applies to the derived side; the source keeps its own name."""
    stem, dot, ext = name.rpartition(".")
    if not stem:  # extensionless ("README ") or a dotfile (".env ")
        return name.strip()
    return f"{stem.strip()}{dot}{ext.strip()}"


def derived_note_relpath(source_relative_path: str) -> str:
    """Repo-relative path (under ``archive/processed`` or ``knowledge/index``)
    of a source's generated Markdown note.

    Keeps the source's OWN extension and appends ``.md``, so ``report.pdf`` ->
    ``report.pdf.md`` and ``report.docx`` -> ``report.docx.md`` never collide
    at ``report.md`` (which silently clobbered one source's note, index note
    and assets dir). Extensionless sources are unchanged (``README`` ->
    ``README.md``). The filename is sanitised (see
    ``sanitise_derived_name``); the directories above it are not, since they
    are shared with the source tree. The processed-note wikilink references
    this path WITH the ``.md`` so Obsidian still resolves the embed."""
    head, sep, name = source_relative_path.replace(os.sep, "/").rpartition("/")
    return f"{head}{sep}{sanitise_derived_name(name)}.md"


def derived_assets_dirname(source_relative_path: str) -> str:
    """Assets-dir name next to the processed note: keeps the extension too
    (``report.pdf`` -> ``report.pdf_assets``) so two same-stem sources don't
    share one assets dir, and sanitised the same way as the note name."""
    return sanitise_derived_name(Path(source_relative_path).name) + "_assets"


def _processed_link_path(source_relative_path: str) -> str:
    # Reference the processed note by its full name (incl. the trailing .md)
    # so an Obsidian embed of report.pdf.md resolves — a bare [[report.pdf]]
    # would be read as a literal .pdf file reference, not the .md twin.
    return derived_note_relpath(source_relative_path)


def _strip_extension(path_str: str) -> str:
    p = Path(path_str)
    return str(p.with_suffix("")).replace(os.sep, "/")


def _source_link_path(source_relative_path: str) -> str:
    """Wikilink body for the raw source, under ``archive/raw/``.

    Extensionless per the vault convention — except when dropping the
    extension leaves trailing whitespace. ``archive/raw`` is immutable, so a
    source named ``x .pdf`` keeps that name, and ``[[archive/raw/.../x ]]``
    is stripped back to ``x`` by every wikilink parser and resolves to
    nothing. Linking such a source by its FULL name survives the strip, so
    the one link that can actually resolve is the one we write."""
    stripped = _strip_extension(source_relative_path)
    if stripped != stripped.strip():
        return source_relative_path.replace(os.sep, "/")
    return stripped
