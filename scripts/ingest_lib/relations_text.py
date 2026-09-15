"""Pure text -> text editing over relation frontmatter and log sections.

Split out of :mod:`ingest_lib.relations` (AUD-121): the module docstring
there still owns the full picture. This slice is the "Pure text editing"
bullet — :func:`upsert_relation_in_text` and :func:`append_fact_to_log`
are ``text -> text`` so the MCP entity tools and the consolidation pass
(separate stages) can apply them to any note body without touching the
filesystem here. History is never edited or deleted — closing an entry
and adding a new one is the only mutation (supersede, don't delete). Also
carries :func:`query_relations`, the structured read over the same data.

Frontmatter editing strategy: parse with ``notes._split_frontmatter`` to
read and mutate the relation entries, then splice ONLY the ``relations:``
block back into the fence textually (the same line surgery
``mcp_server/provenance.py`` does for its own keys). Every other key keeps
its source bytes — quote style, key order, comments, and timestamps in the
vault's ``…Z`` form — and so does the body below the fence. A whole-fence
``yaml.safe_dump`` round-trip would rewrite all of that on every relation
write.

Depends on :mod:`ingest_lib.relations_identity` and
:mod:`ingest_lib.relations_model`; must not import
:mod:`ingest_lib.relations` (that would be a cycle back to the module this
was split out of).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import yaml

from .config import VaultPaths
from .notes import (  # private helpers, but module-internal
    _split_frontmatter,
    _split_frontmatter_raw,
)
from .relations_identity import normalize_target
from .relations_model import Relation, _coerce_str, _LOG, entity_notes

_LOG_HEADING = "## Log"
_HEADING_RE = re.compile(r"^#{1,6}\s")
# Blank sections carry this placeholder (AGENTS.md: "leave the heading and
# write _(empty)_"); the first real bullet replaces it.
_EMPTY_PLACEHOLDER = "_(empty)_"
# A Markdown code fence: up to three leading spaces, then 3+ backticks or
# tildes. Lines inside one are content, not document structure.
_CODE_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _relation_entry(relation: Relation) -> dict[str, str]:
    """Serialisable frontmatter entry. Optional keys are emitted only when
    set, matching the frontmatter contract (absent == currently valid)."""
    entry: dict[str, str] = {"rel": relation.rel, "target": relation.target}
    if relation.valid_from:
        entry["valid_from"] = relation.valid_from
    if relation.valid_until:
        entry["valid_until"] = relation.valid_until
    if relation.source:
        entry["source"] = relation.source
    return entry


def _entry_matches(entry: dict[str, object], relation: Relation) -> bool:
    return (
        _coerce_str(entry.get("rel")) == relation.rel
        and normalize_target(_coerce_str(entry.get("target"))) == relation.target
    )


# Top-level ``relations:`` key line. Quote-tolerant, like provenance.py's
# strip matcher: ``"relations":`` parses as the same key, and a splice that
# missed it would append a SECOND relations key and lose every entry.
_RELATIONS_KEY_RE = re.compile(r"""^["']?relations["']?\s*:""")


def _dump_relations(entries: list[object]) -> str:
    """The ``relations:`` block on its own, as block-style YAML.

    ``width`` is effectively unbounded: a folded long ``source:`` or
    ``target:`` round-trips fine but churns the file and defeats the
    line-oriented tools (grep, provenance's line surgery) that read it.
    """
    dumped = yaml.safe_dump(
        {"relations": entries},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=2**31 - 1,
    )
    # PyYAML ships no stubs, so mypy sees Any; without ``stream=`` it always
    # returns str. Narrowed here the way ``notes._yaml_dump_str`` does (that
    # helper only takes bool kwargs, so it can't carry ``width``).
    if not isinstance(dumped, str):
        raise TypeError(f"yaml.safe_dump did not return str: {type(dumped)!r}")
    return dumped


def _splice_relations(text: str, entries: list[object]) -> str:
    """``text`` with only its ``relations:`` block rewritten (F10).

    Every other frontmatter line and the whole body keep their source
    bytes; a note with no fence at all gets a fresh one carrying just the
    relations block, and a note whose fence has no ``relations:`` key yet
    gets the block appended after the existing keys.
    """
    block = _dump_relations(entries)
    raw = _split_frontmatter_raw(text)
    if raw is None:
        return f"---\n{block}---\n{text}"
    yaml_block, body = raw
    fm_lines = yaml_block.splitlines(keepends=True)
    all_lines = text.splitlines(keepends=True)
    open_fence, close_fence = all_lines[0], all_lines[1 + len(fm_lines)]

    start = next(
        (i for i, line in enumerate(fm_lines) if _RELATIONS_KEY_RE.match(line)),
        len(fm_lines),
    )
    # The block runs to the last indented line or column-0 sequence dash
    # after the key ("- rel:" at the key's own indent is legal YAML).
    # Blank lines neither extend it nor end it.
    end = start + 1
    for j in range(start + 1, len(fm_lines)):
        stripped = fm_lines[j].strip()
        if not stripped:
            continue
        if not (fm_lines[j][0].isspace() or stripped.startswith("-")):
            break
        end = j + 1
    return (
        open_fence
        + "".join(fm_lines[:start])
        + block
        + "".join(fm_lines[end:])
        + close_fence
        + body
    )


def _frontmatter_fence_error(text: str) -> str | None:
    """None if ``text`` has no ``---`` fence at all (nothing to check here);
    otherwise a diagnostic string if the fence fails to parse as a YAML
    mapping, else None for a fence that parses cleanly.

    Mirrors ``notes._split_frontmatter``'s own fence-extraction so the two
    stay in lockstep: this only ever reports on the same inputs for which
    that function falls back to its "no mapping" identity return.
    """
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return None
    lines = text.splitlines(keepends=True)
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return "no closing '---' fence found"
    yaml_block = "".join(lines[1:end])
    try:
        loaded = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as exc:
        return str(exc)
    if not isinstance(loaded, dict):
        return f"top-level frontmatter is a {type(loaded).__name__}, not a mapping"
    return None


_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _as_day(raw: str) -> date | None:
    """A relation bound (or ``as_of``) as a date, or None when it isn't one.

    Tolerant of the one non-canonical form YAML itself produces: an
    unquoted ``valid_from: 2025-03-01T00:00:00Z`` loads as a datetime whose
    ``str()`` is ``'2025-03-01 00:00:00+00:00'``, so the leading ten
    characters are still the day. Anything else — ``2025-1-1``, prose — is
    None: comparing those as raw text silently mis-answers ``as_of`` (F12).
    """
    head = raw.strip()[:10]
    if not _DAY_RE.match(head):
        return None
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


def _day_before(iso_date: str) -> str:
    """Best-effort ISO "day before" ``iso_date``, used to auto-close a
    superseded open span the moment a new one starts (see
    ``upsert_relation_in_text``). Falls back to ``iso_date`` unchanged when
    it isn't a parseable date — format validation is the sweep's job, not
    this module's (mirrors the trade-off in ``Relation``'s docstring)."""
    try:
        return (date.fromisoformat(iso_date) - timedelta(days=1)).isoformat()
    except ValueError:
        return iso_date


def upsert_relation_in_text(text: str, relation: Relation) -> tuple[str, str]:
    """Apply one relation to a note's frontmatter. Pure: text -> (text, action).

    Semantics (matching = same rel + same normalised target):

    - ``relation.valid_until`` set and an OPEN matching entry exists ->
      close that entry (set valid_until, keep its valid_from): the entry
      whose ``valid_from`` equals ``relation.valid_from``, or — when
      ``relation.valid_from`` is empty — the open entry with the latest
      ``valid_from``, never blindly the first open match. Result: ``"closed"``.
    - an identical entry already exists (same valid_from — both empty
      counts — and same open/closed state) -> ``"noop"``, text unchanged.
    - adding a new OPEN entry (``relation.valid_until`` empty) while an
      open entry already exists for the same (rel, target): that existing
      entry is superseded — its ``valid_until`` is set to the day before
      the new entry's ``valid_from`` — so two open spans for one edge
      never coexist. (One sentence, per the module's supersede-never-
      delete contract: the OLD span's end is inferred, never its start.)
    - otherwise append the relation as a new entry: ``"added"``.

    Existing entries — including malformed or unknown-rel ones — are never
    rewritten or removed; only a close/supersede sets one new key on one
    entry. A note with no ``---`` fence at all gets a fresh one prepended.
    A note whose fence IS present but does not parse as a YAML mapping is
    refused (raises ``ValueError``) instead of being silently buried under
    a second fence (F2) — every other key would otherwise drop out of the
    frontmatter with the tool reporting success.
    """
    fence_error = _frontmatter_fence_error(text)
    if fence_error is not None:
        raise ValueError(
            "frontmatter fence present but does not parse as a YAML "
            f"mapping: {fence_error}"
        )
    frontmatter, _body = _split_frontmatter(text)

    raw = frontmatter.get("relations")
    if isinstance(raw, list):
        entries: list[object] = raw
    elif raw in (None, "", []):
        entries = []
    else:
        # Truthy non-list (someone wrote a scalar): wrap it instead of
        # discarding — history is never deleted, even malformed history.
        entries = [raw]

    matches = [e for e in entries if isinstance(e, dict) and _entry_matches(e, relation)]

    if relation.valid_until:
        open_matches = [e for e in matches if not _coerce_str(e.get("valid_until"))]
        target_entry: dict[str, object] | None = None
        if open_matches:
            if relation.valid_from:
                # Close the open span the caller actually named, never
                # blindly the first (F3): the one whose own valid_from
                # equals relation.valid_from.
                target_entry = next(
                    (e for e in open_matches
                     if _coerce_str(e.get("valid_from")) == relation.valid_from),
                    None,
                )
            else:
                # No valid_from given ("this ended on Y", the natural close
                # form): with several open spans (legacy bad data), close
                # the one that started most recently, not entries[0].
                target_entry = max(
                    open_matches, key=lambda e: _coerce_str(e.get("valid_from"))
                )
        if target_entry is not None:
            target_entry["valid_until"] = relation.valid_until
            return _splice_relations(text, entries), "closed"
        for e in matches:
            if _coerce_str(e.get("valid_until")) != relation.valid_until:
                continue
            # A close carrying NO valid_from ("this ended on Y") is the
            # natural way to close: an empty relation.valid_from must match
            # the existing entry's own valid_from (whatever it is) so a
            # *retried* close is a noop, not a bogus duplicate. An explicit
            # different valid_from still distinguishes (falls through to
            # append), preserving the exact-match path.
            if not relation.valid_from or (
                _coerce_str(e.get("valid_from")) == relation.valid_from
            ):
                return text, "noop"   # this closed span already recorded
    else:
        for e in matches:
            if _coerce_str(e.get("valid_until")):
                continue   # ended span: a new open entry may coexist
            if _coerce_str(e.get("valid_from")) == relation.valid_from:
                return text, "noop"
        if relation.valid_from:
            # Adding a new OPEN span: any existing open entry for this
            # (rel, target) is superseded, never left open alongside it
            # (F3 — never two open spans for one edge). Close it the day
            # before this one starts.
            for e in matches:
                if _coerce_str(e.get("valid_until")):
                    continue
                e["valid_until"] = _day_before(relation.valid_from)

    entries.append(_relation_entry(relation))
    return _splice_relations(text, entries), "added"


@dataclass(frozen=True)
class RelationHit:
    """One relation entry matched by :func:`query_relations`, carrying the
    declaring entity (node id) plus the edge's fields and provenance."""

    entity: str          # node id of the note that declares the relation
    rel: str
    target: str
    valid_from: str
    valid_until: str
    source: str


def query_relations(
    paths: VaultPaths,
    *,
    rel: str | None = None,
    entity: str | None = None,
    target: str | None = None,
    as_of: str | None = None,
    include_closed: bool = False,
    limit: int = 50,
) -> list[RelationHit]:
    """Structured, time-aware query over the typed relation graph.

    Filters (all optional, ANDed):
    - ``rel``: relation name (closed vocabulary).
    - ``entity`` / ``target``: node ids (tolerant of the ``knowledge/`` and
      ``.md`` forms via ``normalize_target``); ``entity`` is the declaring
      note, ``target`` the pointed-at node (reverse lookup).
    - ``as_of`` (YYYY-MM-DD): keep only relations whose interval CONTAINS the
      date — the supersede-never-delete history made queryable ("where did X
      work last spring?"). Bounds are parsed to real dates before comparison,
      so a bound YAML handed back as ``'2025-03-01 00:00:00+00:00'`` still
      answers correctly; a relation carrying a bound that is no date at all
      is reported and excluded rather than compared as text (F12). Open
      bounds are treated as ±infinity. When ``as_of`` is given it decides
      membership; ``include_closed`` is ignored. ``ValueError`` if ``as_of``
      itself is not YYYY-MM-DD.
    - ``include_closed`` (no ``as_of``): include relations that have a
      ``valid_until`` (ended). Default returns only currently-open relations.

    Deterministic: pure read over ``entity_notes`` + ``parse_relations``.
    Results are sorted and capped at ``limit``.
    """
    ent_norm = normalize_target(entity) if entity else None
    tgt_norm = normalize_target(target) if target else None
    as_of_day = _as_day(as_of) if as_of else None
    if as_of and as_of_day is None:
        raise ValueError(f"as_of must be YYYY-MM-DD, got {as_of!r}")

    hits: list[RelationHit] = []
    for node_id, info in entity_notes(paths).items():
        if ent_norm is not None and node_id != ent_norm:
            continue
        for r in info.relations:
            if rel is not None and r.rel != rel:
                continue
            if tgt_norm is not None and r.target != tgt_norm:
                continue
            if as_of_day is not None:
                start = _as_day(r.valid_from) if r.valid_from else None
                end = _as_day(r.valid_until) if r.valid_until else None
                if (r.valid_from and start is None) or (r.valid_until and end is None):
                    _LOG.warning(
                        "relations: %s: %s -> %s has a bound that is not "
                        "YYYY-MM-DD ([%s..%s]) — excluded from the as_of query",
                        info.rel_path, r.rel, r.target,
                        r.valid_from or "open", r.valid_until or "open",
                    )
                    continue
                if start is not None and start > as_of_day:
                    continue
                if end is not None and end < as_of_day:
                    continue
            elif not include_closed and r.valid_until:
                continue  # ended relation, and no as_of asked for history
            hits.append(RelationHit(
                entity=node_id, rel=r.rel, target=r.target,
                valid_from=r.valid_from, valid_until=r.valid_until, source=r.source,
            ))
    hits.sort(key=lambda h: (h.entity, h.rel, h.target, h.valid_from, h.valid_until))
    return hits[: max(0, limit)]


def _code_fence_flags(lines: list[str]) -> list[bool]:
    """Per-line "this line belongs to a fenced code block" flags.

    The fence lines themselves count as inside, so neither the opener nor
    the closer can be mistaken for a heading or a bullet. A fence closes on
    a line of the same character, at least as long, with nothing after it;
    an unterminated fence runs to end of text, exactly as Markdown reads it.
    """
    flags: list[bool] = []
    marker = ""
    for line in lines:
        match = _CODE_FENCE_RE.match(line)
        if not marker:
            marker = match.group(1) if match is not None else ""
            flags.append(bool(marker))
            continue
        flags.append(True)
        if (
            match is not None
            and match.group(1)[0] == marker[0]
            and len(match.group(1)) >= len(marker)
            and not line.strip()[len(match.group(1)):]
        ):
            marker = ""
    return flags


def append_fact_to_log(text: str, fact_line: str) -> str:
    """Append ``- <fact_line>`` as the last bullet of the ``## Log``
    section; create the section at end of file if absent.

    Idempotent on an EXACT-duplicate bullet: if the rendered ``- <fact_line>``
    already appears verbatim as a line inside the existing ``## Log`` section,
    the text is returned unchanged. This is crash-/retry-safety — a
    consolidation pass that crashes mid-flight and reruns, or a retried
    ``entity_append_fact``, must not double-apply the same fact. Distinct
    fact lines (different date, text, or source) still both append. Pure:
    text -> text.

    Fenced code blocks are content, not structure (F11): a ``#`` comment or
    a ``## Log`` line inside one never ends or starts the section, so a
    bullet can't land inside a fence — ahead of the real last fact, which
    would break "newest LAST, never rewritten". The first bullet REPLACES a
    lone ``_(empty)_`` placeholder: a section with a fact in it is not empty.
    """
    bullet = f"- {fact_line}"

    lines = text.split("\n")
    in_fence = _code_fence_flags(lines)
    heading_idx = -1
    for i, line in enumerate(lines):
        if not in_fence[i] and line.strip() == _LOG_HEADING:
            heading_idx = i
            break

    if heading_idx < 0:
        base = text.rstrip("\n")
        if base:
            return f"{base}\n\n{_LOG_HEADING}\n\n{bullet}\n"
        return f"{_LOG_HEADING}\n\n{bullet}\n"

    # Section ends at the next heading (any level) or end of file — a '#'
    # line inside a fenced block is code, not a heading.
    end = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        if not in_fence[j] and _HEADING_RE.match(lines[j]):
            end = j
            break

    body_range = range(heading_idx + 1, end)

    # Idempotency: the exact bullet already recorded in THIS section is a
    # no-op (crash-safe consolidation reruns / retried appends). Compare with
    # a trailing '\r' stripped so a note that picked up CRLF endings (Windows
    # editor / some sync tools) doesn't defeat the dedup and double-apply.
    # A copy inside a code fence is example text, not a recorded fact.
    if any(
        not in_fence[j] and lines[j].rstrip("\r") == bullet for j in body_range
    ):
        return text

    # First real bullet: take the placeholder's place rather than sitting
    # under it, so the section never claims to be both empty and dated.
    if not any(
        not in_fence[j] and lines[j].lstrip().startswith("- ") for j in body_range
    ):
        for j in body_range:
            if not in_fence[j] and lines[j].strip() == _EMPTY_PLACEHOLDER:
                lines[j] = bullet
                return "\n".join(lines)

    # Insert after the section's last non-blank line; an empty section
    # gets a blank line between heading and first bullet.
    for j in range(end - 1, heading_idx, -1):
        if lines[j].strip():
            lines.insert(j + 1, bullet)
            break
    else:
        lines.insert(heading_idx + 1, "")
        lines.insert(heading_idx + 2, bullet)
    return "\n".join(lines)
