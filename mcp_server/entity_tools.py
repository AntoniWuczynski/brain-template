"""Structured entity-memory write tools: relations, facts, meetings.

These are the typed counterparts to the free-text note verbs in
``mcp_server.tools``: instead of handing an agent a blank page, each tool
enforces the entity-memory contract from stage A —

- relations come from the closed ``RELATION_VOCAB`` and point at node ids
  (``knowledge/``-relative paths without extension, e.g.
  ``people/anna-kowalska``) whose notes must already exist;
- facts are single dated, source-linked lines appended under ``## Log``;
- meetings are one note plus an ``attended`` relation on every attendee,
  written all-or-nothing in a single commit.

The write mechanics (rate bucket, write lock, atomic write, commit,
async push, background reindex, audit) are reused from ``tools`` so every
write path in the server behaves identically.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

# Make the ingest_lib package importable. Same shim the CLI scripts use.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ingest_lib.concepts import slugify as _slugify  # type: ignore[import-not-found]  # noqa: E402
from ingest_lib.relations import (  # type: ignore[import-not-found]  # noqa: E402
    RELATION_VOCAB,
    Relation,
    append_fact_to_log,
    is_valid_node_id,
    node_id_for_note,
    normalize_target,
    note_path_for_node,
    query_relations,
    upsert_relation_in_text,
)

from . import tools as _tools
from .config import MAX_NOTE_BYTES, ServerConfig
from .git_ops import CommitOutcome
from .identity import current_agent
from .provenance import stamp_provenance
from .runtime import Runtime
# _resolve_inside_vault is private to safety, but module-internal reuse is
# the established pattern (relations imports notes._split_frontmatter the
# same way): existence checks on caller-supplied node ids MUST go through
# the traversal/symlink/control-character gauntlet, or a target like
# ``people/../../secret`` would probe paths outside the vault.
from .safety import _resolve_inside_vault, resolve_write_under_allowlist
from .tools import ToolError, WriteResult

# Cap on one fact line. Facts are meant to be distilled single sentences;
# anything longer belongs in a note body, not the log.
MAX_FACT_CHARS: int = 500


class EntityWriteResult(WriteResult):
    """WriteResult plus what the upsert actually did to the relation."""

    action: str  # "added" | "closed" | "noop"


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

def _validate_date(value: str, *, field: str) -> str:
    """Strict YYYY-MM-DD. The round-trip check rejects non-canonical forms
    strptime would tolerate (``2026-1-1``) — dates land in filenames and
    frontmatter, where one canonical spelling keeps everything sortable."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ToolError(f"{field} must be a YYYY-MM-DD date, got {value!r}") from None
    if parsed.strftime("%Y-%m-%d") != value:
        raise ToolError(f"{field} must be a canonical YYYY-MM-DD date, got {value!r}")
    return value


def _validate_source(vault_root: Path, source: str) -> None:
    """A relation/fact source is a vault-relative path: ``knowledge/...``
    without extension (its ``.md`` twin must exist) or ``archive/...``
    verbatim (raw/processed sources keep their real extension).

    The resolved path must stay UNDER the prefix the caller declared: a
    ``..`` that escapes it (``archive/../.env``) is reported identically to
    a missing file, so this can't become a whole-repo file-existence oracle
    (the read allowlist exists precisely to deny that enumeration)."""
    if source.startswith("knowledge/"):
        if source.endswith(".md"):
            raise ToolError(
                f"source must be the no-extension wikilink form, got {source!r} "
                "(drop the .md)"
            )
        rel = source + ".md"
        prefix = "knowledge/"
    elif source.startswith("archive/"):
        rel = source
        prefix = "archive/"
    else:
        raise ToolError(
            f"source must start with knowledge/ or archive/, got {source!r}"
        )
    resolved = _resolve_inside_vault(vault_root, rel)
    posix = resolved.relative_to(vault_root).as_posix()
    if not posix.startswith(prefix) or not resolved.is_file():
        raise ToolError(f"source does not exist: {source!r}")


def _resolve_existing_entity(cfg: ServerConfig, entity_path: str) -> tuple[Path, str]:
    """Resolve an entity note under the write allowlist; it must already
    exist — creating entities is vault_create_note's job, and these tools
    refusing to create keeps 'who made this note' unambiguous."""
    resolved = resolve_write_under_allowlist(cfg.vault_root, entity_path)
    rel_path = resolved.relative_to(cfg.vault_root).as_posix()
    if not rel_path.endswith(".md"):
        raise ToolError(f"entity_path must be a Markdown note (.md), got {entity_path!r}")
    # PROFILE.md is byte-budgeted and written only through profile_update;
    # the entity verbs (append_fact / upsert_relation) would otherwise grow
    # or mutate it past the budget, so refuse it here too.
    _tools._refuse_if_profile(rel_path)
    if not resolved.is_file():
        raise ToolError(
            f"entity note does not exist: {entity_path!r} — "
            "creating entities is vault_create_note's job; create it first"
        )
    return resolved, rel_path


def _require_target_note(vault_root: Path, target_id: str) -> None:
    target_rel = note_path_for_node(target_id)
    if not _resolve_inside_vault(vault_root, target_rel).is_file():
        raise ToolError(
            f"target note does not exist: {target_rel!r} — "
            "create the entity note first (vault_create_note), then add the relation"
        )


# ---------------------------------------------------------------------------
# relations_query (read-only)
# ---------------------------------------------------------------------------

class RelationHitOut(BaseModel):
    entity: str                 # node id declaring the relation
    rel: str
    target: str
    valid_from: str = ""
    valid_until: str = ""       # empty == currently open
    source: str = ""


class RelationsQueryOut(BaseModel):
    relations: list[RelationHitOut] = Field(default_factory=list)


def tool_relations_query(
    cfg: ServerConfig,
    runtime: Runtime,
    rel: str = "",
    entity: str = "",
    target: str = "",
    as_of: str = "",
    include_closed: bool = False,
    limit: int = 50,
) -> RelationsQueryOut:
    """Structured, time-aware query over the typed relation graph.

    All filters optional. ``as_of`` (YYYY-MM-DD) returns relations whose
    interval contains that date (the supersede history, queryable). Without
    ``as_of``, only currently-open relations unless ``include_closed``.
    """
    _tools._rate_check_read()
    if rel and rel not in RELATION_VOCAB:
        raise ToolError(
            f"unknown rel {rel!r}; the closed vocabulary is: "
            f"{', '.join(sorted(RELATION_VOCAB))}"
        )
    if as_of:
        _validate_date(as_of, field="as_of")
    if not 1 <= limit <= 500:
        raise ToolError("limit must be in [1, 500]")

    paths = _tools._paths_for_root(cfg.vault_root)
    hits = query_relations(
        paths,
        rel=rel or None,
        entity=entity or None,
        target=target or None,
        as_of=as_of or None,
        include_closed=include_closed,
        limit=limit,
    )
    runtime.audit.access_event(
        agent=current_agent(), tool="relations_query", paths=[],
        query=f"rel={rel} entity={entity} target={target} as_of={as_of}",
    )
    return RelationsQueryOut(
        relations=[
            RelationHitOut(
                entity=h.entity, rel=h.rel, target=h.target,
                valid_from=h.valid_from, valid_until=h.valid_until, source=h.source,
            )
            for h in hits
        ]
    )


# ---------------------------------------------------------------------------
# entity_upsert_relation
# ---------------------------------------------------------------------------

def tool_entity_upsert_relation(
    cfg: ServerConfig,
    runtime: Runtime,
    entity_path: str,
    rel: str,
    target: str,
    valid_from: str = "",
    valid_until: str = "",
    source: str = "",
) -> EntityWriteResult:
    """Add or close one typed relation in an entity note's frontmatter.

    Semantics come from ``relations.upsert_relation_in_text``: a set
    ``valid_until`` closes the matching open entry, an identical entry is
    a noop, anything else appends. History is superseded, never deleted.
    """
    def _do() -> EntityWriteResult:
        _tools._rate_check_write()
        agent = current_agent()
        if rel not in RELATION_VOCAB:
            raise ToolError(
                f"unknown rel {rel!r}; the closed vocabulary is: "
                f"{', '.join(sorted(RELATION_VOCAB))}"
            )
        resolved, rel_path = _resolve_existing_entity(cfg, entity_path)
        node = node_id_for_note(rel_path)

        target_id = normalize_target(target)
        if not is_valid_node_id(target_id):
            raise ToolError(
                f"target {target!r} is not a valid node id — want the "
                "knowledge/-relative path without extension and no '..' "
                "segments, e.g. 'people/anna-kowalska'"
            )
        _require_target_note(cfg.vault_root, target_id)
        for field, value in (("valid_from", valid_from), ("valid_until", valid_until)):
            if value:
                _validate_date(value, field=field)
        if source:
            _validate_source(cfg.vault_root, source)

        relation = Relation(
            rel=rel, target=target_id,
            valid_from=valid_from, valid_until=valid_until, source=source,
        )
        with _tools._write_lock:
            if not resolved.is_file():  # raced away since validation
                raise ToolError(f"entity note does not exist: {entity_path!r}")
            try:
                text = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise ToolError("entity note could not be read") from None
            new_text, action = upsert_relation_in_text(text, relation)
            if action == "noop":
                # Nothing to write, nothing to commit: tell the agent the
                # graph already says this instead of minting empty commits.
                return EntityWriteResult(
                    path=rel_path, action="noop", bytes_written=0,
                    committed=False, push_state="skipped", index_refresh="skipped",
                    warning="relation already present",
                )
            stamped = stamp_provenance(
                new_text, agent=agent, mode="replace",
                memory_area=_tools._is_memory_area(rel_path), prior=text,
            )
            _tools._atomic_write_text(resolved, stamped)
            outcome = _tools._commit(
                cfg, [resolved],
                _tools._commit_message(agent, f"relation {rel} {node} -> {target_id}"),
            )
        # Relations ARE graph input — always flag the rebuild.
        push_state, index_refresh = _tools._finish_write(
            runtime, rel=rel_path, outcome=outcome, graph_changed=True, reindex=True,
        )
        return EntityWriteResult(
            path=rel_path, action=action, bytes_written=len(stamped.encode()),
            commit_sha=outcome.sha, committed=outcome.committed, pushed=outcome.pushed,
            push_state=push_state, index_refresh=index_refresh,
            warning=None if outcome.committed else outcome.detail,
        )

    return _tools._audited_write(
        runtime, tool="entity_upsert_relation", path=entity_path, fn=_do
    )


# ---------------------------------------------------------------------------
# entity_append_fact
# ---------------------------------------------------------------------------

def tool_entity_append_fact(
    cfg: ServerConfig,
    runtime: Runtime,
    entity_path: str,
    text: str,
    source: str,
    date: str = "",
) -> WriteResult:
    """Append one dated, source-linked fact bullet to ``## Log``."""
    def _do() -> WriteResult:
        _tools._rate_check_write()
        agent = current_agent()
        if "\n" in text or "\r" in text:
            raise ToolError("text must be a single line — one fact per call")
        fact = text.strip()
        if not fact:
            raise ToolError("text must be a non-empty single line")
        if len(fact) > MAX_FACT_CHARS:
            raise ToolError(
                f"text is {len(fact)} characters; facts are capped at "
                f"{MAX_FACT_CHARS} — distil it or write a note instead"
            )
        if not source:
            raise ToolError(
                "source is required: the vault-relative no-extension path of "
                "the note the fact was learned from"
            )
        _validate_source(cfg.vault_root, source)
        if date:
            when = _validate_date(date, field="date")
        else:
            # The date is CONTENT — when the fact was learned — not a
            # generated timestamp. AGENTS.md's no-timestamps-in-bodies rule
            # exists so REBUILD paths stay deterministic (same inputs ->
            # same bytes); this is an append-only tool whose output is
            # never regenerated, so recording today's date IS the fact's
            # data, the same way a human would date a logbook line.
            when = datetime.now(timezone.utc).date().isoformat()

        resolved, rel_path = _resolve_existing_entity(cfg, entity_path)
        node = node_id_for_note(rel_path)
        fact_line = f"{when} — {fact} ([[{source}]])"
        with _tools._write_lock:
            if not resolved.is_file():  # raced away since validation
                raise ToolError(f"entity note does not exist: {entity_path!r}")
            try:
                existing = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise ToolError("entity note could not be read") from None
            combined = append_fact_to_log(existing, fact_line)
            stamped = stamp_provenance(
                combined, agent=agent, mode="append",
                memory_area=_tools._is_memory_area(rel_path), prior=existing,
            )
            if len(stamped.encode()) > MAX_NOTE_BYTES:
                raise ToolError(
                    f"note would exceed {MAX_NOTE_BYTES} bytes; "
                    "the log needs consolidating before more facts land"
                )
            _tools._atomic_write_text(resolved, stamped)
            outcome = _tools._commit(
                cfg, [resolved], _tools._commit_message(agent, f"fact -> {node}")
            )
        # A log bullet never touches frontmatter -> no graph rebuild; the
        # embed refresh still keeps memory_search fresh.
        push_state, index_refresh = _tools._finish_write(
            runtime, rel=rel_path, outcome=outcome, graph_changed=False, reindex=True,
        )
        return _tools._write_result(
            rel_path, len(fact_line.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _tools._audited_write(
        runtime, tool="entity_append_fact", path=entity_path, fn=_do
    )


# ---------------------------------------------------------------------------
# meeting_create
# ---------------------------------------------------------------------------

def _yaml_squote(value: str) -> str:
    """Single-quoted YAML scalar (the vault's frontmatter style); single
    quotes escape by doubling."""
    return "'" + value.replace("'", "''") + "'"


def _meeting_note_text(
    *,
    title: str,
    date: str,
    attendee_ids: list[str],
    project_id: str,
    body: str,
    now_iso: str,
) -> str:
    """Render the meeting note. Matches knowledge/index/templates/meeting.md:
    frontmatter holds node ids; body wikilinks are full vault-relative
    no-extension paths so they resolve from anywhere. Timestamps live in
    frontmatter only (created/updated), per AGENTS.md."""
    attendees_yaml = "[" + ", ".join(_yaml_squote(a) for a in attendee_ids) + "]"
    project_yaml = _yaml_squote(project_id) if project_id else "''"
    notes_section = body.strip() if body.strip() else "_(empty)_"
    links = [f"- Attendee: [[knowledge/{a}]]" for a in attendee_ids]
    if project_id:
        links.append(f"- Project: [[knowledge/{project_id}]]")
    return (
        "---\n"
        f"title: {_yaml_squote(title)}\n"
        "type: meeting\n"
        f"date: {_yaml_squote(date)}\n"
        f"attendees: {attendees_yaml}\n"
        f"project: {project_yaml}\n"
        "topics: []\n"
        f"created: {_yaml_squote(now_iso)}\n"
        f"updated: {_yaml_squote(now_iso)}\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        "## Agenda\n"
        "\n"
        "_(empty)_\n"
        "\n"
        "## Notes\n"
        "\n"
        f"{notes_section}\n"
        "\n"
        "## Decisions\n"
        "\n"
        "_(empty)_\n"
        "\n"
        "## Actions\n"
        "\n"
        "_(empty)_\n"
        "\n"
        "## Links\n"
        "\n" + "\n".join(links) + "\n"
    )


def _finish_multi_write(
    runtime: Runtime, *, rels: list[str], outcome: CommitOutcome
) -> tuple[str, str]:
    """``tools._finish_write`` for a multi-note commit: one push request,
    one reindex enqueue per written note. Every touched note changed its
    relations, so the whole batch is graph-relevant."""
    if not outcome.committed:
        return "skipped", "skipped"
    push_state = runtime.push_worker.request_push()
    index_refresh = "skipped"
    for rel in rels:
        index_refresh = runtime.refresher.enqueue(rel, graph_changed=True)
    return push_state, index_refresh


def tool_meeting_create(
    cfg: ServerConfig,
    runtime: Runtime,
    date: str,
    title: str,
    attendees: list[str],
    project: str = "",
    body: str = "",
) -> WriteResult:
    """Create one meeting note and record ``attended`` on every attendee.

    All-or-nothing: every input is validated and every new text computed
    BEFORE the first byte hits disk, then everything lands in one commit —
    a half-written meeting (note without attendee edges, or vice versa)
    would be a graph lie no later write repairs automatically.
    """
    def _do() -> WriteResult:
        _tools._rate_check_write()
        _tools._check_note_size(body)
        agent = current_agent()
        _validate_date(date, field="date")
        clean_title = title.strip()
        if not clean_title:
            raise ToolError("title must be non-empty")
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in clean_title):
            raise ToolError("title must not contain control characters")
        slug = _slugify(clean_title)
        if not slug:
            raise ToolError(
                f"title {title!r} slugifies to nothing — give it at least one "
                "alphanumeric character"
            )
        year = date[:4]
        node = f"meetings/{year}/{date}-{slug}"
        rel_path = f"knowledge/meetings/{year}/{date}-{slug}.md"
        resolved = resolve_write_under_allowlist(cfg.vault_root, rel_path)

        # --- validate EVERYTHING before writing anything ------------------
        attendee_ids: list[str] = []
        for raw in attendees:
            aid = normalize_target(raw)
            # is_valid_node_id blocks '..' traversal (people/../../archive/x)
            # BEFORE the people/ prefix check, so an attendee id can never
            # resolve to and rewrite a file outside knowledge/people/.
            if not is_valid_node_id(aid) or not aid.startswith("people/"):
                raise ToolError(
                    f"attendee {raw!r} is not a people/ node id "
                    "(want e.g. 'people/anna-kowalska')"
                )
            if aid not in attendee_ids:  # dedupe, keep caller order
                attendee_ids.append(aid)
        if not attendee_ids:
            raise ToolError("attendees must list at least one people/ node id")
        # Collect ALL missing attendee notes so the agent can create the
        # stubs in one pass instead of failing one-at-a-time.
        missing = [
            note_path_for_node(aid) for aid in attendee_ids
            if not _resolve_inside_vault(cfg.vault_root, note_path_for_node(aid)).is_file()
        ]
        if missing:
            raise ToolError(
                "missing attendee note(s): " + ", ".join(repr(m) for m in missing)
                + " — create the person stubs first (vault_create_note), then retry"
            )
        project_id = ""
        if project:
            project_id = normalize_target(project)
            if not is_valid_node_id(project_id) or not project_id.startswith("projects/"):
                raise ToolError(
                    f"project {project!r} is not a projects/ node id "
                    "(want e.g. 'projects/fyp')"
                )
            project_rel = note_path_for_node(project_id)
            if not _resolve_inside_vault(cfg.vault_root, project_rel).is_file():
                raise ToolError(
                    f"project note does not exist: {project_rel!r} — create it first"
                )

        # --- compute every new text -----------------------------------------
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meeting_text = _meeting_note_text(
            title=clean_title, date=date, attendee_ids=attendee_ids,
            project_id=project_id, body=body, now_iso=now_iso,
        )
        if project_id:
            # The meeting's own frontmatter carries the typed edge to the
            # project, so the graph sees it without parsing the body.
            meeting_text, _action = upsert_relation_in_text(
                meeting_text, Relation(rel="related_to", target=project_id)
            )
        meeting_text = stamp_provenance(
            meeting_text, agent=agent, mode="create",
            memory_area=_tools._is_memory_area(rel_path),
        )
        attended = Relation(
            rel="attended", target=node, valid_from=date, source=f"knowledge/{node}"
        )

        with _tools._write_lock:
            if resolved.exists():
                raise ToolError(f"meeting note already exists: {rel_path!r}")
            # Pass 1 — read & transform every attendee note. Any failure
            # here aborts with zero writes.
            pending: list[tuple[Path, str]] = []
            for aid in attendee_ids:
                note_abs = _resolve_inside_vault(cfg.vault_root, note_path_for_node(aid))
                if not note_abs.is_file():  # raced away since validation
                    raise ToolError(f"missing attendee note: {note_path_for_node(aid)!r}")
                try:
                    text = note_abs.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    raise ToolError(
                        f"attendee note could not be read: {note_path_for_node(aid)!r}"
                    ) from None
                new_text, action = upsert_relation_in_text(text, attended)
                if action == "noop":
                    continue  # already recorded (e.g. a retried call)
                pending.append((
                    note_abs,
                    stamp_provenance(new_text, agent=agent, mode="replace",
                                     memory_area=False, prior=text),
                ))
            # Pass 2 — all inputs validated and transformed; now write.
            resolved.parent.mkdir(parents=True, exist_ok=True)
            _tools._atomic_write_text(resolved, meeting_text)
            for note_abs, stamped in pending:
                _tools._atomic_write_text(note_abs, stamped)
            touched = [resolved, *(p for p, _text in pending)]
            outcome = _tools._commit(
                cfg, touched, _tools._commit_message(agent, f"meeting {date}-{slug}")
            )
        push_state, index_refresh = _finish_multi_write(
            runtime,
            rels=[p.relative_to(cfg.vault_root).as_posix() for p in touched],
            outcome=outcome,
        )
        return _tools._write_result(
            rel_path, len(meeting_text.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    # Best-effort audit label; validation failures still record the path
    # the call was aiming at.
    audit_path = f"knowledge/meetings/{date[:4]}/{date}-{_slugify(title)}.md"
    return _tools._audited_write(runtime, tool="meeting_create", path=audit_path, fn=_do)
