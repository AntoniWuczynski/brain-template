"""Decision-record write tools: ``decision_record`` and ``decision_resolve``.

A decision record lives at
``knowledge/{projects,chats}/<slug>/decisions/<NNNN>-<title-slug>.md`` and
its path, frontmatter and body shape are owned by ``ingest_lib.decisions``
(this module only validates, allocates numbers and writes). The general
verbs refuse that folder (``safety.refuse_protected_path``), so these two
tools are its only doors: an accepted decision never changes except by
being superseded, and a proposed one is settled exactly once.

The write mechanics (rate bucket, write lock, provenance, atomic write,
commit, async push, reindex, audit) are reused from ``tools`` and
``entity_tools``. Every write goes through
``entity_tools._write_all_or_nothing``: a supersede lands the new record and
the old one in one commit or neither, and a failed resolve leaves the record
proposed (restored and unstaged) so it can be retried. A re-render passes
the record's current text as ``prior``, so hand-added frontmatter keys,
preamble text and Links lines survive it.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, UTC
from pathlib import Path
from typing import Literal

# Importing ``.tools`` first runs its sys.path shim, which is what makes
# ``ingest_lib`` importable below.
from . import entity_tools as _entity_tools
from . import tools as _tools
from ingest_lib import paths_for_root as _paths_for_root
from ingest_lib.decisions import (
    DECISION_PATH_RE,
    PROJECT_ID_RE,
    DecisionNote,
    DecisionSections,
    decision_rel_path,
    iter_decisions,
    next_number,
    parse_decision,
    parse_sections,
    render_decision_note,
)
from ingest_lib.relations import normalize_target, note_path_for_node

from .config import MAX_NOTE_BYTES, ServerConfig
from .errors import ToolError
from .identity import current_agent
from .provenance import stamp_provenance
from .runtime import Runtime
from .safety import _resolve_inside_vault, resolve_write_under_allowlist
from .tools import WriteResult

MAX_TITLE_CHARS = 120


def _today_or(date: str | None) -> str:
    if date is None:
        return datetime.now(UTC).date().isoformat()
    return _entity_tools._validate_date(date, field="date")


def _render(note: DecisionNote, sections: DecisionSections, prior: str | None = None) -> str:
    try:
        text = render_decision_note(note, sections, prior=prior)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    if len(text.encode()) > MAX_NOTE_BYTES:
        raise ToolError(f"decision record would exceed {MAX_NOTE_BYTES} bytes")
    return text


def _validate_title(title: str) -> str:
    clean = title.strip()
    if not clean:
        raise ToolError("title must be non-empty")
    # splitlines() also breaks on U+2028/U+2029/U+0085, which parse_sections
    # would read as a new line of the body.
    if len(clean.splitlines()) != 1 or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in clean):
        raise ToolError("title must be a single line with no control characters")
    if len(clean) > MAX_TITLE_CHARS:
        raise ToolError(f"title is {len(clean)} characters; the cap is {MAX_TITLE_CHARS}")
    return clean


def _project_id(cfg: ServerConfig, project: str) -> str:
    """The project's node id (``projects/<slug>/<slug>``); its overview must
    exist. The short ``projects/<slug>`` form resolves to the overview."""
    project_id = normalize_target(project, vault_root=cfg.vault_root)
    if not PROJECT_ID_RE.match(project_id):
        raise ToolError(
            f"project {project!r} is not a project node id "
            "(want 'projects/<slug>/<slug>' or 'chats/<slug>/<slug>')"
        )
    overview = note_path_for_node(project_id)
    if not _resolve_inside_vault(cfg.vault_root, overview).is_file():
        raise ToolError(f"project overview does not exist: {overview!r} — create it first")
    return project_id


def tool_decision_record(
    cfg: ServerConfig,
    runtime: Runtime,
    project: str,
    title: str,
    context: str,
    decision: str,
    consequences: str = "",
    alternatives: str = "",
    status: Literal["accepted", "proposed"] = "accepted",
    supersedes: int | None = None,
    date: str | None = None,
) -> WriteResult:
    """Record decision NNNN (max existing + 1) in ``project``'s decisions/
    folder. With ``supersedes``, the accepted record of that number in the
    same project flips to ``superseded`` in the same commit."""
    def _do() -> WriteResult:
        _tools._rate_check_write()
        for text in (context, decision, consequences, alternatives):
            _tools._check_note_size(text)
        agent = current_agent()
        if supersedes is not None and status != "accepted":
            raise ToolError(
                "only an accepted decision supersedes another; record the "
                "proposal without supersedes, or record it accepted"
            )
        when = _today_or(date)
        clean_title = _validate_title(title)
        if not context.strip() or not decision.strip():
            raise ToolError("context and decision must be non-empty")
        project_id = _project_id(cfg, project)
        sections = DecisionSections(context, decision, consequences, alternatives)

        with _tools._write_lock:
            paths = _paths_for_root(cfg.vault_root)
            number = next_number(paths, project_id)
            try:
                rel_path = decision_rel_path(project_id, number, clean_title)
            except ValueError as exc:
                raise ToolError(str(exc)) from None
            resolved = resolve_write_under_allowlist(cfg.vault_root, rel_path)
            if resolved.exists():
                raise ToolError(f"decision record already exists: {rel_path!r}")

            writes: list[tuple[Path, str]] = []
            old: DecisionNote | None = None
            if supersedes is not None:
                matches = [
                    d for d in iter_decisions(paths)
                    if d.folder_project == project_id and d.file_number == supersedes
                ]
                if len(matches) != 1:
                    raise ToolError(
                        f"no single decision {supersedes:04d} in {project_id!r} to supersede"
                    )
                old = matches[0]
                if old.status != "accepted":
                    raise ToolError(
                        f"decision {old.rel_path!r} is {old.status or 'without a status'}; "
                        "only an accepted decision can be superseded"
                    )

            note = DecisionNote(
                rel_path=rel_path, project=project_id, number=number,
                title=clean_title, status=status, date=when,
                decided=when if status == "accepted" else None,
                supersedes=old.ref if old else None, superseded_by=None,
            )
            writes.append((resolved, stamp_provenance(
                _render(note, sections), agent=agent, mode="create", memory_area=False,
            )))
            if old is not None:
                old_abs = resolve_write_under_allowlist(cfg.vault_root, old.rel_path)
                try:
                    old_text = old_abs.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    raise ToolError(f"decision {old.rel_path!r} could not be read") from None
                flipped = dataclasses.replace(old, status="superseded", superseded_by=note.ref)
                writes.append((old_abs, stamp_provenance(
                    _render(flipped, parse_sections(old_text), old_text),
                    agent=agent, mode="replace", memory_area=False, prior=old_text,
                )))
            action = f"decision {rel_path}"
            if old is not None:
                action += f" (supersedes {old.file_number:04d})"
            outcome = _entity_tools._write_all_or_nothing(
                cfg, writes, _tools._commit_message(agent, action)
            )
        push_state, index_refresh = _entity_tools._finish_multi_write(
            runtime,
            rels=[p.relative_to(cfg.vault_root).as_posix() for p, _text in writes],
            outcome=outcome, graph_changed=False,
        )
        return _tools._write_result(
            rel_path, len(writes[0][1].encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    # No path label: the record's number is only allocated under the lock,
    # and an unlabelled write treats any standing pull conflict as its own
    # (sync.touches) — refusing beats guessing which record it would hit.
    return _tools._audited_write(runtime, tool="decision_record", path=None, fn=_do)


def tool_decision_resolve(
    cfg: ServerConfig,
    runtime: Runtime,
    path: str,
    status: Literal["accepted", "rejected"],
    decision: str | None = None,
    consequences: str | None = None,
    date: str | None = None,
) -> WriteResult:
    """Settle a ``proposed`` decision record: set ``status`` and
    ``decided``, optionally replacing its Decision / Consequences text."""
    def _do() -> WriteResult:
        _tools._rate_check_write()
        for text in (decision, consequences):
            if text is not None:
                _tools._check_note_size(text)
        agent = current_agent()
        if decision is not None and not decision.strip():
            raise ToolError("decision, when given, must be non-empty")
        when = _today_or(date)
        resolved = resolve_write_under_allowlist(cfg.vault_root, path)
        rel = resolved.relative_to(cfg.vault_root).as_posix()
        if not DECISION_PATH_RE.match(rel):
            raise ToolError(
                f"{path!r} is not a decision record "
                "(knowledge/{projects,chats}/<slug>/decisions/<NNNN>-<slug>.md)"
            )
        with _tools._write_lock:
            if not resolved.is_file():
                raise ToolError(f"decision record does not exist: {path!r}")
            try:
                text = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise ToolError("decision record could not be read") from None
            note = parse_decision(rel, text)
            if note.status != "proposed":
                raise ToolError(
                    f"{rel!r} is {note.status or 'without a status'}; only a proposed "
                    "decision can be resolved (an accepted one changes only by being "
                    "superseded through decision_record)"
                )
            sections = parse_sections(text)
            sections = dataclasses.replace(
                sections,
                decision=sections.decision if decision is None else decision,
                consequences=sections.consequences if consequences is None else consequences,
            )
            body = stamp_provenance(
                _render(dataclasses.replace(note, status=status, decided=when), sections, text),
                agent=agent, mode="replace", memory_area=False, prior=text,
            )
            outcome = _entity_tools._write_all_or_nothing(
                cfg, [(resolved, body)],
                _tools._commit_message(agent, f"resolve decision {rel} {status}"),
            )
        push_state, index_refresh = _tools._finish_write(
            runtime, rel=rel, outcome=outcome, graph_changed=False, reindex=True,
        )
        return _tools._write_result(
            rel, len(body.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _tools._audited_write(runtime, tool="decision_resolve", path=path, fn=_do)
