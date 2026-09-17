"""Dream-pass meeting filing: the LLM fallback for :mod:`ingest_lib.meeting_projects`.

The promotion pass files a meeting only when the rules single out one
project. Every other meeting note without a project reaches the dream packet
as ``unfiled_meetings`` together with a ``project_catalogue``; the session
picks a project where it is confident, and :func:`write_meeting_filings`
turns each pick into an ``approved: false`` fact note proposing
``related_to`` from the meeting to the project. ``scripts/consolidate.py``
applies it only once a human approves.

One proposal per meeting, ever: the proposal's filename is pinned to the
meeting alone (``identity=[]``), so a meeting with a proposal in the inbox,
or one a human already archived (rejected), never reaches the packet again
and a second pick for it is refused.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TypedDict

from .config import VaultPaths
from .meeting_projects import filed_project, is_project_overview, score_projects, project_profiles
from .notes import _split_frontmatter
from .propose import ProposeResult, propose_fact
from .relations import EntityInfo, Relation, entity_notes, normalize_target, parse_relations

MEETING_PROJECT_KIND = "dream-meeting-project"
MEETING_FILING_BUDGET = 10
_EXCERPT_CHARS = 800
_MARKER_RE = re.compile(r"<!--.*?-->")
_SPACE_RE = re.compile(r"\s+")


class ProjectCatalogueEntry(TypedDict):
    project_id: str
    title: str
    aliases: list[str]
    topics: list[str]


class RuleScore(TypedDict):
    project_id: str
    score: int


class UnfiledMeeting(TypedDict):
    node_id: str
    note_path: str
    title: str
    date: str
    attendees: list[str]
    excerpt: str
    rule_scores: list[RuleScore]


class MeetingFiling(TypedDict):
    """One pick the dream session hands back."""

    node_id: str
    project_id: str
    reason: str


class MeetingFilingOutcome(TypedDict):
    node_id: str
    project_id: str
    proposal_path: str
    action: str


def _str_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, str)]


def _read(paths: VaultPaths, rel_path: str) -> str:
    try:
        return (paths.root / rel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def project_catalogue(
    paths: VaultPaths, entities: dict[str, EntityInfo]
) -> list[ProjectCatalogueEntry]:
    out: list[ProjectCatalogueEntry] = []
    for node_id in sorted(entities):
        info = entities[node_id]
        if info.superseded_by or not is_project_overview(node_id):
            continue
        frontmatter, _body = _split_frontmatter(_read(paths, info.rel_path))
        out.append(ProjectCatalogueEntry(
            project_id=node_id, title=info.title,
            aliases=list(info.aliases), topics=_str_list(frontmatter.get("topics")),
        ))
    return out


def _propose(
    paths: VaultPaths, *, node_id: str, project_id: str, reason: str,
    now: datetime | None, dry_run: bool,
) -> ProposeResult:
    meeting_link = f"knowledge/{node_id}"
    return propose_fact(
        paths,
        kind=MEETING_PROJECT_KIND,
        title=f"project candidate: {node_id} -> {project_id}",
        target=node_id,
        source=meeting_link,
        reason=reason,
        relations=[Relation(rel="related_to", target=project_id, source=meeting_link)],
        identity=[],
        now=now,
        dry_run=dry_run,
    )


def _unfiled(paths: VaultPaths, info: EntityInfo) -> tuple[dict[str, object], str] | None:
    """(frontmatter, body) of a meeting note with no project yet, else None."""
    if info.superseded_by or info.type != "meeting" or not info.node_id.startswith("meetings/"):
        return None
    frontmatter, body = _split_frontmatter(_read(paths, info.rel_path))
    relations, _problems = parse_relations(frontmatter)
    if filed_project(frontmatter, [
        normalize_target(r.target) for r in relations
        if r.rel == "related_to" and not r.valid_until
    ]):
        return None
    return frontmatter, body


def unfiled_meetings(
    paths: VaultPaths,
    entities: dict[str, EntityInfo],
    *,
    budget: int = MEETING_FILING_BUDGET,
) -> list[UnfiledMeeting]:
    """Meeting notes with no project and no proposal yet, newest first."""
    profiles = project_profiles(entities)
    found: list[UnfiledMeeting] = []
    for node_id in sorted(entities):
        info = entities[node_id]
        unfiled = _unfiled(paths, info)
        if unfiled is None:
            continue
        pending = _propose(
            paths, node_id=node_id, project_id="", reason="", now=None, dry_run=True
        )
        if pending.action != "proposed":
            continue
        frontmatter, body = unfiled
        attendees = [normalize_target(a) for a in _str_list(frontmatter.get("attendees"))]
        text = _SPACE_RE.sub(" ", _MARKER_RE.sub(" ", body)).strip()
        found.append(UnfiledMeeting(
            node_id=node_id,
            note_path=info.rel_path,
            title=info.title,
            date=str(frontmatter.get("date") or ""),
            attendees=attendees,
            excerpt=text[:_EXCERPT_CHARS],
            rule_scores=[
                RuleScore(project_id=pid, score=score)
                for pid, score in score_projects(
                    profiles, title=info.title, summary=body, attendee_ids=attendees
                )
            ],
        ))
    found.sort(key=lambda m: (m["date"], m["node_id"]), reverse=True)
    return found[:budget]


def parse_meeting_filings(raw: object) -> list[MeetingFiling]:
    """Validate the session's picks. Strict: this is an LLM -> writer boundary."""
    if not isinstance(raw, list):
        raise ValueError("meeting filings must be a JSON array")
    out: list[MeetingFiling] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"filing {i}: expected a JSON object")
        values: dict[str, str] = {}
        for key in ("node_id", "project_id", "reason"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"filing {i}: {key!r} must be a non-empty string")
            values[key] = value.strip()
        out.append(MeetingFiling(
            node_id=values["node_id"], project_id=values["project_id"],
            reason=values["reason"],
        ))
    return out


def write_meeting_filings(
    paths: VaultPaths,
    filings: list[MeetingFiling],
    *,
    now: datetime,
    logger: logging.Logger,
) -> list[MeetingFilingOutcome]:
    """Propose each pick. Every pick is checked before any is written: the
    meeting must be an unfiled meeting note, the project a live project
    overview. A meeting that already has a proposal answers ``exists`` or
    ``archived`` and nothing new is written for it."""
    entities = entity_notes(paths)
    for f in filings:
        info = entities.get(f["node_id"])
        if info is None or _unfiled(paths, info) is None:
            raise ValueError(f"{f['node_id']!r} is not an unfiled meeting note")
        project = entities.get(f["project_id"])
        if project is None or project.superseded_by or not is_project_overview(f["project_id"]):
            raise ValueError(f"{f['project_id']!r} is not a live project overview note")
    outcomes: list[MeetingFilingOutcome] = []
    for f in filings:
        result = _propose(
            paths, node_id=f["node_id"], project_id=f["project_id"],
            reason=f["reason"], now=now, dry_run=False,
        )
        logger.info("dream meeting filing %s: %s", result.action, result.path)
        outcomes.append(MeetingFilingOutcome(
            node_id=f["node_id"], project_id=f["project_id"],
            proposal_path=result.path, action=result.action,
        ))
    return outcomes


__all__ = [
    "MEETING_FILING_BUDGET",
    "MEETING_PROJECT_KIND",
    "MeetingFiling",
    "MeetingFilingOutcome",
    "ProjectCatalogueEntry",
    "UnfiledMeeting",
    "parse_meeting_filings",
    "project_catalogue",
    "unfiled_meetings",
    "write_meeting_filings",
]
