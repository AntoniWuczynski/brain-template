"""Deterministic meeting -> project matching (zero-LLM).

The meeting-promotion pass (:mod:`ingest_lib.meetings`) asks this module
which project a meeting belongs to, and files it only on strong evidence.
Anything it cannot settle is left unfiled for the dream pass
(:mod:`ingest_lib.dream_meetings`), which only ever PROPOSES a project.

Evidence, scored per project (live project overview notes only):

- a project name term in the meeting TITLE: 3
- a project name term in the meeting summary: 1
- each resolved attendee whose person note carries an open
  ``member_of``/``collaborator_on`` relation to the project: 2

Name terms are the project's aliases (always trusted: a human wrote them),
its full title, the parts of a compound title (``"Kern — Minimalist Phone"``
gives ``kern`` and ``minimalist phone``, ``"Home server (x)"`` gives
``home server`` and ``x``) and a hyphenated folder slug. A single-word term
taken from a title is dropped unless it is also an alias: ``Check`` or
``brain`` as a bare word would match half the vault. Terms are accent-folded
and matched on whole words.

A meeting is filed when the best score is at least 3 and beats the runner-up
by at least 2, so the user's own membership of every project (equal points
everywhere) never decides it. Ties and near-ties are ``ambiguous``.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .concepts import fold_ascii
from .relations import EntityInfo, normalize_target

TITLE_POINTS = 3
SUMMARY_POINTS = 1
MEMBER_POINTS = 2
FILE_THRESHOLD = 3
FILE_MARGIN = 2

_MEMBER_RELS = frozenset({"member_of", "collaborator_on"})
_NON_WORD_RE = re.compile(r"[^0-9a-z]+")
_TITLE_SPLIT_RE = re.compile(r"\s+[—–-]\s+|:\s+|\(|\)")

MatchStatus = Literal["filed", "ambiguous", "unmatched"]


@dataclass(frozen=True)
class ProjectProfile:
    node_id: str                 # projects/<slug>/<slug> (or projects/<slug>)
    terms: tuple[str, ...]       # folded name terms
    members: frozenset[str]      # live people/ node ids linked to the project


@dataclass(frozen=True)
class ProjectMatch:
    status: MatchStatus
    project_id: str                          # "" unless status == "filed"
    scores: tuple[tuple[str, int], ...]      # non-zero scores, best first


def fold(text: str) -> str:
    """Lowercase, accent-stripped, punctuation collapsed to single spaces."""
    return _NON_WORD_RE.sub(" ", fold_ascii(text).casefold()).strip()


def is_project_overview(node_id: str) -> bool:
    parts = node_id.split("/")
    return parts[0] == "projects" and (
        len(parts) == 2 or (len(parts) == 3 and parts[1] == parts[2])
    )


def _name_terms(info: EntityInfo) -> tuple[str, ...]:
    terms: set[str] = {t for a in info.aliases if (t := fold(a))}
    title_parts = [info.title, *_TITLE_SPLIT_RE.split(info.title)]
    for part in title_parts:
        term = fold(part)
        if " " in term:
            terms.add(term)
    slug = info.node_id.split("/")[-1]
    if "-" in slug:
        terms.add(fold(slug))
    return tuple(sorted(terms))


def project_profiles(entities: Mapping[str, EntityInfo]) -> tuple[ProjectProfile, ...]:
    """One profile per live project overview note, sorted by node id."""
    members: dict[str, set[str]] = {}
    for info in entities.values():
        if not info.node_id.startswith("people/") or info.superseded_by:
            continue
        for r in info.relations:
            if r.rel in _MEMBER_RELS and not r.valid_until:
                members.setdefault(normalize_target(r.target), set()).add(info.node_id)
    profiles: list[ProjectProfile] = []
    for node_id in sorted(entities):
        info = entities[node_id]
        if info.superseded_by or not is_project_overview(node_id):
            continue
        # A relation may name the short folder form (projects/<slug>).
        short = "/".join(node_id.split("/")[:2])
        linked = members.get(node_id, set()) | members.get(short, set())
        profiles.append(ProjectProfile(
            node_id=node_id, terms=_name_terms(info), members=frozenset(linked),
        ))
    return tuple(profiles)


def _mentions(folded_text: str, terms: Sequence[str]) -> bool:
    padded = f" {folded_text} "
    return any(f" {term} " in padded for term in terms)


def score_projects(
    profiles: Sequence[ProjectProfile],
    *,
    title: str,
    summary: str,
    attendee_ids: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    """Non-zero scores, best first (ties broken by node id)."""
    folded_title, folded_summary = fold(title), fold(summary)
    attendees = set(attendee_ids)
    scores: list[tuple[str, int]] = []
    for p in profiles:
        score = MEMBER_POINTS * len(attendees & p.members)
        if _mentions(folded_title, p.terms):
            score += TITLE_POINTS
        if _mentions(folded_summary, p.terms):
            score += SUMMARY_POINTS
        if score:
            scores.append((p.node_id, score))
    return tuple(sorted(scores, key=lambda s: (-s[1], s[0])))


def match_project(
    profiles: Sequence[ProjectProfile],
    *,
    title: str,
    summary: str,
    attendee_ids: Sequence[str],
) -> ProjectMatch:
    scores = score_projects(
        profiles, title=title, summary=summary, attendee_ids=attendee_ids
    )
    if not scores or scores[0][1] < FILE_THRESHOLD:
        return ProjectMatch(status="unmatched", project_id="", scores=scores)
    runner_up = scores[1][1] if len(scores) > 1 else 0
    if scores[0][1] - runner_up < FILE_MARGIN:
        return ProjectMatch(status="ambiguous", project_id="", scores=scores)
    return ProjectMatch(status="filed", project_id=scores[0][0], scores=scores)


def filed_project(frontmatter: Mapping[str, object], relations_targets: Sequence[str]) -> str:
    """The project a meeting note is already filed under, or "".

    ``project:`` wins; otherwise the first open ``related_to`` target that is
    a project (what an approved dream proposal leaves behind).
    """
    project = frontmatter.get("project")
    if isinstance(project, str) and project.strip():
        return project.strip()
    for target in relations_targets:
        if target.startswith("projects/"):
            return target
    return ""


__all__ = [
    "FILE_MARGIN",
    "FILE_THRESHOLD",
    "MatchStatus",
    "ProjectMatch",
    "ProjectProfile",
    "filed_project",
    "fold",
    "is_project_overview",
    "match_project",
    "project_profiles",
    "score_projects",
]
