"""Decision records: the one place their path, frontmatter and body shape live.

A decision record is identified BY PATH, never by ``type:`` (legacy
``type: decision`` curated notes outside ``decisions/`` are not records)::

    knowledge/{projects,chats}/<slug>/decisions/<NNNN>-<title-slug>.md

API (shared by the MCP decision tools, ``sweep`` and ``open_loops``):

- ``DECISION_PATH_RE``      vault-relative record path; groups 1-3 are the
                            folder kind, the project slug and the 4-digit
                            number.
- ``PROJECT_ID_RE``         a project node id, ``projects/<slug>/<slug>`` or
                            ``chats/<slug>/<slug>`` (the ``project:`` form).
- ``DecisionStatus`` / ``DECISION_STATUSES``  the closed status vocabulary.
- ``DecisionNote``          frozen frontmatter view of one record, plus the
                            path-derived ``ref`` / ``folder_project`` /
                            ``file_number`` properties.
- ``DecisionSections``      the four free-text body sections.
- ``parse_decision(rel_path, text)`` / ``parse_sections(text)``  read a
                            record (lenient: a missing or malformed key reads
                            as ``None`` / ``""``, never raises over content).
- ``iter_decisions(paths)`` every record in the vault, sorted by path.
- ``next_number(paths, project_id)``  max filename number in the project's
                            ``decisions/`` folder + 1 (1 when empty).
- ``decision_rel_path(project_id, number, title)``  where a new record goes.
- ``render_decision_note(note, sections, prior=None)``  the full note text
                            (frontmatter and body). Re-rendering a parsed
                            record is how a status change is written, so the
                            Links section is always derived from the
                            frontmatter; with ``prior`` (the record's current
                            text) the hand-added parts survive: unknown
                            frontmatter keys verbatim, text between the H1 and
                            the first section, and extra Links lines.
                            Provenance stamps are the MCP layer's job.

``supersedes`` / ``superseded_by`` hold the record's ``ref``: the
``knowledge/``-prefixed, extension-less path (the ``superseded_by`` form
consolidate writes); ``project:`` keeps the node-id form meeting notes use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from .concepts import slugify
from .config import VaultPaths
from .meetings import _yaml_squote  # the meeting notes' frontmatter style
from .notes import _split_frontmatter, _split_frontmatter_raw, _top_level_raw_blocks

DecisionStatus = Literal["proposed", "accepted", "rejected", "superseded"]
DECISION_STATUSES: tuple[DecisionStatus, ...] = (
    "proposed", "accepted", "rejected", "superseded",
)

DECISION_PATH_RE = re.compile(
    r"^knowledge/(projects|chats)/([a-z0-9-]+)/decisions/(\d{4})-[a-z0-9-]+\.md$"
)
PROJECT_ID_RE = re.compile(r"^(projects|chats)/([a-z0-9-]+)/\2$")
_FILE_RE = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")

_EMPTY = "_(empty)_"
_SECTION_TITLES = ("Context", "Decision", "Consequences", "Alternatives considered")
_HEADINGS = frozenset(f"## {t}" for t in (*_SECTION_TITLES, "Links"))
# The frontmatter keys and Links lines render_decision_note derives; a
# re-render with ``prior`` carries everything else through.
_OWNED_KEYS = frozenset((
    "title", "type", "project", "number", "status", "date", "decided",
    "supersedes", "superseded_by",
))
_DERIVED_LINKS = ("- Project: ", "- Supersedes: ", "- Superseded by: ")


@dataclass(frozen=True)
class DecisionNote:
    rel_path: str
    project: str
    number: int | None
    title: str
    status: DecisionStatus | None  # None: absent or outside the vocabulary
    date: str | None
    decided: str | None
    supersedes: str | None
    superseded_by: str | None

    @property
    def ref(self) -> str:
        """The value other records' supersedes/superseded_by point at."""
        return self.rel_path.removesuffix(".md")

    @property
    def folder_project(self) -> str:
        """The project node id the record's folder implies."""
        _knowledge, kind, slug, *_rest = self.rel_path.split("/")
        return f"{kind}/{slug}/{slug}"

    @property
    def file_number(self) -> int:
        return int(self.rel_path.rsplit("/", 1)[1][:4])


@dataclass(frozen=True)
class DecisionSections:
    context: str
    decision: str
    consequences: str = ""
    alternatives: str = ""


def _str(value: object) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _status(value: object) -> DecisionStatus | None:
    for status in DECISION_STATUSES:
        if value == status:
            return status
    return None


def parse_decision(rel_path: str, text: str) -> DecisionNote:
    """Frontmatter view of the record at ``rel_path`` (must match
    ``DECISION_PATH_RE``; ValueError otherwise)."""
    if not DECISION_PATH_RE.match(rel_path):
        raise ValueError(f"not a decision record path: {rel_path}")
    fm, _body = _split_frontmatter(text)
    number = fm.get("number")
    return DecisionNote(
        rel_path=rel_path,
        project=_str(fm.get("project")) or "",
        number=number if isinstance(number, int) and not isinstance(number, bool) else None,
        title=_str(fm.get("title")) or "",
        status=_status(fm.get("status")),
        date=_str(fm.get("date")),
        decided=_str(fm.get("decided")),
        supersedes=_str(fm.get("supersedes")),
        superseded_by=_str(fm.get("superseded_by")),
    )


def _body_parts(text: str) -> tuple[str, dict[str, list[str]]]:
    """The body text before the first record heading (minus the H1) and
    each record heading's lines."""
    _fm, body = _split_frontmatter(text)
    preamble: list[str] = []
    found: dict[str, list[str]] = {}
    current = preamble
    for line in body.splitlines():
        heading = line.rstrip()
        if heading in _HEADINGS:
            current = found.setdefault(heading, [])
        else:
            current.append(line)
    h1 = next((i for i, line in enumerate(preamble) if line.startswith("# ")), None)
    if h1 is not None:
        del preamble[h1]
    return "\n".join(preamble).strip(), found


def parse_sections(text: str) -> DecisionSections:
    """The four free-text sections of a rendered record (``""`` when a
    section is absent or holds the empty placeholder)."""
    _preamble, found = _body_parts(text)

    def section(title: str) -> str:
        content = "\n".join(found.get(f"## {title}", [])).strip()
        return "" if content == _EMPTY else content

    context, decision, consequences, alternatives = (section(t) for t in _SECTION_TITLES)
    return DecisionSections(context, decision, consequences, alternatives)


def iter_decisions(paths: VaultPaths) -> list[DecisionNote]:
    """Every decision record in the vault, sorted by path. Unreadable files
    are skipped."""
    notes: list[DecisionNote] = []
    for kind in ("projects", "chats"):
        for path in (paths.knowledge / kind).glob("*/decisions/*.md"):
            rel = path.relative_to(paths.root).as_posix()
            if not DECISION_PATH_RE.match(rel):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            notes.append(parse_decision(rel, text))
    return sorted(notes, key=lambda n: n.rel_path)


def _decisions_dir_rel(project_id: str) -> str:
    match = PROJECT_ID_RE.match(project_id)
    if match is None:
        raise ValueError(f"not a project node id: {project_id}")
    return f"knowledge/{match.group(1)}/{match.group(2)}/decisions"


def next_number(paths: VaultPaths, project_id: str) -> int:
    """The next free record number in ``project_id``'s ``decisions/`` folder,
    from the filenames (the record's identity is its path)."""
    folder = paths.root / _decisions_dir_rel(project_id)
    numbers = [
        int(m.group(1))
        for p in (folder.iterdir() if folder.is_dir() else ())
        if (m := _FILE_RE.match(p.name))
    ]
    return max(numbers, default=0) + 1


def decision_rel_path(project_id: str, number: int, title: str) -> str:
    """Vault-relative path for record ``number`` titled ``title``."""
    slug = slugify(title)
    if not slug:
        raise ValueError(f"title has no slug-able characters: {title!r}")
    return f"{_decisions_dir_rel(project_id)}/{number:04d}-{slug}.md"


def _carried(prior: str) -> tuple[str, str, list[str]]:
    """What a re-render keeps from ``prior``: its frontmatter blocks for
    keys this module does not own (verbatim), its preamble, and its Links
    lines that are not derived from the frontmatter."""
    raw = _split_frontmatter_raw(prior)
    blocks = _top_level_raw_blocks(raw[0]) if raw is not None else {}
    extra_fm = "".join(
        block if block.endswith("\n") else block + "\n"
        for key, block in blocks.items() if key not in _OWNED_KEYS
    )
    preamble, found = _body_parts(prior)
    links = "\n".join(
        line for line in found.get("## Links", []) if not line.startswith(_DERIVED_LINKS)
    ).strip("\n")
    return extra_fm, preamble, links.splitlines()


def render_decision_note(
    note: DecisionNote, sections: DecisionSections, *, prior: str | None = None
) -> str:
    """Full record text. ValueError when number, status or date is missing,
    the title is not a single line, or a section contains a line equal to
    one of the record's own headings (it would not parse back). ``prior``,
    the record's current text, carries its hand-added parts through."""
    if note.number is None or note.status is None or note.date is None:
        raise ValueError("a decision record needs a number, a status and a date")
    if len(note.title.splitlines()) != 1:
        raise ValueError("a decision record's title must be a single line")
    extra_fm, preamble, extra_links = _carried(prior) if prior is not None else ("", "", [])
    texts = zip(
        _SECTION_TITLES,
        (sections.context, sections.decision, sections.consequences, sections.alternatives),
        strict=True,
    )
    body = [f"# {note.number:04d} — {note.title}", ""]
    if preamble:
        body += [preamble, ""]
    for title, content in texts:
        if any(line.rstrip() in _HEADINGS for line in content.splitlines()):
            raise ValueError(f"{title} must not contain a decision-record heading line")
        body += [f"## {title}", "", content.strip() or _EMPTY, ""]

    fm = [
        f"title: {_yaml_squote(note.title)}",
        "type: decision",
        f"project: {_yaml_squote(note.project)}",
        f"number: {note.number}",
        f"status: {note.status}",
        f"date: {_yaml_squote(note.date)}",
    ]
    links = [f"- Project: [[knowledge/{note.project}]]"]
    if note.decided:
        fm.append(f"decided: {_yaml_squote(note.decided)}")
    if note.supersedes:
        fm.append(f"supersedes: {_yaml_squote(note.supersedes)}")
        links.append(f"- Supersedes: [[{note.supersedes}]]")
    if note.superseded_by:
        fm.append(f"superseded_by: {_yaml_squote(note.superseded_by)}")
        links.append(f"- Superseded by: [[{note.superseded_by}]]")
    body += ["## Links", "", *links, *extra_links]
    return "---\n" + "\n".join(fm) + "\n" + extra_fm + "---\n\n" + "\n".join(body) + "\n"


__all__ = [
    "DECISION_PATH_RE",
    "DECISION_STATUSES",
    "PROJECT_ID_RE",
    "DecisionNote",
    "DecisionSections",
    "DecisionStatus",
    "decision_rel_path",
    "iter_decisions",
    "next_number",
    "parse_decision",
    "parse_sections",
    "render_decision_note",
]
