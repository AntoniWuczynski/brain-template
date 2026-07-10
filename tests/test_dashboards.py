"""Entity dashboards under knowledge/index/entities/: deterministic tables
over relations.entity_notes — open relations only, meetings most-recent
first, empty groups write nothing, skip-unchanged rebuilds, and user
tails preserved across rewrites (mirroring the concept-note contract)."""
from __future__ import annotations

import logging
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.dashboards import rebuild_dashboards

_LOG = logging.getLogger("test")

PERSON = """---
title: Anna Kowalska
type: person
updated: '2026-06-01T00:00:00Z'
relations:
  - rel: works_at
    target: organisations/acme
    valid_from: "2025-03-01"
  - rel: works_at
    target: organisations/old-corp
    valid_from: "2020-01-01"
    valid_until: "2024-12-31"
---

# Anna Kowalska
"""

ORG = """---
title: ACME
type: organisation
---

# ACME
"""

PROJECT = """---
title: Kern
type: project
status: active
topics: [search, embeddings]
updated: '2026-05-01T00:00:00Z'
---

# Kern
"""

MEETING_JUNE = """---
title: Kern call
type: meeting
date: "2026-06-12"
attendees: [people/anna-kowalska]
project: projects/kern
---

# Kern call
"""

MEETING_JAN = """---
title: Kickoff
type: meeting
date: "2026-01-05"
attendees: []
project: ""
---

# Kickoff
"""


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _seed(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(tmp_path, "knowledge/people/anna-kowalska.md", PERSON)
    _write(tmp_path, "knowledge/organisations/acme.md", ORG)
    _write(tmp_path, "knowledge/projects/kern.md", PROJECT)
    _write(tmp_path, "knowledge/meetings/2026/2026-06-12-kern-call.md", MEETING_JUNE)
    _write(tmp_path, "knowledge/meetings/2026/2026-01-05-kickoff.md", MEETING_JAN)
    return paths


def _dashboard(paths: VaultPaths, name: str) -> str:
    return (paths.knowledge_index / "entities" / name).read_text(encoding="utf-8")


def test_people_dashboard_shows_only_open_relations(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    stats = rebuild_dashboards(paths, logger=_LOG)

    assert stats.written == 4
    assert stats.written_paths == (
        "knowledge/index/entities/meetings.md",
        "knowledge/index/entities/organisations.md",
        "knowledge/index/entities/people.md",
        "knowledge/index/entities/projects.md",
    )
    people = _dashboard(paths, "people.md")
    assert "type: dashboard" in people
    assert "count: 1" in people
    assert "| [[knowledge/people/anna-kowalska]] " in people
    assert "works_at -> [[knowledge/organisations/acme]]" in people
    # The closed old-corp span must not appear in "Current relations".
    assert "old-corp" not in people
    assert "| 2026-06-01T00:00:00Z |" in people


def test_org_without_relations_renders_placeholders(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    rebuild_dashboards(paths, logger=_LOG)

    orgs = _dashboard(paths, "organisations.md")
    # No open relations and no updated: timestamp -> em-dash cells.
    assert "| [[knowledge/organisations/acme]] | — | — |" in orgs


def test_project_dashboard_columns(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    rebuild_dashboards(paths, logger=_LOG)

    projects = _dashboard(paths, "projects.md")
    assert "| Note | Status | Topics | Updated |" in projects
    assert (
        "| [[knowledge/projects/kern]] | active | search, embeddings "
        "| 2026-05-01T00:00:00Z |"
    ) in projects


def test_meetings_sorted_most_recent_first(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    rebuild_dashboards(paths, logger=_LOG)

    meetings = _dashboard(paths, "meetings.md")
    june = meetings.index("[[knowledge/meetings/2026/2026-06-12-kern-call]]")
    jan = meetings.index("[[knowledge/meetings/2026/2026-01-05-kickoff]]")
    assert june < jan, "most recent meeting must come first"
    assert "[[knowledge/people/anna-kowalska]]" in meetings
    assert "[[knowledge/projects/kern]]" in meetings
    # The project-less January meeting renders an em-dash, not [[knowledge/]].
    assert "| 2026-01-05 | — | — |" in meetings


def test_empty_group_writes_nothing(tmp_path: Path) -> None:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    _write(tmp_path, "knowledge/people/anna-kowalska.md", PERSON)
    _write(tmp_path, "knowledge/organisations/acme.md", ORG)

    stats = rebuild_dashboards(paths, logger=_LOG)

    assert stats.written == 2
    entities = paths.knowledge_index / "entities"
    assert (entities / "people.md").exists()
    assert (entities / "organisations.md").exists()
    assert not (entities / "projects.md").exists()
    assert not (entities / "meetings.md").exists()


def test_second_rebuild_is_byte_identical_noop(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    first = rebuild_dashboards(paths, logger=_LOG)
    assert first.written == 4
    snapshot = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted((paths.knowledge_index / "entities").glob("*.md"))
    }

    second = rebuild_dashboards(paths, logger=_LOG)

    assert second.written == 0
    assert second.unchanged == 4
    assert second.written_paths == ()
    after = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted((paths.knowledge_index / "entities").glob("*.md"))
    }
    # Byte-for-byte identical, including the `updated:` timestamps.
    assert after == snapshot


def test_user_tail_preserved_across_rewrite(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    rebuild_dashboards(paths, logger=_LOG)

    target = paths.knowledge_index / "entities" / "people.md"
    text = target.read_text(encoding="utf-8")
    edited = text.replace(
        "_(Your hand-written notes about these entities go here. "
        "Preserved across re-runs.)_",
        "My own notes about these people.",
    )
    assert edited != text
    target.write_text(edited, encoding="utf-8")

    # A new person changes the table, forcing a rewrite of people.md.
    _write(
        tmp_path,
        "knowledge/people/borys-nowak.md",
        "---\ntitle: Borys Nowak\ntype: person\n---\n\n# Borys Nowak\n",
    )
    stats = rebuild_dashboards(paths, logger=_LOG)

    assert "knowledge/index/entities/people.md" in stats.written_paths
    rewritten = target.read_text(encoding="utf-8")
    assert "[[knowledge/people/borys-nowak]]" in rewritten
    assert "My own notes about these people." in rewritten


def test_deleted_end_marker_refuses_rewrite_and_preserves_user_tail(tmp_path: Path) -> None:
    # Same data-loss guard as concept notes: a deleted END marker must make
    # the writer refuse, not silently clobber the hand-written tail.
    paths = _seed(tmp_path)
    rebuild_dashboards(paths, logger=_LOG)

    target = paths.knowledge_index / "entities" / "people.md"
    text = target.read_text(encoding="utf-8")
    sentinel = "# Irreplaceable\n\nHand-written dashboard notes.\n"
    text = text + "\n" + sentinel
    text = text.replace("<!-- AUTO-GENERATED-END -->\n", "")
    target.write_text(text, encoding="utf-8")

    # Force a table change so the writer would otherwise rewrite people.md.
    _write(
        tmp_path,
        "knowledge/people/borys-nowak.md",
        "---\ntitle: Borys Nowak\ntype: person\n---\n\n# Borys Nowak\n",
    )
    stats = rebuild_dashboards(paths, logger=_LOG)

    assert "knowledge/index/entities/people.md" not in stats.written_paths
    assert sentinel in target.read_text(encoding="utf-8")
