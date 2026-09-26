"""Meeting -> project filing: the deterministic rules at promotion time
(ingest_lib.meeting_projects via ingest_lib.meetings) and the dream-pass
fallback that only ever proposes (ingest_lib.dream_meetings)."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.consolidate import consolidate
from ingest_lib.dream_meetings import (
    parse_meeting_filings,
    unfiled_meetings,
    write_meeting_filings,
)
from ingest_lib.meeting_projects import match_project, project_profiles
from ingest_lib.meetings import promote_meetings
from ingest_lib.metadata import IndexRecord, append_record
from ingest_lib.notes import _split_frontmatter
from ingest_lib.relations import entity_notes

_LOG = logging.getLogger("test")
_NOW = datetime(2026, 9, 17, 12, 0, 0)
_MEETING = "knowledge/meetings/2026/2026-09-16-weekly-sync.md"


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    dest = paths.root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")


def _project(paths: VaultPaths, slug: str, title: str, aliases: str = "[]") -> None:
    _write(paths, f"knowledge/projects/{slug}/{slug}.md",
           f"---\ntitle: \"{title}\"\ntype: project\naliases: {aliases}\n"
           "topics: [hosting]\n---\n\n# x\n")


def _person(paths: VaultPaths, slug: str, title: str, member_of: str = "") -> None:
    rels = (
        f"relations:\n- rel: member_of\n  target: {member_of}\n" if member_of else ""
    )
    _write(paths, f"knowledge/people/{slug}.md",
           f"---\ntitle: {title}\ntype: person\n{rels}---\n\n## Log\n")


def _snapshot(
    paths: VaultPaths, *, title: str, summary: str = "", attendees: list[str] | None = None,
) -> None:
    rel = "meetings/granola/2026-09-16-weekly-sync-abcd1234.json"
    raw = paths.archive_raw / rel
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({
        "connector": "granola", "id": "not_1", "title": title, "date": "2026-09-16",
        "attendees": attendees or [], "summary": summary, "transcript": "",
    }), encoding="utf-8")
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path=rel, source_hash="a" * 64, size_bytes=raw.stat().st_size,
        extension=".json", extractor="meeting", status="processed",
        raw_path=f"archive/raw/{rel}", processed_path=f"archive/processed/{rel}.md",
        index_note_path=f"knowledge/index/{rel}.md",
        created_at="2026-09-16T18:00:00Z", updated_at="2026-09-16T18:00:00Z",
    ))


def _base(paths: VaultPaths) -> None:
    _project(paths, "kern", "Kern — Minimalist Phone", "[kern, \"Kern Ltd\"]")
    _project(paths, "server", "Home server (wuczynskiserver0)")
    _project(paths, "check", "Check")
    _project(paths, "klinika-ambroziak-website", "Klinika Ambroziak website", "[strona]")


def _fm(paths: VaultPaths, rel: str) -> dict[str, object]:
    return _split_frontmatter((paths.root / rel).read_text(encoding="utf-8"))[0]


# ------------------------------------------------------------ the rules

@pytest.mark.parametrize(("title", "summary", "expected"), [
    ("Weekly Sync", "", ("unmatched", "")),
    ("Kern weekly", "", ("filed", "projects/kern/kern")),
    ("Home-server backup plan", "", ("filed", "projects/server/server")),
    ("Łódź call about Strona", "", ("filed", "projects/klinika-ambroziak-website/klinika-ambroziak-website")),
    # A bare title word that is no alias never counts.
    ("Check in with Sam", "", ("unmatched", "")),
    # Summary evidence alone is too weak to file.
    ("Weekly Sync", "We discussed kern pricing.", ("unmatched", "")),
    # Two projects named in the title: a tie.
    ("Kern and home server", "", ("ambiguous", "")),
])
def test_match_project_rules(
    tmp_path: Path, title: str, summary: str, expected: tuple[str, str],
) -> None:
    paths = _vault(tmp_path)
    _base(paths)
    profiles = project_profiles(entity_notes(paths))
    match = match_project(profiles, title=title, summary=summary, attendee_ids=[])
    assert (match.status, match.project_id) == expected


def test_membership_evidence_needs_a_margin(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _base(paths)
    _person(paths, "me", "Me", member_of="projects/kern")
    _person(paths, "brit", "Brit", member_of="projects/kern/kern")
    profiles = project_profiles(entity_notes(paths))
    # Two Kern members (short and full target forms both count): 4 vs 0.
    both = match_project(profiles, title="Sync", summary="", attendee_ids=["people/me", "people/brit"])
    assert (both.status, both.project_id) == ("filed", "projects/kern/kern")
    # One member plus a server mention in the summary: 2 vs 1, below threshold.
    one = match_project(profiles, title="Sync", summary="home server", attendee_ids=["people/me"])
    assert one.status == "unmatched"


# ------------------------------------------------------------ promotion

def test_promotion_files_a_clear_match_once(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _base(paths)
    _snapshot(paths, title="Kern weekly")
    report = promote_meetings(paths, now=_NOW)
    promo = report.promotions[0]
    assert (promo.project_filing, promo.project_id) == ("filed", "projects/kern/kern")
    rel = "knowledge/meetings/2026/2026-09-16-kern-weekly.md"
    fm = _fm(paths, rel)
    assert fm["project"] == "projects/kern/kern"
    assert fm["relations"] == [{
        "rel": "related_to", "target": "projects/kern/kern",
        "source": "archive/raw/meetings/granola/2026-09-16-weekly-sync-abcd1234",
    }]
    before = (paths.root / rel).read_bytes()
    again = promote_meetings(paths, now=datetime(2026, 9, 18))
    assert again.promotions[0].project_filing == "already"
    assert (paths.root / rel).read_bytes() == before


def test_dry_run_reports_the_filing_without_writing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _base(paths)
    _snapshot(paths, title="Kern weekly")
    report = promote_meetings(paths, dry_run=True, now=_NOW)
    assert report.promotions[0].project_filing == "filed"
    assert not (paths.knowledge / "meetings").exists()


def test_promotion_never_refiles_or_touches_foreign_notes(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _base(paths)
    _snapshot(paths, title="Weekly sync")
    promote_meetings(paths, now=_NOW)
    note = paths.root / _MEETING
    # A human files it by hand under another project: the rules never override.
    note.write_text(note.read_text(encoding="utf-8").replace(
        "project: ''", "project: 'projects/server/server'"), encoding="utf-8")
    report = promote_meetings(paths, now=_NOW)
    assert (report.promotions[0].project_filing, report.promotions[0].project_id) == (
        "already", "projects/server/server")

    # A note this pass did not write is left for the dream pass.
    _write(paths, _MEETING, "---\ntitle: 'Kern weekly'\ntype: meeting\n"
           "author: 'agent:default'\nproject: ''\nsource_id: not_1\n"
           "source_file: archive/raw/meetings/granola/2026-09-16-weekly-sync-abcd1234.json\n"
           "---\n\n# Kern weekly\n")
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].project_filing == "not_owned"
    assert _fm(paths, _MEETING)["project"] == ""


def test_ambiguous_match_is_left_unfiled(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _base(paths)
    _snapshot(paths, title="Weekly sync", summary="kern and the home server")
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].project_filing == "unmatched"
    assert report.promotions[0].project_scores == (
        ("projects/kern/kern", 1), ("projects/server/server", 1))
    assert _fm(paths, _MEETING)["project"] == ""


# ------------------------------------------------------------ dream fallback

def test_dream_lists_unfiled_and_proposes_once(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _base(paths)
    _snapshot(paths, title="Weekly sync", summary="Hosting access issues")
    promote_meetings(paths, now=_NOW)

    listed = unfiled_meetings(paths, entity_notes(paths))
    assert [m["node_id"] for m in listed] == ["meetings/2026/2026-09-16-weekly-sync"]
    assert listed[0]["date"] == "2026-09-16"
    assert "Hosting access issues" in listed[0]["excerpt"]

    picks = parse_meeting_filings([{
        "node_id": "meetings/2026/2026-09-16-weekly-sync",
        "project_id": "projects/server/server",
        "reason": "Hosting access is the home server's remote-access work.",
    }])
    [outcome] = write_meeting_filings(paths, picks, now=_NOW, logger=_LOG)
    assert outcome["action"] == "proposed"
    # Pending proposal: no longer listed, and a second pick is not a new note.
    assert unfiled_meetings(paths, entity_notes(paths)) == []
    other = parse_meeting_filings([{**picks[0], "project_id": "projects/kern/kern"}])
    [second] = write_meeting_filings(paths, other, now=_NOW, logger=_LOG)
    assert (second["action"], second["proposal_path"]) == ("exists", outcome["proposal_path"])

    # Approval applies it to the meeting note, which then reads as filed.
    proposal = paths.root / outcome["proposal_path"]
    proposal.write_text(proposal.read_text(encoding="utf-8").replace(
        "approved: false", "approved: true"), encoding="utf-8")
    assert consolidate(paths, logger=_LOG, as_of=date(2026, 9, 17)).promoted == 1
    report = promote_meetings(paths, now=_NOW)
    assert (report.promotions[0].project_filing, report.promotions[0].project_id) == (
        "already", "projects/server/server")


@pytest.mark.parametrize(("node_id", "project_id", "error"), [
    ("meetings/2026/nope", "projects/kern/kern", "not an unfiled meeting"),
    ("meetings/2026/2026-09-16-weekly-sync", "projects/kern", "not a live project"),
    ("meetings/2026/2026-09-16-weekly-sync", "people/me", "not a live project"),
])
def test_dream_filing_refuses_bad_picks(
    tmp_path: Path, node_id: str, project_id: str, error: str,
) -> None:
    paths = _vault(tmp_path)
    _base(paths)
    _snapshot(paths, title="Weekly sync")
    promote_meetings(paths, now=_NOW)
    picks = parse_meeting_filings([{"node_id": node_id, "project_id": project_id, "reason": "r"}])
    with pytest.raises(ValueError, match=error):
        write_meeting_filings(paths, picks, now=_NOW, logger=_LOG)
    assert not list((paths.knowledge / "assistant" / "inbox").glob("*.md"))


def test_parse_meeting_filings_is_strict() -> None:
    with pytest.raises(ValueError, match="array"):
        parse_meeting_filings({"node_id": "x"})
    with pytest.raises(ValueError, match="reason"):
        parse_meeting_filings([{"node_id": "a", "project_id": "b", "reason": " "}])
