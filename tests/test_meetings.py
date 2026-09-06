"""Meeting promotion: connector snapshots -> first-class meeting notes plus
proposed ``attended`` edges (ingest_lib.meetings), and the snapshot schema
read both it and the extractor share (ingest_lib.extractors.meeting).

The vault's posture is what most of these assert: the meeting note is
derived output and may be written, but every edge on a PERSON is only ever
PROPOSED, and an attendee matching no existing note mints nothing.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.connectors.base import (
    DISPLAY_NAME_MAX,
    normalize_attendees,
    safe_display_name,
)
from ingest_lib.connectors.runner import _versioned_relpath
from ingest_lib.consolidate import consolidate
from ingest_lib.extractors import meeting as meeting_ex
from ingest_lib.meetings import (
    find_meeting_candidates,
    promote_meetings,
)
from ingest_lib.metadata import IndexRecord, Status, append_record
from ingest_lib.notes import _split_frontmatter
from ingest_lib.pipeline import _relative_to_logical_root
from ingest_lib.wikilinks import find_candidates
from mcp_server.entity_tools import _meeting_note_text

_LOG = logging.getLogger("test")
_NOW = datetime(2026, 9, 6, 12, 0, 0)


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def _person(paths: VaultPaths, slug: str, title: str, extra: str = "") -> Path:
    dest = paths.knowledge / "people" / f"{slug}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"---\ntitle: {title}\ntype: person\n{extra}---\n\n## Log\n", encoding="utf-8")
    return dest


def _snapshot(
    paths: VaultPaths,
    *,
    name: str = "2026-07-12-kern-weekly-abcd1234.json",
    status: Status = "processed",
    connector: str = "granola",
    title: str = "Kern weekly",
    snap_date: str = "2026-07-12",
    attendees: list[str] | None = None,
    summary: str = "Agreed to ship the promotion pass.",
    native_id: str | None = None,
    ingested_at: str = "2026-07-12T18:00:00Z",
) -> str:
    """Archive one connector snapshot and record it the way ingest does.

    The record's ``relative_path`` is ``meetings/<connector>/<name>``, NOT
    ``inbox/meetings/...``: ingest makes the path relative to ``inbox/``
    (``pipeline._relative_to_logical_root``) before recording it, so a
    fixture carrying the prefix would assert a ``source_file`` shape no real
    run ever writes.
    """
    rel = f"meetings/{connector}/{name}"
    raw = paths.archive_raw / rel
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({
        "connector": connector, "id": name if native_id is None else native_id,
        "title": title, "date": snap_date,
        "attendees": attendees if attendees is not None else [],
        "summary": summary, "transcript": "Alice: hello",
    }), encoding="utf-8")
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path=rel, source_hash="a" * 64, size_bytes=raw.stat().st_size,
        extension=".json", extractor="meeting", status=status,
        raw_path=f"archive/raw/{rel}", processed_path=f"archive/processed/{rel}.md",
        index_note_path=f"knowledge/index/{rel}.md",
        created_at=ingested_at, updated_at=ingested_at,
    ))
    return rel


def _inbox_names(paths: VaultPaths) -> list[str]:
    inbox = paths.knowledge / "assistant" / "inbox"
    return sorted(p.name for p in inbox.glob("*.md")) if inbox.is_dir() else []


# --------------------------------------------------------------- schema

def test_snapshot_parses_dict_attendees_like_the_connectors_do(tmp_path: Path) -> None:
    """Both readers go through connectors.base.normalize_attendees, so the
    ``{"name": ...}`` shape the connectors normalise FROM reads identically."""
    src = tmp_path / "m.json"
    src.write_bytes(json.dumps({
        "title": "T", "attendees": [{"name": "Alice Smith"}, {"email": "x@y"}, "Bob Jones"],
    }).encode("utf-8"))
    snapshot, error = meeting_ex.load_snapshot(src)
    assert error == ""
    assert snapshot is not None
    assert snapshot.attendees == ("Alice Smith", "Bob Jones")


def test_bad_snapshot_returns_an_error_not_an_exception(tmp_path: Path) -> None:
    src = tmp_path / "m.json"
    src.write_bytes(b"[1, 2, 3]")
    snapshot, error = meeting_ex.load_snapshot(src)
    assert snapshot is None
    assert "not a JSON object" in error


def test_attendee_slug_folds_accents_to_the_live_slug_shape() -> None:
    """The live vault's person notes are slugged accent-folded
    (people/antoni-wuczynski, people/lluis-masanes); punching the accent out
    as a separator produces a slug that resolves to nothing."""
    assert meeting_ex.attendee_slug("Antoni Wuczyński") == "antoni-wuczynski"
    assert meeting_ex.attendee_slug("Lluís Masanes") == "lluis-masanes"
    assert meeting_ex.attendee_slug("日本語") == ""


def test_extractor_links_attendees_with_the_folded_slug(tmp_path: Path) -> None:
    src = tmp_path / "m.json"
    src.write_bytes(json.dumps({
        "connector": "granola", "title": "T", "date": "2026-07-12",
        "attendees": ["Antoni Wuczyński"], "summary": "s",
    }).encode("utf-8"))
    res = meeting_ex.extract(src, tmp_path / "a")
    assert "[[knowledge/people/antoni-wuczynski]]" in res.markdown


# ------------------------------------------------------------ promotion

def test_promotion_writes_note_and_proposes_one_relation_per_attendee(
    tmp_path: Path,
) -> None:
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _person(paths, "bob-jones", "Bob Jones")
    _snapshot(paths, attendees=["Alice Smith", "Bob Jones"])

    report = promote_meetings(paths, now=_NOW)
    assert len(report.promotions) == 1
    promotion = report.promotions[0]
    assert promotion.note_action == "written"
    assert promotion.candidate.node_id == "meetings/2026/2026-07-12-kern-weekly"

    note = paths.knowledge / "meetings" / "2026" / "2026-07-12-kern-weekly.md"
    frontmatter, body = _split_frontmatter(note.read_text(encoding="utf-8"))
    assert frontmatter["type"] == "meeting"
    assert frontmatter["date"] == "2026-07-12"
    assert frontmatter["attendees"] == ["people/alice-smith", "people/bob-jones"]
    assert frontmatter["source_file"] == (
        "archive/raw/meetings/granola/2026-07-12-kern-weekly-abcd1234.json"
    )
    assert frontmatter["written_via"] == "script"
    assert "Agreed to ship the promotion pass." in body
    for heading in ("## Agenda", "## Notes", "## Decisions", "## Actions", "## Links"):
        assert heading in body

    # The edges themselves are PROPOSED, never written onto the person.
    assert [r.action for r in promotion.proposals] == ["proposed", "proposed"]
    assert len(_inbox_names(paths)) == 2
    for slug in ("alice-smith", "bob-jones"):
        person = (paths.knowledge / "people" / f"{slug}.md").read_text(encoding="utf-8")
        assert "attended" not in person


def test_proposal_promotes_into_a_real_attended_relation(tmp_path: Path) -> None:
    """End to end: the proposal is exactly the shape consolidate.py reads."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith"])
    promote_meetings(paths, now=_NOW)

    proposal = next((paths.knowledge / "assistant" / "inbox").glob("*.md"))
    proposal.write_text(
        proposal.read_text(encoding="utf-8").replace("approved: false", "approved: true"),
        encoding="utf-8",
    )
    stats = consolidate(paths, logger=_LOG, as_of=date(2026, 9, 6))
    assert stats.promoted == 1
    person = (paths.knowledge / "people" / "alice-smith.md").read_text(encoding="utf-8")
    assert "rel: attended" in person
    assert "target: meetings/2026/2026-07-12-kern-weekly" in person

    # And the pass goes quiet on it: the edge now exists, so nothing is re-proposed.
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].proposals == ()
    assert report.promotions[0].already_related == ("people/alice-smith",)


def test_unknown_attendee_mints_no_node_and_is_listed_by_name(tmp_path: Path) -> None:
    """The email-twin split came from minting a node per attendee id. An
    attendee with no existing note gets NO node, no edge, and a line in the
    note a human can act on."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith", "Nobody Known"])

    report = promote_meetings(paths, now=_NOW)
    promotion = report.promotions[0]
    assert promotion.candidate.unresolved_names == ("Nobody Known",)
    assert promotion.candidate.resolved_ids == ("people/alice-smith",)
    assert not (paths.knowledge / "people" / "nobody-known.md").exists()
    assert sorted(p.name for p in (paths.knowledge / "people").glob("*.md")) == [
        "alice-smith.md"
    ]

    note = (paths.knowledge / "meetings" / "2026" / "2026-07-12-kern-weekly.md").read_text(
        encoding="utf-8"
    )
    assert "- Nobody Known (unresolved attendee: no people/ note)\n" in note
    assert "[[knowledge/people/nobody-known]]" not in note
    assert len(promotion.proposals) == 1


def test_the_note_body_has_no_section_the_meeting_template_lacks(
    tmp_path: Path,
) -> None:
    """A promoted meeting is not a second body shape: an unresolved attendee
    is a line in ``## Links``, not a section of its own. The template
    (knowledge/index/templates/meeting.md) and ``meeting_create`` both write
    exactly these five headings."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith", "Nobody Known"])
    promote_meetings(paths, now=_NOW)
    note = (paths.knowledge / "meetings" / "2026" / "2026-07-12-kern-weekly.md").read_text(
        encoding="utf-8"
    )
    assert [ln for ln in note.splitlines() if ln.startswith("## ")] == [
        "## Agenda", "## Notes", "## Decisions", "## Actions", "## Links",
    ]


def test_attendee_named_by_email_resolves_by_title(tmp_path: Path) -> None:
    """The live vault's email-slugged nodes (people/alexasymmetricsecuritycom)
    do not match the address's slug, but their title is the address verbatim."""
    paths = _vault(tmp_path)
    _person(paths, "alexasymmetricsecuritycom", "alex@asymmetricsecurity.com")
    _snapshot(paths, attendees=["alex@asymmetricsecurity.com"])
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].candidate.resolved_ids == (
        "people/alexasymmetricsecuritycom",
    )


def test_alias_resolves_onto_the_merge_survivor(tmp_path: Path) -> None:
    """AGENTS.md's merge keeps the address as an alias on the named node and
    supersedes the twin. A snapshot still carrying the address must land on
    the survivor, never grow a new edge on the history."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith", "aliases: [alice@example.com]\n")
    _person(
        paths, "aliceexamplecom", "alice@example.com",
        "superseded_by: knowledge/people/alice-smith\n",
    )
    _snapshot(paths, attendees=["alice@example.com"])
    report = promote_meetings(paths, now=_NOW)
    # The email node's own title matches first and is followed through
    # superseded_by; either route must end on the survivor.
    assert report.promotions[0].candidate.resolved_ids == ("people/alice-smith",)


def test_accent_folded_name_resolves_to_the_live_slug(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _person(paths, "antoni-wuczynski", "Antoni Wuczyński")
    _snapshot(paths, attendees=["Antoni Wuczyński"])
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].candidate.resolved_ids == ("people/antoni-wuczynski",)


def test_two_people_with_one_name_are_ambiguous_not_guessed(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _person(paths, "alice-smith-2", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith"])
    report = promote_meetings(paths, now=_NOW)
    promotion = report.promotions[0]
    assert promotion.candidate.resolved_ids == ()
    assert promotion.candidate.unresolved_names == ("Alice Smith",)
    assert [a.status for a in promotion.candidate.attendees] == ["ambiguous"]
    assert promotion.proposals == ()
    # The slug layer DOES hit people/alice-smith here; it must not be
    # allowed to break the tie the two identical titles create.
    assert (paths.knowledge / "people" / "alice-smith.md").exists()


def test_rerun_over_an_unchanged_vault_writes_and_proposes_nothing(
    tmp_path: Path,
) -> None:
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith"])
    promote_meetings(paths, now=_NOW)
    note = paths.knowledge / "meetings" / "2026" / "2026-07-12-kern-weekly.md"
    before = note.read_text(encoding="utf-8")
    inbox_before = _inbox_names(paths)

    report = promote_meetings(paths, now=datetime(2027, 1, 1))
    assert report.promotions[0].note_action == "exists"
    assert [r.action for r in report.promotions[0].proposals] == ["exists"]
    assert note.read_text(encoding="utf-8") == before
    assert _inbox_names(paths) == inbox_before


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith"])
    report = promote_meetings(paths, dry_run=True, now=_NOW)
    assert report.promotions[0].note_action == "written"
    assert [r.action for r in report.promotions[0].proposals] == ["proposed"]
    assert not (paths.knowledge / "meetings").exists()
    assert _inbox_names(paths) == []


def test_a_different_snapshot_on_the_same_id_is_a_conflict(tmp_path: Path) -> None:
    """Two DIFFERENT meetings can share <date>-<slug>. The second must not
    adopt the first's note, and must not hang its attendees off it.

    "Different" is the snapshot's own native id, not its archive path — the
    filename the connector produces carries an 8-hex tag of that id
    (``connectors.base.meeting_filename``), so two ids is what this fixture
    pair really differs by.
    """
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith"], native_id="granola-1")
    promote_meetings(paths, now=_NOW)
    inbox_before = _inbox_names(paths)
    note = paths.knowledge / "meetings" / "2026" / "2026-07-12-kern-weekly.md"
    before = note.read_text(encoding="utf-8")

    _person(paths, "bob-jones", "Bob Jones")
    _snapshot(paths, name="2026-07-12-kern-weekly-99999999.json",
              attendees=["Bob Jones"], native_id="granola-2")
    report = promote_meetings(paths, now=_NOW)
    conflicts = [p for p in report.promotions if p.note_action == "conflict"]
    assert len(conflicts) == 1
    assert conflicts[0].proposals == ()
    assert conflicts[0].conflict_source.endswith("2026-07-12-kern-weekly-abcd1234.json")
    assert conflicts[0].conflict_source_id == "granola-1"
    assert _inbox_names(paths) == inbox_before
    assert note.read_text(encoding="utf-8") == before


def test_a_repulled_snapshot_updates_the_note_instead_of_conflicting(
    tmp_path: Path,
) -> None:
    """The shape the connector actually produces for a CHANGED re-pull.

    ``connectors.runner._resolve_dest`` never overwrites the snapshot already
    in the inbox: it routes the new bytes to a hash-versioned sibling
    (``x.<12hex>.json``), so one meeting reaches the archive under two paths
    with one native id. Comparing archive paths made that a permanent false
    conflict — the note was never updated, the added attendee was never
    proposed, and every nightly run logged it again.
    """
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _person(paths, "dave-morgan", "Dave Morgan")
    _snapshot(paths, attendees=["Alice Smith"], native_id="granola-1")
    promote_meetings(paths, now=_NOW)

    versioned = _versioned_relpath(
        "inbox/meetings/granola/2026-07-12-kern-weekly-abcd1234.json", "4392ff0d7e3b" * 6
    )
    _snapshot(
        paths, name=Path(versioned).name, native_id="granola-1",
        attendees=["Alice Smith", "Dave Morgan"],
        ingested_at="2026-07-13T18:00:00Z",
    )
    report = promote_meetings(paths, now=datetime(2026, 9, 7, 9, 0, 0))
    # One meeting, one candidate: the newer record owns the note, so the two
    # archived versions cannot take turns rewriting it.
    assert [p.note_action for p in report.promotions] == ["updated"]

    note = paths.knowledge / "meetings" / "2026" / "2026-07-12-kern-weekly.md"
    frontmatter, body = _split_frontmatter(note.read_text(encoding="utf-8"))
    assert frontmatter["attendees"] == ["people/alice-smith", "people/dave-morgan"]
    assert frontmatter["source_id"] == "granola-1"
    # The note now names the re-pulled bytes, and only them.
    assert frontmatter["source_file"] == (
        "archive/raw/meetings/granola/2026-07-12-kern-weekly-abcd1234.4392ff0d7e3b.json"
    )
    assert frontmatter["created"] == "2026-09-06T12:00:00Z"
    assert frontmatter["updated"] == "2026-09-07T09:00:00Z"
    assert "- Attendee: [[knowledge/people/dave-morgan]]" in body
    # …and the newly-seen attendee is proposed, not silently dropped.
    assert [r.action for r in report.promotions[0].proposals] == ["exists", "proposed"]
    assert any("dave-morgan" in name for name in _inbox_names(paths))


def test_note_without_provenance_is_this_meeting_and_is_left_alone(
    tmp_path: Path,
) -> None:
    """Every meeting note in the live vault was written by ``meeting_create``,
    which stamps no ``source_file``. Promoting the snapshot behind one must
    neither rewrite it nor call it a different meeting — it fills the
    attendee gap by proposal only."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith"])
    note = paths.knowledge / "meetings" / "2026" / "2026-07-12-kern-weekly.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("---\ntitle: Kern weekly\ntype: meeting\n---\n\nhand-written\n",
                    encoding="utf-8")

    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].note_action == "exists"
    assert note.read_text(encoding="utf-8").endswith("hand-written\n")
    assert [r.action for r in report.promotions[0].proposals] == ["proposed"]


def test_snapshot_without_a_canonical_date_is_skipped_not_guessed(
    tmp_path: Path,
) -> None:
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, snap_date="2026-7-12", attendees=["Alice Smith"])
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions == ()
    assert len(report.skipped) == 1
    assert "no canonical YYYY-MM-DD date" in report.skipped[0].reason
    assert not (paths.knowledge / "meetings").exists()


def test_failed_extraction_is_not_promoted(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _snapshot(paths, status="manual_review", attendees=[])
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions == ()
    assert [s.reason for s in report.skipped] == ["record status is manual_review"]


def test_non_meeting_records_are_ignored(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    append_record(paths.metadata_index_jsonl, IndexRecord(
        relative_path="notes/a.txt", source_hash="b" * 64, size_bytes=1,
        extension=".txt", extractor="text", status="processed",
        raw_path="archive/raw/notes/a.txt", processed_path=None,
        index_note_path=None,
    ))
    candidates, skipped = find_meeting_candidates(paths)
    assert candidates == []
    assert skipped == []


def test_source_links_use_the_resolvable_form(tmp_path: Path) -> None:
    """AGENTS.md rule 3. The raw snapshot is linked extensionless (a bare
    [[....json]] is read as a literal .json file and resolves to nothing);
    the index note keeps its full ``.json.md`` name for the same reason."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith"])
    promote_meetings(paths, now=_NOW)
    note = (paths.knowledge / "meetings" / "2026" / "2026-07-12-kern-weekly.md").read_text(
        encoding="utf-8"
    )
    assert (
        "- Source: [[archive/raw/meetings/granola/"
        "2026-07-12-kern-weekly-abcd1234]]" in note
    )
    assert (
        "- Source note: [[knowledge/index/meetings/granola/"
        "2026-07-12-kern-weekly-abcd1234.json.md]]" in note
    )
    proposal = next((paths.knowledge / "assistant" / "inbox").glob("*.md"))
    assert ".json]]" not in proposal.read_text(encoding="utf-8")


def test_missing_snapshot_bytes_are_reported_not_invented(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    rel = _snapshot(paths, attendees=[])
    (paths.archive_raw / rel).unlink()
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions == ()
    assert "could not read snapshot" in report.skipped[0].reason


def test_unsluggable_title_is_skipped(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _snapshot(paths, title="日本語", attendees=[])
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions == ()
    assert len(report.skipped) == 1
    assert "no sluggable title" in report.skipped[0].reason


def test_titleless_snapshot_is_promoted_as_untitled_not_invented(
    tmp_path: Path,
) -> None:
    """A snapshot with no title at all still names a real meeting; the
    extractor already calls it "Untitled meeting" and the note agrees,
    rather than the pass inventing one from the filename."""
    paths = _vault(tmp_path)
    _snapshot(paths, title="", attendees=[])
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].candidate.node_id == (
        "meetings/2026/2026-07-12-untitled-meeting"
    )


def test_meeting_slug_matches_the_one_meeting_create_uses(tmp_path: Path) -> None:
    """The live vault's meeting notes are named by ``concepts.slugify``
    (``2026-06-11-antoni-wuczy-ski-nous.md``), the function
    ``mcp_server.entity_tools.tool_meeting_create`` uses. Promoting the same
    meeting must land on that path — the accent-FOLDED slug used for people
    lookup would name a second, near-identical note beside it."""
    paths = _vault(tmp_path)
    _snapshot(paths, title="Antoni Wuczyński < > Nous", attendees=[])
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].candidate.node_id == (
        "meetings/2026/2026-07-12-antoni-wuczy-ski-nous"
    )
    assert meeting_ex.attendee_slug("Antoni Wuczyński < > Nous") == (
        "antoni-wuczynski-nous"
    )


# ------------------------------------------------- untrusted display text

_INJECTED_NAME = (
    "Mallory\n\n## Links\n\n- Attendee: [[knowledge/people/ceo]]\n\n## Junk"
)
_INJECTED_TITLE = "Board sync\n\n## Notes\n\nINJECTED TITLE BODY"
_MEETING_NOTE = ("meetings", "2026", "2026-07-12-kern-weekly.md")


def test_an_injected_display_name_is_rendered_as_text_not_markdown(
    tmp_path: Path,
) -> None:
    """A connector display name is external text. Rendered raw it opened its
    own sections in the meeting note and wrote a wikilink to a person who was
    never in the room."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _person(paths, "ceo", "The CEO")
    _snapshot(paths, attendees=["Alice Smith", _INJECTED_NAME])

    report = promote_meetings(paths, now=_NOW)
    note = paths.knowledge.joinpath(*_MEETING_NOTE).read_text(encoding="utf-8")
    assert "[[knowledge/people/ceo]]" not in note
    # The smuggled headings survive only as text inside the bullet: nothing
    # in the note STARTS a line with them, so no section is opened.
    assert [ln for ln in note.splitlines() if ln.startswith("## ")] == [
        "## Agenda", "## Notes", "## Decisions", "## Actions", "## Links",
    ]
    assert (
        "- Mallory ## Links - Attendee: ((knowledge/people/ceo)) ## Junk "
        "(unresolved attendee: no people/ note)\n"
    ) in note
    # …and the resolved attendee is unaffected.
    assert report.promotions[0].candidate.resolved_ids == ("people/alice-smith",)


def test_the_wikilink_pass_proposes_nothing_from_an_injected_name(
    tmp_path: Path,
) -> None:
    """The teeth of it: the nightly wikilink pass reads every graph node's
    note for links and proposes the edges it finds. A link smuggled in
    through an attendee name would become a ``related_to`` proposal between
    the meeting and a person who never attended, sourced entirely from
    outside the vault."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _person(paths, "ceo", "The CEO")
    _snapshot(paths, attendees=["Alice Smith", _INJECTED_NAME])
    promote_meetings(paths, now=_NOW)

    meeting = "meetings/2026/2026-07-12-kern-weekly"
    assert [
        (c.source, c.target)
        for c in find_candidates(paths)
        if meeting in (c.source, c.target)
    ] == []


def test_a_title_with_control_characters_is_refused_not_promoted(
    tmp_path: Path,
) -> None:
    """``mcp_server.entity_tools.tool_meeting_create`` rejects a title with
    control characters; a snapshot's title is the same untrusted string and
    gets the same answer — reported, not silently reshaped into a node id
    naming whatever was smuggled in."""
    paths = _vault(tmp_path)
    _snapshot(paths, title=_INJECTED_TITLE, attendees=[])
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions == ()
    assert len(report.skipped) == 1
    assert "title contains control characters" in report.skipped[0].reason
    assert not (paths.knowledge / "meetings").exists()


def test_a_title_carrying_link_syntax_cannot_write_a_wikilink(
    tmp_path: Path,
) -> None:
    """One line, but still a link: the heading and the frontmatter title both
    render it inert, so the wikilink pass sees nothing to propose."""
    paths = _vault(tmp_path)
    _person(paths, "ceo", "The CEO")
    _snapshot(paths, title="Board [[knowledge/people/ceo]] sync", attendees=[])
    report = promote_meetings(paths, now=_NOW)
    candidate = report.promotions[0].candidate
    assert candidate.title == "Board ((knowledge/people/ceo)) sync"
    text = (paths.root / candidate.rel_path).read_text(encoding="utf-8")
    assert "[[knowledge/people/ceo]]" not in text
    assert find_candidates(paths) == []


def test_a_display_name_is_capped_and_stripped_of_markdown_syntax() -> None:
    """The sanitiser itself, on the payload shapes the refuter used."""
    assert safe_display_name("### Alice") == "Alice"
    assert safe_display_name(_INJECTED_NAME) == (
        "Mallory ## Links - Attendee: ((knowledge/people/ceo)) ## Junk"
    )
    assert safe_display_name("[[knowledge/people/ceo]]") == "((knowledge/people/ceo))"
    assert len(safe_display_name("x" * 5000)) == DISPLAY_NAME_MAX
    # A "name" that is only Markdown syntax names nobody, so it is dropped.
    assert normalize_attendees(["###", "  ", "Alice Smith"]) == ["Alice Smith"]


# --------------------------------------------- refreshing an existing note

def test_creating_the_missing_person_note_and_rerunning_fixes_the_note(
    tmp_path: Path,
) -> None:
    """The module's own promise: an unresolved attendee is "listed by display
    name so a human can create the person note and re-run". Before, the
    re-run proposed the edge while the note went on listing the person as
    unresolved forever — the note and the graph contradicting each other."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith", "Nobody Here"])
    promote_meetings(paths, now=_NOW)
    note = paths.knowledge.joinpath(*_MEETING_NOTE)
    assert "- Nobody Here (unresolved attendee: no people/ note)" in note.read_text(
        encoding="utf-8"
    )

    _person(paths, "nobody-here", "Nobody Here")
    report = promote_meetings(paths, now=datetime(2026, 9, 7, 9, 0, 0))
    assert report.promotions[0].note_action == "updated"
    frontmatter, body = _split_frontmatter(note.read_text(encoding="utf-8"))
    assert frontmatter["attendees"] == ["people/alice-smith", "people/nobody-here"]
    assert "- Attendee: [[knowledge/people/nobody-here]]" in body
    assert "unresolved" not in body
    assert [r.action for r in report.promotions[0].proposals] == ["exists", "proposed"]

    # …and it settles: a third run over the now-unchanged vault is a no-op.
    before = note.read_text(encoding="utf-8")
    third = promote_meetings(paths, now=datetime(2028, 1, 1))
    assert third.promotions[0].note_action == "exists"
    assert note.read_text(encoding="utf-8") == before


def test_refreshing_a_note_leaves_hand_written_regions_byte_identical(
    tmp_path: Path,
) -> None:
    """Only the fenced bullets and the derived frontmatter keys move. What a
    human filled in — Agenda, Notes, Decisions, Actions, a tail below the
    fence, and the append-only ``## Log`` — is not this pass's to rewrite."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith", "Nobody Here"])
    promote_meetings(paths, now=_NOW)
    note = paths.knowledge.joinpath(*_MEETING_NOTE)

    edited = note.read_text(encoding="utf-8").replace(
        "## Agenda\n\n_(empty)_", "## Agenda\n\n- Budget\n- Hiring"
    ).replace("## Decisions\n\n_(empty)_", "## Decisions\n\n- Ship on Friday")
    edited += "\n## Log\n\n- 2026-09-06 — minuted by hand\n"
    note.write_text(edited, encoding="utf-8")

    _person(paths, "nobody-here", "Nobody Here")
    assert promote_meetings(paths, now=_NOW).promotions[0].note_action == "updated"
    after = note.read_text(encoding="utf-8")
    assert "## Agenda\n\n- Budget\n- Hiring" in after
    assert "## Decisions\n\n- Ship on Friday" in after
    assert after.endswith("## Log\n\n- 2026-09-06 — minuted by hand\n")
    assert "- Attendee: [[knowledge/people/nobody-here]]" in after


def test_a_log_moved_inside_the_fence_stops_the_refresh(tmp_path: Path) -> None:
    """``## Log`` is append-only history (AGENTS.md). One that has been moved
    inside the generated block is not re-rendered away — the refresh gives up
    on the note instead."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith", "Nobody Here"])
    promote_meetings(paths, now=_NOW)
    note = paths.knowledge.joinpath(*_MEETING_NOTE)
    note.write_text(
        note.read_text(encoding="utf-8").replace(
            "- Source: ", "## Log\n\n- 2026-09-06 — a fact\n\n- Source: "
        ),
        encoding="utf-8",
    )
    before = note.read_text(encoding="utf-8")

    _person(paths, "nobody-here", "Nobody Here")
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].note_action == "exists"
    assert note.read_text(encoding="utf-8") == before


def test_a_note_this_pass_did_not_write_is_never_refreshed(tmp_path: Path) -> None:
    """No ``author: script:meeting-promotion``, no rewrite: a note
    ``meeting_create`` or a human wrote is theirs, even when its attendee
    list is stale."""
    paths = _vault(tmp_path)
    _person(paths, "alice-smith", "Alice Smith")
    _snapshot(paths, attendees=["Alice Smith"])
    note = paths.knowledge.joinpath(*_MEETING_NOTE)
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "---\ntitle: 'Kern weekly'\ntype: meeting\nattendees: []\n---\n\n"
        "## Links\n\n<!-- MEETING-DERIVED-START -->\n- Source: [[x]]\n"
        "<!-- MEETING-DERIVED-END -->\n",
        encoding="utf-8",
    )
    before = note.read_text(encoding="utf-8")
    report = promote_meetings(paths, now=_NOW)
    assert report.promotions[0].note_action == "exists"
    assert note.read_text(encoding="utf-8") == before
    # The gap is still reported the only way a graph write may be: a proposal.
    assert [r.action for r in report.promotions[0].proposals] == ["proposed"]


# ----------------------------------------------------------- resolution

def test_a_half_merged_duplicate_grows_no_new_edge(tmp_path: Path) -> None:
    """AGENTS.md's merge supersedes a duplicate rather than deleting it, and
    the duplicate must not collect new ``attended`` history. When its
    ``superseded_by`` names a note that is not on disk, following the chain
    dead-ends AT the duplicate — resolving to it anyway is exactly the
    email-twin growth the merge shape exists to stop."""
    paths = _vault(tmp_path)
    _person(
        paths, "frankexamplecom", "frank@example.com",
        "superseded_by: knowledge/people/frank-example\n",
    )
    _snapshot(paths, attendees=["frank@example.com"])
    report = promote_meetings(paths, now=_NOW)
    promotion = report.promotions[0]
    assert promotion.candidate.resolved_ids == ()
    assert promotion.candidate.unresolved_names == ("frank@example.com",)
    assert promotion.proposals == ()

    # With the survivor on disk the same name resolves onto it.
    _person(paths, "frank-example", "Frank Example")
    again = promote_meetings(paths, now=_NOW)
    assert again.promotions[0].candidate.resolved_ids == ("people/frank-example",)


# ------------------------------------------------------------- note shape

def test_the_frontmatter_is_spelled_the_way_meeting_create_spells_it(
    tmp_path: Path,
) -> None:
    """The note claims the shape ``meeting_create`` writes, so the shared
    keys must be byte-identical, not merely equivalent YAML: ``fm_scalar``
    left a plain title unquoted where ``_meeting_note_text`` always
    single-quotes it."""
    paths = _vault(tmp_path)
    _snapshot(paths, title="Adversarial attendees", attendees=[])
    promote_meetings(paths, now=_NOW)
    text = (
        paths.knowledge / "meetings" / "2026" / "2026-07-12-adversarial-attendees.md"
    ).read_text(encoding="utf-8")
    reference = _meeting_note_text(
        title="Adversarial attendees", date="2026-07-12", attendee_ids=[],
        project_id="", body="Agreed to ship the promotion pass.",
        now_iso="2026-09-06T12:00:00Z",
    )
    for key in ("title:", "type:", "date:", "attendees:", "project:", "topics:",
                "created:", "updated:"):
        line = next(ln for ln in reference.splitlines() if ln.startswith(key))
        assert line in text.splitlines()


def test_the_fixture_record_is_the_path_shape_ingest_records(
    tmp_path: Path,
) -> None:
    """Guards the fixture itself: ingest records a snapshot relative to
    ``inbox/``, so ``inbox/`` never appears in a record path or in the
    ``source_file`` these tests assert."""
    paths = _vault(tmp_path)
    rel = _snapshot(paths, attendees=[])
    assert _relative_to_logical_root(
        paths.inbox / "meetings" / "granola" / "2026-07-12-kern-weekly-abcd1234.json",
        paths, from_archive=False,
    ) == rel
    candidates, _skipped = find_meeting_candidates(paths)
    assert candidates[0].source_file == f"archive/raw/{rel}"
    assert "inbox/" not in candidates[0].source_file
