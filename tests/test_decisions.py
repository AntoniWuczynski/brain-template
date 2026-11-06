"""Decision records (ingest_lib.decisions) and the sweep's
decision-integrity check."""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.decisions import (
    DECISION_PATH_RE,
    DecisionNote,
    DecisionSections,
    DecisionStatus,
    decision_rel_path,
    iter_decisions,
    next_number,
    parse_decision,
    parse_sections,
    render_decision_note,
)
from ingest_lib.sweep import _check_decision_integrity, run_sweep

PROJECT = "projects/brain/brain"
OLD_REL = "knowledge/projects/brain/decisions/0001-use-uv.md"
NEW_REL = "knowledge/projects/brain/decisions/0002-use-uv-workspaces.md"


def _note(
    rel: str,
    number: int,
    *,
    project: str = PROJECT,
    status: DecisionStatus | None = "accepted",
    decided: str | None = "2026-10-11",
    supersedes: str | None = None,
    superseded_by: str | None = None,
) -> DecisionNote:
    return DecisionNote(
        rel_path=rel, project=project, number=number, title=f"Decision {number}",
        status=status, date="2026-10-11", decided=decided,
        supersedes=supersedes, superseded_by=superseded_by,
    )


SECTIONS = DecisionSections(context="Why.", decision="Use uv.")


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _write(paths: VaultPaths, note: DecisionNote) -> None:
    target = paths.root / note.rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_decision_note(note, SECTIONS), encoding="utf-8")


def _supersede_pair() -> tuple[DecisionNote, DecisionNote]:
    old = _note(OLD_REL, 1, status="superseded", superseded_by=NEW_REL[:-3])
    new = _note(NEW_REL, 2, supersedes=OLD_REL[:-3])
    return old, new


def test_render_shape() -> None:
    old, new = _supersede_pair()
    sections = DecisionSections("Why.", "Use uv workspaces.", "", "Poetry.")
    assert render_decision_note(new, sections) == (
        "---\n"
        "title: 'Decision 2'\n"
        "type: decision\n"
        "project: 'projects/brain/brain'\n"
        "number: 2\n"
        "status: accepted\n"
        "date: '2026-10-11'\n"
        "decided: '2026-10-11'\n"
        "supersedes: 'knowledge/projects/brain/decisions/0001-use-uv'\n"
        "---\n"
        "\n"
        "# 0002 — Decision 2\n"
        "\n"
        "## Context\n\nWhy.\n\n"
        "## Decision\n\nUse uv workspaces.\n\n"
        "## Consequences\n\n_(empty)_\n\n"
        "## Alternatives considered\n\nPoetry.\n\n"
        "## Links\n\n"
        "- Project: [[knowledge/projects/brain/brain]]\n"
        "- Supersedes: [[knowledge/projects/brain/decisions/0001-use-uv]]\n"
    )
    assert "- Superseded by: [[knowledge/projects/brain/decisions/0002-use-uv-workspaces]]" in (
        render_decision_note(old, sections)
    )


def test_render_parse_round_trip() -> None:
    old, _new = _supersede_pair()
    old = replace(old, title="It's: a title")
    sections = DecisionSections("Line one.\n\n- bullet", "Use uv.", "Faster.", "")
    text = render_decision_note(old, sections)
    assert parse_decision(OLD_REL, text) == old
    assert parse_sections(text) == sections


def test_render_refuses_heading_lines_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="Context"):
        render_decision_note(_note(OLD_REL, 1), DecisionSections("x\n## Decision\ny", "d"))
    with pytest.raises(ValueError, match="number"):
        render_decision_note(replace(_note(OLD_REL, 1), number=None), SECTIONS)
    # A sub-heading that is not one of the record's own is fine.
    text = render_decision_note(_note(OLD_REL, 1), DecisionSections("### Detail\nx", "d"))
    assert parse_sections(text).context == "### Detail\nx"


@pytest.mark.parametrize("sep", ["\u2028", "\u2029", "\x85", "\x1e", "\v", "\f"])
def test_render_refuses_a_multi_line_title(sep: str) -> None:
    note = replace(_note(OLD_REL, 1), title=f"Pick X{sep}## Decision{sep}INJECTED")
    with pytest.raises(ValueError, match="title"):
        render_decision_note(note, SECTIONS)


def test_render_with_prior_carries_hand_added_content() -> None:
    note = _note(OLD_REL, 1)
    prior = render_decision_note(note, SECTIONS).replace(
        "type: decision\n", "type: decision\ntags: [x]  # kept\nstatus: proposed\n", 1,
    ).replace("## Context", "Preamble.\n\n## Context").rstrip("\n") + "\n- Extra: [[y]]\n"
    flipped = replace(note, status="superseded", superseded_by=NEW_REL[:-3])
    text = render_decision_note(flipped, parse_sections(prior), prior=prior)
    assert "tags: [x]  # kept\n" in text
    assert text.count("status:") == 1 and parse_decision(OLD_REL, text) == flipped
    assert "# 0001 — Decision 1\n\nPreamble.\n\n## Context" in text
    assert text.endswith(
        "- Project: [[knowledge/projects/brain/brain]]\n"
        f"- Superseded by: [[{NEW_REL[:-3]}]]\n"
        "- Extra: [[y]]\n"
    )
    assert parse_sections(text) == SECTIONS


def test_parse_decision_is_lenient() -> None:
    text = "---\nstatus: done\nnumber: true\ndate: 2026-10-11\n---\n\nbody\n"
    note = parse_decision(OLD_REL, text)
    assert (note.status, note.number, note.date, note.title, note.project) == (
        None, None, "2026-10-11", "", ""
    )
    assert (note.ref, note.folder_project, note.file_number) == (
        OLD_REL[:-3], PROJECT, 1
    )
    with pytest.raises(ValueError):
        parse_decision("knowledge/projects/brain/harness-decision.md", text)


def test_decision_rel_path() -> None:
    assert decision_rel_path("chats/worksearch/worksearch", 7, "Use Łódź office!") == (
        "knowledge/chats/worksearch/decisions/0007-use-lodz-office.md"
    )
    assert DECISION_PATH_RE.match(decision_rel_path(PROJECT, 12, "A b"))
    with pytest.raises(ValueError):
        decision_rel_path("projects/brain/other", 1, "x")
    with pytest.raises(ValueError):
        decision_rel_path(PROJECT, 1, "!!!")


def test_next_number_reads_filenames(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    assert next_number(paths, PROJECT) == 1
    folder = paths.root / "knowledge/projects/brain/decisions"
    folder.mkdir(parents=True)
    for name in ("0001-a.md", "0003-b.md", "9-misnamed.md", "0042-x.txt"):
        (folder / name).write_text("x", encoding="utf-8")
    assert next_number(paths, PROJECT) == 4


def test_iter_decisions_is_by_path(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    old, new = _supersede_pair()
    chat = _note("knowledge/chats/worksearch/decisions/0001-x.md", 1,
                 project="chats/worksearch/worksearch")
    for n in (new, old, chat):
        _write(paths, n)
    legacy = paths.root / "knowledge/projects/brain/harness-decision.md"
    legacy.write_text("---\ntype: decision\n---\n", encoding="utf-8")
    (paths.root / "knowledge/projects/brain/decisions/Notes.md").write_text("x", encoding="utf-8")
    assert iter_decisions(paths) == [chat, old, new]


def _details(paths: VaultPaths) -> list[tuple[str, str]]:
    return [(f.path, f.detail) for f in _check_decision_integrity(paths)]


def test_sweep_consistent_records_are_clean(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    for n in _supersede_pair():
        _write(paths, n)
    _write(paths, _note("knowledge/projects/brain/decisions/0003-p.md", 3,
                        status="proposed", decided=None))
    assert _details(paths) == []


def test_sweep_flags_each_integrity_problem(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    bad = [
        _note("knowledge/projects/brain/decisions/0001-a.md", 1, status="superseded"),
        _note("knowledge/projects/brain/decisions/0002-b.md", 5),
        _note("knowledge/projects/brain/decisions/0003-c.md", 3,
              supersedes="knowledge/projects/brain/decisions/0002-b"),
        _note("knowledge/projects/brain/decisions/0003-d.md", 3, project="projects/x/x"),
        _note("knowledge/projects/brain/decisions/0004-e.md", 4,
              superseded_by="knowledge/projects/brain/decisions/0009-gone"),
    ]
    for n in bad:
        _write(paths, n)
    # Status outside the vocabulary cannot be rendered; write it by hand.
    raw = paths.root / "knowledge/projects/brain/decisions/0006-f.md"
    raw.write_text(
        render_decision_note(_note(str(raw.relative_to(paths.root)), 6), SECTIONS)
        .replace("status: accepted", "status: done"),
        encoding="utf-8",
    )
    d = "knowledge/projects/brain/decisions/"
    assert _details(paths) == [
        (d + "0001-a.md", "status superseded but no superseded_by"),
        (d + "0002-b.md", "number 5 does not match filename number 2"),
        (d + "0003-c.md", "number 3 also used by " + d + "0003-d.md"),
        (d + "0003-c.md", f"supersedes {d}0002-b, whose superseded_by does not point back"),
        (d + "0003-d.md", "number 3 also used by " + d + "0003-c.md"),
        (d + "0003-d.md", "project 'projects/x/x' is not its folder 'projects/brain/brain'"),
        (d + "0004-e.md", "superseded_by set but status is accepted"),
        (d + "0004-e.md", f"superseded_by {d}0009-gone, which is not a decision record"),
        (d + "0006-f.md", "status is not one of proposed, accepted, rejected, superseded"),
    ]


def test_run_sweep_reports_decision_integrity(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _write(paths, _note("knowledge/projects/brain/decisions/0001-a.md", 2))
    report = run_sweep(paths, logger=logging.getLogger("test"), as_of=date(2026, 10, 11))
    assert report.counts.get("decision-integrity") == 1
