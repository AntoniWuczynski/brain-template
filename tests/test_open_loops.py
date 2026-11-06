"""Open Loops note (ingest_lib.open_loops)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ingest_lib import open_loops
from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.open_loops import collect, rebuild_open_loops, render

TODAY = date(2026, 10, 11)


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path / "vault")
    paths.ensure()
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    target = paths.root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _decision(paths: VaultPaths, rel: str, status: str) -> None:
    _write(paths, rel, (
        f"---\ntitle: 'Use {status}'\ntype: decision\nproject: 'projects/brain/brain'\n"
        f"number: 1\nstatus: {status}\ndate: '2026-10-01'\n---\n\n# body\n"
    ))


def _populate(paths: VaultPaths) -> None:
    _decision(paths, "knowledge/projects/brain/decisions/0001-use-proposed.md", "proposed")
    _decision(paths, "knowledge/projects/brain/decisions/0002-use-accepted.md", "accepted")
    # A legacy curated `type: decision` note outside decisions/ is not a record.
    _decision(paths, "knowledge/projects/brain/old-decision.md", "proposed")

    log = "knowledge/projects/brain/log"
    _write(paths, f"{log}/2026-09-01.md", "# s\n\n## Next\n- old brain item\n")
    _write(paths, f"{log}/2026-09-20.md", (
        "---\ntitle: s\n---\n# s\n\n## Next\n- superseded within the file\n\n"
        "## 10:00Z\n\n### Next\n- wrapped item that\n  continues here\n"
        "  - nested detail\n- second item\n\n"
        "```\n## Next\n- fenced item\n```\n"
        "> ## Next\n> - quoted item\n\n"
        "## Open Loops — feature write-up\n- not a loop\n"
    ))
    _write(paths, f"{log}/2026-09-30.md", "# s\n\n## What was done\n- nothing next\n")
    _write(paths, "knowledge/chats/worksearch/log/2026-08-30.md", (
        "# s\n\n## Open at session end\nSubmit the application\nafter review.\n"
    ))

    meetings = "knowledge/meetings/2026"
    _write(paths, f"{meetings}/2026-10-06-interview.md", (
        "---\ntitle: 'Interview'\ntype: meeting\ndate: '2026-10-06'\n---\n\n# Interview\n\n"
        "## Agenda\n\n_(empty)_\n\n## Notes\n\n# Context\n\n- a fact\n\n"
        "# Next Steps\n\n- **Call the co-founder**\n\n  Detail paragraph.\n"
        "- Prepare\n\n---\n\nChat with meeting transcript: link\n\n"
        "## Decisions\n\n_(empty)_\n\n## Actions\n\n- Send the CV\n\n## Links\n\n- x\n"
    ))
    _write(paths, f"{meetings}/2026-09-16-site.md", (
        "---\ntitle: 'Site'\ndate: 2026-09-16\n---\n\n## Notes\n\n### Następne Kroki\n\n"
        "- Złożyć zamówienie\n\n## Actions\n\n_(empty)_\n"
    ))
    _write(paths, f"{meetings}/2026-08-01-too-old.md", (
        "---\ntitle: 'Old'\ndate: '2026-08-01'\n---\n\n## Notes\n\n# Next Steps\n\n- stale\n"
    ))
    # Next steps outside ## Notes are not collected.
    _write(paths, f"{meetings}/2026-10-07-outside.md", (
        "---\ntitle: 'Outside'\ndate: '2026-10-07'\n---\n\n## Agenda\n\n# Next Steps\n\n- no\n"
    ))
    # The archive duplicate of a meeting is never read.
    _write(paths, "archive/processed/meetings/granola/x.json.md",
           "## Notes\n\n# Next Steps\n\n- duplicate\n")

    inbox = "knowledge/assistant/inbox"
    _write(paths, f"{inbox}/a.md", "---\ntitle: a\napproved: false\n---\nbody\n")
    _write(paths, f"{inbox}/b.md", "---\ntitle: b\napproved: false\n---\nbody\n")
    _write(paths, f"{inbox}/c.md", "---\ntitle: c\napproved: true\n---\nbody\n")

    reports = "knowledge/notes/dreams/reports"
    _write(paths, f"{reports}/2026-10-09.md", "## For a human\n\n1. **Old ask.**\n")
    _write(paths, f"{reports}/2026-10-10.md", (
        "# r\n\n## For a human\n\nTwo things:\n\n1. **Rotate the token.** It was\n"
        "   readable.\n2. **Decide** the fleet version.\n\nAlso outstanding.\n\n## Links\n\n- y\n"
    ))
    _write(paths, "knowledge/notes/dreams/questions.md", (
        "---\ntitle: Open questions\ndreamed: 2026-10-10\n---\n\n# Open questions\n\n"
        "## Closed or corrected this pass\n\n## New — the token is readable\n\n- detail\n\n"
        "## Carried and escalated — meeting filing\n\n## Carried — brain\n\n"
        "### New — not a level-2 heading\n\n## Links\n"
    ))
    # knowledge/index is generated and never scanned.
    _write(paths, "knowledge/index/Now.md", "## Next\n- from the index\n")


def test_collect_sections(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _populate(paths)
    loops = collect(paths, TODAY)

    assert [n.rel_path for n in loops.proposed] == [
        "knowledge/projects/brain/decisions/0001-use-proposed.md"
    ]

    # Newest log WITH a Next section, its last Next heading; newest first.
    brain, chat = loops.project_next
    assert brain.source == "knowledge/projects/brain/log/2026-09-20"
    assert brain.date == "2026-09-20" and brain.label == "brain"
    assert brain.items == ("wrapped item that continues here", "second item")
    assert chat.label == "worksearch"
    assert chat.items == ("Submit the application after review.",)

    interview, site = loops.meeting_next
    assert interview.label == "Interview" and interview.date == "2026-10-06"
    assert interview.items == ("**Call the co-founder**", "Prepare", "Send the CV")
    assert site.date == "2026-09-16" and site.items == ("Złożyć zamówienie",)

    assert loops.awaiting_approval == 2

    assert loops.for_human is not None
    assert loops.for_human.source == "knowledge/notes/dreams/reports/2026-10-10"
    assert loops.for_human.item_count == 2
    assert loops.for_human.text.startswith("Two things:")
    assert "1. **Rotate the token.** It was\n   readable." in loops.for_human.text
    assert loops.for_human.text.endswith("Also outstanding.")

    assert loops.questions is not None
    assert loops.questions.date == "2026-10-10"
    assert loops.questions.items == (
        "New — the token is readable",
        "Carried and escalated — meeting filing",
        "Carried — brain",
    )
    assert loops.item_count == 1 + 3 + 4 + 2 + 2 + 3


def test_render_links_dates_and_exclusions(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _populate(paths)
    body = render(collect(paths, TODAY))

    assert "15 open item(s)." in body
    assert (
        "- [[knowledge/projects/brain/decisions/0001-use-proposed|0001 — Use proposed]]"
        " — brain, 2026-10-01"
    ) in body
    assert "### [[knowledge/projects/brain/log/2026-09-20|brain]] — 2026-09-20" in body
    assert "### [[knowledge/meetings/2026/2026-10-06-interview|Interview]] — 2026-10-06" in body
    assert "## Awaiting your approval (2)" in body
    assert "[[knowledge/index/Memory inbox.base|Memory inbox]]" in body
    assert "From [[knowledge/notes/dreams/reports/2026-10-10|dream report 2026-10-10]]:" in body
    assert "### [[knowledge/notes/dreams/questions|Open questions]] — 2026-10-10" in body
    for excluded in (
        "superseded within the file", "old brain item", "fenced item", "quoted item",
        "not a loop", "stale", "duplicate", "from the index", "Old ask", "a fact",
        "Chat with meeting transcript", "Use accepted", "old-decision", "- no\n",
        "Closed or corrected", "not a level-2 heading", "nested detail",
    ):
        assert excluded not in body, excluded


def test_empty_vault_renders_none(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    loops = collect(paths, TODAY)
    assert loops.item_count == 0
    body = render(loops)
    assert body.count("_(none)_") == 5
    assert "## Awaiting your approval (0)" in body


def test_meeting_window_follows_today(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _populate(paths)
    later = collect(paths, date(2026, 11, 10))
    assert [g.label for g in later.meeting_next] == ["Interview"]


def test_rebuild_is_deterministic_and_keeps_hand_tail(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    _populate(paths)
    target = paths.knowledge_index / "Open Loops.md"

    assert rebuild_open_loops(paths, TODAY) == (True, 15)
    first = target.read_text(encoding="utf-8")
    assert "type: dashboard" in first
    assert "python -m ingest_lib.open_loops" in first
    assert "metadata/index.jsonl" not in first

    target.write_text(first.replace("_(Your hand-written", "kept\n_(Your hand-written"),
                      encoding="utf-8")
    edited = target.read_text(encoding="utf-8")
    # Same inputs (and a different today inside the same window): no rewrite.
    assert rebuild_open_loops(paths, date(2026, 10, 12)) == (False, 15)
    assert target.read_text(encoding="utf-8") == edited
    assert render(collect(paths, TODAY)) == render(collect(paths, TODAY))

    _decision(paths, "knowledge/projects/brain/decisions/0003-use-proposed.md", "proposed")
    assert rebuild_open_loops(paths, TODAY) == (True, 16)
    assert "kept\n" in target.read_text(encoding="utf-8")


def test_cli_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _vault(tmp_path)
    _populate(paths)
    monkeypatch.setattr(open_loops, "default_paths", lambda: paths)

    assert open_loops.main(["--dry-run", "--today", "2026-10-11"]) == 0
    assert "## Proposed decisions (1)" in capsys.readouterr().out
    assert not (paths.knowledge_index / "Open Loops.md").exists()

    assert open_loops.main(["--today", "2026-10-11"]) == 0
    assert "15 item(s), note written" in capsys.readouterr().out
    assert (paths.knowledge_index / "Open Loops.md").is_file()


def _meeting(paths: VaultPaths, name: str, body: str) -> None:
    _write(paths, f"knowledge/meetings/2026/{name}.md",
           f"---\ntitle: {name}\ndate: '{name[:10]}'\n---\n\n{body}")


def test_meeting_heading_must_be_exactly_next_steps(tmp_path: Path) -> None:
    """F-1: a combined heading ("Role Fit, Salary, and Next Steps") is
    background, not actions; only a heading that IS the phrase counts."""
    paths = _vault(tmp_path)
    _meeting(paths, "2026-10-07-b", (
        "## Notes\n\n### Role Fit, Salary, and Next Steps\n\n- Salary: turned down £75k\n"
        "- Visa: sponsorship needed\n\n# Next Steps:\n\n- Send the offer\n\n"
        "## Actions\n\n_(empty)_\n"
    ))
    (meeting,) = collect(paths, TODAY).meeting_next
    assert meeting.items == ("Send the offer",)


def test_lead_in_item_keeps_its_nested_items(tmp_path: Path) -> None:
    """F-2: a "…:" lead-in bullet carries its nested items instead of
    rendering as a bare 'Next steps:'."""
    paths = _vault(tmp_path)
    _meeting(paths, "2026-10-08-d", (
        "## Notes\n\n### Next steps\n\n"
        "- Next steps:\n  - Nicholas sends the take-home\n  - Culture-fit with Matteo\n"
        "- 4 stages total:\n\n  1. Recruiter\n  2. Technical\n"
    ))
    (meeting,) = collect(paths, TODAY).meeting_next
    assert meeting.items == (
        "Next steps:\n  - Nicholas sends the take-home\n  - Culture-fit with Matteo",
        "4 stages total:\n  - Recruiter\n  - Technical",
    )
    body = render(collect(paths, TODAY))
    assert "- Next steps:\n  - Nicholas sends the take-home\n  - Culture-fit with Matteo\n" in body


def test_multiline_comments_checked_items_and_h1_runaway(tmp_path: Path) -> None:
    """F-3: a multi-line HTML comment is skipped, a checked item is not open,
    and an H1 next-steps heading inside ## Notes stops at ## Decisions."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/p/log/2026-10-10.md", (
        "# s\n\n## Next\n<!--\n- secret draft item inside a comment\n-->\n"
        "- [x] already done\n- [X] also done\n- [ ] still open\n"
    ))
    _meeting(paths, "2026-10-06-a", (
        "# A\n\n## Agenda\n\n_(empty)_\n\n## Notes\n\n# Next Steps\n\n- **Call Patrick**\n\n"
        "## Decisions\n\n_(empty)_\n\n## Actions\n\n- **Send the deck**\n\n"
        "## Links\n\n- Attendee: [[knowledge/people/x]]\n- Source: [[archive/raw/meetings/a]]\n"
    ))
    loops = collect(paths, TODAY)
    (log,) = loops.project_next
    assert log.items == ("[ ] still open",)
    (meeting,) = loops.meeting_next
    assert meeting.items == ("**Call Patrick**", "**Send the deck**")


def test_comment_opener_inside_fence_is_code(tmp_path: Path) -> None:
    """F-3: ``<!--`` inside a fence starts no comment; outside fences a
    multi-line comment is still stripped."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/p/log/2026-10-10.md", (
        "## Next\n```\nfoo <!-- \n```\n- real item\n<!-- one\n- hidden\n-->\n"
        "- kept <!-- inline --> text\n## Other\n-->\n- later\n"
    ))
    (log,) = collect(paths, TODAY).project_next
    assert log.items == ("real item", "kept  text")


def test_all_checked_section_has_no_items(tmp_path: Path) -> None:
    """F-3: checked items never resurface through the paragraph fallback."""
    paths = _vault(tmp_path)
    _write(paths, "knowledge/projects/p/log/2026-10-09.md", "## Next\n- older open item\n")
    _write(paths, "knowledge/projects/p/log/2026-10-10.md", "## Next\n- [x] done\n- [X] also done\n")
    _write(paths, "knowledge/projects/q/log/2026-10-10.md", "## Next\nA paragraph\n  - [x] nested done\n")
    groups = {g.label: g.items for g in collect(paths, TODAY).project_next}
    assert groups == {"p": ("older open item",), "q": ("A paragraph",)}
