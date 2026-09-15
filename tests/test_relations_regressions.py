"""Tests for ingest_lib.relations — pinned regressions AUD-028 / AUD-029 /
AUD-030 / AUD-035 / AUD-096 (frontmatter-preserving splices, fenced-code
safety in the log appender, YAML-datetime as_of bounds, write-path node-id
invariants enforced on read, and AGENTS.md as the vocabulary source)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.notes import _split_frontmatter
from ingest_lib.relations import (
    RELATION_VOCAB,
    Relation,
    append_fact_to_log,
    normalize_target,
    parse_relations,
    upsert_relation_in_text,
)



def _body(text: str) -> str:
    return _split_frontmatter(text)[1]


def _rels(fm: dict[str, object]) -> list[dict[str, object]]:
    """Narrow ``fm["relations"]`` for assertions: YAML round-trips hand back
    plain ``object`` values, and the asserts double as shape checks."""
    raw = fm["relations"]
    assert isinstance(raw, list)
    entries: list[dict[str, object]] = []
    for entry in raw:
        assert isinstance(entry, dict)
        entries.append({str(key): value for key, value in entry.items()})
    return entries


def _vault(tmp_path: Path) -> Path:
    for sub in (
        "knowledge/people", "knowledge/organisations", "knowledge/projects",
        "knowledge/meetings", "knowledge/index", "knowledge/concepts",
        "metadata", "inbox", "logs",
        "archive/raw", "archive/processed", "archive/failed",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")



# ---------------------------------------------------------------------------
# AUD-028 / AUD-029 / AUD-030 / AUD-035 / AUD-096
# ---------------------------------------------------------------------------

def test_upsert_leaves_every_other_frontmatter_key_byte_for_byte() -> None:
    # AUD-028: the old whole-fence yaml.safe_dump round-trip rewrote every
    # OTHER key on a relation write — `created: 2026-06-09T09:50:42Z` came
    # back as `2026-06-09 09:50:42+00:00`, flow lists exploded to block
    # style, quotes normalised and a long title folded onto a continuation
    # line. Only the relations block may change.
    note = (
        "---\n"
        "title: A very long title that goes on and on and on and really must "
        "not be folded onto a continuation line by the dumper\n"
        "created: 2026-06-09T09:50:42Z\n"
        "updated: 2026-06-09T09:50:42Z\n"
        "topics: [alpha, beta]\n"
        'type: "person"\n'
        "# a hand-written comment\n"
        "---\n"
        "# Body\n"
    )
    new_text, action = upsert_relation_in_text(
        note, Relation(rel="works_at", target="organisations/acme")
    )
    assert action == "added"
    for line in note.splitlines():
        if line != "---":
            assert line in new_text.splitlines()
    assert "created: 2026-06-09T09:50:42Z" in new_text
    assert "topics: [alpha, beta]" in new_text
    assert '# a hand-written comment' in new_text
    assert _rels(_split_frontmatter(new_text)[0])[0]["target"] == "organisations/acme"
    assert _body(new_text) == "# Body\n"


def test_upsert_rewrites_only_the_relations_block_of_an_existing_list() -> None:
    # The splice must replace the whole existing block (key line plus its
    # indented entries), never append a second `relations:` key.
    note = (
        "---\n"
        "title: Anna\n"
        "relations:\n"
        "  - rel: works_at\n"
        "    target: organisations/acme\n"
        "    valid_from: \"2025-01-01\"\n"
        "aliases: [Ania]\n"
        "---\n"
        "# Anna\n"
    )
    new_text, action = upsert_relation_in_text(
        note, Relation(rel="member_of", target="organisations/initech")
    )
    assert action == "added"
    assert new_text.count("relations:") == 1
    assert "title: Anna\n" in new_text
    assert "aliases: [Ania]\n" in new_text
    entries = _rels(_split_frontmatter(new_text)[0])
    assert [(e["rel"], e["target"]) for e in entries] == [
        ("works_at", "organisations/acme"),
        ("member_of", "organisations/initech"),
    ]


def test_upsert_does_not_fold_a_long_value_in_a_preserved_entry() -> None:
    # width must be unbounded: an existing malformed entry is kept verbatim
    # (supersede, never delete), and folding its long prose value onto a
    # continuation line churns the file for no gain.
    prose = (
        "closed when Anna left, per the handover note of the twelfth of June, "
        "which nobody has since rewritten as a date"
    )
    note = (
        "---\n"
        "title: Anna\n"
        "relations:\n"
        "  - rel: works_at\n"
        "    target: organisations/acme\n"
        f"    valid_until: {prose}\n"
        "---\n"
        "# Anna\n"
    )
    new_text, _action = upsert_relation_in_text(
        note, Relation(rel="member_of", target="organisations/initech")
    )
    assert f"  valid_until: {prose}\n" in new_text


def test_append_fact_never_lands_inside_a_fenced_code_block() -> None:
    # AUD-029: a '#' comment inside a ``` fence ended the section, so the
    # bullet was inserted into the code block ahead of the real last fact.
    note = (
        "## Log\n"
        "\n"
        "- 2026-01-01 — a\n"
        "```sh\n"
        "# comment\n"
        "```\n"
        "- 2026-01-02 — b\n"
    )
    result = append_fact_to_log(note, "2026-01-03 — c")
    assert result == note + "- 2026-01-03 — c\n"


def test_append_fact_ignores_a_log_heading_inside_a_fence() -> None:
    note = (
        "# T\n"
        "\n"
        "~~~md\n"
        "## Log\n"
        "~~~\n"
        "\n"
        "## Log\n"
        "\n"
        "- 2026-01-01 — a\n"
    )
    result = append_fact_to_log(note, "2026-01-02 — b")
    assert result.endswith("- 2026-01-01 — a\n- 2026-01-02 — b\n")
    assert result.count("## Log") == 2


def test_append_fact_replaces_the_empty_placeholder() -> None:
    # The templates ship `_(empty)_`; the first real fact takes its place
    # rather than sitting under it (a dated section is not empty).
    note = "## Log\n\n<!-- guidance -->\n\n_(empty)_\n"
    result = append_fact_to_log(note, "2026-01-03 — c")
    assert "_(empty)_" not in result
    assert result == "## Log\n\n<!-- guidance -->\n\n- 2026-01-03 — c\n"


def test_append_fact_keeps_a_placeholder_in_another_section() -> None:
    note = "## Notes\n\n_(empty)_\n\n## Log\n\n_(empty)_\n"
    result = append_fact_to_log(note, "2026-01-03 — c")
    assert result == "## Notes\n\n_(empty)_\n\n## Log\n\n- 2026-01-03 — c\n"


def test_append_fact_second_fact_leaves_the_first_alone() -> None:
    once = append_fact_to_log("## Log\n\n_(empty)_\n", "2026-01-03 — c")
    twice = append_fact_to_log(once, "2026-01-04 — d")
    assert twice == "## Log\n\n- 2026-01-03 — c\n- 2026-01-04 — d\n"


def _bad_date_vault(tmp_path: Path, valid_from: str) -> VaultPaths:
    root = _vault(tmp_path)
    _write(
        root, "knowledge/people/anna.md",
        "---\ntitle: Anna\nrelations:\n"
        "  - rel: works_at\n    target: organisations/acme\n"
        f"    valid_from: {valid_from}\n---\n",
    )
    _write(root, "knowledge/organisations/acme.md", "---\ntitle: Acme\n---\n")
    return paths_for_root(root)


def test_query_relations_as_of_handles_a_yaml_datetime_bound(tmp_path: Path) -> None:
    # AUD-030: an unquoted `valid_from: 2025-03-01T00:00:00Z` loads as a
    # datetime whose str() is '2025-03-01 00:00:00+00:00'; comparing that
    # as raw text made it sort AFTER '2025-03-01' and the relation vanished
    # from an as_of query on its own start day.
    from ingest_lib.relations import query_relations
    paths = _bad_date_vault(tmp_path, "2025-03-01T00:00:00Z")
    hits = query_relations(paths, as_of="2025-03-01")
    assert [h.target for h in hits] == ["organisations/acme"]
    assert query_relations(paths, as_of="2025-02-28") == []


def test_query_relations_as_of_excludes_an_unparseable_bound(tmp_path: Path) -> None:
    from ingest_lib.relations import query_relations
    paths = _bad_date_vault(tmp_path, '"sometime in spring"')
    assert query_relations(paths, as_of="2025-03-01") == []
    # Without as_of the relation is still an open edge, unchanged.
    assert [h.target for h in query_relations(paths)] == ["organisations/acme"]


def test_query_relations_rejects_a_non_date_as_of(tmp_path: Path) -> None:
    from ingest_lib.relations import query_relations
    paths = _bad_date_vault(tmp_path, '"2025-03-01"')
    with pytest.raises(ValueError, match="as_of must be YYYY-MM-DD"):
        query_relations(paths, as_of="last spring")


@pytest.mark.parametrize(
    "target",
    ["acme", "assistant/PROFILE", "people/Łukasz", "people/../../archive/x"],
)
def test_parse_relations_rejects_targets_the_write_path_rejects(target: str) -> None:
    # AUD-035: node-id invariants were enforced on write (entity_upsert_relation
    # calls is_valid_node_id) but not on read, so a hand-edited note could put
    # a node the write path can never address into the graph.
    relations, problems = parse_relations(
        {"relations": [{"rel": "related_to", "target": target}]}
    )
    assert relations == []
    assert len(problems) == 1 and "invalid target node id" in problems[0]


def test_normalize_target_drops_a_heading_fragment() -> None:
    # A '#…' fragment names a heading inside a note, not a node.
    assert normalize_target("[[knowledge/people/anna.md#Log]]") == "people/anna"
    relations, problems = parse_relations(
        {"relations": [{"rel": "related_to", "target": "people/anna#Log"}]}
    )
    assert problems == []
    assert relations[0].target == "people/anna"


def test_agents_md_is_the_relation_vocabulary_source() -> None:
    # AUD-096: the vocabulary is hand-copied into several prose files with
    # no import. AGENTS.md is the documented source; this pins the code to it.
    agents_md = Path(__file__).resolve().parent.parent / "AGENTS.md"
    block = agents_md.read_text(encoding="utf-8").split("<!-- relation-vocab -->")[1]
    documented = set(re.findall(r"`([a-z_]+)`", block.split("<!-- /relation-vocab -->")[0]))
    assert documented == set(RELATION_VOCAB)
