"""Tests for ingest_lib.relations — node-id canonicalisation: target-string
normalisation, note-path round-tripping, and the one definition of node
identity (canonical_entity_id / owning_entity_id / live_entity_id)."""
from __future__ import annotations

from pathlib import Path

from ingest_lib.config import paths_for_root
from ingest_lib.relations import (
    EntityInfo,
    Relation,
    canonical_entity_id,
    entity_notes,
    folder_overview_id,
    live_entity_id,
    node_id_for_note,
    normalize_target,
    note_path_for_node,
    owning_entity_id,
)



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
# target / node-id canonicalisation
# ---------------------------------------------------------------------------

def test_normalize_target_collapses_all_tolerated_forms() -> None:
    for raw in (
        "people/x",
        "knowledge/people/x",
        "people/x.md",
        "knowledge/people/x.md",
        "[[knowledge/people/x]]",
        "[[knowledge/people/x|Display Name]]",
        "  people/x  ",
        "/knowledge/people/x",
    ):
        assert normalize_target(raw) == "people/x", raw


def test_node_id_round_trip() -> None:
    assert node_id_for_note("knowledge/people/anna-kowalska.md") == "people/anna-kowalska"
    assert note_path_for_node("people/anna-kowalska") == "knowledge/people/anna-kowalska.md"
    for node in ("people/anna", "organisations/acme", "meetings/2026/2026-06-12-kern-call"):
        assert node_id_for_note(note_path_for_node(node)) == node


def test_normalize_target_without_vault_root_is_pure_string_work(tmp_path: Path) -> None:
    """F9: the folder-overview resolution is opt-in. Callers that pass no
    ``vault_root`` (Relation.__post_init__, query filters, the MCP tools)
    keep the old behaviour even when the overview note exists on disk."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")

    assert normalize_target("projects/server") == "projects/server"
    assert normalize_target("[[knowledge/projects/server]]") == "projects/server"


def test_normalize_target_resolves_short_form_to_folder_overview(tmp_path: Path) -> None:
    """F9: a project is a folder, so ``projects/server`` names no note.
    With a vault_root it resolves to the overview that does exist."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")

    for raw in (
        "projects/server",
        "knowledge/projects/server",
        "[[knowledge/projects/server]]",
        "projects/server.md",
    ):
        assert normalize_target(raw, vault_root=root) == "projects/server/server", raw
    # Already canonical: returned untouched (the flat-note check matches it).
    assert normalize_target("projects/server/server", vault_root=root) == "projects/server/server"


def test_normalize_target_leaves_short_form_alone_when_flat_note_exists(tmp_path: Path) -> None:
    """A real note at the short path wins outright — the folder overview is
    only consulted when nothing sits at ``knowledge/<node_id>.md``."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/rex.md", "---\ntitle: Rex\n---\n")
    # A same-named folder overview also exists: the flat note still wins.
    _write(root, "knowledge/projects/rex/rex.md", "---\ntitle: Rex folder\n---\n")

    assert normalize_target("projects/rex", vault_root=root) == "projects/rex"


def test_normalize_target_unresolvable_target_is_returned_unchanged(tmp_path: Path) -> None:
    """Neither a flat note nor a folder overview: no guessing — the id comes
    back as written so the caller reports the miss."""
    root = _vault(tmp_path)
    (root / "knowledge/projects/ghost").mkdir(parents=True, exist_ok=True)
    _write(root, "knowledge/projects/ghost/notes.md", "---\ntitle: Notes\n---\n")

    assert normalize_target("projects/ghost", vault_root=root) == "projects/ghost"
    assert normalize_target("people/nobody", vault_root=root) == "people/nobody"


def test_normalize_target_never_resolves_a_traversal(tmp_path: Path) -> None:
    """A ``..`` target is not a valid node id, so resolution is skipped
    entirely — it must never be joined onto the vault root, and it must come
    back unchanged for ``is_valid_node_id`` downstream to reject."""
    root = _vault(tmp_path)
    _write(root, "AGENTS.md", "ORIGINAL\n")
    (root / "knowledge/AGENTS").mkdir(parents=True, exist_ok=True)
    _write(root, "knowledge/AGENTS/AGENTS.md", "decoy\n")

    for raw in ("../AGENTS", "projects/../../AGENTS", "people/./x"):
        assert normalize_target(raw, vault_root=root) == raw, raw
    # A single segment is a concept slug, not an entity: never expanded.
    assert normalize_target("server", vault_root=root) == "server"


def test_folder_overview_id_is_purely_textual() -> None:
    assert folder_overview_id("projects/server") == "projects/server/server"
    assert folder_overview_id("projects/a/b") == "projects/a/b/b"


def test_relation_dataclass_normalises_target_on_construction() -> None:
    r = Relation(rel="works_at", target="[[knowledge/organisations/acme]]")
    assert r.target == "organisations/acme"


# ---------------------------------------------------------------------------
# canonical_entity_id / live_entity_id — the one definition of node identity
# ---------------------------------------------------------------------------

def test_canonical_entity_id_flat_entity_notes(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    assert canonical_entity_id(
        "knowledge/people/anna-kowalska.md", vault_root=root
    ) == "people/anna-kowalska"
    assert canonical_entity_id(
        "knowledge/organisations/acme.md", vault_root=root
    ) == "organisations/acme"
    # Negative: a note nested BELOW the flat folder is not a node of its own.
    assert canonical_entity_id(
        "knowledge/people/old/anna-kowalska.md", vault_root=root
    ) is None


def test_canonical_entity_id_meetings_need_a_year_folder(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    assert canonical_entity_id(
        "knowledge/meetings/2026/2026-06-12-kern-call.md", vault_root=root
    ) == "meetings/2026/2026-06-12-kern-call"
    # Negative: the year folder is part of the layout, not decoration.
    assert canonical_entity_id(
        "knowledge/meetings/kern-call.md", vault_root=root
    ) is None
    assert canonical_entity_id(
        "knowledge/meetings/drafts/kern-call.md", vault_root=root
    ) is None


def test_canonical_entity_id_project_overview_and_its_notes(tmp_path: Path) -> None:
    """A project's dated log entry belongs to the project's overview — it is
    not a graph node of its own (the bug that made dated log notes
    entities). A CURATED note beside the overview is the opposite case:
    AGENTS.md says it keeps its own path, and 36 of the live vault's stored
    relation targets are exactly such notes, so folding it into the project
    makes every consumer name the wrong node."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")
    _write(root, "knowledge/projects/server/log/2026-09-03.md", "# log\n")
    _write(root, "knowledge/projects/server/stack.md", "# stack\n")

    assert canonical_entity_id(
        "knowledge/projects/server/server.md", vault_root=root
    ) == "projects/server/server"
    assert canonical_entity_id(
        "knowledge/projects/server/log/2026-09-03.md", vault_root=root
    ) == "projects/server/server"
    assert canonical_entity_id(
        "knowledge/projects/server/stack.md", vault_root=root
    ) == "projects/server/stack"
    # Deeper than any documented shape, and not a log entry: no node.
    _write(root, "knowledge/projects/server/notes/deep/x.md", "# x\n")
    assert canonical_entity_id(
        "knowledge/projects/server/notes/deep/x.md", vault_root=root
    ) is None


def test_canonical_entity_id_flat_project_note(tmp_path: Path) -> None:
    """AGENTS.md tolerates a flat knowledge/projects/<slug>.md and one is
    live, so it is a node; the SAME two-segment id with a folder behind it
    instead is the short form normalize_target rewrites — no note, no node."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/zymbly-role.md", "---\ntitle: Zymbly\n---\n")
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")

    assert canonical_entity_id(
        "knowledge/projects/zymbly-role.md", vault_root=root
    ) == "projects/zymbly-role"
    assert canonical_entity_id("knowledge/projects/server.md", vault_root=root) is None


def test_canonical_entity_id_chats_are_folder_backed_like_projects(tmp_path: Path) -> None:
    """knowledge/chats/<slug>/ follows the project folder shape (AGENTS.md,
    "Folder-backed entities"): the overview is the node, a dated log entry
    resolves to it, a curated note beside it keeps its own id, and the bare
    short form normalises to the overview. knowledge/personal/ is a
    free-form area like notes/ and owns no nodes at all."""
    root = _vault(tmp_path)
    _write(root, "knowledge/chats/worksearch/worksearch.md", "---\ntitle: WorkSearch\n---\n")
    _write(root, "knowledge/chats/worksearch/log/2026-08-30.md", "# log\n")
    _write(root, "knowledge/chats/worksearch/drive-engineer-application.md", "# app\n")
    _write(root, "knowledge/personal/leases/flat-2026.md", "# lease\n")

    assert canonical_entity_id(
        "knowledge/chats/worksearch/worksearch.md", vault_root=root
    ) == "chats/worksearch/worksearch"
    assert canonical_entity_id(
        "knowledge/chats/worksearch/log/2026-08-30.md", vault_root=root
    ) == "chats/worksearch/worksearch"
    assert canonical_entity_id(
        "knowledge/chats/worksearch/drive-engineer-application.md", vault_root=root
    ) == "chats/worksearch/drive-engineer-application"
    assert owning_entity_id(
        "knowledge/chats/worksearch/drive-engineer-application.md", vault_root=root
    ) == "chats/worksearch/worksearch"
    assert normalize_target("chats/worksearch", vault_root=root) == "chats/worksearch/worksearch"
    assert canonical_entity_id(
        "knowledge/personal/leases/flat-2026.md", vault_root=root
    ) is None


def test_owning_entity_id_attributes_a_projects_own_notes_to_the_project(
    tmp_path: Path,
) -> None:
    """The other identity question: which entity does prose written in this
    note belong to. A project's curated note is its own node but a document
    ABOUT the project, so its content is attributed to the overview;
    everything else owns what it says."""
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/server/server.md", "---\ntitle: Server\n---\n")
    _write(root, "knowledge/projects/server/stack.md", "# stack\n")
    _write(root, "knowledge/projects/server/log/2026-09-03.md", "# log\n")
    _write(root, "knowledge/people/anna-kowalska.md", "---\ntitle: Anna\n---\n")

    for rel in (
        "knowledge/projects/server/server.md",
        "knowledge/projects/server/stack.md",
        "knowledge/projects/server/log/2026-09-03.md",
    ):
        assert owning_entity_id(rel, vault_root=root) == "projects/server/server", rel
    assert owning_entity_id(
        "knowledge/people/anna-kowalska.md", vault_root=root
    ) == "people/anna-kowalska"
    assert owning_entity_id("knowledge/concepts/graphs.md", vault_root=root) is None
    # No overview to be attributed to: the note keeps its own content.
    _write(root, "knowledge/projects/orphan/stack.md", "# stack\n")
    assert owning_entity_id(
        "knowledge/projects/orphan/stack.md", vault_root=root
    ) == "projects/orphan/stack"


def test_canonical_entity_id_project_folder_without_an_overview(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/orphan/log/2026-09-03.md", "# log\n")
    assert canonical_entity_id(
        "knowledge/projects/orphan/log/2026-09-03.md", vault_root=root
    ) is None


def test_canonical_entity_id_shared_folder_is_not_a_project(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(root, "knowledge/projects/shared/shared.md", "---\ntitle: Shared\n---\n")
    _write(root, "knowledge/projects/shared/hiring.md", "# hiring\n")
    assert canonical_entity_id(
        "knowledge/projects/shared/hiring.md", vault_root=root
    ) is None
    assert canonical_entity_id(
        "knowledge/projects/shared/shared.md", vault_root=root
    ) is None


def test_canonical_entity_id_non_entity_areas(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    for rel in (
        "knowledge/concepts/graphs.md",
        "knowledge/index/people.md",
        "knowledge/notes/dreams/questions.md",
        "knowledge/assistant/inbox/2026-09-03-fact.md",
        "knowledge/university/ucl.md",
        "knowledge/research/idea.md",
        "archive/processed/paper.md",
        "knowledge/projects/server.md",   # flat form with no note behind it
    ):
        assert canonical_entity_id(rel, vault_root=root) is None, rel


def test_entity_notes_reads_superseded_by(tmp_path: Path) -> None:
    root = _vault(tmp_path)
    _write(
        root, "knowledge/people/annakowalskaexamplecom.md",
        "---\ntitle: anna@example.com\n"
        "superseded_by: knowledge/people/anna-kowalska\n---\n",
    )
    _write(root, "knowledge/people/anna-kowalska.md", "---\ntitle: Anna\n---\n")

    entities = entity_notes(paths_for_root(root))
    dup = entities["people/annakowalskaexamplecom"]
    assert dup.superseded_by == "people/anna-kowalska"
    assert entities["people/anna-kowalska"].superseded_by is None


def _entity(node_id: str, superseded_by: str | None = None) -> EntityInfo:
    return EntityInfo(
        node_id=node_id,
        rel_path=f"knowledge/{node_id}.md",
        title=node_id,
        type="",
        aliases=(),
        relations=(),
        updated="",
        superseded_by=superseded_by,
    )


def test_live_entity_id_follows_a_chain_and_is_cycle_safe() -> None:
    entities = {
        "people/a": _entity("people/a", "people/b"),
        "people/b": _entity("people/b", "people/c"),
        "people/c": _entity("people/c"),
    }
    assert live_entity_id("people/a", entities) == "people/c"
    assert live_entity_id("people/c", entities) == "people/c"
    # Not an entity at all.
    assert live_entity_id("people/nobody", entities) is None
    # A pointer at a node that isn't an entity stops at the last live node.
    dangling = {"people/a": _entity("people/a", "people/gone")}
    assert live_entity_id("people/a", dangling) == "people/a"


def test_live_entity_id_never_returns_a_dangling_successor() -> None:
    """A superseded_by pointing at a note that does not exist (or that
    parsing dropped as an invalid id) must resolve to the last node that
    really is an entity — handing back the dangling id would let a merged
    duplicate grow edges on a target nothing can resolve."""
    entities = {
        "people/a": _entity("people/a", "people/b"),
        "people/b": _entity("people/b", "people/nowhere"),
    }
    for start in ("people/a", "people/b"):
        live = live_entity_id(start, entities)
        assert live == "people/b"
        assert live in entities


def test_live_entity_id_survives_a_superseded_cycle() -> None:
    entities = {
        "people/a": _entity("people/a", "people/b"),
        "people/b": _entity("people/b", "people/a"),
    }
    assert live_entity_id("people/a", entities) in {"people/a", "people/b"}
    assert live_entity_id("people/b", entities) in {"people/a", "people/b"}
