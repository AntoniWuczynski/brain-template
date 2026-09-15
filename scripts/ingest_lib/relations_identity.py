"""Node-id identity for the typed relation graph.

Split out of :mod:`ingest_lib.relations` (AUD-121): the module docstring
there still owns the full picture. This slice is the "Identity" bullet —
an entity's node id is its ``knowledge/``-relative path without the
``.md`` extension (``people/anna-kowalska``). Node ids always contain a
``/`` (the subdirectory), so they can never collide with concept slugs,
which never do. A *folder-backed* entity — a project laid out as
``knowledge/projects/<slug>/`` with an overview note beside its curated
notes — is identified by that overview note's own path,
``projects/<slug>/<slug>``; the folder itself is not a node.
:func:`normalize_target` resolves the ambiguous short form
(``projects/<slug>``) to it when given a ``vault_root`` to check.
:func:`canonical_entity_id` is the single answer to "is this note a
graph node, and which" (a project's ``log/`` entry is not a node of its
own — it belongs to the project's overview — while a curated note beside
the overview keeps its own id, as AGENTS.md says it does), and
:func:`owning_entity_id` answers the neighbouring question "which entity
does prose written in this note belong to". Nothing should re-derive
either of them from a path prefix.

This module has no dependency on the rest of ``relations`` — the sibling
modules depend on it, not the other way round.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final


def normalize_target(raw: str, *, vault_root: Path | None = None) -> str:
    """Canonicalise a relation target to a node id (``people/x``).

    Tolerant of the forms humans (and earlier notes) actually write:
    ``knowledge/people/x``, ``people/x.md``, ``[[knowledge/people/x]]``,
    ``[[knowledge/people/x|Anna]]``, ``[[people/x#Log]]`` — all collapse
    to ``people/x``. A ``#…`` fragment names a heading INSIDE a note, not
    a node of its own, so it is dropped (F19).

    Pure string work by default. Pass ``vault_root`` (the vault root, not
    ``knowledge/``) to additionally resolve the ambiguous short form of a
    folder-backed entity against the filesystem: ``projects/server``
    becomes ``projects/server/server`` when ``knowledge/projects/server.md``
    does NOT exist but ``knowledge/projects/server/server.md`` does. A
    flat note always wins, so an id that already names a real note is
    returned untouched, and an id resolving to neither is left alone for
    the caller to report.
    """
    t = raw.strip()
    if t.startswith("[[") and t.endswith("]]"):
        t = t[2:-2].strip()
    t = t.split("|", 1)[0].strip()  # wikilink display alias
    t = t.split("#", 1)[0].strip()  # heading fragment
    if t.endswith(".md"):
        t = t[: -len(".md")]
    t = t.lstrip("/")
    if t.startswith("knowledge/"):
        t = t[len("knowledge/") :]
    if vault_root is None:
        return t
    return _resolve_folder_entity(t, vault_root)


# A canonical node id is one or more path segments of [a-z0-9._-] joined by
# single slashes, e.g. ``people/anna-kowalska`` or ``meetings/2026/2026-06-12-x``.
# It must NOT contain a ``.``/``..`` path segment: those survive
# ``normalize_target`` unchanged and, once expanded to ``knowledge/<id>.md``,
# resolve OUTSIDE the knowledge/ tree (``people/../../archive/...``), which
# would let a relation target or meeting attendee escape the write allowlist.
_NODE_ID_RE = re.compile(r"^(?![./])[a-z0-9._/-]+$")


def is_valid_node_id(node_id: str) -> bool:
    """True iff ``node_id`` is a safe, canonical node id (see _NODE_ID_RE).

    Rejects empty ids, leading ``/`` or ``.``, and any ``.``/``..`` segment,
    so a caller can never smuggle a path traversal through a node id.
    """
    if not node_id or "/" not in node_id:
        return False
    segments = node_id.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        return False
    return bool(_NODE_ID_RE.match(node_id))


def node_id_for_note(rel_path: str) -> str:
    """Vault-relative note path -> node id.

    ``knowledge/people/anna.md`` -> ``people/anna``.
    """
    return normalize_target(rel_path)


def note_path_for_node(node_id: str) -> str:
    """Node id -> vault-relative note path (inverse of node_id_for_note).

    ``people/anna`` -> ``knowledge/people/anna.md``.
    """
    return f"knowledge/{node_id}.md"


def folder_overview_id(node_id: str) -> str:
    """The node id of ``node_id``'s folder overview note.

    ``projects/server`` -> ``projects/server/server``. Purely textual —
    says nothing about whether either note exists.
    """
    return f"{node_id}/{node_id.rsplit('/', 1)[-1]}"


def _resolve_folder_entity(node_id: str, vault_root: Path) -> str:
    """Short form -> folder-overview node id, when that is what exists.

    A project entity lives in a FOLDER (``knowledge/projects/<slug>/``)
    whose overview note is ``<slug>.md`` inside it, so its honest id is
    ``projects/<slug>/<slug>``. Notes (and the brain-project-note skill,
    historically) also wrote the bare ``projects/<slug>``, which expands
    to ``knowledge/projects/<slug>.md`` — a path with no note behind it,
    so a relation or a ``promote.target`` using it can never resolve.

    Resolution order, both by existence, never by guesswork:

    1. a flat note at ``knowledge/<node_id>.md`` wins outright — a real
       note keeps its own id (``projects/forward-deployed-engineer.md``);
    2. otherwise the folder overview ``knowledge/<node_id>/<slug>.md``,
       returned as ``<node_id>/<slug>``;
    3. otherwise ``node_id`` unchanged, so the caller reports the miss.

    Ids that aren't valid node ids (``..`` traversal, no ``/``) are
    returned untouched WITHOUT any filesystem access: nothing here may
    build a path out of a segment ``is_valid_node_id`` would reject.
    """
    if not is_valid_node_id(node_id):
        return node_id
    if (vault_root / note_path_for_node(node_id)).is_file():
        return node_id
    overview = folder_overview_id(node_id)
    if (vault_root / note_path_for_node(overview)).is_file():
        return overview
    return node_id


_FLAT_ENTITY_DIRS: Final[tuple[str, ...]] = ("people", "organisations")
# Folder-backed entity roots (AGENTS.md, "Folder-backed entities"): a
# <slug>/ folder whose overview note is <slug>.md inside it. projects/ and
# chats/ share the shape; only projects/ has a cross-project shared/ folder.
_FOLDER_ENTITY_DIRS: Final[tuple[str, ...]] = ("projects", "chats")
_YEAR_RE = re.compile(r"^\d{4}$")
# The cross-project folder under knowledge/projects/ holds topics shared by
# several projects (AGENTS.md, "Folder-backed entities"); it is not a project
# and owns no overview.
_SHARED_PROJECT_DIR: Final[str] = "shared"
# A project's dated diary entries: knowledge/projects/<slug>/log/<date>.md.
_PROJECT_LOG_DIR: Final[str] = "log"


def _folder_overview(area: str, slug: str, *, vault_root: Path) -> str | None:
    """``<area>/<slug>/<slug>`` when that overview note is on disk."""
    overview = folder_overview_id(f"{area}/{slug}")
    if (vault_root / note_path_for_node(overview)).is_file():
        return overview
    return None


def _project_overview(slug: str, *, vault_root: Path) -> str | None:
    """``projects/<slug>/<slug>`` when that overview note is on disk."""
    return _folder_overview("projects", slug, vault_root=vault_root)


def canonical_entity_id(rel_path: str, *, vault_root: Path) -> str | None:
    """The graph node a knowledge note belongs to, or None if it is not one.

    THE single definition of "is this note an entity node, and which" —
    every consumer (graph builders, the wikilink pass, sweep, the dream
    pass) must ask here rather than re-deriving it from a path prefix, which
    is how dated project log notes ended up as nodes of their own.

    ``rel_path`` is vault-relative and includes ``.md``
    (``knowledge/people/anna.md``). Given the layout in AGENTS.md
    ("Entity notes and typed relations", "Folder-backed entities"):

    - a flat note directly under ``knowledge/people/`` or
      ``knowledge/organisations/`` -> its own node id;
    - a meeting at ``knowledge/meetings/<YYYY>/<note>.md`` -> its own node id;
    - ``knowledge/projects/<slug>/<slug>.md`` -> ``projects/<slug>/<slug>``,
      and ``knowledge/chats/<slug>/<slug>.md`` -> ``chats/<slug>/<slug>`` —
      chats/ is the same folder shape (the rules below say "project" but
      apply to every root in ``_FOLDER_ENTITY_DIRS``);
    - a curated note beside it, ``knowledge/projects/<slug>/<note>.md`` ->
      its OWN id: AGENTS.md is explicit that "curated notes beside it keep
      their own path (``projects/fps/stack-and-architecture``)", and 36 of
      the live vault's stored relation targets are exactly such notes;
    - a dated diary entry, ``knowledge/projects/<slug>/log/<note>.md`` ->
      the project's overview id, because a log entry is not a node of its
      own — provided the overview note exists (checked on disk, the way
      :func:`_resolve_folder_entity` checks it); otherwise None, because
      there is no node to speak for it;
    - a flat ``knowledge/projects/<slug>.md`` -> its own id when that file
      really is on disk (AGENTS.md tolerates the shape, and one such
      project is live). The same two-segment id with a FOLDER behind it
      instead is the short form ``normalize_target`` rewrites, and no note
      at all, so it stays None;
    - ``knowledge/projects/shared/…`` -> None (the cross-project folder is
      not a project — AGENTS.md, "Folder-backed entities");
    - anything nested deeper under a project than those documented shapes,
      and everything else — concepts, index, notes, personal, assistant,
      university, research, anything outside ``knowledge/`` — None.
    """
    node_id = node_id_for_note(rel_path)
    if not rel_path.strip().lstrip("/").startswith("knowledge/"):
        return None
    if not is_valid_node_id(node_id):
        return None
    parts = node_id.split("/")
    if len(parts) == 2 and parts[0] in _FLAT_ENTITY_DIRS:
        return node_id
    if len(parts) == 3 and parts[0] == "meetings" and _YEAR_RE.match(parts[1]):
        return node_id
    if parts[0] not in _FOLDER_ENTITY_DIRS:
        return None
    if len(parts) == 2:
        return node_id if (vault_root / note_path_for_node(node_id)).is_file() else None
    if parts[0] == "projects" and parts[1] == _SHARED_PROJECT_DIR:
        return None
    if len(parts) == 4 and parts[2] == _PROJECT_LOG_DIR:
        return _folder_overview(parts[0], parts[1], vault_root=vault_root)
    if len(parts) == 3:
        return node_id
    return None


def owning_entity_id(rel_path: str, *, vault_root: Path) -> str | None:
    """The entity a note's CONTENT is attributed to, or None.

    :func:`canonical_entity_id` answers "which node IS this note" — the
    identity question a link target, a ``promote.target`` or an evidence
    hint asks. This answers the other one: "which entity does something
    written in this note belong to". They differ for exactly one shape, a
    project's own notes: ``projects/kern/stack.md`` is a node in its own
    right, but a sentence written there is a statement about the PROJECT,
    so a pass mining that prose attributes it to ``projects/kern/kern``
    (with the note itself kept as provenance). Everything else — people,
    organisations, meetings, a flat project note — owns its own content.

    A project note whose folder has no overview has no project to be
    attributed to, so it keeps its own id rather than losing its content.
    """
    node_id = canonical_entity_id(rel_path, vault_root=vault_root)
    if node_id is None:
        return None
    parts = node_id.split("/")
    if len(parts) == 3 and parts[0] in _FOLDER_ENTITY_DIRS:
        return _folder_overview(parts[0], parts[1], vault_root=vault_root) or node_id
    return node_id
