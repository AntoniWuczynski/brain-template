"""Typed, dated relationships between knowledge entities.

Entity notes (people, organisations, projects, meetings, …) can declare
how they relate to each other in frontmatter::

    relations:
      - rel: works_at
        target: organisations/acme
        valid_from: "2025-03-01"      # optional, YYYY-MM-DD
        valid_until: ""               # optional; absent/empty = currently valid
        source: knowledge/meetings/2026/2026-06-12-kern-call   # provenance

This module owns three things, all deterministic and LLM-free:

- **Parsing**: a tolerant reader (:func:`parse_relations`) that skips
  malformed entries with a problem string instead of raising, and a vault
  scanner (:func:`entity_notes`) that turns every hand-edited knowledge
  note into an :class:`EntityInfo`.
- **Identity**: an entity's node id is its ``knowledge/``-relative path
  without the ``.md`` extension (``people/anna-kowalska``). Node ids
  always contain a ``/`` (the subdirectory), so they can never collide
  with concept slugs, which never do. A *folder-backed* entity — a
  project laid out as ``knowledge/projects/<slug>/`` with an overview
  note beside its curated notes — is identified by that overview note's
  own path, ``projects/<slug>/<slug>``; the folder itself is not a node.
  :func:`normalize_target` resolves the ambiguous short form
  (``projects/<slug>``) to it when given a ``vault_root`` to check.
  :func:`canonical_entity_id` is the single answer to "is this note a
  graph node, and which" (a project's ``log/`` entry is not a node of its
  own — it belongs to the project's overview — while a curated note beside
  the overview keeps its own id, as AGENTS.md says it does),
  :func:`owning_entity_id` answers the neighbouring question "which entity
  does prose written in this note belong to", and :func:`live_entity_id`
  follows ``superseded_by`` from a merged duplicate to the survivor.
  Nothing should re-derive any of them from a path prefix.
- **Pure text editing**: :func:`upsert_relation_in_text` and
  :func:`append_fact_to_log` are ``text -> text`` so the MCP entity tools
  and the consolidation pass (separate stages) can apply them to any note
  body without touching the filesystem here. History is never edited or
  deleted — closing an entry and adding a new one is the only mutation
  (supersede, don't delete).

Frontmatter editing strategy: parse with ``notes._split_frontmatter`` to
read and mutate the relation entries, then splice ONLY the ``relations:``
block back into the fence textually (the same line surgery
``mcp_server/provenance.py`` does for its own keys). Every other key keeps
its source bytes — quote style, key order, comments, and timestamps in the
vault's ``…Z`` form — and so does the body below the fence. A whole-fence
``yaml.safe_dump`` round-trip would rewrite all of that on every relation
write.

AUD-121: this module's body now lives in three sibling modules, split
along the section boundaries described above —
:mod:`ingest_lib.relations_identity` (Identity),
:mod:`ingest_lib.relations_model` (Parsing: the dataclasses,
``parse_relations``, ``entity_notes``, and ``live_entity_id``, which walks
the ``EntityInfo`` records those build), and
:mod:`ingest_lib.relations_text` (Pure text editing, plus the
``query_relations`` read). This module re-exports every name — public and
private alike — that used to live here directly, so every existing
importer (``from ingest_lib.relations import ...``, including the private
helpers a few call sites deliberately reach for, and ``relations.<name>``
attribute access) keeps working unchanged.
"""
from __future__ import annotations

from .relations_identity import canonical_entity_id as canonical_entity_id
from .relations_identity import folder_overview_id as folder_overview_id
from .relations_identity import is_valid_node_id as is_valid_node_id
from .relations_identity import node_id_for_note as node_id_for_note
from .relations_identity import normalize_target as normalize_target
from .relations_identity import note_path_for_node as note_path_for_node
from .relations_identity import owning_entity_id as owning_entity_id
from .relations_identity import _FLAT_ENTITY_DIRS as _FLAT_ENTITY_DIRS
from .relations_identity import _NODE_ID_RE as _NODE_ID_RE
from .relations_identity import _PROJECT_LOG_DIR as _PROJECT_LOG_DIR
from .relations_identity import _SHARED_PROJECT_DIR as _SHARED_PROJECT_DIR
from .relations_identity import _YEAR_RE as _YEAR_RE
from .relations_identity import _project_overview as _project_overview
from .relations_identity import _resolve_folder_entity as _resolve_folder_entity

from .relations_model import EntityInfo as EntityInfo
from .relations_model import RELATION_VOCAB as RELATION_VOCAB
from .relations_model import Relation as Relation
from .relations_model import entity_notes as entity_notes
from .relations_model import live_entity_id as live_entity_id
from .relations_model import parse_relations as parse_relations
from .relations_model import _ASSISTANT_HISTORY_PREFIXES as _ASSISTANT_HISTORY_PREFIXES
from .relations_model import _LOG as _LOG
from .relations_model import _MAX_SUPERSEDED_HOPS as _MAX_SUPERSEDED_HOPS
from .relations_model import _RELATION_ENTRY_KEYS as _RELATION_ENTRY_KEYS
from .relations_model import _coerce_aliases as _coerce_aliases
from .relations_model import _coerce_str as _coerce_str
from .relations_model import _coerce_superseded_by as _coerce_superseded_by

from .relations_text import RelationHit as RelationHit
from .relations_text import append_fact_to_log as append_fact_to_log
from .relations_text import query_relations as query_relations
from .relations_text import upsert_relation_in_text as upsert_relation_in_text
from .relations_text import _CODE_FENCE_RE as _CODE_FENCE_RE
from .relations_text import _DAY_RE as _DAY_RE
from .relations_text import _EMPTY_PLACEHOLDER as _EMPTY_PLACEHOLDER
from .relations_text import _HEADING_RE as _HEADING_RE
from .relations_text import _LOG_HEADING as _LOG_HEADING
from .relations_text import _RELATIONS_KEY_RE as _RELATIONS_KEY_RE
from .relations_text import _as_day as _as_day
from .relations_text import _code_fence_flags as _code_fence_flags
from .relations_text import _day_before as _day_before
from .relations_text import _dump_relations as _dump_relations
from .relations_text import _entry_matches as _entry_matches
from .relations_text import _frontmatter_fence_error as _frontmatter_fence_error
from .relations_text import _relation_entry as _relation_entry
from .relations_text import _splice_relations as _splice_relations

__all__ = [
    "RELATION_VOCAB",
    "Relation",
    "EntityInfo",
    "normalize_target",
    "is_valid_node_id",
    "node_id_for_note",
    "note_path_for_node",
    "folder_overview_id",
    "canonical_entity_id",
    "owning_entity_id",
    "live_entity_id",
    "parse_relations",
    "entity_notes",
    "RelationHit",
    "query_relations",
    "upsert_relation_in_text",
    "append_fact_to_log",
]
