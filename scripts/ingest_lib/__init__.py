"""brain ingestion library.

Public surface used by ``scripts/ingest.py`` and by future MCP handlers.
Internals live in submodules; this file only re-exports.
"""
from .caption import CaptionStats, rebuild_captions
from .chat import AnswerResult, ask
from .concepts import ConceptStats, rebuild_concepts
from .config import VaultPaths, default_paths, paths_for_root
from .connections import (
    ConnectionStats,
    rebuild_connections,
    related_concepts,
    related_entities,
)
from .consolidate import ConsolidateStats, consolidate
from .dashboards import DashboardStats, rebuild_dashboards
from .describe import DescribeStats, rebuild_descriptions
from .extractors import ExtractionResult, dispatch_extractor, registered_extensions
from .metadata import IndexRecord, append_record, iter_records, latest_records_by_path
from .notes import write_index_note, write_processed_note
from .pipeline import IngestPlan, IngestStats, backfill_summaries, plan_ingest, run_ingest
from .recency import MemoryHit, memory_search
from .relations import RELATION_VOCAB, EntityInfo, Relation, entity_notes, parse_relations
from .semantic import (
    SearchHit,
    build_index as build_search_index,
    chunks_for_source,
    search as semantic_search,
    upsert_notes,
)
from .sweep import SweepReport, run_sweep

__all__ = [
    "AnswerResult",
    "CaptionStats",
    "ConceptStats",
    "ConnectionStats",
    "ConsolidateStats",
    "DashboardStats",
    "DescribeStats",
    "EntityInfo",
    "ExtractionResult",
    "IndexRecord",
    "IngestPlan",
    "IngestStats",
    "MemoryHit",
    "RELATION_VOCAB",
    "Relation",
    "SearchHit",
    "SweepReport",
    "VaultPaths",
    "append_record",
    "ask",
    "backfill_summaries",
    "build_search_index",
    "consolidate",
    "default_paths",
    "paths_for_root",
    "dispatch_extractor",
    "entity_notes",
    "iter_records",
    "latest_records_by_path",
    "memory_search",
    "parse_relations",
    "plan_ingest",
    "rebuild_captions",
    "rebuild_concepts",
    "rebuild_connections",
    "rebuild_dashboards",
    "rebuild_descriptions",
    "related_concepts",
    "related_entities",
    "registered_extensions",
    "run_ingest",
    "run_sweep",
    "semantic_search",
    "chunks_for_source",
    "upsert_notes",
    "write_index_note",
    "write_processed_note",
]
