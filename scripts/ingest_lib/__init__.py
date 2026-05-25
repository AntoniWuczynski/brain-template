"""brain ingestion library.

Public surface used by ``scripts/ingest.py`` and by future MCP handlers.
Internals live in submodules; this file only re-exports.
"""
from .chat import AnswerResult, ask
from .concepts import ConceptStats, rebuild_concepts
from .config import VaultPaths, default_paths
from .extractors import ExtractionResult, dispatch_extractor, registered_extensions
from .metadata import IndexRecord, append_record, iter_records, latest_records_by_path
from .notes import write_index_note, write_processed_note
from .pipeline import IngestPlan, IngestStats, backfill_summaries, plan_ingest, run_ingest
from .semantic import SearchHit, build_index as build_search_index, search as semantic_search

__all__ = [
    "AnswerResult",
    "ConceptStats",
    "ExtractionResult",
    "IndexRecord",
    "IngestPlan",
    "IngestStats",
    "SearchHit",
    "VaultPaths",
    "append_record",
    "ask",
    "backfill_summaries",
    "build_search_index",
    "default_paths",
    "dispatch_extractor",
    "iter_records",
    "latest_records_by_path",
    "plan_ingest",
    "rebuild_concepts",
    "registered_extensions",
    "run_ingest",
    "semantic_search",
    "write_index_note",
    "write_processed_note",
]
