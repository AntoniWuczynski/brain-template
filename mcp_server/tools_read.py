"""The six read MCP tools, plus their input/output models.

Split out of ``mcp_server.tools`` (AUD-012) to keep both modules under the
line-count limit. Shares the write-side ``_RateBucket`` class and the
``current_agent``/``safety``/``runtime`` collaborators from its siblings;
``entity_tools`` and ``memory_tools`` import the search/read rate checks
and ``_hit_gate_path`` back from here.
"""
from __future__ import annotations

import logging
import sys
import threading
from collections import defaultdict
from pathlib import Path
from types import TracebackType
from typing import Final, Literal

from pydantic import BaseModel, Field

# Make the ingest_lib package importable. Same shim the CLI scripts use.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ingest_lib import (  # noqa: E402
    IndexRecord as _IndexRecord,
    VaultPaths as _VaultPaths,
    chunks_for_source as _chunks_for_source,
    latest_records_by_path as _latest_records_by_path,
    paths_for_root as _paths_for_root,
    related_concepts as _related_concepts,
    semantic_search as _semantic_search,
)
from ingest_lib.concepts import slugify as _slugify  # noqa: E402
from ingest_lib.knowledge import (  # noqa: E402
    KNOWLEDGE_EXTRACTOR as _KNOWLEDGE_EXTRACTOR,
    knowledge_records as _knowledge_records,
)
from ingest_lib.notes import (  # noqa: E402
    derived_note_relpath as _derived_note_relpath,
)
from ingest_lib.relations import (  # noqa: E402
    EntityInfo as _EntityInfo,
    canonical_entity_id as _canonical_entity_id,
    entity_notes as _entity_notes,
    live_entity_id as _live_entity_id,
)
from ingest_lib.sweep import (  # noqa: E402
    _differs_only_in_numbers as _sweep_differs_only_in_numbers,
    _levenshtein as _sweep_levenshtein,
)

from .errors import ToolError
from .config import MAX_NOTE_BYTES, ServerConfig
from .identity import current_agent
from .runtime import Runtime
from .safety import SafetyError, resolve_read
from .tools import _RateBucket

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiting + concurrency guards (search/read only — writes use
# tools._write_bucket)
# ---------------------------------------------------------------------------

# Search runs a model encode + matmul in a threadpool worker; an
# authenticated agent firing many searches could otherwise exhaust the
# pool and wedge all tools. Plain reads are cheaper and get a higher
# bucket of their own (_READ_RATE_PER_MINUTE below).
_SEARCH_RATE_PER_MINUTE: int = 60
_search_bucket = _RateBucket(_SEARCH_RATE_PER_MINUTE)


def _rate_check_search() -> None:
    if not _search_bucket.allow(current_agent()):
        raise ToolError(
            f"search rate limit exceeded ({_SEARCH_RATE_PER_MINUTE}/minute); back off and retry"
        )


_READ_RATE_PER_MINUTE: int = 120
_read_bucket = _RateBucket(_READ_RATE_PER_MINUTE)


def _rate_check_read() -> None:
    if not _read_bucket.allow(current_agent()):
        raise ToolError(
            f"read rate limit exceeded ({_READ_RATE_PER_MINUTE}/minute); back off and retry"
        )


class _ConcurrencyGuard:
    """Fail-fast bounded concurrency. Rate buckets cap calls-per-minute but
    not how many run at once; without this a burst of slow ops (torch encode,
    large reads) parks every threadpool worker and wedges all tools. Acquire
    is non-blocking: over the limit, fail immediately rather than queue."""

    def __init__(self, n: int, label: str) -> None:
        self._sem = threading.BoundedSemaphore(n)
        self._label = label

    def __enter__(self) -> _ConcurrencyGuard:
        if not self._sem.acquire(blocking=False):
            raise ToolError(f"server busy ({self._label}); retry shortly")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._sem.release()


_search_guard = _ConcurrencyGuard(4, "search")
_read_guard = _ConcurrencyGuard(8, "read")


# ---------------------------------------------------------------------------
# Input / output schemas
# ---------------------------------------------------------------------------

# Fragmentation rule for the evidence hint, ported from
# ``ingest_lib.sweep._check_fragmentation`` (the same "close enough a human
# would call it the same thing" heuristic the nightly sweep uses for concept
# slugs) rather than inventing a second, looser threshold here: a tighter
# edit distance (1, not 2) and a minimum slug length (8) because a short
# slug ("Rex", "Rox") is one or two edits from countless unrelated words,
# plus ``_differs_only_in_numbers`` so two dated or versioned notes are
# never flagged as the same entity.
_ENTITY_NEAR_MAX_DISTANCE: Final[int] = 1
_ENTITY_NEAR_MIN_SLUG_LEN: Final[int] = 8


class EvidenceHintOut(BaseModel):
    """Whether a search hit's subject already has an entity note, so an
    agent can link the existing node instead of minting a duplicate (see
    knowledge/index/templates/memory-fact.md and AGENTS.md's typed-relations
    contract). Built ONLY from the canonical entity graph —
    ``relations.canonical_entity_id`` restricts this to people,
    organisations, projects (the overview, a flat project note, and the
    curated notes beside an overview, which AGENTS.md says keep their own
    path), and meetings notes, never an inbox proposal, the assistant
    profile, a dream note, or a project's own dated log entry (those map
    to their project's overview note, not to themselves).
    Every hit resolves through ``relations.live_entity_id``, so a
    superseded note's hint always names its merge survivor, never the dead
    node.

    ``exists``: the hit itself is a canonical entity note (or maps to one,
    as a project's dated log entry maps to its overview), or its
    title/alias/slug matches
    one exactly. ``probable``: a near-miss on the entity slug — edit
    distance <= 1, minimum slug length 8, never two notes differing only in
    their numbers — scored across every candidate; the nearest wins, ties
    broken by node id. ``unknown``: no candidate found. ``node_id`` is set
    for exists/probable so the agent can act on the hint without a second
    lookup (e.g. via relations_query or entity_upsert_relation)."""

    status: Literal["exists", "probable", "unknown"]
    node_id: str | None = None


class _EntityIndex:
    """Evidence-hint lookup built once per search call (never once per
    hit) from the CANONICAL entity graph — ``relations.entity_notes``
    filtered down to the nodes ``relations.canonical_entity_id`` actually
    recognises, so an inbox proposal, the assistant PROFILE, a dream note,
    or a project's dated log entry never becomes a link target, and a log
    entry's hint (via the path branch) resolves to its project's overview
    note instead of minting a node of its own. A curated note beside an
    overview is a node in its own right and stays one — 36 of the live
    vault's stored relation targets are exactly such notes, and folding
    them into their project made the hint name the wrong node. Every id
    handed back is run
    through ``relations.live_entity_id`` first, so a superseded entity's
    hint always names the merge survivor.

    Loading + parsing every entity note is the expensive part (callers
    skip the build entirely when there are no hits to annotate — see
    ``build_entity_index``'s callers); the per-hit lookups against the
    built index are dict lookups plus a bounded scan of same-length-ish
    slug buckets. Cheaper than the build, but real cost, not free by
    comparison (review/2026-09-05/refute-hints.md, F9)."""

    def __init__(self, entities: dict[str, _EntityInfo], *, vault_root: Path) -> None:
        canonical = {
            node_id: info
            for node_id, info in entities.items()
            if _canonical_entity_id(info.rel_path, vault_root=vault_root) == node_id
        }
        exact: dict[str, str] = {}
        slug_exact: dict[str, str] = {}
        near_by_len: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for node_id, info in canonical.items():
            for name in (info.title, *info.aliases):
                key = name.strip().casefold()
                if key and key not in exact:
                    exact[key] = node_id
            slug = _slugify(info.title)
            if slug and slug not in slug_exact:
                slug_exact[slug] = node_id
            if slug and len(slug) >= _ENTITY_NEAR_MIN_SLUG_LEN:
                near_by_len[len(slug)].append((slug, node_id))
        self._entities = canonical
        self._vault_root = vault_root
        self._exact = exact
        self._slug_exact = slug_exact
        self._near_by_len: dict[int, list[tuple[str, str]]] = dict(near_by_len)

    def _live(self, node_id: str) -> str | None:
        return _live_entity_id(node_id, self._entities)

    def _near_candidates(self, slug: str) -> list[tuple[int, str]]:
        """Every candidate within the fragmentation rule, scored by edit
        distance — bucketed by slug length so a hit's cost is bounded by
        the (small) number of entities within ``_ENTITY_NEAR_MAX_DISTANCE``
        characters of its own slug length, not the whole entity graph."""
        if len(slug) < _ENTITY_NEAR_MIN_SLUG_LEN:
            return []
        scored: list[tuple[int, str]] = []
        lengths = {
            len(slug) - _ENTITY_NEAR_MAX_DISTANCE,
            len(slug),
            len(slug) + _ENTITY_NEAR_MAX_DISTANCE,
        }
        for length in lengths:
            for cand_slug, cand_id in self._near_by_len.get(length, ()):
                if _sweep_differs_only_in_numbers(slug, cand_slug):
                    continue
                distance = _sweep_levenshtein(slug, cand_slug)
                similar = (
                    slug == cand_slug + "s"
                    or cand_slug == slug + "s"
                    or distance <= _ENTITY_NEAR_MAX_DISTANCE
                )
                if similar:
                    scored.append((distance, cand_id))
        return scored

    def hint(self, source_relative_path: str, title: str) -> EvidenceHintOut:
        canonical_id = _canonical_entity_id(
            source_relative_path, vault_root=self._vault_root
        )
        if canonical_id is not None and canonical_id in self._entities:
            return EvidenceHintOut(status="exists", node_id=self._live(canonical_id))
        key = title.strip().casefold()
        node_id = self._exact.get(key)
        if node_id is not None:
            return EvidenceHintOut(status="exists", node_id=self._live(node_id))
        slug = _slugify(title)
        if slug:
            slug_node_id = self._slug_exact.get(slug)
            if slug_node_id is not None:
                return EvidenceHintOut(status="exists", node_id=self._live(slug_node_id))
            candidates = self._near_candidates(slug)
            if candidates:
                _, best_id = min(candidates, key=lambda c: (c[0], c[1]))
                return EvidenceHintOut(status="probable", node_id=self._live(best_id))
        return EvidenceHintOut(status="unknown", node_id=None)


def build_entity_index(cfg: ServerConfig) -> _EntityIndex:
    """Load and index every entity note once. Call this once per
    ``vault_search``/``memory_search`` invocation, never per hit — and
    only when there is at least one hit to annotate; callers skip this
    entirely on an empty hit list (F8)."""
    paths: _VaultPaths = _paths_for_root(cfg.vault_root)
    return _EntityIndex(_entity_notes(paths), vault_root=paths.root)


class SearchHitOut(BaseModel):
    score: float
    source_relative_path: str
    title: str
    chunk_idx: int
    snippet: str
    evidence: EvidenceHintOut


class SearchOut(BaseModel):
    hits: list[SearchHitOut]


class ChunkOut(BaseModel):
    chunk_idx: int
    text: str
    is_target: bool


class ChunkContextOut(BaseModel):
    source_relative_path: str
    total_chunks: int
    chunks: list[ChunkOut]


class ReadOut(BaseModel):
    path: str
    content: str
    size_bytes: int


class ListEntry(BaseModel):
    name: str
    is_dir: bool
    size_bytes: int | None = None


class ListOut(BaseModel):
    path: str
    entries: list[ListEntry]


class RecordOut(BaseModel):
    relative_path: str
    source_hash: str
    status: str
    extractor: str
    extension: str
    size_bytes: int
    summary: str | None = None
    topics: list[str] = Field(default_factory=list)
    processed_path: str | None = None
    index_note_path: str | None = None


class MetadataQueryOut(BaseModel):
    records: list[RecordOut]


class RelatedConceptOut(BaseModel):
    slug: str
    display: str
    kinds: list[str]              # which signals link them: cooccurrence, semantic
    cooccurrence: float           # shared-document count (0 if none)
    semantic: float               # centroid cosine (0 if none)


class RelatedOut(BaseModel):
    concept: str                  # the resolved concept slug
    related: list[RelatedConceptOut]


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

# The embedding model truncates to ~512 tokens anyway, so a legitimate
# query is never more than a few hundred chars. Cap it well above that so a
# hostile client can't tie up a search worker tokenizing a multi-MB string
# (note bodies cap at 5 MB, the request at 160 MB — without this a query
# could ride that ceiling straight into model.encode).
MAX_QUERY_CHARS: int = 4096


def _check_query_len(query: str) -> None:
    if len(query) > MAX_QUERY_CHARS:
        raise ToolError(
            f"query is {len(query)} characters; max is {MAX_QUERY_CHARS} "
            "(the embedder truncates to a few hundred tokens anyway)"
        )


def require_search_index(cfg: ServerConfig) -> None:
    """Raise unless the semantic index exists on disk.

    Distinguish "no index built" (fresh clone — the index is gitignored)
    from "no matches": returning an empty hit list for both makes agents
    conclude the vault is empty. Every search-shaped tool fails loudly with
    the same message (AUD-039) — ``memory_search`` used to swallow the
    missing index and answer "you know nothing about X".
    """
    metadata = _paths_for_root(cfg.vault_root).metadata
    if not (metadata / "embeddings.npy").exists() or not (
        metadata / "embeddings_meta.jsonl"
    ).exists():
        raise ToolError(
            "search index not built; run "
            "'python scripts/ingest.py --rebuild-search-index' first"
        )


def note_still_exists(source_relative_path: str, origin: str, gated: Path) -> bool:
    """False when a KNOWLEDGE-NOTE hit's note has been deleted on disk.

    There is no MCP delete verb, so a note deleted in Obsidian leaves its
    rows in the index until the next rebuild; ``recency`` then reads no
    frontmatter, weighs the ghost 1.0 and floats it above its live
    replacement (AUD-050). One ``is_file()`` per hit — at most ``top_k``
    (<= 50) stats per query, microseconds against the ~10 ms query encode.

    Deliberately NOT applied to ingested sources: their gate path is
    *derived* (``archive/processed/<source>.<ext>.md``) and does not have to
    name a real file — vaults built before that convention hold notes at the
    old ``<source>.md`` path, and on this vault 624 of 645 archive gate paths
    do not exist. Existence there is a naming signal, not a deletion signal.
    """
    is_note = origin == _KNOWLEDGE_EXTRACTOR or (
        not origin and source_relative_path.startswith("knowledge/")
    )
    if not is_note or gated.is_file():
        return True
    log.warning(
        "index drift: dropping hit for deleted note %s", source_relative_path
    )
    return False


def tool_search(
    cfg: ServerConfig, runtime: Runtime, query: str, top_k: int = 10,
    mode: str = "hybrid",
) -> SearchOut:
    """Search over the vault. Returns the top-k matching chunks. ``mode`` is
    dense (embeddings), lexical (BM25, exact identifiers), or hybrid (both).
    Each hit carries an ``evidence`` hint (exists/probable/unknown, plus a
    node_id when known) saying whether its subject already has an entity
    note under knowledge/people|organisations|projects|meetings/ — check it
    before creating a new entity note so you link the existing one instead
    of splitting the person/project graph."""
    if not query or not query.strip():
        raise ToolError("query must be non-empty")
    _check_query_len(query)
    if not 1 <= top_k <= 50:
        raise ToolError("top_k must be in [1, 50]")
    if mode not in ("dense", "lexical", "hybrid"):
        raise ToolError("mode must be one of: dense, lexical, hybrid")
    _rate_check_search()

    paths = _paths_for_root(cfg.vault_root)
    require_search_index(cfg)
    with _search_guard:
        hits = _semantic_search(paths, query, top_k=top_k, mode=mode, logger=log)
    # Gate hits by the read policy: only return a hit whose backing
    # artifact the agent could read directly (see _hit_gate_path).
    safe = []
    for h in hits:
        origin = getattr(h, "origin", "")
        try:
            resolved = resolve_read(
                cfg.vault_root, _hit_gate_path(h.source_relative_path, origin)
            )
        except SafetyError:
            # The hit's backing artifact isn't readable under the policy —
            # drop that one hit, don't fail the whole search. (_hit_gate_path
            # is pure string ops and resolve_read only raises SafetyError, so
            # no other exception reaches here.)
            continue
        if not note_still_exists(h.source_relative_path, origin, resolved):
            continue
        safe.append(h)
    hits = safe
    # Audit the query plus the GATED hit paths — what the agent actually saw.
    runtime.audit.access_event(
        agent=current_agent(),
        tool="vault_search",
        paths=[h.source_relative_path for h in hits],
        query=query,
    )
    if not hits:
        return SearchOut(hits=[])
    entity_index = build_entity_index(cfg)
    return SearchOut(
        hits=[
            SearchHitOut(
                score=h.score,
                source_relative_path=h.source_relative_path,
                title=h.title,
                chunk_idx=h.chunk_idx,
                snippet=h.snippet,
                evidence=entity_index.hint(h.source_relative_path, h.title),
            )
            for h in hits
        ]
    )


def _hit_gate_path(source_relative_path: str, origin: str) -> str:
    """Map a search hit's source label to the path the read policy gates on.

    Curated knowledge/ notes are embedded directly — the note IS the
    readable artifact, so gate on it as-is. Ingested sources are labelled
    by their raw source path; their readable artifact is the processed
    markdown twin under archive/processed/.

    ``origin`` (the indexing record's extractor, stored per meta row)
    decides which case applies — NOT the path prefix: an ingested source
    dropped at inbox/knowledge/x.pdf is labelled "knowledge/x.pdf" but is
    not a vault note. Rows from indexes built before ``origin`` existed
    carry "" and degrade to the prefix heuristic until the next rebuild.
    """
    if origin == _KNOWLEDGE_EXTRACTOR or (
        not origin and source_relative_path.startswith("knowledge/")
    ):
        return source_relative_path
    # Same derivation the pipeline uses (keeps the source extension), so the
    # gate path matches the real processed twin. Gating only checks the read
    # POLICY (archive/processed is an allowed area), so old-convention notes
    # still pass regardless — this just keeps the two in sync.
    return "archive/processed/" + _derived_note_relpath(source_relative_path)


def tool_chunk_context(
    cfg: ServerConfig,
    runtime: Runtime,
    source_relative_path: str,
    chunk_idx: int,
    before: int = 1,
    after: int = 1,
) -> ChunkContextOut:
    """Return a search hit's neighbouring chunks — cheaper than reading the
    whole backing file just to see the context around one snippet. Gated by
    the same read policy as ``vault_search``: a hit whose backing artifact you
    could not read directly returns nothing here either."""
    if not source_relative_path or not source_relative_path.strip():
        raise ToolError("source_relative_path must be non-empty")
    if chunk_idx < 0:
        raise ToolError("chunk_idx must be >= 0")
    if not 0 <= before <= 20 or not 0 <= after <= 20:
        raise ToolError("before/after must be in [0, 20]")
    _rate_check_read()

    paths = _paths_for_root(cfg.vault_root)
    rows = _chunks_for_source(paths, source_relative_path)
    if not rows:
        raise ToolError("no indexed chunks for that source (rebuild the index?)")
    # Gate on the read policy, keyed by the source's own origin (as search does).
    origin = rows[0]["origin"]
    try:
        resolve_read(cfg.vault_root, _hit_gate_path(source_relative_path, origin))
    except SafetyError:
        raise ToolError("not found or not readable") from None

    lo, hi = chunk_idx - before, chunk_idx + after
    # chunk_idx is always an int on a MetaRow (narrowed once, on load, by
    # ingest_lib.semantic._row_from_json).
    indexed = [(r, r["chunk_idx"]) for r in rows]
    window = [
        ChunkOut(chunk_idx=idx, text=r["text"], is_target=idx == chunk_idx)
        for r, idx in indexed
        if lo <= idx <= hi
    ]
    runtime.audit.access_event(
        agent=current_agent(),
        tool="vault_chunk_context",
        paths=[source_relative_path],
        query=f"chunk {chunk_idx} ±({before},{after})",
    )
    return ChunkContextOut(
        source_relative_path=source_relative_path,
        total_chunks=len(rows),
        chunks=window,
    )


def tool_read(cfg: ServerConfig, runtime: Runtime, path: str) -> ReadOut:
    """Read a UTF-8 text file from the vault. Reads are allowlist-based: only
    paths under READ_ALLOW_PREFIXES (knowledge/, archive/, inbox/, metadata/)
    or the READ_ALLOW_ROOT_FILES vault docs are permitted, with the DENY_*
    lists (secrets, logs, the embeddings index) applied on top. Everything
    else — and any refusal — returns the same opaque 'not found or not
    readable'. Binary and oversize files are refused."""
    _rate_check_read()
    resolved = resolve_read(cfg.vault_root, path)
    if not resolved.is_file():
        raise ToolError("not found or not readable")
    try:
        size = resolved.stat().st_size
    except OSError:
        raise ToolError("not found or not readable") from None
    if size > MAX_NOTE_BYTES:
        raise ToolError(f"file is {size} bytes; reads are capped at {MAX_NOTE_BYTES}")
    with _read_guard:
        try:
            data = resolved.read_bytes()
        except OSError:
            raise ToolError("read failed") from None
    if b"\x00" in data[:4096]:
        raise ToolError(
            "refusing to return binary content; use semantic search or list+filename"
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError("file is not valid UTF-8") from None
    runtime.audit.access_event(agent=current_agent(), tool="vault_read", paths=[path])
    return ReadOut(path=path, content=text, size_bytes=len(data))


def tool_list(cfg: ServerConfig, runtime: Runtime, path: str = "") -> ListOut:
    """List entries under a directory. Empty path lists the vault root."""
    _rate_check_read()
    if path:
        resolved = resolve_read(cfg.vault_root, path)
    else:
        resolved = cfg.vault_root
    if not resolved.is_dir():
        raise ToolError("not found or not readable")
    entries: list[ListEntry] = []
    # iterdir on an unreadable directory raises OSError carrying the absolute
    # path; convert to the same opaque error tool_read uses so the filesystem
    # layout never leaks to the agent.
    try:
        children = sorted(resolved.iterdir())
    except OSError:
        raise ToolError("not found or not readable") from None
    for child in children:
        if child.name.startswith("."):
            continue
        # Only list entries the read allowlist would accept, so the root
        # listing can't reveal the names of non-readable trees (scripts/,
        # mcp_server/, etc.) that a subdir listing would refuse.
        try:
            resolve_read(cfg.vault_root, child.relative_to(cfg.vault_root).as_posix())
        except SafetyError:
            continue
        if child.is_dir():
            entries.append(ListEntry(name=child.name, is_dir=True))
        elif child.is_file():
            try:
                size = child.stat().st_size
            except OSError:
                size = None
            entries.append(ListEntry(name=child.name, is_dir=False, size_bytes=size))
    runtime.audit.access_event(agent=current_agent(), tool="vault_list", paths=[path])
    return ListOut(path=path, entries=entries)


def tool_metadata_query(
    cfg: ServerConfig,
    runtime: Runtime,
    by: str = "status",
    value: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> MetadataQueryOut:
    """Query metadata/index.jsonl. Filter by status, extension, or extractor."""
    _rate_check_read()
    if by not in {"status", "extension", "extractor", "path_prefix", "all"}:
        raise ToolError(
            "by must be one of: status, extension, extractor, path_prefix, all"
        )
    if by != "all" and not value:
        raise ToolError(f"value is required when by={by!r}")
    if not 1 <= limit <= 500:
        raise ToolError("limit must be in [1, 500]")
    if offset < 0:
        raise ToolError("offset must be >= 0")

    paths = _paths_for_root(cfg.vault_root)
    # index.jsonl covers ingested sources only. The hand-edited notes are
    # indexed and searchable as virtual records carrying the same
    # ``knowledge-note`` extractor tag ``vault_search`` reports, so the
    # metadata view has to know about them too — without this,
    # by=extractor value=knowledge-note answered "nothing" about 368 live
    # notes (AUD-049). An index.jsonl record wins any path collision.
    ingested = _latest_records_by_path(paths.metadata_index_jsonl)
    records = list(ingested.values()) + [
        r for r in _knowledge_records(paths) if r.relative_path not in ingested
    ]
    # Sort so `offset` names a stable position across calls: index.jsonl
    # order is append order, and merging in the virtual records would
    # otherwise reshuffle the tail on every note edit.
    records.sort(key=lambda r: r.relative_path)

    def matches(r: _IndexRecord) -> bool:
        if by == "all":
            return True
        if by == "status":
            return r.status == value
        if by == "extension":
            return r.extension == value
        if by == "extractor":
            return r.extractor == value
        if by == "path_prefix":
            return r.relative_path.startswith(value or "")
        return False

    filtered = [r for r in records if matches(r)][offset:offset + limit]
    runtime.audit.access_event(
        agent=current_agent(),
        tool="vault_metadata_query",
        paths=[],
        query=f"by={by} value={value or ''} offset={offset}",
    )
    return MetadataQueryOut(
        records=[
            RecordOut(
                relative_path=r.relative_path,
                source_hash=r.source_hash,
                status=r.status,
                extractor=r.extractor,
                extension=r.extension,
                size_bytes=r.size_bytes,
                summary=r.summary or None,
                topics=list(r.topics or []),
                processed_path=r.processed_path,
                index_note_path=r.index_note_path,
            )
            for r in filtered
        ]
    )


def tool_related(cfg: ServerConfig, runtime: Runtime, concept: str, limit: int = 8) -> RelatedOut:
    """Return the concepts most related to ``concept`` (slug or display name),
    from the persisted relationship graph (co-occurrence + semantic)."""
    if not concept or not concept.strip():
        raise ToolError("concept must be non-empty")
    _check_query_len(concept)
    if not 1 <= limit <= 50:
        raise ToolError("limit must be in [1, 50]")
    _rate_check_read()
    paths = _paths_for_root(cfg.vault_root)
    if not (paths.metadata / "connections.jsonl").exists():
        raise ToolError(
            "connection graph not built; run "
            "'python scripts/ingest.py --rebuild-connections' first"
        )
    slug, rels = _related_concepts(paths, concept, top_n=limit)
    if not slug:
        raise ToolError(f"unknown concept {concept!r}; not a topic in the vault")
    runtime.audit.access_event(
        agent=current_agent(), tool="vault_related", paths=[], query=concept
    )
    return RelatedOut(
        concept=slug,
        related=[
            RelatedConceptOut(
                slug=r.slug,
                display=r.display,
                kinds=list(r.kinds),
                cooccurrence=r.cooccurrence,
                semantic=r.semantic,
            )
            for r in rels
        ],
    )
