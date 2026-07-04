"""All MCP tools, plus their input/output models.

The tools share helpers from this module's siblings (safety, git_ops,
provenance) and reuse the ingest_lib package for search + metadata reads.
Every tool takes ``(cfg, runtime, ...)``: cfg is pure env-derived data,
runtime carries the stateful collaborators (audit log, async push
worker, background index refresher).

Each tool runs through this contract:
  1. Validate inputs (Pydantic does most of this).
  2. Resolve any paths via mcp_server.safety. SafetyError -> MCP error.
  3. Do the work. Return a small structured result.
  4. For writes: stamp provenance, commit via git_ops (the PUSH is
     async — see push_queue), enqueue a background reindex, and audit
     the outcome. Surface the SHA in the response so the agent can
     refer back to its own change.

Errors raised from inside a tool are caught by FastMCP and surfaced
as MCP error responses. No stack traces leak.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

from pydantic import BaseModel, Field

# Make the ingest_lib package importable. Same shim the CLI scripts use.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ingest_lib import (  # type: ignore[import-not-found]  # noqa: E402
    latest_records_by_path as _latest_records_by_path,
    paths_for_root as _paths_for_root,
    related_concepts as _related_concepts,
    semantic_search as _semantic_search,
)
from ingest_lib.knowledge import (  # type: ignore[import-not-found]  # noqa: E402
    KNOWLEDGE_EXTRACTOR as _KNOWLEDGE_EXTRACTOR,
)
from ingest_lib.notes import (  # type: ignore[import-not-found]  # noqa: E402
    derived_note_relpath as _derived_note_relpath,
)

from .config import (
    MAX_INBOX_BYTES,
    MAX_NOTE_BYTES,
    PROFILE_NOTE_PATH,
    ServerConfig,
    WRITE_RATE_PER_MINUTE,
)
from .git_ops import CommitOutcome, GitError, commit_paths
from .identity import current_agent
from .provenance import frontmatter_signature, stamp_provenance
from .runtime import Runtime
from .safety import (
    SafetyError,
    resolve_inbox,
    resolve_read,
    resolve_write_concept,
    resolve_write_under_allowlist,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiting (writes only)
# ---------------------------------------------------------------------------

class _RateBucket:
    """Sliding-window rate limiter. One instance for the whole server."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        # Tool calls run in a threadpool; the deque must not be mutated
        # from multiple workers concurrently.
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            while self._hits and self._hits[0] < cutoff:
                self._hits.popleft()
            if len(self._hits) >= self._max:
                return False
            self._hits.append(now)
            return True


_write_bucket = _RateBucket(WRITE_RATE_PER_MINUTE)
# Search runs a model encode + matmul in a threadpool worker; an
# authenticated agent firing many searches could otherwise exhaust the
# pool and wedge all tools. Plain reads are cheaper and get a higher
# bucket of their own (_READ_RATE_PER_MINUTE below).
_SEARCH_RATE_PER_MINUTE: int = 60
_search_bucket = _RateBucket(_SEARCH_RATE_PER_MINUTE)


def _rate_check_write() -> None:
    if not _write_bucket.allow():
        raise ToolError(
            f"write rate limit exceeded ({WRITE_RATE_PER_MINUTE}/minute); back off and retry"
        )


def _rate_check_search() -> None:
    if not _search_bucket.allow():
        raise ToolError(
            f"search rate limit exceeded ({_SEARCH_RATE_PER_MINUTE}/minute); back off and retry"
        )


_READ_RATE_PER_MINUTE: int = 120
_read_bucket = _RateBucket(_READ_RATE_PER_MINUTE)


def _rate_check_read() -> None:
    if not _read_bucket.allow():
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

    def __enter__(self) -> "_ConcurrencyGuard":
        if not self._sem.acquire(blocking=False):
            raise ToolError(f"server busy ({self._label}); retry shortly")
        return self

    def __exit__(self, *exc) -> None:
        self._sem.release()


_search_guard = _ConcurrencyGuard(4, "search")
_read_guard = _ConcurrencyGuard(8, "read")

# Serializes the read-modify-write-commit critical section of every write
# tool. Writes are rate-limited (30/min) and git is already serialized, so a
# single global lock adds negligible latency while removing three hazards:
# (a) the lost-update race when two appends/updates race the same note
#     (both read the old body, the later write drops the earlier one);
# (b) the create/drop exists-then-write TOCTOU;
# (c) unbounded concurrent write threads (e.g. many large inbox uploads at
#     once) parking the worker pool.
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Tool error type — converted to MCP error by FastMCP automatically
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Raised for any user-visible tool error (safety, size, rate, etc.)."""


# ---------------------------------------------------------------------------
# Input / output schemas
# ---------------------------------------------------------------------------

class SearchHitOut(BaseModel):
    score: float
    source_relative_path: str
    title: str
    chunk_idx: int
    snippet: str


class SearchOut(BaseModel):
    hits: list[SearchHitOut]


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


class WriteResult(BaseModel):
    path: str
    bytes_written: int
    commit_sha: str | None = None
    committed: bool = False        # did the change reach a git commit?
    # True only in legacy sync flows; the push is async now, so at return
    # time this is always False — see push_state for the queue's answer.
    pushed: bool = False
    push_state: str = ""           # "queued" | "disabled" | "skipped"
    index_refresh: str = ""        # "queued" | "off" | "skipped"
    warning: str | None = None     # set when committed is False


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


def tool_search(
    cfg: ServerConfig, runtime: Runtime, query: str, top_k: int = 10,
    mode: str = "hybrid",
) -> SearchOut:
    """Search over the vault. Returns the top-k matching chunks. ``mode`` is
    dense (embeddings), lexical (BM25, exact identifiers), or hybrid (both)."""
    if not query or not query.strip():
        raise ToolError("query must be non-empty")
    _check_query_len(query)
    if not 1 <= top_k <= 50:
        raise ToolError("top_k must be in [1, 50]")
    if mode not in ("dense", "lexical", "hybrid"):
        raise ToolError("mode must be one of: dense, lexical, hybrid")
    _rate_check_search()

    paths = _paths_for_root(cfg.vault_root)
    # Distinguish "no index built" (fresh clone — gitignored) from "no
    # matches": returning [] for both makes agents conclude the vault is
    # empty. Fail loudly like tool_related does for its graph file.
    if not (paths.metadata / "embeddings.npy").exists() or not (
        paths.metadata / "embeddings_meta.jsonl"
    ).exists():
        raise ToolError(
            "search index not built; run "
            "'python scripts/ingest.py --rebuild-search-index' first"
        )
    with _search_guard:
        hits = _semantic_search(paths, query, top_k=top_k, mode=mode, logger=log)
    # Gate hits by the read policy: only return a hit whose backing
    # artifact the agent could read directly (see _hit_gate_path).
    safe = []
    for h in hits:
        try:
            resolve_read(
                cfg.vault_root,
                _hit_gate_path(h.source_relative_path, getattr(h, "origin", "")),
            )
        # ValueError guards a degenerate row (empty source_relative_path ->
        # Path('').with_suffix): drop that one hit, don't fail the search.
        except (SafetyError, ValueError):
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
    return SearchOut(
        hits=[
            SearchHitOut(
                score=h.score,
                source_relative_path=h.source_relative_path,
                title=h.title,
                chunk_idx=h.chunk_idx,
                snippet=h.snippet,
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


def tool_read(cfg: ServerConfig, runtime: Runtime, path: str) -> ReadOut:
    """Read a file from the vault. Allowed paths cover everything outside
    the deny-list (system files, secrets, logs). Binary files refused."""
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
    for child in sorted(resolved.iterdir()):
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

    paths = _paths_for_root(cfg.vault_root)
    records = list(_latest_records_by_path(paths.metadata_index_jsonl).values())

    def matches(r) -> bool:
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

    filtered = [r for r in records if matches(r)][:limit]
    runtime.audit.access_event(
        agent=current_agent(),
        tool="vault_metadata_query",
        paths=[],
        query=f"by={by} value={value or ''}",
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


# ---------------------------------------------------------------------------
# Write tools
# ---------------------------------------------------------------------------

def tool_create_note(cfg: ServerConfig, runtime: Runtime, path: str, content: str) -> WriteResult:
    """Create a new Markdown note under one of the user-writable knowledge subdirs.

    Refuses to overwrite an existing file. Use append_to_note to extend.
    Provenance (author/written_via, plus memory_status in the memory
    areas) is stamped server-side; client-supplied values are overridden.
    """
    def _do() -> WriteResult:
        _rate_check_write()
        _check_note_size(content)
        agent = current_agent()
        resolved = resolve_write_under_allowlist(cfg.vault_root, path)
        rel = resolved.relative_to(cfg.vault_root).as_posix()
        _refuse_if_profile(rel)
        body = content
        graph_changed = False
        if _is_knowledge_md(rel):
            # Stamping adds ~100 bytes over the size check above — slack
            # the 5 MB cap absorbs without a second error path.
            body = stamp_provenance(
                content, agent=agent, mode="create", memory_area=_is_memory_area(rel)
            )
            topics, relations = frontmatter_signature(body)
            graph_changed = bool(topics or relations)
        with _write_lock:
            if resolved.exists():
                raise ToolError(
                    f"file already exists: {path!r} "
                    "(use append_to_note to extend, or replace_note to rewrite in full)"
                )
            resolved.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(resolved, body)
            outcome = _commit(cfg, [resolved], _commit_message(agent, f"create note {path}"))
        push_state, index_refresh = _finish_write(
            runtime, rel=rel, outcome=outcome,
            graph_changed=graph_changed, reindex=_is_knowledge_md(rel),
        )
        return _write_result(
            path, len(body.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _audited_write(runtime, tool="vault_create_note", path=path, fn=_do)


def tool_replace_note(cfg: ServerConfig, runtime: Runtime, path: str, content: str) -> WriteResult:
    """Overwrite an existing user-writable note with new content.

    Refuses to create a missing file (use create_note for that) and
    refuses paths outside the knowledge/ write allowlist. Use this to
    regenerate a note in full; use append_to_note to extend one.
    Stamps last_written_by/written_via; the create-time author survives.
    """
    def _do() -> WriteResult:
        _rate_check_write()
        _check_note_size(content)
        agent = current_agent()
        resolved = resolve_write_under_allowlist(cfg.vault_root, path)
        rel = resolved.relative_to(cfg.vault_root).as_posix()
        _refuse_if_profile(rel)
        body = content
        with _write_lock:
            if not resolved.is_file():
                raise ToolError(f"file does not exist: {path!r} (use create_note for a new note)")
            graph_changed = False
            if _is_knowledge_md(rel):
                try:
                    old: str | None = resolved.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    old = None  # unreadable old text: assume the graph moved
                # Stamp with the OLD note as prior so author/memory_status
                # are re-asserted from what the server last wrote, never
                # from the client's (forgeable) new content.
                body = stamp_provenance(
                    content, agent=agent, mode="replace",
                    memory_area=_is_memory_area(rel), prior=old,
                )
                graph_changed = (
                    old is None
                    or frontmatter_signature(old) != frontmatter_signature(body)
                )
            _atomic_write_text(resolved, body)
            outcome = _commit(cfg, [resolved], _commit_message(agent, f"replace note {path}"))
        push_state, index_refresh = _finish_write(
            runtime, rel=rel, outcome=outcome,
            graph_changed=graph_changed, reindex=_is_knowledge_md(rel),
        )
        return _write_result(
            path, len(body.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _audited_write(runtime, tool="vault_replace_note", path=path, fn=_do)


def tool_append_to_note(cfg: ServerConfig, runtime: Runtime, path: str, content: str) -> WriteResult:
    """Append content to an existing user-writable note. Adds a leading
    newline if needed. Stamps last_written_by/written_via on the combined
    text; the create-time author survives."""
    def _do() -> WriteResult:
        _rate_check_write()
        _check_note_size(content)
        agent = current_agent()
        resolved = resolve_write_under_allowlist(cfg.vault_root, path)
        rel = resolved.relative_to(cfg.vault_root).as_posix()
        _refuse_if_profile(rel)
        with _write_lock:
            if not resolved.is_file():
                raise ToolError(f"file does not exist: {path!r} (use create_note first)")
            try:
                existing = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise ToolError("existing note could not be read") from None
            separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
            combined = existing + separator + content
            graph_changed = False
            if _is_knowledge_md(rel):
                combined = stamp_provenance(
                    combined, agent=agent, mode="append",
                    memory_area=_is_memory_area(rel), prior=existing,
                )
                # Appends rarely touch frontmatter, but a client may send
                # a full new fence on a fenceless note — compare to be sure.
                graph_changed = (
                    frontmatter_signature(existing) != frontmatter_signature(combined)
                )
            if len(combined.encode()) > MAX_NOTE_BYTES:
                raise ToolError(
                    f"appended content would exceed {MAX_NOTE_BYTES} bytes; "
                    "split into a separate note instead"
                )
            _atomic_write_text(resolved, combined)
            outcome = _commit(cfg, [resolved], _commit_message(agent, f"append to {path}"))
        push_state, index_refresh = _finish_write(
            runtime, rel=rel, outcome=outcome,
            graph_changed=graph_changed, reindex=_is_knowledge_md(rel),
        )
        return _write_result(
            path, len(content.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _audited_write(runtime, tool="vault_append_to_note", path=path, fn=_do)


_USER_SECTION_MARKER = "<!-- AUTO-GENERATED-END -->"


def tool_update_concept_user_section(
    cfg: ServerConfig,
    runtime: Runtime,
    slug: str,
    content: str,
) -> WriteResult:
    """Replace the user-editable section of a concept note (everything
    below the AUTO-GENERATED-END marker). Auto-generated content above
    the marker is preserved. No provenance stamping: the edit is scoped
    below the marker, and attribution lives in the commit + audit log."""
    def _do() -> WriteResult:
        _rate_check_write()
        _check_note_size(content)
        agent = current_agent()
        resolved = resolve_write_concept(cfg.vault_root, slug)
        rel = resolved.relative_to(cfg.vault_root).as_posix()
        with _write_lock:
            if not resolved.is_file():
                raise ToolError(
                    f"concept note does not exist: {slug!r} "
                    "(it gets created by the ingest pipeline when a topic first appears)"
                )
            try:
                full = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise ToolError("concept note could not be read") from None
            marker_pos = full.find(_USER_SECTION_MARKER)
            if marker_pos < 0:
                raise ToolError(
                    f"concept note {slug!r} has no AUTO-GENERATED-END marker; "
                    "refusing to write (would clobber the whole file)"
                )
            auto_part = full[: marker_pos + len(_USER_SECTION_MARKER)]
            new_full = auto_part + "\n\n" + content.lstrip() + ("\n" if not content.endswith("\n") else "")
            _atomic_write_text(resolved, new_full)
            outcome = _commit(
                cfg,
                [resolved],
                _commit_message(agent, f"update concept user section: {slug}"),
            )
        # Concept notes are derived state — the rebuild path regenerates them;
        # the user tail gets embedded on the next full index rebuild.
        push_state, index_refresh = _finish_write(
            runtime, rel=rel, outcome=outcome, graph_changed=False, reindex=False,
        )
        return _write_result(
            rel, len(content.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _audited_write(
        runtime, tool="vault_update_concept_user_section", path=slug, fn=_do
    )


def tool_drop_inbox_file(
    cfg: ServerConfig,
    runtime: Runtime,
    path: str,
    content_base64: str,
) -> WriteResult:
    """Drop a file under inbox/<path>. Content is base64-encoded so the
    JSON transport handles binary cleanly. Refuses if a file already
    exists at that path. No provenance stamping (often binary) and no
    reindex — the ingest pipeline owns inbox content."""
    def _do() -> WriteResult:
        import base64

        _rate_check_write()
        try:
            data = base64.b64decode(content_base64, validate=True)
        except Exception:
            raise ToolError("content_base64 is not valid base64") from None
        if len(data) > MAX_INBOX_BYTES:
            raise ToolError(
                f"file is {len(data)} bytes; max is {MAX_INBOX_BYTES} for inbox uploads"
            )
        if len(data) == 0:
            raise ToolError("refusing to drop a zero-byte file")

        agent = current_agent()
        resolved = resolve_inbox(cfg.vault_root, path)
        rel = resolved.relative_to(cfg.vault_root).as_posix()
        with _write_lock:
            if resolved.exists():
                raise ToolError(f"inbox file already exists: {path!r}")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(resolved, data)
            outcome = _commit(cfg, [resolved], _commit_message(agent, f"drop inbox file {path}"))
        push_state, index_refresh = _finish_write(
            runtime, rel=rel, outcome=outcome, graph_changed=False, reindex=False,
        )
        return _write_result(
            rel, len(data), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _audited_write(runtime, tool="vault_drop_inbox_file", path=path, fn=_do)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_knowledge_md(rel: str) -> bool:
    """Markdown notes under knowledge/ get provenance stamps and a
    background reindex; anything else (binary inbox drops, the odd
    non-.md asset) does not. Casefold the extension test so ``x.MD`` on a
    case-insensitive FS can't dodge provenance stamping/reindex."""
    return rel.startswith("knowledge/") and rel.casefold().endswith(".md")


def _is_memory_area(rel: str) -> bool:
    """knowledge/assistant/ notes carry the consolidation lifecycle
    (memory_status) — except PROFILE.md, whose dedicated profile tool
    manages its own provenance."""
    return rel.startswith("knowledge/assistant/") and not _is_profile_path(rel)


def _is_profile_path(rel: str) -> bool:
    """Casefolded PROFILE.md test. On a case-insensitive FS (macOS APFS —
    the live deployment) ``Path.resolve()`` preserves the requested casing,
    so ``knowledge/assistant/profile.md`` opens the real PROFILE.md while an
    exact-case compare misses it — the same bypass safety.resolve_read
    already casefolds its DENY checks against."""
    return rel.casefold() == PROFILE_NOTE_PATH.casefold()


def _refuse_if_profile(rel: str) -> None:
    """The assistant PROFILE lives under knowledge/assistant/ — inside the
    write allowlist — so the general note verbs would otherwise accept it
    and silently dodge the byte budget + forced memory_status that
    tool_profile_update enforces. Refuse it here so profile_update stays
    the only door (it writes PROFILE.md on its own atomic-write/commit
    path, never through these three tools, so it isn't caught by this)."""
    if _is_profile_path(rel):
        raise ToolError(
            f"{PROFILE_NOTE_PATH} is byte-budgeted; use profile_update "
            "(the general note verbs would bypass the budget)"
        )


def _commit_message(agent: str, action: str) -> str:
    """``mcp(<agent>): <action>``. The agent name is slug-validated at
    config load and the path inside ``action`` already passed safety's
    control-character checks; sanitizing both anyway is defense in depth
    against commit-message injection."""
    return (
        f"mcp({_sanitize_for_commit_message(agent)}): "
        f"{_sanitize_for_commit_message(action)}"
    )


def _audited_write(runtime: Runtime, *, tool: str, path: str | None, fn) -> WriteResult:
    """Run one write-tool body and record how it ended in the audit log.

    Refusals (rate limit, safety, size, exists/missing) are part of the
    protocol and land as ``refused: <reason>``; anything unexpected lands
    as ``error``. Both re-raise so FastMCP surfaces them unchanged."""
    agent = current_agent()
    try:
        result = fn()
    except (ToolError, SafetyError) as exc:
        runtime.audit.tool_event(
            agent=agent, tool=tool, path=path,
            outcome=f"refused: {str(exc)[:200]}",
        )
        raise
    except Exception as exc:  # noqa: BLE001 — audit, then let FastMCP handle it
        runtime.audit.tool_event(
            agent=agent, tool=tool, path=path, outcome="error",
            detail=f"{type(exc).__name__}: {exc}"[:200],
        )
        raise
    runtime.audit.tool_event(
        agent=agent, tool=tool, path=path, outcome="ok",
        detail=f"commit={result.commit_sha[:8] if result.commit_sha else 'none'}",
    )
    return result


def _finish_write(
    runtime: Runtime,
    *,
    rel: str,
    outcome: CommitOutcome,
    graph_changed: bool,
    reindex: bool,
) -> tuple[str, str]:
    """Post-commit bookkeeping shared by every write tool: request an
    async push and enqueue the background reindex. Returns
    ``(push_state, index_refresh)``. Nothing here can fail the write —
    both collaborators only flip flags and poke worker threads."""
    if not outcome.committed:
        # Nothing new on the branch (commit failed, or a no-op write):
        # neither a push nor a reindex has anything to act on.
        return "skipped", "skipped"
    push_state = runtime.push_worker.request_push()
    index_refresh = (
        runtime.refresher.enqueue(rel, graph_changed=graph_changed)
        if reindex
        else "skipped"
    )
    return push_state, index_refresh


def _check_note_size(content: str) -> None:
    if len(content.encode()) > MAX_NOTE_BYTES:
        raise ToolError(
            f"content is {len(content.encode())} bytes; max for notes is {MAX_NOTE_BYTES}"
        )


def _sanitize_for_commit_message(text: str) -> str:
    """Make a string safe to inline into a git commit message.

    safety._resolve_inside_vault already rejects control characters in
    paths; this helper is defense-in-depth so that even if that check
    is ever loosened, the commit message format stays intact: no spoofed
    trailers like ``Co-Authored-By:`` on a fresh line, no log-line
    confusion, no overlong subject.
    """
    # Drop any character a git or web commit-view treats as a line break
    # or formatting control: C0 controls (incl. CR/LF/tab), DEL, NEL
    # (U+0085), LINE/PARAGRAPH SEPARATOR (U+2028/U+2029), and the BOM /
    # zero-width no-break space (U+FEFF). Codepoint filtering avoids
    # embedding invisible characters in this source file.
    _line_breakers = {0x0085, 0x2028, 0x2029, 0xFEFF}
    cleaned = "".join(
        ch for ch in text
        if not (ord(ch) < 0x20 or ord(ch) == 0x7F or ord(ch) in _line_breakers)
    )
    if len(cleaned) > 120:
        cleaned = cleaned[:117] + "..."
    return cleaned


def _atomic_write_text(target: Path, text: str) -> None:
    _atomic_write_bytes(target, text.encode("utf-8"))


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Atomic write with no symlink-prediction window.

    Uses ``tempfile.mkstemp`` in the target's parent directory, which
    creates the temp file with ``O_CREAT|O_EXCL`` and a random suffix.
    Three properties matter:

    - Random suffix → an attacker can't pre-create a symlink at the
      temp path waiting to be opened.
    - ``O_EXCL`` → if the path somehow already exists, ``mkstemp`` fails
      rather than silently following.
    - Same-directory placement → ``os.replace`` is atomic (POSIX rename).

    ``os.replace`` itself doesn't follow symlinks at the destination —
    if the target was a symlink, the rename swaps the directory entry,
    leaving whatever the symlink pointed to untouched. So this routine
    is safe even when the destination path is or becomes a symlink.
    """
    import os
    import tempfile

    parent = target.parent
    try:
        # mkdir + mkstemp are INSIDE the try: a disk-full/permission failure
        # here raised a raw OSError carrying the absolute vault path and temp
        # naming scheme straight to the agent, which the generic 'write
        # failed' below is meant to prevent.
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
    except OSError as exc:
        log.warning("atomic write setup failed for %s: %s", target.name, exc)
        raise ToolError("write failed") from None
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except OSError as exc:
        # Clean up, then return a generic error. The raw OSError carries
        # the random temp path and absolute fs paths; neither should reach
        # the agent. Log the detail server-side instead.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        log.warning("atomic write failed for %s: %s", target.name, exc)
        raise ToolError("write failed") from None
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _commit(cfg: ServerConfig, paths: list[Path], message: str) -> CommitOutcome:
    """Commit only — the push is asynchronous via the runtime's PushWorker.
    The old synchronous push could hold the write path for up to 15s on a
    black-holed network; the worker moves that latency off every write."""
    try:
        return commit_paths(cfg.vault_root, paths=paths, message=message)
    except GitError as exc:
        # The file is already written to disk, but git refused. Report the
        # truth rather than a fake success. Detail (which may carry remote
        # URLs / ssh hints) stays server-side.
        log.warning("commit failed: %s", exc)
        return CommitOutcome(None, False, False, "written to disk but git commit failed")


def _write_result(
    path: str,
    nbytes: int,
    outcome: CommitOutcome,
    *,
    push_state: str,
    index_refresh: str,
) -> WriteResult:
    """Build a WriteResult that tells the agent whether the change was
    actually committed (not just written to disk) and where the async
    push and reindex stand."""
    warning = None if outcome.committed else outcome.detail
    return WriteResult(
        path=path,
        bytes_written=nbytes,
        commit_sha=outcome.sha,
        committed=outcome.committed,
        pushed=outcome.pushed,
        push_state=push_state,
        index_refresh=index_refresh,
        warning=warning,
    )
