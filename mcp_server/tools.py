"""The six write MCP tools, plus shared infrastructure both this module
and ``tools_read`` build on.

The tools share helpers from this module's siblings (safety, git_ops,
provenance) and reuse the ingest_lib package for metadata reads. Every
tool takes ``(cfg, runtime, ...)``: cfg is pure env-derived data, runtime
carries the stateful collaborators (audit log, async push worker,
background index refresher).

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

The read tools (``vault_search``, ``vault_chunk_context``, ``vault_read``,
``vault_list``, ``vault_metadata_query``, ``vault_related``) live in
``mcp_server.tools_read``; this module keeps ``vault_create_note``,
``vault_replace_note``, ``vault_append_to_note``,
``vault_update_concept_user_section``, ``vault_update_compiled_truth``,
``vault_drop_inbox_file``, the
``WriteResult`` model, and the rate-bucket / write-commit primitives
``tools_read`` and ``entity_tools``/``memory_tools`` import back from here.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

# Make the ingest_lib package importable. Same shim the CLI scripts use.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ingest_lib.atomic import atomic_write_bytes  # noqa: E402
from ingest_lib.dream import (  # noqa: E402 — one definition of the fence
    COMPILED_TRUTH_END,
    COMPILED_TRUTH_START,
    compiled_truth_span,
    marker_state,
)
from ingest_lib.notes import (  # noqa: E402
    _split_frontmatter,
    _split_frontmatter_raw,
)
from ingest_lib.relations import (  # noqa: E402
    _HEADING_RE,
    _LOG_HEADING,
    Relation,
    parse_relations,
)

from .errors import ToolError
from .config import (
    MAX_INBOX_BYTES,
    MAX_NOTE_BYTES,
    PROFILE_NOTE_PATH,
    ServerConfig,
    WRITE_RATE_PER_MINUTE,
)
from .git_ops import CommitOutcome, GitError, commit_paths, current_branch
from .identity import current_agent
from .provenance import frontmatter_signature, stamp_provenance
from .runtime import Runtime
from .safety import (
    SafetyError,
    resolve_inbox,
    resolve_write_concept,
    resolve_write_under_allowlist,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiting (writes only)
# ---------------------------------------------------------------------------

class _RateBucket:
    """Sliding-window rate limiter. Per-agent windows: one deque per agent
    name, so a runaway or compromised agent only starves its own budget."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        # Tool calls run in a threadpool; the dict/deques must not be
        # mutated from multiple workers concurrently.
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            hits = self._hits.get(key)
            if hits is not None:
                while hits and hits[0] < cutoff:
                    hits.popleft()
                if not hits:
                    # Window is clear: drop the empty deque so the dict
                    # doesn't grow unboundedly across distinct agent names.
                    del self._hits[key]
                    hits = None
            if hits is not None and len(hits) >= self._max:
                return False
            self._hits.setdefault(key, deque()).append(now)
            return True


_write_bucket = _RateBucket(WRITE_RATE_PER_MINUTE)


def _rate_check_write() -> None:
    if not _write_bucket.allow(current_agent()):
        raise ToolError(
            f"write rate limit exceeded ({WRITE_RATE_PER_MINUTE}/minute); back off and retry"
        )


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
# Input / output schemas
# ---------------------------------------------------------------------------

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
            _refuse_bad_relations(content)
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
                _refuse_bad_relations(content)
                try:
                    old: str | None = resolved.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    old = None  # unreadable old text: assume the graph moved
                if old is not None:
                    # Before anything reaches disk: a rewrite may add and
                    # supersede, never drop relation intervals or Log lines.
                    _refuse_history_loss(old, content)
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
            # _check_note_size only bounded the client content; the composed
            # note (auto-generated header + content) can still exceed the cap.
            if len(new_full.encode()) > MAX_NOTE_BYTES:
                raise ToolError(
                    f"composed concept note would exceed {MAX_NOTE_BYTES} bytes; "
                    "trim the user section"
                )
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


def _with_first_compiled_truth_fence(full: str, block: str) -> str:
    """``full`` with its first compiled-truth fence inserted directly below
    the frontmatter, carrying ``block``.

    Above the note's own prose, never appended: an entity note's last
    section is usually ``## Log``, and ``dream._log_bullets`` reads a
    section to the next heading — a fence appended at end of file therefore
    lands inside the Log, and its lines read back as Log bullets.
    """
    raw = _split_frontmatter_raw(full)
    if raw is None:
        head, rest = "", full
    else:
        _yaml_block, rest = raw
        head = full[: len(full) - len(rest)]
        if not head.endswith("\n"):
            head += "\n"
        head += "\n"
    fence = f"{COMPILED_TRUTH_START}\n{block}\n{COMPILED_TRUTH_END}\n"
    body = rest.lstrip("\n")
    return f"{head}{fence}\n{body}" if body else f"{head}{fence}"


def tool_update_compiled_truth(
    cfg: ServerConfig,
    runtime: Runtime,
    path: str,
    content: str,
) -> WriteResult:
    """Replace the compiled-truth block of an entity note — the text
    between the COMPILED-TRUTH markers, and nothing else.

    The dream pass's compiled-truth job owns that block and only that
    block. Sent through ``vault_replace_note`` the promise was prose: the
    history guard catches a dropped relation entry or ``## Log`` bullet,
    but a rewrite that ate the note's hand-written Overview / Stack /
    Architecture sections was accepted, committed and pushed. Here the
    scope is mechanical — the server splices the new text between the
    markers and every other byte of the note is the one already on disk,
    bar the provenance keys it stamps on every write (below).

    A note with NO fence gets its first one here, inserted directly after
    the frontmatter and above the hand-written body — the server picks the
    place because the alternative (an append, which is all a skill can do)
    lands the fence at end of file, i.e. inside whatever the last section
    happens to be, and a fence inside ``## Log`` makes the pass read its
    own summary back as Log evidence. An AMBIGUOUS fence (one marker
    without the other, out of order, duplicated) is still refused: there is
    no safe place to write, and guessing where the block ends is how a
    hand-written section gets eaten. Refuses the assistant memory areas
    outright — a memory-fact note has no compiled truth, and its lifecycle
    keys are consolidate's.

    ``content`` may not itself contain either marker, and the composed note
    is re-checked before anything reaches disk: a payload carrying a marker
    (the skill's own example block, copied) would otherwise splice a second
    fence into the note, and a note with two fences is one this tool refuses
    for ever after — permanently unwritable, and permanently counted as a
    real change by the dream gate.

    Provenance is stamped like every other write
    (``last_written_by``/``written_via``, AGENTS.md's server contract). The
    values are derived from the agent, not the clock, so a nightly refresh
    by the same agent leaves the frontmatter byte-identical and
    ``dream._only_compiled_truth_changed`` still sees a compiled-truth-only
    edit.
    """
    def _do() -> WriteResult:
        _rate_check_write()
        _check_note_size(content)
        agent = current_agent()
        resolved = resolve_write_under_allowlist(cfg.vault_root, path)
        rel = resolved.relative_to(cfg.vault_root).as_posix()
        _refuse_if_profile(rel)
        if not _is_knowledge_md(rel):
            raise ToolError(
                f"{path!r} is not a knowledge/ Markdown note; compiled truth "
                "lives on entity notes (people/, organisations/, projects/)"
            )
        if _is_memory_area(rel):
            raise ToolError(
                "knowledge/assistant/ notes carry the memory lifecycle, not a "
                "compiled-truth block; propose a fact note instead"
            )
        if COMPILED_TRUTH_START in content or COMPILED_TRUTH_END in content:
            raise ToolError(
                "the compiled-truth body must not contain the fence markers "
                f"({COMPILED_TRUTH_START} / {COMPILED_TRUTH_END}) — send the "
                "block's text only; the server owns the markers"
            )
        with _write_lock:
            if not resolved.is_file():
                raise ToolError(f"file does not exist: {path!r}")
            try:
                full = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise ToolError("note could not be read") from None
            block = content.strip("\n")
            span = compiled_truth_span(full)
            if span is None:
                if marker_state(full) != "absent":
                    raise ToolError(
                        f"{path!r} has an ambiguous compiled-truth fence (one "
                        "marker without the other, out of order, or duplicated); "
                        "refusing to write — a human repairs the markers"
                    )
                new_full = _with_first_compiled_truth_fence(full, block)
            else:
                start, end = span
                new_full = full[:start] + "\n" + block + "\n" + full[end:]
            new_full = stamp_provenance(
                new_full, agent=agent, mode="replace",
                memory_area=False, prior=full,
            )
            if marker_state(new_full) != "ok":
                raise ToolError(
                    "the composed note would not have exactly one compiled-truth "
                    "fence; refusing to write (nothing on disk was changed)"
                )
            if len(new_full.encode()) > MAX_NOTE_BYTES:
                raise ToolError(
                    f"composed note would exceed {MAX_NOTE_BYTES} bytes; "
                    "trim the compiled-truth block"
                )
            _atomic_write_text(resolved, new_full)
            outcome = _commit(
                cfg, [resolved], _commit_message(agent, f"update compiled truth {path}")
            )
        # Only provenance keys can have moved in the frontmatter — the
        # splice never reaches it and stamping touches nothing else — so
        # relations/topics are unchanged and the graph cannot have moved;
        # the body did change, so reindex.
        push_state, index_refresh = _finish_write(
            runtime, rel=rel, outcome=outcome, graph_changed=False, reindex=True,
        )
        return _write_result(
            rel, len(content.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _audited_write(
        runtime, tool="vault_update_compiled_truth", path=path, fn=_do
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
        import binascii

        _rate_check_write()
        try:
            data = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError):
            # Only the bad-input error. A bare `except Exception` here
            # relabelled MemoryError on a huge payload as "not valid
            # base64", sending the caller to fix input that was fine.
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


def _refuse_bad_relations(content: str) -> None:
    """Refuse a client-supplied ``relations:`` block that parse_relations
    (the reader every consumer uses) would silently drop edges from —
    the generic write tools must not persist what entity_upsert_relation
    would refuse to write."""
    frontmatter, _body = _split_frontmatter(content)
    if "relations" not in frontmatter:
        return
    _relations, problems = parse_relations(frontmatter)
    if problems:
        raise ToolError(
            "relations frontmatter rejected: " + "; ".join(problems) +
            " — see AGENTS.md 'Relations frontmatter shape': a list of "
            "{rel, target} mappings, rel from the closed vocabulary, target "
            "a node id like projects/<slug> (no knowledge/ prefix, no .md)"
        )


def _log_section(text: str) -> tuple[bool, list[str]]:
    """``(has_log_section, bullet_lines)`` for a note's ``## Log``.

    Same section detection ``relations.append_fact_to_log`` writes with:
    the first line that is exactly ``## Log``, ending at the next heading
    of any level. Bullets are compared stripped (and with a trailing
    ``\\r`` dropped, as the append path does) so re-indentation or CRLF
    endings do not read as a deleted line.
    """
    lines = text.split("\n")
    heading_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == _LOG_HEADING:
            heading_idx = i
            break
    if heading_idx < 0:
        return False, []
    end = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        if _HEADING_RE.match(lines[j]):
            end = j
            break
    bullets = []
    for j in range(heading_idx + 1, end):
        stripped = lines[j].rstrip("\r").strip()
        if stripped.startswith("- "):
            bullets.append(stripped)
    return True, bullets


def _edge_key(relation: Relation) -> tuple[str, str, str, str]:
    """The fields that identify one interval of one typed edge. ``source``
    is provenance, not identity — re-sourcing an entry is not a deletion."""
    return (relation.rel, relation.target, relation.valid_from, relation.valid_until)


def _describe(relation: Relation) -> str:
    span = relation.valid_from or "(open start)"
    span += f"..{relation.valid_until}" if relation.valid_until else ".. (open)"
    return f"{relation.rel} -> {relation.target} [{span}]"


def _refuse_history_loss(old: str, new: str) -> None:
    """Refuse a full-note replace that would drop relation history or
    ``## Log`` lines the note already carries.

    AGENTS.md is explicit that the closed intervals ARE the queryable
    history and that ``## Log`` is append-only, but those rules were
    enforced only by the typed tools (``entity_upsert_relation``,
    ``entity_append_fact``). The dream pass rewrites entity notes through
    ``vault_replace_note``, regenerating the relation block freehand — a
    dropped entry or bullet was silent, and the tool reported success.

    Superseding still passes: adding entries is free, and an entry that
    was open may come back carrying a ``valid_until`` (closing an interval
    is history, not loss). Only notes that ARE entity notes (relations or
    a ``## Log``) are guarded; an ordinary note is untouched.
    """
    old_fm, _old_body = _split_frontmatter(old)
    old_relations, _old_problems = parse_relations(old_fm)
    has_log, old_bullets = _log_section(old)
    if not old_relations and not has_log:
        return   # not an entity note: nothing to preserve

    new_fm, _new_body = _split_frontmatter(new)
    new_relations, _new_problems = parse_relations(new_fm)
    new_edges = {_edge_key(r) for r in new_relations}
    # (rel, target, valid_from) triples the new content closes: an OLD open
    # entry showing up here has been superseded, not deleted.
    superseded = {
        (r.rel, r.target, r.valid_from) for r in new_relations if r.valid_until
    }
    lost_relations = [
        r for r in old_relations
        if _edge_key(r) not in new_edges
        and not (not r.valid_until and (r.rel, r.target, r.valid_from) in superseded)
    ]
    _new_has_log, new_bullets = _log_section(new)
    kept = set(new_bullets)
    lost_bullets = [b for b in old_bullets if b not in kept]
    if not lost_relations and not lost_bullets:
        return

    parts = []
    if lost_relations:
        parts.append(
            "relation entries: " + "; ".join(_describe(r) for r in lost_relations)
        )
    if lost_bullets:
        parts.append("## Log lines: " + "; ".join(repr(b) for b in lost_bullets))
    raise ToolError(
        "refusing to replace: the new content drops history already on the note — "
        + " | ".join(parts)
        + " — AGENTS.md 'supersede, never delete': closed intervals are the "
        "queryable history and ## Log lines are never rewritten. Append to the "
        "note (vault_append_to_note / entity_append_fact) or supersede the entry "
        "(entity_upsert_relation with valid_until) rather than rewriting it."
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


def _require_configured_branch(runtime: Runtime) -> None:
    """Refuse BEFORE any disk write when the vault has a branch other than
    the configured one checked out. ``commit_paths`` guards the commit too,
    but by then the file is on disk and the tool would report a
    written-but-uncommitted result; refusing up front keeps the shared
    working tree clean and makes the cause visible to the client."""
    expected = runtime.push_worker.branch
    current = current_branch(runtime.push_worker.vault_root)
    if current != expected:
        raise ToolError(
            f"refusing to write: the vault has {current!r} checked out, not the "
            f"configured {expected!r}; a write now would be stranded off {expected!r}"
        )


def _audited_write(
    runtime: Runtime, *, tool: str, path: str | None, fn: Callable[[], WriteResult]
) -> WriteResult:
    """Run one write-tool body and record how it ended in the audit log.

    Refusals (rate limit, safety, size, exists/missing) are part of the
    protocol and land as ``refused: <reason>``; anything unexpected lands
    as ``error``. Both re-raise so FastMCP surfaces them unchanged."""
    agent = current_agent()
    try:
        _require_configured_branch(runtime)
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
    """Atomic write with no symlink-prediction window, plus error redaction.

    The mechanics are ``ingest_lib.atomic.atomic_write_bytes`` — mkdir,
    ``tempfile.mkstemp`` in the target's parent, write → flush → fsync →
    ``os.replace``, unlink-and-re-raise on any ``BaseException``. Three
    properties of that mkstemp matter to the server:

    - Random suffix → an attacker can't pre-create a symlink at the
      temp path waiting to be opened.
    - ``O_EXCL`` → if the path somehow already exists, ``mkstemp`` fails
      rather than silently following.
    - Same-directory placement → ``os.replace`` is atomic (POSIX rename).

    ``os.replace`` itself doesn't follow symlinks at the destination —
    if the target was a symlink, the rename swaps the directory entry,
    leaving whatever the symlink pointed to untouched. So this routine
    is safe even when the destination path is or becomes a symlink.

    What this wrapper adds is redaction. Every ``OSError`` — from the mkdir
    and mkstemp setup as much as from the write itself — carries the
    absolute vault path, the temp naming scheme, or both; none of that may
    reach the agent, so it is logged server-side and re-raised as a generic
    ``ToolError``. The mode stays mkstemp's 0600, matching every other note
    writer in the vault (``ingest_lib.notes._atomic_write``), so a note's
    permissions don't flip depending on which writer touched it last.
    """
    try:
        atomic_write_bytes(
            target, data, prefix=f".{target.name}.", suffix=".tmp"
        )
    except OSError as exc:
        log.warning("atomic write failed for %s: %s", target.name, exc)
        raise ToolError("write failed") from None


def _commit(cfg: ServerConfig, paths: list[Path], message: str) -> CommitOutcome:
    """Commit only — the push is asynchronous via the runtime's PushWorker.
    The old synchronous push could hold the write path for up to 15s on a
    black-holed network; the worker moves that latency off every write."""
    try:
        return commit_paths(
            cfg.vault_root, paths=paths, message=message, expected_branch=cfg.git_branch
        )
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
