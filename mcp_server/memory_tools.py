"""Memory-flavoured MCP tools: recency-aware search and the profile.

``memory_search`` answers "what do I currently know about X" — the same
semantic index as ``vault_search``, re-ranked by recency (half-life decay
on each note's ``updated`` date) and memory status (superseded notes
sink). ``profile_update`` rewrites the assistant's standing profile of
the user under a hard byte budget: the profile rides into every session,
so its size is a token cost, not a storage cost.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from pydantic import BaseModel

# Make the ingest_lib package importable. Same shim the CLI scripts use.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Module (not function) import so tests can monkeypatch
# ``recency.memory_search`` and the patch is visible through here.
from ingest_lib import recency as _recency  # type: ignore[import-not-found]  # noqa: E402
from ingest_lib.config import paths_for_root  # type: ignore[import-not-found]  # noqa: E402

from . import tools as _tools
from .config import PROFILE_NOTE_PATH, ServerConfig
from .identity import current_agent
from .provenance import stamp_provenance
from .runtime import Runtime
from .safety import SafetyError, resolve_read, resolve_write_under_allowlist
from .tools import ToolError, WriteResult

log = logging.getLogger(__name__)


class MemoryHitOut(BaseModel):
    score: float            # cosine * recency * status_weight
    cosine: float           # raw similarity
    recency: float          # half-life decay factor, in (0, 1]
    status_weight: float    # 1.0 unless memory_status demotes the note
    source_relative_path: str
    title: str
    chunk_idx: int
    snippet: str
    updated: str            # the timestamp the decay was computed from


class MemorySearchOut(BaseModel):
    hits: list[MemoryHitOut]


def tool_memory_search(
    cfg: ServerConfig,
    runtime: Runtime,
    query: str,
    top_k: int = 10,
    recency_halflife_days: float = 30.0,
    types: list[str] | None = None,
) -> MemorySearchOut:
    """Semantic search re-ranked by recency and memory status."""
    if not query or not query.strip():
        raise ToolError("query must be non-empty")
    _tools._check_query_len(query)
    if not 1 <= top_k <= 50:
        raise ToolError("top_k must be in [1, 50]")
    if not 1.0 <= recency_halflife_days <= 3650.0:
        raise ToolError("recency_halflife_days must be in [1, 3650]")
    _tools._rate_check_search()

    paths = paths_for_root(cfg.vault_root)
    with _tools._search_guard:
        try:
            hits = _recency.memory_search(
                paths, query,
                top_k=top_k,
                halflife_days=recency_halflife_days,
                types=types,
                logger=log,
            )
        except ValueError as exc:
            # Unknown type token: a protocol-level refusal with the valid
            # vocabulary in the message, not a server error.
            raise ToolError(str(exc)) from None

    # Gate hits by the read policy, exactly like tool_search: only return
    # a hit whose backing artifact the agent could read directly.
    safe = []
    for h in hits:
        try:
            resolve_read(
                cfg.vault_root,
                _tools._hit_gate_path(h.source_relative_path, h.origin),
            )
        except SafetyError:
            continue
        safe.append(h)
    # Audit the query plus the GATED hit paths — what the agent actually saw.
    runtime.audit.access_event(
        agent=current_agent(),
        tool="memory_search",
        paths=[h.source_relative_path for h in safe],
        query=query,
    )
    return MemorySearchOut(
        hits=[
            MemoryHitOut(
                score=h.score,
                cosine=h.cosine,
                recency=h.recency,
                status_weight=h.status_weight,
                source_relative_path=h.source_relative_path,
                title=h.title,
                chunk_idx=h.chunk_idx,
                snippet=h.snippet,
                updated=h.updated,
            )
            for h in safe
        ]
    )


_MEMORY_STATUS_LINE = re.compile(r"^memory_status:")


def _ensure_consolidated(text: str) -> str:
    """Force ``memory_status: consolidated`` into the frontmatter fence
    (guaranteed present — ``stamp_provenance`` ran first). The profile is
    consolidated BY DEFINITION: every write is a curation pass, so any
    client-supplied lifecycle value would be wrong and is overridden."""
    lines = text.splitlines(keepends=True)
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx < 0:  # unreachable after stamping; fail safe anyway
        return f"---\nmemory_status: consolidated\n---\n{text}"
    kept = [ln for ln in lines[1:close_idx] if not _MEMORY_STATUS_LINE.match(ln)]
    return (
        lines[0] + "".join(kept) + "memory_status: consolidated\n"
        + "".join(lines[close_idx:])
    )


def tool_profile_update(cfg: ServerConfig, runtime: Runtime, content: str) -> WriteResult:
    """Rewrite the assistant's profile note in full, under the byte budget.

    Create-or-replace: this is the ONLY tool that may create
    ``knowledge/assistant/PROFILE.md`` — the general note verbs treat it
    like any other note but the profile's lifecycle (always consolidated,
    always within budget) is enforced here.
    """
    def _do() -> WriteResult:
        _tools._rate_check_write()
        agent = current_agent()
        resolved = resolve_write_under_allowlist(cfg.vault_root, PROFILE_NOTE_PATH)
        with _tools._write_lock:
            # Read the current profile (if any) as prior so any server-owned
            # keys survive from what the server last wrote, not the client
            # body. mode="replace" always: the profile is create-or-replace
            # and its lifecycle is forced to consolidated just below.
            try:
                prior: str | None = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                prior = None
            # memory_area=False: the consolidation lifecycle is handled right
            # below — the profile never passes through unconsolidated.
            body = stamp_provenance(
                content, agent=agent, mode="replace", memory_area=False, prior=prior
            )
            body = _ensure_consolidated(body)
            # Enforce the budget on the FINAL note (content + provenance +
            # memory_status), not the pre-stamp content: the profile rides
            # into every session in full, so the whole note is the token cost.
            final_bytes = len(body.encode())
            if final_bytes > cfg.profile_max_bytes:
                raise ToolError(
                    f"profile note is {final_bytes} bytes after provenance "
                    f"stamping; the cap is {cfg.profile_max_bytes} — the "
                    "profile is a token budget, not a notebook: curate, don't "
                    "accumulate"
                )
            _tools._atomic_write_text(resolved, body)
            outcome = _tools._commit(
                cfg, [resolved], _tools._commit_message(agent, "profile update")
            )
        push_state, index_refresh = _tools._finish_write(
            runtime, rel=PROFILE_NOTE_PATH, outcome=outcome,
            graph_changed=False, reindex=True,
        )
        return _tools._write_result(
            PROFILE_NOTE_PATH, len(body.encode()), outcome,
            push_state=push_state, index_refresh=index_refresh,
        )

    return _tools._audited_write(
        runtime, tool="profile_update", path=PROFILE_NOTE_PATH, fn=_do
    )
