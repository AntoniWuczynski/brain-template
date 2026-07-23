"""FastAPI shell + FastMCP mount + bearer-token middleware.

The shell hands MCP traffic to FastMCP's Streamable HTTP transport at
``/mcp``. Everything else is just /health for liveness checks.

Tools are registered against FastMCP via ``@mcp.tool()`` decorators
below; the actual implementations live in ``mcp_server.tools`` and
this module only adapts them.
"""
from __future__ import annotations

import functools
import logging
from contextlib import asynccontextmanager

import anyio

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from . import entity_tools as _entity_tools
from . import memory_tools as _memory_tools
from . import tools as _tools
from .auth import BearerAuthMiddleware
from .config import ServerConfig, load_config
from .runtime import Runtime, build_runtime


def build_app() -> FastAPI:
    """Construct the FastAPI app with auth + MCP wired in."""
    cfg = load_config()
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)sZ %(levelname)-7s %(name)s | %(message)s",
    )
    log = logging.getLogger(__name__)
    log.info(
        "brain MCP server starting (vault=%s, host=%s:%d, git_push=%s)",
        cfg.vault_root, cfg.bind_host, cfg.bind_port, cfg.git_push_on_write,
    )
    # One runtime for the process: audit appender, async push worker,
    # background index refresher. Tools receive it alongside cfg.
    runtime = build_runtime(cfg)

    # DNS-rebinding / Host-header guard. FastMCP defaults to localhost-only,
    # which would REJECT traffic arriving through the Cloudflare Tunnel with
    # the public Host header. Add the operator-configured public host(s) on
    # top of the localhost defaults so the tunnel works and the guard stays on.
    from mcp.server.transport_security import TransportSecuritySettings
    _hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", *cfg.allowed_hosts]
    _origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
    for h in cfg.allowed_hosts:
        _origins += [f"https://{h}", f"http://{h}"]
    mcp = FastMCP(
        "brain-vault",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_hosts,
            allowed_origins=_origins,
        ),
    )
    _register_tools(mcp, cfg, runtime)

    # FastMCP's streamable HTTP needs its session manager's task group
    # running for the duration of the app. When mounted as a sub-app,
    # FastAPI doesn't run the sub-app's lifespan, so we explicitly
    # adopt it here. On shutdown, flush the background workers: the
    # refresher FIRST (its derived-notes commit must exist before the
    # final push), then the push worker's last best-effort push.
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                await anyio.to_thread.run_sync(runtime.refresher.stop)
                await anyio.to_thread.run_sync(runtime.push_worker.stop)

    app = FastAPI(
        title="brain MCP server",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # FastMCP's streamable HTTP app already serves itself at /mcp; mount
    # it at root so the final URL stays /mcp (not /mcp/mcp).
    app.mount("/", mcp.streamable_http_app())
    app.add_middleware(
        BearerAuthMiddleware,
        tokens={token: agent for token, agent in cfg.tokens},
    )

    return app


def _register_tools(mcp: FastMCP, cfg: ServerConfig, runtime: Runtime) -> None:
    """Adapter layer that binds tool functions to FastMCP.

    FastMCP infers tool name and signature from the Python function;
    docstrings become the tool's description visible to the agent.

    Every adapter is ``async`` and offloads the (blocking) tool body to a
    worker thread via ``anyio.to_thread.run_sync``. FastMCP calls sync
    tool functions directly on the event loop, so without this a slow
    git push or the one-off model load would freeze /health and every
    other session. Offloading also makes the in-tool locks and
    concurrency guards behave as intended (real threads to bound).
    """

    async def _offload(fn, *args, **kwargs):
        result = await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))
        return result.model_dump()

    @mcp.tool()
    async def vault_search(query: str, top_k: int = 10, mode: str = "hybrid") -> dict:
        """Search the vault. Returns up to ``top_k`` matching chunks with
        their source paths, scores, and snippets. ``mode``: 'hybrid'
        (default — embeddings + BM25, best for both paraphrase and exact
        identifiers like course codes / error strings), 'dense' (embeddings
        only), or 'lexical' (BM25 only). Use this for "what does the vault
        say about X" questions."""
        return await _offload(_tools.tool_search, cfg, runtime, query=query, top_k=top_k, mode=mode)

    @mcp.tool()
    async def vault_read(path: str) -> dict:
        """Read a text file from the vault by its relative path. Returns
        the full content as UTF-8 text. Refuses binary files, secrets,
        and anything under .git/ or logs/."""
        return await _offload(_tools.tool_read, cfg, runtime, path)

    @mcp.tool()
    async def vault_chunk_context(
        source_relative_path: str, chunk_idx: int, before: int = 1, after: int = 1,
    ) -> dict:
        """Expand one ``vault_search`` hit into its neighbouring chunks —
        cheaper and more focused than ``vault_read`` on the whole file. Pass a
        hit's ``source_relative_path`` and ``chunk_idx``; returns that chunk
        plus ``before``/``after`` neighbours and the source's total chunk
        count. Gated by the same read policy as search."""
        return await _offload(
            _tools.tool_chunk_context, cfg, runtime,
            source_relative_path=source_relative_path, chunk_idx=chunk_idx,
            before=before, after=after,
        )

    @mcp.tool()
    async def vault_list(path: str = "") -> dict:
        """List entries under a vault directory. Empty path lists the root.
        Hidden files (those starting with .) are omitted."""
        return await _offload(_tools.tool_list, cfg, runtime, path)

    @mcp.tool()
    async def vault_metadata_query(
        by: str = "status",
        value: str | None = None,
        limit: int = 50,
    ) -> dict:
        """Query metadata/index.jsonl. ``by`` is one of: status, extension,
        extractor, path_prefix, all. ``value`` is the filter value
        (required unless by=all). Returns up to ``limit`` records with
        per-source metadata (hash, summary, topics, paths)."""
        return await _offload(_tools.tool_metadata_query, cfg, runtime, by=by, value=value, limit=limit)

    @mcp.tool()
    async def vault_related(concept: str, limit: int = 8) -> dict:
        """Given a concept — its slug or display name (e.g. 'packet-switching'
        or 'Packet Switching') — return the concepts most related to it, by
        co-occurrence (tagged on the same documents) and semantic similarity.
        Use this to explore how topics in the vault connect."""
        return await _offload(_tools.tool_related, cfg, runtime, concept=concept, limit=limit)

    @mcp.tool()
    async def vault_create_note(path: str, content: str) -> dict:
        """Create a new Markdown note. ``path`` must live under one of
        knowledge/{notes,projects,research,people,organisations,university,meetings,assistant}.
        Refuses to overwrite an existing note — vault_append_to_note
        extends one, vault_replace_note rewrites one in full.
        knowledge/assistant/PROFILE.md is excluded — it is byte-budgeted,
        write it through profile_update.

        A project is not limited to its knowledge/projects/<slug>/<slug>.md
        overview and log/<date>.md entries: when a durable decision, design,
        artefact or sub-topic deserves its own home, create a focused curated
        note flat beside the overview
        (knowledge/projects/<slug>/<descriptive-kebab-name>.md). Put notes that
        span several projects under knowledge/projects/shared/ and link each
        project with a related_to relation. Give such notes topics: and
        relations: frontmatter so they join the concept and relation graph. Do
        not write into a project's notes/ subdir — that is the human's area."""
        return await _offload(_tools.tool_create_note, cfg, runtime, path=path, content=content)

    @mcp.tool()
    async def vault_replace_note(path: str, content: str) -> dict:
        """Overwrite an existing note under one of the knowledge/ subdirs
        with new content (full replace). Refuses to create a missing file
        — use vault_create_note for that. Use this to regenerate a note in
        full; vault_append_to_note only extends an existing one."""
        return await _offload(_tools.tool_replace_note, cfg, runtime, path=path, content=content)

    @mcp.tool()
    async def vault_append_to_note(path: str, content: str) -> dict:
        """Append ``content`` to an existing note under one of the
        knowledge/ subdirs. Adds a blank-line separator if needed."""
        return await _offload(_tools.tool_append_to_note, cfg, runtime, path=path, content=content)

    @mcp.tool()
    async def vault_update_concept_user_section(slug: str, content: str) -> dict:
        """Replace the user-editable section of a concept note (everything
        below the AUTO-GENERATED-END marker). Auto-generated content
        above the marker is preserved. ``slug`` is the filename without
        the .md extension (e.g. 'tcp-congestion-control')."""
        return await _offload(_tools.tool_update_concept_user_section, cfg, runtime, slug=slug, content=content)

    @mcp.tool()
    async def vault_drop_inbox_file(path: str, content_base64: str) -> dict:
        """Drop a file at inbox/<path> for later ingestion. ``content_base64``
        is the file content base64-encoded. Up to 100 MB. Refuses if a
        file already exists at that path. Use this to hand the ingest
        pipeline a new PDF / DOCX / etc. — the user runs ingest later."""
        return await _offload(_tools.tool_drop_inbox_file, cfg, runtime, path=path, content_base64=content_base64)

    @mcp.tool()
    async def entity_upsert_relation(
        entity_path: str,
        rel: str,
        target: str,
        valid_from: str = "",
        valid_until: str = "",
        source: str = "",
    ) -> dict:
        """Add or close one typed relation in an EXISTING entity note's
        frontmatter. ``rel`` comes from a closed vocabulary: works_at,
        member_of, attended, stakeholder_in, collaborator_on, met_at,
        related_to. ``target`` is a node id — the knowledge/-relative path
        without extension (e.g. 'organisations/acme') — whose note must
        already exist (create entity stubs with vault_create_note first).
        Dates are YYYY-MM-DD; setting ``valid_until`` closes the matching
        open relation (history is superseded, never deleted). ``source``
        is optional provenance: the vault-relative no-extension path of
        the note the relation was learned from."""
        return await _offload(
            _entity_tools.tool_entity_upsert_relation, cfg, runtime,
            entity_path=entity_path, rel=rel, target=target,
            valid_from=valid_from, valid_until=valid_until, source=source,
        )

    @mcp.tool()
    async def entity_append_fact(
        entity_path: str, text: str, source: str, date: str = ""
    ) -> dict:
        """Append one dated, source-linked fact to the '## Log' section of
        an existing entity note. ``text`` is a single line (max 500 chars
        — distil it); ``source`` is required, the vault-relative
        no-extension path the fact was learned from (e.g.
        'knowledge/meetings/2026/2026-06-12-kern-call'); ``date`` is when
        the fact was learned, YYYY-MM-DD, defaulting to today UTC. Facts
        accumulate newest-last and are never rewritten."""
        return await _offload(
            _entity_tools.tool_entity_append_fact, cfg, runtime,
            entity_path=entity_path, text=text, source=source, date=date,
        )

    @mcp.tool()
    async def relations_query(
        rel: str = "",
        entity: str = "",
        target: str = "",
        as_of: str = "",
        include_closed: bool = False,
        limit: int = 50,
    ) -> dict:
        """Query the typed relation graph (read-only). All filters optional
        and ANDed: ``rel`` (closed vocab: works_at, member_of, attended,
        stakeholder_in, collaborator_on, met_at, related_to), ``entity`` (the
        node declaring it, e.g. 'people/anna-kowalska'), ``target`` (reverse
        lookup, e.g. 'organisations/acme' -> who works there). ``as_of``
        (YYYY-MM-DD) returns relations whose interval contains that date —
        the supersede history made queryable ('where did X work last
        spring?'). Without ``as_of``, only currently-open relations unless
        ``include_closed``. Returns each edge with its interval and source."""
        return await _offload(
            _entity_tools.tool_relations_query, cfg, runtime,
            rel=rel, entity=entity, target=target, as_of=as_of,
            include_closed=include_closed, limit=limit,
        )

    @mcp.tool()
    async def meeting_create(
        date: str,
        title: str,
        attendees: list[str],
        project: str = "",
        body: str = "",
    ) -> dict:
        """Create a meeting note at knowledge/meetings/<YYYY>/<date>-<slug>.md
        and record an 'attended' relation on every attendee. ``attendees``
        are people/ node ids (e.g. 'people/anna-kowalska') whose notes must
        already exist — missing ones are all listed in the error so you can
        create the stubs in one pass. ``project`` (optional) is a projects/
        node id; it lands as a related_to relation in the meeting's own
        frontmatter. ``body`` goes under '## Notes'. Everything is written
        and committed together, or not at all."""
        return await _offload(
            _entity_tools.tool_meeting_create, cfg, runtime,
            date=date, title=title, attendees=attendees,
            project=project, body=body,
        )

    @mcp.tool()
    async def memory_search(
        query: str,
        top_k: int = 10,
        recency_halflife_days: float = 30.0,
        types: list[str] | None = None,
    ) -> dict:
        """Memory-flavoured semantic search: same index as vault_search but
        re-ranked by recency (half-life decay on each note's 'updated'
        date; default halflife 30 days) and memory status (superseded
        notes sink). Use this for "what do I currently know about X";
        use vault_search for timeless archival lookups. ``types``
        optionally filters to knowledge subdirs (people, organisations,
        projects, meetings, notes, research, university, assistant)
        and/or 'archive' for ingested sources."""
        return await _offload(
            _memory_tools.tool_memory_search, cfg, runtime,
            query=query, top_k=top_k,
            recency_halflife_days=recency_halflife_days, types=types,
        )

    @mcp.tool()
    async def profile_update(content: str) -> dict:
        """Replace the assistant's standing profile of the user
        (knowledge/assistant/PROFILE.md) in full. The profile rides into
        every session, so it has a hard byte budget (default 4096): curate,
        don't accumulate — fold in what changed, drop what no longer
        matters. This is the only tool that may create the file; it is
        always stamped memory_status: consolidated."""
        return await _offload(_memory_tools.tool_profile_update, cfg, runtime, content=content)


# No module-level ``app = build_app()``: that would call load_config()
# at import time and crash any import without the env set. uvicorn uses
# factory mode (`mcp_server.app:build_app`, --factory); see __main__.py.
