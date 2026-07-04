# mcp/ — the brain MCP server

> **Status: implemented.** The server lives in [`mcp_server/`](../mcp_server/).
> This document is the contract it exposes; deployment lives in
> [`DEPLOY.md`](DEPLOY.md).

The Model Context Protocol (MCP) lets agents (Claude, Codex, others) talk to
tools over a uniform interface. The server exposes this vault as one such
resource so any MCP-aware agent can search, read, write, and hand files to
ingestion without per-agent custom integrations.

## Goals

1. **Single source of truth.** The vault filesystem (this repo) is
   authoritative; the MCP server is a thin adapter on top. Every write is
   committed to git (and pushed asynchronously when configured).
2. **Multi-agent.** Multiple agents may connect concurrently, each with its
   own bearer token and identity; writes are serialised behind a global lock
   and a git lock.
3. **Safe by construction.** Reads are allowlist-gated; writes are confined
   to a small set of `knowledge/` subtrees. `archive/raw/` is never writable.
4. **Determinism.** Identical inputs produce identical note bodies.

## Tools

Fifteen tools, registered in [`mcp_server/app.py`](../mcp_server/app.py) and
implemented in [`mcp_server/tools.py`](../mcp_server/tools.py) (read/write),
[`mcp_server/entity_tools.py`](../mcp_server/entity_tools.py) (entity) and
[`mcp_server/memory_tools.py`](../mcp_server/memory_tools.py) (memory).

### Read

| Name | Description | I/O |
|---|---|---|
| `vault_search` | Semantic search over processed sources **and curated knowledge notes**. Knowledge hits carry their vault path (`knowledge/...`) as `source_relative_path` and are read-gated on that path. | `query: string, top_k?: int=10` → `{hits: [{score, source_relative_path, title, chunk_idx, snippet}]}` |
| `vault_read` | Read a UTF-8 text file by vault-relative path. Refuses binary, secrets, `.git/`, `logs/`. | `path: string` → `{path, content, size_bytes}` |
| `vault_list` | List a directory (hidden entries omitted). | `path?: string=""` → `{path, entries: [{name, is_dir, size_bytes?}]}` |
| `vault_metadata_query` | Query `metadata/index.jsonl`. | `by: "status"\|"extension"\|"extractor"\|"path_prefix"\|"all", value?: string, limit?: int=50` → `{records: [...]}` |
| `vault_related` | Concepts most related to a concept, by co-occurrence + semantic similarity. | `concept: string, limit?: int=8` → `{concept, related: [...]}` |
| `relations_query` | Structured, time-aware query over the typed relation graph. Filter by `rel` (closed vocab), `entity` (declaring node), `target` (reverse lookup — who points here). `as_of` (YYYY-MM-DD) returns relations whose interval contains that date — the supersede history, queryable; without it, only open relations unless `include_closed`. | `rel?: string, entity?: string, target?: string, as_of?: string, include_closed?: bool=false, limit?: int=50` → `{relations: [{entity, rel, target, valid_from, valid_until, source}]}` |
| `memory_search` | Same index as `vault_search`, re-ranked for **memory**: `score = cosine × recency × status_weight`. Recency is half-life decay on each note's `updated` date; `memory_status: superseded` notes sink (×0.2). `types` filters to knowledge subdirs (`people`, `organisations`, `projects`, `meetings`, `notes`, `research`, `university`, `assistant`) and/or `archive`; an unknown token is refused, not ignored. Use this for "what do I currently know about X"; `vault_search` for timeless archival lookups. | `query: string, top_k?: int=10, recency_halflife_days?: float=30, types?: [string]` → `{hits: [{score, cosine, recency, status_weight, source_relative_path, title, chunk_idx, snippet, updated}]}` |

### Write

Every write commits to the vault's git branch and returns a `WriteResult`:

```
{path, bytes_written, commit_sha, committed,
 pushed,         # legacy — the push is async now, always false at return
 push_state,     # "queued" | "disabled" | "skipped"
 index_refresh,  # "queued" | "off" | "skipped"
 warning}        # set when committed is false
```

`committed` tells the agent the change reached a commit; `push_state` and
`index_refresh` report where the **asynchronous** push and background
reindex stand (see "After a write" below). `pushed` is kept for
compatibility only.

| Name | Description | I/O |
|---|---|---|
| `vault_create_note` | Create a **new** note. Refuses to overwrite. | `path, content` → `WriteResult` |
| `vault_replace_note` | **Overwrite an existing** note in full. Refuses to create a missing file. | `path, content` → `WriteResult` |
| `vault_append_to_note` | Append to an existing note. | `path, content` → `WriteResult` |
| `vault_update_concept_user_section` | Replace the user-editable section of a concept note (below the `AUTO-GENERATED-END` marker). | `slug, content` → `WriteResult` |
| `vault_drop_inbox_file` | Drop a (possibly binary) file under `inbox/` for later ingestion. | `path, content_base64` → `WriteResult` |
| `profile_update` | Create-or-replace the assistant's standing profile (`knowledge/assistant/PROFILE.md`) in full. Refuses content over the byte budget (`BRAIN_PROFILE_MAX_BYTES`, default 4096) — the profile rides into every session, so its size is a token cost. The only tool that may create the file; always stamped `memory_status: consolidated`. | `content` → `WriteResult` |

### Entity

Typed counterparts to the free-text write verbs — each enforces the
entity-memory contract from [`AGENTS.md`](../AGENTS.md): relations come from
the closed vocabulary (`works_at`, `member_of`, `attended`,
`stakeholder_in`, `collaborator_on`, `met_at`, `related_to`), targets are
node ids (`knowledge/`-relative paths without extension, e.g.
`organisations/acme`) whose notes must already exist, and history is
superseded, never deleted. None of these tools creates entity notes —
that is `vault_create_note`'s job, so "who made this note" stays unambiguous.

| Name | Description | I/O |
|---|---|---|
| `entity_upsert_relation` | Add or close one typed relation in an existing entity note's frontmatter. A set `valid_until` closes the matching open entry; an identical entry is a no-op (no commit); anything else appends. Refuses: unknown `rel`, missing entity or target note, non-canonical dates (strict `YYYY-MM-DD`), a `source` that doesn't exist. | `entity_path, rel, target, valid_from?, valid_until?, source?` → `WriteResult + {action: "added"\|"closed"\|"noop"}` |
| `entity_append_fact` | Append one dated, source-linked fact bullet to the entity note's `## Log` (newest last, never rewritten). Refuses: multi-line or empty text, facts over 500 chars, a missing/invalid `source` (required — vault-relative no-extension path), bad dates. `date` defaults to today UTC. | `entity_path, text, source, date?` → `WriteResult` |
| `meeting_create` | Create `knowledge/meetings/<YYYY>/<date>-<slug>.md` and record an `attended` relation on every attendee — all-or-nothing in **one commit**. Attendees are `people/` node ids; all missing attendee notes are listed in one error so stubs can be created in one pass. Optional `project` (a `projects/` node id) lands as `related_to` in the meeting's frontmatter. Refuses to overwrite an existing meeting. | `date, title, attendees: [string], project?, body?` → `WriteResult` |

**Write allowlist** (`mcp_server/config.py`): `create`/`replace`/`append`
notes may only touch
`knowledge/{notes,projects,research,people,organisations,university,meetings,assistant}`.
Concept notes use the dedicated `update_concept_user_section` tool.
`knowledge/assistant/PROFILE.md` lives inside the allowlist but the three
general verbs **refuse** it — that is what makes `profile_update`'s byte
budget (`BRAIN_PROFILE_MAX_BYTES`) real, since it would otherwise be
trivially bypassable by a direct `vault_replace_note`. `inbox/` is writable
only through `vault_drop_inbox_file`. Everything else — `archive/`,
`metadata/`, `scripts/`, the server's own code — is refused.

`create` vs `replace` vs `append`: there is no implicit "upsert". To
regenerate a note in full an agent reads it, rebuilds the body (preserving
any frontmatter it wants to keep), then calls `vault_replace_note` — an
explicit verb, never a silent truncation.

## After a write: provenance, commits, push, reindex

- **Provenance is server-asserted.** Every Markdown note written under
  `knowledge/` gets frontmatter stamped by the server, never trusted from
  the client: `author: 'agent:<name>'` + `written_via: mcp` on create
  (plus `memory_status: unconsolidated` under `knowledge/assistant/`);
  `last_written_by` + `written_via` on replace/append, leaving the
  create-time `author` and any `memory_status` alone. Client-supplied
  values for these keys are overridden.
- **Commits are attributed**: `mcp(<agent>): <action>`, where `<agent>` is
  the name the presented token maps to. The background reindex commits
  derived notes as `mcp: refresh derived notes`.
- **The push is asynchronous.** The commit lands before the tool returns;
  a single background worker pushes the branch and retries failures on a
  capped backoff (30s → 5min). `push_state: "queued"` means a push is owed,
  not done — hence `pushed` is always false at return.
- **The reindex is automatic.** A background refresher re-embeds written
  notes into the semantic index (debounced, batched), and — when a write
  changed `topics`/`relations` frontmatter — rebuilds the connection graph,
  concept notes, and entity dashboards. Notes written over MCP are
  searchable within seconds; no manual rebuild.

## Authentication and isolation

- **Bearer token** on every request (`Authorization: Bearer …`), checked in
  `mcp_server/auth.py` with a constant-time compare. `/health` is the only
  unauthenticated path.
- **Per-agent identity**: `BRAIN_MCP_TOKENS=name=token,name2=token2` maps
  each token to a named agent (lowercase slug, max 32 chars; tokens min 24
  chars). The legacy single `BRAIN_MCP_BEARER_TOKEN` still works and maps to
  the agent `default`; both may coexist while migrating clients, but a
  shared token or a second `default` is refused at startup — identity must
  be unambiguous. The agent name lands in commit messages, frontmatter
  provenance, and the audit logs.
- **Audit logs** (gitignored, append-only JSONL): `logs/mcp-audit.jsonl`
  records every write/mutating call and its outcome (ok / refused / error);
  `logs/mcp-access.jsonl` records reads and searches with the vault paths
  the agent actually saw. Logging is fail-open — telemetry loss never
  breaks a tool call.
- **Transport security**: DNS-rebinding / Host-header guard is on; allowed
  hosts are configured for the deployment (localhost by default, plus any
  public host behind a Cloudflare Tunnel).
- **Rate + concurrency limits**: writes 30/min, search 60/min, reads 120/min,
  with bounded concurrency so a burst of slow ops can't wedge every tool.
- **Cloudflare Access** is the intended outer auth ring for remote
  deployments; the bearer token is the inner ring. See `DEPLOY.md`.

## Running it

- **Locally for Claude Code** — `mcp_server/run-local.sh` binds
  `127.0.0.1:8765`, persists a token at `~/.brain-mcp-token`, and prints the
  `claude mcp add` command to register it (push is off by default).
- **As a service** — `mcp_server/systemd/brain-mcp.service`; see `DEPLOY.md`.
- **Smoke tests** — `uv run python -m mcp_server.manual_test` (full stack,
  rewinds its own writes) and `uv run python -m mcp_server.test_replace_note`
  (isolated, safe on a dirty tree).

## Non-goals (deliberately)

- Editing `archive/raw/`. Not supported even with auth — re-add the source
  via `inbox/` and let ingestion overwrite the processed copy.
- Running ingestion over MCP. The server hands files to `inbox/`; the user
  runs `scripts/ingest.py` separately. Long, weight-downloading runs don't
  belong behind a tool call.
- A web UI. Obsidian is the human interface.
