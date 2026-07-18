# AGENTS.md — rules for any agent operating in this repo

This file applies to **every** agent: Claude Code, Codex, future MCP-driven
agents, or any human-driven script. If you cannot satisfy these rules,
**stop and ask** instead of doing the wrong thing.

---

## Hard rules (do not violate)

1. **`archive/raw/**` is immutable.** Never modify, rename, delete or
   overwrite anything under it. Re-ingestion always writes to
   `archive/processed/` instead.
2. **No destructive operations on raw source files** (those in `inbox/`
   *or* `archive/raw/`). If you think one needs to move, ask the user.
3. **Always link outputs to their source.** Every generated Markdown note
   carries `source_file:` in its frontmatter and a "Links → Source:"
   wikilink in the body. No exceptions. The one carve-out: dream-pass notes
   (`knowledge/notes/dreams/`, `generated_by: dream-pass`) have no single
   source, so their provenance is the `generated_by` key plus a mandatory
   `## Links` section wikilinking every note they derive from instead.
4. **Update `metadata/index.jsonl` for every processed file.** One JSON
   object per line; never rewrite the whole file.
5. **Log every operation.** Append to `logs/ingest-YYYYMMDDTHHMMSS.log`
   for ingestion runs, or to a similarly named log for other tools. Logs
   are append-only.
6. **Mark uncertainty explicitly.** If extraction is incomplete use
   `status: partial`; if it failed use `status: manual_review` and move
   the file to `archive/failed/`. **Never invent a summary** to cover for
   missing content.
7. **Be deterministic.** The same input must produce the same output.
   Avoid timestamps and random IDs in note bodies; restrict them to
   frontmatter and logs. This binds the dream pass's gate/packet/state
   layer (`scripts/ingest_lib/dream.py`), not the LLM-written bodies of
   dream notes — those are audited via the per-run dream report instead.
8. **Operate statelessly.** Do not assume any prior conversation context
   exists. Everything you need must be readable from files, logs, and
   `metadata/index.jsonl`.
9. **Respect user-edited frontmatter.** Re-ingestion merges frontmatter:
   generated keys (`title`, `type`, `source_file`, `source_hash`,
   `created`, `updated`, `status`) are refreshed — hand edits to `title`
   and `type` are NOT preserved on re-ingest; user-added keys (`topics`,
   `aliases`, anything else) are preserved.

---

## Soft rules (good practice)

- Idempotency: skip files whose `source_hash` already appears in
  `metadata/index.jsonl` with `status: processed`.
- Use atomic writes: write to a temp file, fsync, then rename.
- Never embed binary content in Markdown notes; reference assets by path.
- Keep generated Markdown agent-readable: stable headings, no novelty
  formatting, ASCII where possible.
- Wikilinks (`[[relative/path/without/extension]]`) over absolute paths;
  this keeps the vault portable.
- Any agent (not just Claude) may execute the dream pass by following
  `.claude/skills/dream-pass/SKILL.md` literally — it is written to be
  runner-agnostic. Its hard limits are binding.

---

## Note format (canonical)

Every generated index note must use this frontmatter:

```yaml
---
title: "<human-readable title>"
type: source_note            # source_note | concept | project | person | organisation | meeting | memory_fact | digest | dashboard
source_file: archive/raw/<rel/path>
source_hash: <sha256>
created: <ISO 8601 UTC>
updated: <ISO 8601 UTC>
status: processed            # processed | partial | manual_review
topics: []
aliases: []
---
```

Body skeleton:

```markdown
# Summary
# Key points
# Extracted content
# Links
- Source: [[archive/raw/<rel/path>]]
# Processing notes
```

If a section has no content, leave the heading and write `_(empty)_` so
later agents know it was deliberate.

---

## Entity notes and typed relations

People, organisations, projects and meetings are **graph nodes**, not just
prose. Blank starting points live in `knowledge/index/templates/`.

- **Node ids.** An entity's id is its `knowledge/`-relative path without
  the `.md` extension: `people/anna-kowalska`, `organisations/acme`.
  Node ids always contain a `/`, so they can never collide with concept
  slugs. Relation targets, `promote.target` values and attendee lists all
  use this form.
- **Closed relation vocabulary.** `works_at`, `member_of`, `attended`,
  `stakeholder_in`, `collaborator_on`, `met_at`, `related_to` — nothing
  else. Unknown rel values are reported by tooling and excluded from the
  graph, never silently stored. Don't invent synonyms (`employed_by`);
  pick the closest existing rel.
- **Relations frontmatter shape** (on the entity note):

  ```yaml
  relations:
    - rel: works_at
      target: organisations/acme
      valid_from: "2025-03-01"      # optional, YYYY-MM-DD
      valid_until: ""               # optional; absent/empty = currently valid
      source: knowledge/meetings/2026/2026-06-12-kern-call   # provenance
  ```

- **Supersede, never delete.** When a relation ends, set `valid_until` on
  the open entry and (if a new state replaces it) append a fresh entry.
  Never rewrite or remove existing entries — the closed intervals are the
  queryable history. Typed relations flow into
  `metadata/connections.jsonl` as `kind: typed` edges.
- **Meetings are first-class notes** at
  `knowledge/meetings/<YYYY>/<YYYY-MM-DD>-<slug>.md`, joining people,
  organisations and projects: every attendee gets an `attended` relation,
  and an optional project lands as `related_to` in the meeting's own
  frontmatter.
- **The `## Log` section** on an entity note collects dated,
  source-linked facts, one line each, newest **last**, never rewritten:

  ```markdown
  - 2026-06-12 — Anna moved to the platform team ([[knowledge/meetings/2026/2026-06-12-kern-call]])
  ```

---

## Provenance and the memory lifecycle

- **Provenance keys are server-asserted.** The MCP server stamps
  `author`, `written_via` and `last_written_by` into the frontmatter of
  every note it writes (and overrides any client-supplied values). Do
  not hand-fake these keys — a note claiming `written_via: mcp` that
  never passed through the server is a provenance lie. Hand-edited notes
  simply carry none of them.
- **Memory lifecycle.** Notes under `knowledge/assistant/` carry
  `memory_status: unconsolidated | consolidated | superseded`. Assistant
  facts land in `knowledge/assistant/inbox/` as `unconsolidated`
  (contract: `knowledge/index/templates/memory-fact.md`); the
  deterministic consolidation pass (`scripts/consolidate.py`) promotes
  approved/confirmed facts into entity notes (relations merged, fact line
  appended to `## Log`) and moves the originals to
  `knowledge/assistant/archive/<YYYY-MM>/` — moved, never deleted. Facts
  that linger unconsolidated are swept into monthly digests under
  `knowledge/assistant/digests/`. Mark a note `superseded` when newer
  knowledge replaces it; search down-weights it instead of hiding it.
- **`archive/` and `digests/` are historical.**
  `knowledge/assistant/archive/` and `knowledge/assistant/digests/` hold
  already-consolidated and already-swept material. Only `archive/` is
  excluded from semantic retrieval, so promoted facts don't resurface in
  `memory_search`; digests remain searchable (and are reindexed
  immediately after consolidation). Both `archive/` and `digests/` are
  excluded from the stale-unconsolidated sweep, so they never get
  re-flagged. Treat them as the audit trail, not live memory.
- **The assistant PROFILE is special.**
  `knowledge/assistant/PROFILE.md` is written ONLY through the
  `profile_update` tool, which enforces a hard byte budget and stamps
  `memory_status: consolidated`. The general note verbs
  (`vault_create_note` / `vault_replace_note` / `vault_append_to_note`)
  refuse it so the budget can't be bypassed.
- **`status:` ≠ `memory_status:`.** These are deliberately separate
  vocabularies: `status` (`processed | partial | manual_review`) is the
  **ingest** lifecycle on source notes; `memory_status` is the **memory**
  lifecycle on assistant notes. Never use one where the other belongs.

---

## When you are about to do something risky

Stop and confirm with the user **before**:

- Modifying anything in `archive/raw/`.
- Removing or rewriting entries in `metadata/index.jsonl`.
- Force-pushing, rebasing or amending published commits.
- Bulk-renaming notes in `knowledge/`.
- Installing system-level dependencies (vs. Python deps via uv).
- Downloading multi-GB model weights without acknowledging the size.

---

## When something fails

1. Don't paper over it. Mark the file `manual_review`, log the error
   verbatim, and move on.
2. Append a line under `## Manual review` in [`TODO.md`](TODO.md)
   describing the file and the failure mode.
3. Continue processing other files — partial progress is valuable.
