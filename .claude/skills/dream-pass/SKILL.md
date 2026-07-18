---
name: dream-pass
description: >-
  Use when asked to run the dream pass — the LLM memory-consolidation
  session over the brain vault ("run the dream pass", "/dream-pass",
  "dream over the vault", scheduled headless runs). Finds and writes
  connections, digests, memory merges and open questions from what changed
  since the last dream. Requires the brain MCP server (mcp__brain__*
  tools) and must run with the vault repo as the working directory.
  Arguments: --dry-run (plan only), --force (dream even if the gate says
  skip), --deep (fan out subagents per job).
---

# dream-pass

## Overview

The deterministic layer (`consolidate.py`, `sweep.py`) already handles
rule-based memory promotion. This pass does what counters cannot: read what
changed, articulate connections, write digests, merge duplicate or
contradictory memory, and surface open questions. The design contract is
`docs/superpowers/specs/2026-07-18-dream-pass-design.md`.

**Core principles:**

- The packet is your worklist. `scripts/dream_gate.py --emit-packet` tells
  you exactly what changed and which connections look unexplained. Do not
  crawl the vault rediscovering it.
- All writes go through `mcp__brain__*` tools — never direct file writes to
  the vault. That buys the server's write lock, auto-reindex, and one git
  commit per write.
- Writing nothing is a valid outcome for every job. Never force an insight.
- Honesty over coverage: if two notes only might be related, skip the pair
  and say so in the report.

## Hard limits (tripwires)

- Edit at most **10 existing notes** per run — every replace of a
  pre-existing note counts (consolidation rewrites, digest replacements,
  the questions.md rewrite), creations do not.
- Never delete anything. Supersede: rewrite the losing fact with a pointer
  to what replaced it.
- Never touch `knowledge/assistant/PROFILE.md`, anything under `archive/`,
  `inbox/`, `metadata/`, or `logs/`. State changes happen only via
  `dream_gate.py --mark-done`.
- Unsure about a merge? Skip it and record why in the report.

## Procedure

1. **Gate.** Run `uv run --no-sync python scripts/dream_gate.py`
   (or `.venv/bin/python scripts/dream_gate.py`). Exit 1 and no `--force`
   argument → reply "dream gate: not enough new information" and stop.
   Exit 2 → report the git error and stop.
2. **Packet.** Run `uv run --no-sync python scripts/dream_gate.py
   --emit-packet` and parse the JSON. This is the whole worklist.
3. **Resume check.** Let `today` be the current UTC date. If
   `knowledge/notes/dreams/reports/<today>.md` exists, a run already
   completed — reply so and stop. If that report is absent but dream notes
   exist whose `dreamed:` frontmatter is strictly after the packet's `since`
   date, a previous run died mid-way: treat exactly those notes as already
   done, skip their work, and continue from where it stopped (`dreamed:` is
   the marker, not the file path — paths carry no date stamp, and the
   report path uses today's date even when it's finishing yesterday's
   work, so keying the resume check on today's date alone would miss it).
4. **Connections.** For each `candidate_pairs` entry, read both concepts'
   notes (`mcp__brain__vault_search` the concept names, then
   `mcp__brain__vault_read` the hits). Where a genuine, non-obvious
   relationship exists, create
   `knowledge/notes/dreams/connections/<a>--<b>.md` explaining WHY they
   relate, with wikilinks to both. Skip freely. First check the packet's
   `existing_dream_notes`: skip any pair whose connection note already
   appears there — that list is the dedup guard, and `vault_create_note`
   refuses to overwrite an existing note.
5. **Digests.** For each `active_entities` entry with enough accumulated
   activity (several changed log entries or notes since the entity's last
   digest in `existing_dream_notes`), create or replace
   `knowledge/notes/dreams/digests/<entity-name>.md` — a synthesis of the
   recent activity, not a copy of it.
6. **Consolidation.** For entities in the changeset, read their notes plus
   `mcp__brain__vault_related` context. Where two facts duplicate or
   contradict each other, rewrite the note via
   `mcp__brain__vault_replace_note`: merged fact stated once, superseded
   wording preserved under a "superseded" marker with a pointer. Count
   every replace against the 10-edit cap.
7. **Questions.** Create or replace `knowledge/notes/dreams/questions.md`
   (create on the first ever run, replace after) with the current open
   questions the changeset raises (missing decisions,
   unresolved contradictions you chose not to merge, gaps). Rolling: this
   note is rewritten every run, not appended.
8. **Report.** Create `knowledge/notes/dreams/reports/<today>.md` listing
   every note created or edited (wikilinks), every skip with its reason,
   and the packet's head_commit. This is the audit trail — completeness
   here matters more than elegance.
9. **Mark done.** Run `uv run --no-sync python scripts/dream_gate.py
   --mark-done`. Only after the report exists.
10. **Reply** with a one-paragraph summary: counts per job, anything
    skipped as unsure, and the report note's path.

`--dry-run`: do steps 1–3 — but run the step-1 gate as `dream_gate.py
--dry-run` so no pending marker is recorded — then reply with the plan
(what each job WOULD do) and stop. No writes, no --mark-done: a dry-run
leaves zero state behind.

`--deep`: same contract, but fan out one subagent per job (connections,
digests, consolidation, questions) on the packet, then synthesise their
outputs yourself before writing. Same tripwires, same single report.
Intended for occasional manual or weekly use, not the nightly schedule.

## Note format

Every dream note carries this frontmatter (types from the AGENTS.md closed
vocabulary — no new types):

```markdown
---
title: "<human-readable title>"
type: digest
generated_by: dream-pass
dreamed: <YYYY-MM-DD>
topics: []
---
```

Body ends with a `## Links` section wikilinking every note the content was
derived from (AGENTS.md rule 3 — dream notes have no single source_file,
the links are the provenance).

## Common mistakes

| Mistake | Do instead |
|---|---|
| Crawling the vault to find what changed | The packet already lists it — trust it |
| Writing a connection note for every candidate pair | Most pairs are coincidence — skip freely, note why |
| Direct file writes into knowledge/ | Always mcp__brain__* tools (lock + reindex + commit) |
| Deleting a duplicate fact | Supersede it: keep the wording under a superseded marker |
| Editing metadata/dream.json by hand | Only dream_gate.py --mark-done advances state |
| Marking done before the report note exists | Report first, then --mark-done |
| Padding digests with copied log lines | Digests synthesise — pointers and conclusions, not copies |

---

> **Sync note:** the source of truth for this skill is
> `.claude/skills/dream-pass/SKILL.md` in the private `brain` repo; it
> syncs to the public `brain-template` via `scripts/push_to_upstream.sh`.
