---
name: dream-pass
description: >-
  Use when asked to run the dream pass — the LLM memory-consolidation
  session over the brain vault ("run the dream pass", "/dream-pass",
  "dream over the vault", scheduled headless runs). Finds and writes
  connections, digests, compiled-truth summaries, contradiction
  proposals, memory merges and open questions from what changed since the
  last dream. Requires the brain MCP server (mcp__brain__*
  tools) and must run with the vault repo as the working directory.
  Arguments: --dry-run (plan only), --force (dream even if the gate says
  skip), --deep (fan out subagents per job).
---

# dream-pass

## Overview

The deterministic layer (`consolidate.py`, `sweep.py`) already handles
rule-based memory promotion. This pass does what counters cannot: read what
changed, articulate connections, write digests, keep each entity's
compiled truth current, merge duplicate memory, propose the contradictions
it cannot settle alone, and surface open questions. The design contract is
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
  compiled-truth refreshes, the questions.md rewrite), creations do not.
  The packet's `edit_budget` splits that cap: `compiled_truth` is the
  most notes step 6 may touch and the packet is already truncated to it,
  `reserved_for_other_jobs` is what stays for digests, consolidation and
  questions. Never spend another job's share.
- The compiled-truth block is **fenced**, and step 6 may only rewrite what
  is between the two markers. That is mechanical, not a promise:
  `mcp__brain__vault_update_compiled_truth` splices the block server-side
  and leaves `relations:`, the hand-written body and the whole `## Log`
  exactly as they are on disk — the only frontmatter it touches is the
  provenance the server stamps on every write. Never write compiled truth
  with `mcp__brain__vault_replace_note`.
- Contradiction findings are **proposals only**, and you do not write the
  proposal: step 7 adjudicates and hands the survivors to
  `scripts/dream_gate.py --propose`, which writes them through the shared
  fact-note proposer with `approved: false`. It never edits an entity
  note, a relation or a `## Log`, and it never flips `approved`. A human
  gates every promotion — that is the vault's whole posture, do not
  shortcut it.
- **Unapproved proposals are not facts.** Anything under
  `knowledge/assistant/inbox/` with `type: memory_fact` and
  `approved: false` — this run's, an older dream's, or the wikilink
  pass's — is a question addressed to a human. This pass never promotes
  one: never copy its claim into an entity note, never turn it into a
  relation or a `## Log` line, never set `approved: true`. Approval is a
  person's decision and `scripts/consolidate.py` is what executes it.
- Never delete anything. Supersede: rewrite the losing fact with a pointer
  to what replaced it.
- Never touch `knowledge/assistant/PROFILE.md`, anything under `archive/`,
  the vault-root `inbox/`, `metadata/`, or `logs/`. State changes happen
  only via `dream_gate.py --mark-done`. (`knowledge/assistant/inbox/` is a
  different directory and is step 7's only write target.)
- Unsure about a merge? Skip it and record why in the report.

## Procedure

1. **Gate.** Run `uv run --no-sync python scripts/dream_gate.py --dry-run`
   (or `.venv/bin/python scripts/dream_gate.py --dry-run`). Exit 1 and no
   `--force` argument → reply "dream gate: not enough new information" and
   stop. Exit 2 → report the git error and stop.
   Always `--dry-run` here: same verdict, same exit codes, but it records
   no `metadata/dream.pending` marker. Only `scripts/dream.sh` stamps that
   marker (it is the stall signal for scheduled runs) and only
   `--mark-done` clears it — so a non-dry gate run from inside this
   session would leave a marker behind on every early exit below, and the
   sweep would report a stalled dream that had in fact finished.
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
   appears there in EITHER name order (`<a>--<b>.md` or `<b>--<a>.md` —
   older notes predate the packet's `a <= b` ordering). The packet already
   drops pairs it can see a note for, that list is the second guard, and
   `vault_create_note` refuses to overwrite an existing note. Name every
   new note in the packet's own `a`/`b` order.
5. **Digests.** For each `active_entities` entry with enough accumulated
   activity (several changed log entries or notes since the entity's last
   digest in `existing_dream_notes`), create or replace
   `knowledge/notes/dreams/digests/<entity-name>.md` — a synthesis of the
   recent activity, not a copy of it.
6. **Compiled truth.** The packet's `compiled_truth` lists entity notes
   whose "current best understanding" block is out of date — already
   ranked by how much new evidence landed and already truncated to
   `edit_budget.compiled_truth`. Take that list as given; never widen it.

   For each entry, read the note (`mcp__brain__vault_read`) and write a
   short synthesis of its `## Log` plus the entry's `open_relations`:
   what is true about this entity now, and since when. Read the entry's
   evidence fields to know what moved: `new_log_bullets` counts bullets
   added anywhere under the entity (a project's evidence lands in its
   `log/<date>.md` notes, not in the overview's own `## Log`),
   `changed_notes` counts the changeset's other notes under it, and
   `relations_changed` means the open relations themselves differ from
   the last dream — a relation that was CLOSED adds no bullet anywhere,
   and a block still asserting it is exactly what this job exists to fix.

   The block goes between these two markers, and nowhere else:

   ```
   <!-- COMPILED-TRUTH-START -->
   ...the compiled truth...
   <!-- COMPILED-TRUTH-END -->
   ```

   Write it with `mcp__brain__vault_update_compiled_truth` (`path`, and
   the block's body as `content`) — **never**
   `mcp__brain__vault_replace_note`. That tool replaces the text between
   the markers and nothing else, so the frontmatter (including
   `relations:`), the hand-written body and the whole `## Log` are the
   bytes already on disk and cannot be trimmed by what you send.

   Send the block's TEXT only. Never put the markers themselves in
   `content` — not the pair above, not a copy of an existing block. The
   server owns them, and it refuses a payload carrying either one: a
   second fence spliced into the note makes that note unwritable by this
   tool for ever after.

   `marker_state: ok` → send the new block body straight to that tool.
   `marker_state: absent` → send it the same way. The note has no fence
   yet and the server creates the first one, directly below the
   frontmatter. Do not add a fence yourself with
   `mcp__brain__vault_append_to_note`: an append lands at end of file,
   which on an entity note is inside the `## Log`, and a block inside the
   Log is read back as Log evidence by the next pass.

   Notes in `compiled_truth_blocked` have an ambiguous fence (one marker
   without the other, out of order, or duplicated) and there is no safe
   place to write. Never guess: skip them and list their paths in the
   report so a human can repair the markers.

7. **Contradictions.** The packet's `contradictions` lists claim pairs the
   deterministic detectors think disagree: a relation the note both closes
   and carries as open (`relation_interval_conflict`), an open relation
   while a `## Log` line about that same target carries a termination cue
   (`relation_vs_log`), or one fact stated twice in different words
   (`restated_fact`). These are hints, not verdicts — the cues are
   lexical, so "left the steering group" reads the same as "left Acme".
   Read the note and decide.

   **This job never edits an entity note, and never writes the proposal
   note itself.** You adjudicate; the deterministic half writes. Collect
   the survivors into a JSON array — each entry is the packet's finding
   with `node_id`, `note_path`, `proposal_path`, `proposal_fact` and
   `promote_relations` copied **unaltered**, plus two fields of your own:

   - `title`: `"Contradiction: <one line>"`
   - `body`: both claims verbatim, the detector that paired them, and what
     a human would have to check to settle it.

   Write that array to `~/.cache/brain-dream/survivors.json` (expand `~`
   to the absolute home path — the `Write` tool wants one). That path is
   outside the vault, so writing it is not a vault write, and it is the
   only path a scheduled run is allowed to write: `scripts/dream.sh`
   grants `Write` on that directory and nothing else. Then hand the file
   to the writer:

   ```bash
   uv run --no-sync python scripts/dream_gate.py --propose ~/.cache/brain-dream/survivors.json
   ```

   It writes each one through `scripts/ingest_lib/propose.py`, so the
   memory-fact contract (`approved: false`, `memory_status:
   unconsolidated`, `confirmations: 0`, `created:`) and the honest
   `written_via: script` / `author: script:dream-contradiction`
   provenance have exactly one owner — not frontmatter typed by an agent.
   `approved: false` is the point: `scripts/consolidate.py` promotes
   nothing until a person flips it, so this job cannot move the graph.
   Re-running over the same survivors is a no-op.

   `proposal_fact` and `promote_relations` go in unaltered because the
   file the proposal lands in is a hash of them: editing either addresses
   a different file and breaks the idempotency the packet's inbox and
   archive checks depend on. The CLI refuses the whole run and names the
   mismatch rather than writing a stray note. Your own words go in the
   `title` and the `body`, which the path does not depend on.

   `promote_relations` is how an approval settles the finding rather than
   just recording it: for a `relation_vs_log` hit it carries the SAME
   relation with `valid_until` set to the date on the Log line, so a human
   flipping `approved: true` lets `consolidate.py` close the open span —
   supersede, never delete. What it contains is the packet's call, never
   yours.

   Skip freely, and record every skip with its reason in the report.

8. **Consolidation.** For entities in the changeset, read their notes plus
   `mcp__brain__vault_related` context. Where two facts duplicate or
   contradict each other, merge the prose via
   `mcp__brain__vault_replace_note`: merged fact stated once, superseded
   wording preserved under a "superseded" marker with a pointer. Count
   every replace against the 10-edit cap. A contradiction you are not sure
   enough to merge belongs in step 7's proposals, not here.
   **The server refuses a replace that drops history.** Every typed
   relation entry and every `## Log` bullet already on the note must
   still be present in the text you send, or the write is refused
   (AGENTS.md "supersede, never delete"). So: carry the existing
   `relations:` block and the whole `## Log` section through unchanged,
   and never close or retire a relation by deleting it — set
   `valid_until` on it (that counts as superseding, not losing), or use
   `mcp__brain__entity_upsert_relation`, which is the tool built for it.
   Adding relations or Log lines is always allowed.
9. **Questions.** Create or replace `knowledge/notes/dreams/questions.md`
   (create on the first ever run, replace after) with the current open
   questions the changeset raises (missing decisions,
   unresolved contradictions you chose not to merge, gaps). Rolling: this
   note is rewritten every run, not appended.
10. **Report.** Create `knowledge/notes/dreams/reports/<today>.md` listing
    every note created or edited (wikilinks), every skip with its reason,
    and the packet's head_commit. Also list every `compiled_truth_blocked`
    path and every contradiction proposal you created or declined — those
    are the two things a human has to act on. This is the audit trail —
    completeness here matters more than elegance.
11. **Mark done.** Run `uv run --no-sync python scripts/dream_gate.py
    --mark-done`. Only after the report exists.
12. **Reply** with a one-paragraph summary: counts per job, anything
    skipped as unsure, and the report note's path.

`--dry-run`: do steps 1–3, then reply with the plan (what each job WOULD
do) and stop. No writes, no --mark-done: a dry-run leaves zero state
behind.

`--deep`: same contract, but fan out one subagent per job (connections,
digests, compiled truth, contradictions, consolidation, questions) on the
packet, then synthesise their
outputs yourself before writing. Same tripwires, same single report.
Intended for occasional manual or weekly use, not the nightly schedule.

## Note format

Every dream note carries this frontmatter. `type:` is always `digest` —
for all four output kinds (connection notes, digests, the questions note,
the report). That is the AGENTS.md closed-vocabulary value for a
synthesised note, and the vocabulary has no `connection`, `report`,
`question`, `note` or `log` member: never invent one. The kind of dream
note is carried by its directory plus `generated_by: dream-pass`, not by
`type:`.

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

Two outputs are NOT dream notes and do not take this frontmatter: the
compiled-truth block of step 6 (it lives inside an existing entity note,
whose own frontmatter is untouched) and the contradiction proposals of
step 7 (`type: memory_fact`, written for you by `dream_gate.py --propose`
under the contract in `knowledge/index/templates/memory-fact.md`).

## Common mistakes

| Mistake | Do instead |
|---|---|
| Crawling the vault to find what changed | The packet already lists it — trust it |
| Writing a connection note for every candidate pair | Most pairs are coincidence — skip freely, note why |
| Direct file writes into knowledge/ | Always mcp__brain__* tools (lock + reindex + commit) |
| Deleting a duplicate fact | Supersede it: keep the wording under a superseded marker |
| Editing metadata/dream.json by hand | Only dream_gate.py --mark-done advances state |
| Inventing a `type:` like connection/report/question | Every dream note is `type: digest`; the directory carries the kind |
| Marking done before the report note exists | Report first, then --mark-done |
| Padding digests with copied log lines | Digests synthesise — pointers and conclusions, not copies |
| Sending a whole entity note back to compile truth | `vault_update_compiled_truth` — it splices between the markers, nothing else moves |
| Guessing where a broken compiled-truth fence ends | Skip the note, report the path, let a human repair it |
| Sending the `<!-- COMPILED-TRUTH-... -->` markers as the block's `content` | Send the text only — the server owns the markers and refuses a payload carrying one |
| Appending an empty fence to bootstrap a missing one | Just call `vault_update_compiled_truth`; the server puts the first fence below the frontmatter |
| Writing `survivors.json` into the vault | `~/.cache/brain-dream/survivors.json` — the one path a scheduled run may write |
| Editing an entity note to fix a contradiction | Adjudicate, then `dream_gate.py --propose`; the note keeps `approved: false` |
| Hand-writing a proposal's frontmatter with `vault_create_note` | `dream_gate.py --propose` — `propose.py` owns that contract |
| Rewording a finding's `proposal_fact` or `promote_relations` | Both come from the packet verbatim — the proposal's path is a hash of them |
| Setting `approved: true` on your own proposal | Only a human approves; consolidate.py gates on that flag |
| Treating an `approved: false` inbox note as a fact | It is a question for a human — never promote, copy or approve one |
| Spending the whole 10-edit cap on compiled truth | The packet's `edit_budget` splits it — respect the split |

---

> **Sync note:** the source of truth for this skill is
> `.claude/skills/dream-pass/SKILL.md` in the private `brain` repo; it
> syncs to the public `brain-template` via `scripts/push_to_upstream.sh`.
