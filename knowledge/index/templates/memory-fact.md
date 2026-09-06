---
title: ""
type: memory_fact
created: ""
author: ""
written_via: ""
memory_status: unconsolidated
confirmations: 0
approved: false
promote:
  target: ""
  relations: []
  fact: ""
  source: ""
  # merge:                       # optional; see "Merging a duplicate entity"
  #   duplicate: people/<slug>
  #   survivor: people/<slug>
---

_(Free-form context: why the assistant believes this fact, quotes,
caveats. Preserved verbatim in the archive after promotion.)_

<!--
TEMPLATE — memory fact. The contract for assistant-written facts in
knowledge/assistant/inbox/. Frontmatter is FIRST so a verbatim copy of
this file parses as a real note (the splitter requires the text to start
with the `---` fence); this guidance lives in the body and is ignored by
the parser.

Lifecycle:
1. The assistant drops a fact note into knowledge/assistant/inbox/ with
   memory_status: unconsolidated, confirmations: 0, approved: false.
   Provenance keys (author, written_via, last_written_by) are stamped by
   the MCP server on write.
2. The deterministic consolidation pass promotes entries with
   approved: true OR confirmations >= threshold: `promote.fact` is
   appended to the target entity note's Log
   ("- YYYY-MM-DD — fact ([[source]])"), `promote.relations` are merged
   into its frontmatter, an optional `promote.merge` is performed, and
   the fact note is moved to knowledge/assistant/archive/.
3. Facts that stay unconsolidated are swept into a monthly digest under
   knowledge/assistant/digests/ instead of accumulating forever.

Shapes:
- promote.target: node id — knowledge/-relative path without extension,
  e.g. people/anna-kowalska. A project lives in a FOLDER, so its id is the
  overview note's own path: projects/server/server (the note
  knowledge/projects/server/server.md), never the folder projects/server.
  The short folder form still promotes — consolidate.py resolves it to the
  overview when one exists — but write the canonical form.
- promote.relations: same shape as entity relations; closed vocabulary
  works_at, member_of, attended, stakeholder_in, collaborator_on,
  met_at, related_to; targets are knowledge/-relative no-extension paths.
- promote.fact: a single line — it lands verbatim in the target's Log.
- promote.source: vault-relative no-extension path of the note the fact
  came from. It lands verbatim inside the [[…]] of the Log line, so it
  keeps its knowledge/ prefix — a bare node id there dangles.
- promote.merge: OPTIONAL, and the only shape that changes a note other
  than promote.target. Two node ids:

      promote:
        target: people/anna-kowalska        # == merge.survivor
        merge:
          duplicate: people/annakowalskaexamplecom
          survivor: people/anna-kowalska

  On approval, consolidate.py performs the four steps AGENTS.md's
  "Merging a duplicate entity" paragraph defines, and nothing else: copy
  the duplicate's relations onto the survivor, close every open relation
  on the duplicate with valid_until set to the run's date, add
  superseded_by: knowledge/<survivor> to the duplicate's frontmatter, and
  keep the duplicate's title as an aliases: entry on the survivor. The
  duplicate note stays on disk — supersede, never delete.

  The survivor MUST be promote.target (it is the note the fact promotes
  into). The merge is refused outright, with the fact left in the inbox
  and the reason logged, when the survivor note does not exist, the
  survivor is itself superseded, the two ids name one node, or either id
  is not a graph node. A duplicate that already carries superseded_by is
  an already-merged pair: the merge is a no-op and the fact still
  archives, so re-running is safe.

This directory (knowledge/index/templates/) sits OUTSIDE the enrichment
scan — templates are never embedded, indexed, or surfaced in search.
-->
