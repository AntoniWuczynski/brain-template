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
   into its frontmatter, and the fact note is moved to
   knowledge/assistant/archive/.
3. Facts that stay unconsolidated are swept into a monthly digest under
   knowledge/assistant/digests/ instead of accumulating forever.

Shapes:
- promote.target: node id — knowledge/-relative path without extension,
  e.g. people/anna-kowalska.
- promote.relations: same shape as entity relations; closed vocabulary
  works_at, member_of, attended, stakeholder_in, collaborator_on,
  met_at, related_to; targets are knowledge/-relative no-extension paths.
- promote.fact: a single line — it lands verbatim in the target's Log.
- promote.source: vault-relative no-extension path of the note the fact
  came from.

This directory (knowledge/index/templates/) sits OUTSIDE the enrichment
scan — templates are never embedded, indexed, or surfaced in search.
-->
