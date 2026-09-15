<!--
Append this section to the user's GLOBAL ~/.claude/CLAUDE.md (create the file if
absent). It tells every Claude Code session, in every repo, to treat the brain
vault as durable cross-session memory. It pairs with the SessionStart/Stop hooks
(brain_memory_sync.py) — the hooks nudge deterministically, this section sets the
standing behaviour. Do not paste these HTML comments.
-->

# Brain vault (persistent memory across projects)

A personal "second brain" knowledge vault is available in every project via the
brain MCP server (`mcp__brain__*` tools). Treat it as durable cross-session
memory.

- **Start of a project session:** if the `mcp__brain__*` tools are present, call
  `mcp__brain__memory_search` for the current project (and relevant people /
  meetings / decisions) to load prior context before substantive work. A
  SessionStart hook will usually remind you; do it even if it doesn't.
- **End of a session in which decisions were made, features shipped, plans
  changed, or research concluded:** capture it with the `brain-project-note`
  skill so the next session starts informed. Notes live by subject, not by
  client: `knowledge/projects/<slug>/` for a bounded piece of work,
  `knowledge/chats/<slug>/` for a chat-based project with no repo (same
  folder shape), `knowledge/personal/<area>/` for life admin — recommended
  areas, never forced. Slugs are readable names, never directory or
  chat-project ids. A project isn't only its `<slug>.md` overview and dated
  log — when a durable decision, design or artefact deserves its own home,
  also create a focused curated note beside the overview (or under
  `knowledge/projects/shared/` for a cross-project topic, linked to each
  project with a `related_to` relation).
- **MCP-only:** never read, write, or guess the vault's on-disk location — go
  through the MCP tools. If they aren't connected, say so and continue — don't
  write vault files directly.
