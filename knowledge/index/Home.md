# Home

The Obsidian-facing front door for the vault. Source files live in `archive/raw/`, extracted artefacts in `archive/processed/`, and notes here under `knowledge/`.

For the full system overview see [`../../README.md`](../../README.md). For agent rules see [`../../AGENTS.md`](../../AGENTS.md).

## Principles

- Raw files are never modified.
- Every note links back to its source.
- Notes are generated or curated, never invented when extraction fails.
- Everything important is queryable, by tag (concept notes) or by meaning (semantic search).

## Layout

- `index/` — generated index notes (one per processed source file) plus this file and `Note Template.md`.
- `concepts/` — auto-generated cross-source topic notes. Edit anything below the `AUTO-GENERATED-END` marker; survives regeneration.
- `projects/` — ongoing or finished work.
- `university/` — course-organised material.
- `research/` — papers, lit-review notes, derivations.
- `people/`, `organisations/` — entity notes.
- `meetings/` — one note per meeting, by year.
- `assistant/` — assistant memory: inbox/ archive/ digests/ PROFILE.md.
- `notes/` — anything that doesn't fit the above yet.

## Workflow

1. Drop files into `inbox/`.
2. Run `uv run python scripts/ingest.py --inbox`.
3. Each file is copied to `archive/raw/`, extracted to `archive/processed/`, and gets a generated index note in `knowledge/index/<rel/path>.md`.
4. Curate: move or link the index note into the right subfolder, add tags, write your own thoughts under the auto-generated section.

## Status dashboards

Generated after each ingest; run `uv run python scripts/ingest.py --rebuild-status` to force a refresh.

- [[Now]] — landing view: recently added sources + what needs attention.
- [[Processing Dashboard]] — counts and most-recent runs.
- [[Manual Review]] — files in `archive/failed/` waiting for a human.
