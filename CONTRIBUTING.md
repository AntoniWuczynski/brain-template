# Contributing

`brain-template` is a GitHub template repository — most people don't need to contribute back. Click the green **Use this template** button to spin up your own copy and modify it freely. Your fork is independent.

If you want to improve the framework itself, PRs are welcome.

## Good fits for a PR

- **New extractors** in `scripts/ingest_lib/extractors/`. The pattern is in `README.md` → "Extending". Roughly: write a function with signature `extract(src: Path, assets_dir: Path) -> ExtractionResult`, register it against the file extensions it handles, done.
- **New LLM providers** in `scripts/ingest_lib/summarize.py` (and `chat.py`). Add another `_call_<provider>` and update `_select_provider()` / `_DEFAULT_MODELS`.
- **Bug fixes** in any of the ingest, concept, semantic, or chat modules.
- **Performance improvements** — faster chunking, smaller embedding model with comparable quality, parallelism in the ingest loop, etc.
- **Documentation fixes.** If something tripped you up, fix it.

## Less likely to land

- **Personal workflow opinions.** This is a template — keep changes generic and configurable.
- **Heavy new dependencies.** The stack is intentionally lean. A new dep needs justification in the PR description.
- **MCP server implementation.** There's a design doc at `mcp/README.md` that the maintainer intends to implement themselves. Corrections to the design are fine; large code drops likely aren't.
- **Reformatting / "improving" the code style** without behavioural change. Save the noise.

## Setup

```bash
gh repo clone <your-username>/brain-template brain
cd brain
uv sync
uv pip install --prerelease=allow "mineru[pipeline]"   # only if touching PDF code
```

No formal test suite yet. The expected smoke test for any PR:

```bash
# Drop a small file into inbox/
printf '# Test note\n\nSome content.\n' > inbox/test.md
uv run python scripts/ingest.py --inbox
# Verify archive/processed/test.md and knowledge/index/test.md were produced
```

If your change touches the summarizer or chat: also run `uv run python scripts/ask.py "a question about your test note"` to confirm the round-trip still works.

## License

By contributing you agree your changes are released under the MIT license.

## A note for whoever's running this fork

This template is generated from a private upstream repo via `scripts/sync_template.sh` (one-way: private → public). When you merge a PR here, the change exists only on the public side until you backport it:

```bash
# from inside your private brain repo
git fetch public
git cherry-pick <merge-commit-sha>
# then on next ingest cycle, ./scripts/sync_template.sh will see no diff
```
