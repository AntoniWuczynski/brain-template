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
- **MCP server changes.** The server is implemented in `mcp_server/`, with its contract in `mcp/README.md`. Small fixes and contract corrections are welcome; large redesigns are the maintainer's call — open an issue first.
- **Reformatting / "improving" the code style** without behavioural change. Save the noise.

## Setup

```bash
gh repo clone <your-username>/brain-template brain
cd brain
uv sync
uv pip install --prerelease=allow "mineru[pipeline]==2.7.6" six   # only if touching PDF code
```

Run the test suite (`tests/`, also run in CI) and the smoke test for any PR:

```bash
uv run --no-sync pytest -q
# Then the round-trip smoke test:
# Drop a small file into inbox/
printf '# Test note\n\nSome content.\n' > inbox/test.md
uv run python scripts/ingest.py --inbox
# Verify archive/processed/test.md and knowledge/index/test.md were produced
```

If your change touches the summarizer or chat: also run `uv run python scripts/ask.py "a question about your test note"` to confirm the round-trip still works.

## License

By contributing you agree your changes are released under the MIT license.

## A note for whoever's running this fork privately

If you maintain a private fork of this template that contains your own content, the model is: this template repo is canonical; your private fork is a downstream consumer that pulls framework updates from here. When upstream gets a merged PR, run:

```bash
./scripts/pull_from_upstream.sh
```

inside your private fork to absorb it. When you make framework changes in your private fork that should be shared, run:

```bash
./scripts/push_to_upstream.sh
git push upstream template:main
```

to project them up.
