# brain

> **This is a template repository.** Click the green **Use this template** button at the top of the page to create your own copy. Don't `git clone` it directly unless you intend to track upstream changes.

A personal knowledge vault you can drop files into and query like an extension of your own memory. PDFs, slides, notes, code, datasets — anything you'd want to remember later. Optimised for use by humans (Obsidian), AI agents (Claude, Codex, future MCP clients), and your future self.

## What it gives you

When you drop a file in `inbox/`, the pipeline:

1. Stores a verbatim copy in `archive/raw/` and tracks it by SHA-256.
2. Extracts the content. PDFs go through [MinerU](https://github.com/opendatalab/MinerU) for text, figures, tables, and formulas. Other formats use lighter extractors.
3. Writes a Markdown artefact to `archive/processed/`, with extracted images saved alongside.
4. Generates an Obsidian-friendly index note at `knowledge/index/`, transcluding the processed content.
5. Calls an LLM (optional) to produce a faithful summary, key points, and canonical topic tags.
6. Auto-builds concept notes at `knowledge/concepts/` — one per topic, listing every source *and every hand-written `knowledge/` note* that mentions it. This is the cross-source linking layer.
7. Updates a semantic search index (local embeddings, no API cost) so you can query the whole vault by meaning, not just by tag.

The result is a vault that grows by drop-and-run, organises itself, and stays queryable from the terminal, from Obsidian, and from any agent you point at it.

## How it's structured

```
brain/
├── archive/
│   ├── raw/             ← immutable copies of every source file
│   ├── processed/       ← extracted Markdown + extracted images
│   └── failed/          ← files that couldn't be extracted; need manual review
├── inbox/               ← drop new files here
├── knowledge/
│   ├── index/           ← one Obsidian note per source, with summary + topics
│   ├── concepts/        ← auto-generated cross-source topic notes
│   ├── meetings/        ← one note per meeting, by year
│   ├── assistant/       ← assistant memory: inbox/ archive/ digests/ PROFILE.md
│   ├── projects/        notes/        research/
│   └── people/  organisations/ university/   ← hand-written notes go here
├── metadata/
│   ├── index.jsonl      ← machine record of every processed file
│   └── embeddings.npy   ← semantic search index
├── logs/                ← one log per ingest run
├── scripts/
│   ├── ingest.py        ← the CLI
│   └── ingest_lib/      ← extractors, summarizer, concept builder, search
├── mcp/                 ← MCP server docs + deploy guide
├── mcp_server/          ← the MCP server (FastAPI + FastMCP)
├── .claude/skills/      ← Claude Code skills shipped with the vault
└── pyproject.toml
```

Four layers, four concerns: **archive** is ground truth, **processed** is regenerable extraction, **knowledge** is the curated face, **metadata** is machine state. Agents are expected to read everywhere and write only under `knowledge/`.

## Requirements

- **Python 3.12** (PaddlePaddle, which MinerU uses, doesn't yet ship wheels for 3.13+)
- **macOS or Linux**. Apple Silicon and CUDA both work for MinerU; CPU works but is slow on long PDFs.
- **[uv](https://docs.astral.sh/uv/)** for environment management.
- **Obsidian** if you want the human-facing UI. The vault is plain Markdown, so any editor works, but Obsidian is what the wikilink and transclusion conventions assume.
- **[Optional] An LLM provider** for summaries, key points, and topic tagging. Pick one of:
  - Anthropic Claude (default, set `ANTHROPIC_API_KEY`)
  - OpenAI (set `OPENAI_API_KEY`)
  - Google Gemini (set `GOOGLE_API_KEY` or `GEMINI_API_KEY`)
  - Any OpenAI-compatible local server — Ollama, LM Studio, llama.cpp, vLLM (set `BRAIN_LOCAL_URL` and `BRAIN_LOCAL_MODEL`)

  The summarizer auto-detects from whichever key is present, or you can pin a choice with `BRAIN_LLM_PROVIDER`. Without any provider configured, ingestion still works but the index notes show placeholders instead of summaries.

## Setup

```bash
git clone <your-fork-url> brain
cd brain
uv sync
```

That's enough to ingest text-only files (Markdown, code, notebooks, CSVs). For full PDF extraction with figures and tables:

```bash
uv pip install --prerelease=allow "mineru[pipeline]==2.7.6" six
```

Keep the `==2.7.6` pin: the unpinned latest (mineru 3.4.0) requires `transformers>=4.57.3` but imports a symbol removed in 4.57, so every PDF silently falls back to `pypdf`. `uv sync` will prune MinerU on every subsequent run because it isn't in the lockfile (its transitive deps include pre-releases that break `uv`'s resolver). Re-run the pinned line above after each sync, or wrap both in a `scripts/setup.sh` of your own.

To enable summaries and topic tagging, copy `.env.example` to `.env` and add a provider's credentials:

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (or OPENAI_API_KEY, GOOGLE_API_KEY,
# or BRAIN_LOCAL_URL + BRAIN_LOCAL_MODEL for an OpenAI-compatible local server)
```

## Daily use

### Add files

```bash
cp ~/Downloads/lecture.pdf inbox/university/COMP0123/
cp ~/Notes/meeting.md inbox/projects/acme/
uv run python scripts/ingest.py --inbox
```

Sub-directory structure under `inbox/` is preserved end-to-end. The same file dropped twice is skipped (SHA-256 idempotency).

### Search the vault

```bash
uv run python scripts/ingest.py --search "what happens when a packet is dropped" --top-k 5
```

Returns the most semantically similar passages across every processed source and hand-written `knowledge/` note, with citation paths. The first call after a fresh clone downloads a ~100 MB embedding model to `~/.cache/huggingface/`.

### Chat with the vault

```bash
uv run python scripts/ask.py "What does my vault say about TCP congestion control?"
```

Retrieves the top-k semantic matches and feeds them to whichever LLM provider is configured, returning a concise answer with bracketed citations. Works with any provider — hosted (Anthropic, OpenAI, Gemini) or local (Ollama, LM Studio, llama.cpp via `BRAIN_LOCAL_URL`). The retrieval index is reused across calls; nothing new is written to disk.

For an in-Obsidian chat panel instead of the terminal, install **Copilot for Obsidian** (or **Smart Connections**) from the community plugins, set the chat model to your configured provider (or to Ollama on `http://localhost:11434` for offline use), and use the plugin's "vault chat" mode. The plugin will build its own retrieval index parallel to the one in `metadata/embeddings.npy`; that's wasted disk but otherwise harmless.

### Browse in Obsidian

Open the repo root as an Obsidian vault. `knowledge/index/Home.md` is your entry point. Every concept under `knowledge/concepts/` is a pre-built index of every source that touches that concept. Click into any source's index note and the full extracted content (figures and all) appears inline via transclusion.

### Useful commands

| Command | What it does |
|---|---|
| `--inbox` | Process every supported file under `inbox/` |
| `--raw` | Re-process files already in `archive/raw/` (no copy step) |
| `--path <file>` | Process a single file |
| `--retry-partial` | Re-extract every `partial` record, e.g. after installing MinerU (`archive/processed/` is regenerable) |
| `--dry-run --inbox` | Show the plan, don't write anything |
| `--backfill-summaries` | Add Summary + Key points + Topics to existing records that lack them |
| `--rebuild-concepts` | Regenerate concept notes from current metadata (free, no LLM) |
| `--rebuild-connections` | Rebuild the concept relationship graph in `metadata/connections.jsonl` (free, no LLM) |
| `--rebuild-dashboards` | Regenerate entity dashboards under `knowledge/index/entities/` (free, no LLM) |
| `--rebuild-status` | Regenerate the Processing Dashboard + Manual Review notes under `knowledge/index/` (free, no LLM; also runs after each ingest) |
| `--describe-concepts --limit N` | Write AI-generated, source-grounded descriptions into concept notes (LLM; cached by source hash) |
| `--caption-figures --limit N` | Caption extracted figures/tables with a vision LLM (cached in `metadata/captions.jsonl`) |
| `--rebuild-search-index` | Re-encode every chunk and overwrite the search index |
| `--search "query" --top-k N` | Semantic search the vault |
| `ask.py "question"` (separate script) | Retrieval-augmented chat: top-k chunks + LLM → citation-backed answer |
| `sweep.py --write-report` (separate script) | Lint the vault: orphans, dangling links, relation problems, index drift, stale memory (free, no LLM) |
| `consolidate.py --dry-run` (separate script) | Consolidate assistant memory: promote confirmed facts into entity notes, digest the rest (free, no LLM) |

## Configuration

Everything is via environment variables in `.env`:

| Variable | Effect |
|---|---|
| `BRAIN_LLM_PROVIDER` | `anthropic` / `openai` / `gemini` / `local`. Pin a provider; otherwise auto-detected from whichever key is set. |
| `BRAIN_LLM_MODEL` | Override the model name for the chosen provider. |
| `ANTHROPIC_API_KEY` | Anthropic credentials (default model: `claude-haiku-4-5`). |
| `OPENAI_API_KEY` | OpenAI credentials (default model: `gpt-5-mini`). |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Gemini credentials (default model: `gemini-2.5-flash`). |
| `BRAIN_LOCAL_URL` | OpenAI-compatible local server URL (e.g. `http://localhost:11434/v1` for Ollama). |
| `BRAIN_LOCAL_MODEL` | Model name on your local server (e.g. `llama3.1:8b`). |
| `BRAIN_LOCAL_API_KEY` | Only if your local server enforces auth. |
| `BRAIN_SKIP_SUMMARY=1` | Skip summarization even with a provider configured. |
| `MINERU_DEVICE_MODE` | `cpu` / `mps` / `cuda` for MinerU inference. |
| `BRAIN_EMBED_DEVICE` | Same for the semantic-search embedder. |
| `MINERU_MODEL_SOURCE` | `huggingface` (default) or `modelscope`. |
| `BRAIN_MINERU_FORMULA` | `true` (default) / `false`. Disable MinerU's formula model (it hallucinates LaTeX on handwriting). |
| `BRAIN_MINERU_LANG` | OCR language passed to MinerU (default `en`). |
| `BRAIN_PDF_EXTRACTOR` | Set to `vlm` to route PDFs through the vision-LLM page transcriber (handwritten/scanned notes). |
| `BRAIN_VLM_MODEL` | Vision model for the `vlm` extractor (default `claude-sonnet-4-6`). |
| `BRAIN_VLM_SCALE` | Page render resolution for the `vlm` extractor (default 2.0). |
| `BRAIN_AUTO_DESCRIBE=1` | Auto-run concept descriptions after ingest (costs LLM calls; off by default). |
| `BRAIN_PROFILE_MAX_BYTES` | Byte budget for `knowledge/assistant/PROFILE.md` writes via `profile_update` (default 4096). |

## How agents use it

Two patterns work well today:

**As context for a project.** Drop a small `CLAUDE.md` in your project repo that says `reference vault at ~/brain/`. Then Claude Code sessions in that project will pull from the vault on demand — concepts, summaries, your own notes.

**As an oracle in this repo.** Ask Claude Code from `~/brain/` itself: *"What does my vault say about X?"*, *"Quiz me on COMP0023"*, *"Find sources that connect Y and Z"*. The agent has read access to everything; it can grep, read processed Markdown, follow wikilinks, and synthesise across sources with citations.

**Over MCP, from anywhere.** The [Model Context Protocol](https://modelcontextprotocol.io) server in `mcp_server/` (FastAPI + FastMCP) exposes the vault to any MCP client: semantic search, read, directory listing, metadata/concept-graph queries, and bearer-authenticated writes confined to `knowledge/` (every write is committed to git). Run it locally with `mcp_server/run-local.sh` and register it with `claude mcp add` (contract in `mcp/README.md`; remote deploy behind Cloudflare Access in `mcp/DEPLOY.md`). The bundled **`brain-project-note`** skill in `.claude/skills/` builds on it: from *any* repo on your machine, it summarises the project and session you're working on into `knowledge/projects/<slug>/` — a full-rewrite overview note plus dated session logs. Install it globally with `cp -r .claude/skills/brain-project-note ~/.claude/skills/`.

## Extending

### Add a new file type

Each extractor is a function with this signature:

```python
def extract(src: Path, assets_dir: Path) -> ExtractionResult: ...
```

Put it under `scripts/ingest_lib/extractors/<name>.py`, register it against the extensions it handles in `scripts/ingest_lib/extractors/__init__.py`, and the rest of the pipeline picks it up. The existing `text.py`, `docx.py`, `pptx.py`, `notebook.py`, and `dataset.py` are good references.

### Swap the embedding model

Edit `_MODEL_NAME` in `scripts/ingest_lib/semantic.py` to any `sentence-transformers`-compatible model. The default `BAAI/bge-small-en-v1.5` is a good balance of size and quality for English; `bge-m3` or `bge-large-en-v1.5` are larger and slower but better for retrieval.

### Customise summary style

The system prompt for summarization lives in `scripts/ingest_lib/summarize.py`. The schema is enforced via Pydantic, so changing the prompt won't break parsing as long as the returned JSON still matches `DocSummary`.

### Switch LLM provider

`scripts/ingest_lib/summarize.py` dispatches to one of four backends based on the `BRAIN_LLM_PROVIDER` env var (or auto-detect by which API key is present): `anthropic`, `openai`, `gemini`, `local`. The local backend uses the OpenAI SDK with a custom `base_url`, so anything that speaks the OpenAI Chat Completions API (Ollama, LM Studio, llama.cpp's server, vLLM) works. The structured-output schema is the same Pydantic class across providers — they all return a `DocSummary` and the rest of the pipeline doesn't know or care which one ran.

## Honest limitations

- **PDF extraction quality depends on MinerU.** Scanned PDFs without text layers need OCR, which MinerU handles but is slow. Mathematical typesetting is hit-or-miss.
- **Summaries cost money** if you use a hosted provider. A fraction of a cent per page; under two cents per typical source on Haiku 4.5, GPT-5-mini, or Gemini 2.5 Flash. Free on local providers (Ollama etc.). Disable with `BRAIN_SKIP_SUMMARY=1`.
- **Concept canonicalization isn't perfect.** The summarizer is asked to reuse existing topic names but occasionally drifts. Slugification catches case and punctuation variants; semantic drift across paraphrases doesn't.
- **Remote write access needs the MCP server hosted.** Locally it's one command (`mcp_server/run-local.sh`); exposing it to other machines or to claude.ai means standing up the server with a bearer token behind Cloudflare Access — see `mcp/DEPLOY.md`.
- **Repo size grows with `archive/raw/`.** PDFs aren't diff-friendly, so git history bloats. Long-term, you'll want git-lfs or a separate object store for the raw archive.

## License

MIT. See `LICENSE`.

## Credits

Built on:
- [MinerU](https://github.com/opendatalab/MinerU) for PDF extraction
- [sentence-transformers](https://www.sbert.net) and [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for semantic search
- [Anthropic Claude](https://docs.anthropic.com) for summaries and topic tags
- [Obsidian](https://obsidian.md) for the human interface
- [uv](https://docs.astral.sh/uv/) for environment management
