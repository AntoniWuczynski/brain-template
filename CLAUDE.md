# CLAUDE.md — for Claude Code only

> **This repo is model-agnostic. Do not assume Claude is the only operator.**
> Codex, future MCP-driven agents, and humans also use this vault. Anything
> you write must be readable and reproducible by them. Read
> [`AGENTS.md`](AGENTS.md) — those rules are mandatory for you too.

---

## What this repo is

A long-term personal knowledge system ("second brain"). Sources live in
`archive/raw/` (immutable). Processed Markdown lives in `archive/processed/`.
Hand-edited and generated notes live in `knowledge/`. State is captured in
`metadata/index.jsonl` and `logs/`.

See [`README.md`](README.md) for the full layer breakdown.

---

## What you should focus on

- **Vault structure** — where things live and why.
- **Ingestion** — `scripts/ingest.py` and `scripts/ingest_lib/`.
- **Metadata** — `metadata/index.jsonl` is the source of truth for "what
  has been processed". Always read it before assuming a file is or isn't
  in the system.
- **Entity memory** — typed `relations:` frontmatter (closed vocab,
  supersede-never-delete; see `AGENTS.md`), the `knowledge/assistant/`
  memory lifecycle, and the `scripts/sweep.py` / `scripts/consolidate.py`
  maintenance CLIs.
- **Agent rules** — `AGENTS.md`. Don't drift from them.

You should **not** spend time on:

- Generating "summary" content for documents that failed to extract.
  Mark them `manual_review` and stop.
- Over-built CI: version/OS matrices, coverage gates, packaging/wheels,
  multi-platform anything. See the CI note below for what IS wanted.

**CI is wanted — but lean.** This private repo is the *source* of the
framework that syncs to the **public** `brain-template` repo (via
`scripts/push_to_upstream.sh`), and a public repo needs testing. There is a
single workflow, `.github/workflows/ci.yml` (synced to the template), that
runs pytest + ruff + mypy on **one** runner (Ubuntu, Python 3.12) for pushes
and PRs that touch code. Keep it that way: no matrices, no packaging. It is
gated to skip the constant `mcp(...)` vault-note commits (they change no code
and must not burn Actions minutes). ruff + mypy must stay green — treat a red
run as a real failure, not noise.

---

## Tooling defaults

- **Python**: 3.12, pinned in `pyproject.toml`. PaddlePaddle has no 3.13/3.14
  wheels yet, and MinerU depends on it.
- **Package manager**: `uv`. Never `pip install` outside the venv.
  Use `uv add <pkg>` for new deps, `uv sync` to reproduce the env.
  **Caveat**: `uv sync` prunes any package not in the lockfile, including
  `mineru` and its torch transitives. After every `uv sync`, re-run
  `uv pip install --prerelease=allow "mineru[pipeline]==2.7.6" six`
  to restore PDF full-extraction. (`transformers==4.53.3` is now pinned in
  `pyproject.toml`, so `uv sync` keeps it — you no longer have to re-pin
  transformers after every sync, only reinstall mineru + six.) Pin 2.7.6 —
  the unpinned install now resolves to mineru 3.4.0, which is broken (needs
  `transformers>=4.57.3` but imports `find_pruneable_heads_and_indices`,
  removed in 4.57), so every PDF silently falls back to pypdf and lands as
  `partial`. See `scripts/README.md` for the full explanation.
- **LLM provider for summarizer**: four backends behind one router in
  `scripts/ingest_lib/summarize.py` — `anthropic` (default,
  `claude-haiku-4-5`), `openai` (`gpt-5-mini`), `gemini`
  (`gemini-2.5-flash`), and `local` (any OpenAI-compatible server via
  `BRAIN_LOCAL_URL`). Selected by `BRAIN_LLM_PROVIDER` or auto-detected
  from whichever key is set. Same `DocSummary` Pydantic schema across
  all four.
- **PDF extractor**: MinerU (package `mineru`, CLI invoked as a
  subprocess) — wraps PaddleOCR's PP-Structure for layout, PaddleOCR
  for text OCR, UniMerNet for formulas; outputs Markdown + extracted
  figures/tables. Auto-downloads ~14 GB of weights from Hugging Face on
  first use; no config file required.
- **Fallback PDF extractor**: `pypdf` text-only. Used when the `mineru`
  CLI isn't on PATH (fresh clones, devcontainers without weights yet)
  or when MinerU fails on a specific file. Notes produced via fallback
  are marked `status: partial`.
- **Other extractors**: `python-docx`, `python-pptx`, `nbformat`,
  plain-text reads for code/datasets.

If you change any of these, update both this file and `scripts/README.md`
in the same commit.

---

## Operating principles for this repo

1. **Determinism.** Any agent should be able to re-run ingestion and get
   the same result. No randomness in note bodies; timestamps confined to
   frontmatter and logs.
2. **No destructive ops on `archive/raw/` or `inbox/`** without explicit
   user confirmation in the current conversation.
3. **Idempotency.** Re-running ingestion on already-processed files is a
   no-op. The check is by SHA-256 hash recorded in `metadata/index.jsonl`.
4. **Atomic writes.** When updating `metadata/index.jsonl` write to a
   temp file in the same directory, fsync, then rename.
5. **Honest extraction.** If a PDF fails, you record the failure. You do
   not summarise from the filename. You do not hallucinate.
6. **Statelessness.** Don't expect any persistent context between
   sessions. The only memory is `metadata/index.jsonl`, the logs, and the
   notes themselves.

---

## When the user says "ingest"

1. Read `metadata/index.jsonl` (if present) to know what's already done.
2. Run `uv run python scripts/ingest.py --dry-run --inbox` first. Surface
   the plan to the user (counts by extension, est. time, model-download
   warnings if MinerU needs weights).
3. After confirmation, run without `--dry-run`.
4. After the run, summarise: counts of processed / partial / failed, and
   list anything in `archive/failed/`.

---

## When the user asks a question about the vault

1. Use `metadata/index.jsonl` to find candidates by source path or topic.
2. Read the matching processed Markdown in `archive/processed/`. Cite
   sources by their `archive/raw/...` path or by wikilink to their index
   note in `knowledge/index/`.
3. Never claim something is in the vault without grepping for it first.
