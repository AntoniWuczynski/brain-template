# scripts/ — ingestion tooling

The pipeline turns files in `inbox/` (or files already in `archive/raw/`)
into:

- a verbatim copy at `archive/raw/<rel>` (immutable),
- an extracted Markdown artefact at `archive/processed/<rel>.md`,
- an Obsidian-friendly index note at `knowledge/index/<rel>.md`,
- one JSON line in `metadata/index.jsonl`,
- a per-run log under `logs/ingest-<UTC-timestamp>.log`.

Everything is keyed by SHA-256 hash, so re-running ingestion on
unchanged files is a no-op.

## Running

```bash
uv sync                                                   # one-time setup
uv run python scripts/ingest.py --dry-run --inbox         # see the plan
uv run python scripts/ingest.py --inbox                   # do it
uv run python scripts/ingest.py --raw                     # re-process archive
uv run python scripts/ingest.py --path inbox/foo.pdf      # single file
uv run python scripts/ingest.py --backfill-summaries      # fill missing summaries (LLM)
uv run python scripts/ingest.py --rebuild-concepts        # refresh concept index (free)
uv run python scripts/ingest.py --rebuild-connections     # rebuild concept graph (free)
uv run python scripts/ingest.py --rebuild-dashboards      # refresh entity dashboards (free)
uv run python scripts/ingest.py --describe-concepts --limit 20  # AI concept descriptions (LLM)
uv run python scripts/ingest.py --caption-figures --limit 20    # caption figures (vision LLM)
uv run python scripts/ingest.py --rebuild-search-index    # rebuild semantic index (free)
uv run python scripts/ingest.py --search "query" --top-k 5  # semantic search the vault
uv run python scripts/ask.py "question"                      # chat with the vault (RAG)
uv run python scripts/sweep.py --write-report                # lint the vault (free)
uv run python scripts/consolidate.py --dry-run               # consolidate assistant memory (free)
```

## Semantic search

Complements the canonical-tag concept layer for cases where the
query phrasing doesn't match any tag exactly. Uses
``BAAI/bge-small-en-v1.5`` locally (no API calls, no cost).

**Retrieval modes** (``--mode`` on ``--search``; ``mode`` on the
``vault_search`` tool and ``ask.py``): ``hybrid`` (default) fuses dense
embeddings with a BM25 lexical pass via reciprocal-rank fusion — the dense
half handles paraphrase, the lexical half nails exact identifiers (course
codes like ``COMP0141``, project slugs, error strings) that a small
embedding model ranks poorly. ``dense`` is embeddings only; ``lexical`` is
BM25 only and needs **no** embedding model (works on a machine without the
weights). BM25 runs over the chunk text already in
``metadata/embeddings_meta.jsonl`` (``ingest_lib/lexical.py``, an mtime-cached
in-memory inverted index) — no new files on disk. Measure changes with
``scripts/eval_retrieval.py`` (recall@k / MRR over
``scripts/eval/retrieval_golden.jsonl``).

- First run downloads ~100 MB of model weights to ``~/.cache/huggingface/``.
- **Only ``status: processed`` records are indexed.** ``partial`` notes —
  every PDF extracted by the pypdf fallback when MinerU isn't installed —
  are deliberately **not** searchable, so on a MinerU-less machine
  ``--search`` / ``vault_search`` return nothing for PDF content. The
  excluded count is logged loudly on each rebuild. Install MinerU and
  re-ingest (``--raw``) for full, searchable PDF extraction.
- The index lives at ``metadata/embeddings.npy`` and
  ``metadata/embeddings_meta.jsonl`` and is
  auto-rebuilt after every ``--inbox`` / ``--raw`` run.
- Encoding throughput on an M-series CPU is ~100 chunks/sec on MPS;
  full reindex of 1500 chunks takes ~15 s.
- CLI search has a one-off ~5 s model-load cost per invocation; for
  sub-second queries, embed the same module in a long-running process
  (the MCP server does exactly this).
- Override the inference device with ``BRAIN_EMBED_DEVICE=cpu|mps|cuda``.
- ``semantic.upsert_notes`` patches just the index rows of a few
  knowledge notes in place (the MCP write path), so a note edit doesn't
  pay a full rebuild; the periodic full ``--rebuild-search-index`` is the
  consistency pass. Writers serialise on a cross-process advisory file
  lock (``metadata/.embeddings.lock``, ``flock``), so an MCP-triggered
  upsert can't interleave with a CLI rebuild.
- ``ingest_lib/recency.py`` re-ranks the same index at query time for the
  MCP ``memory_search`` tool: ``score = cosine × recency × status_weight``
  (half-life decay on each note's ``updated`` date; ``memory_status:
  superseded`` notes down-weighted ×0.2). No extra storage, nothing to
  rebuild.

## Curated knowledge notes as enrichment sources

Hand-written notes are first-class inputs to every enrichment layer, not
just ingested sources. ``ingest_lib/knowledge.py`` scans
``knowledge/{assistant,meetings,notes,organisations,people,projects,research,university}``
(mirroring the MCP write allowlist) and synthesizes a virtual record per
Markdown note:

- **Topics** come from the note's frontmatter ``topics:`` list and group
  into concept notes / the co-occurrence graph exactly like summarizer
  topics on ingested documents.
- **Body text** is chunked and embedded into the semantic index (leading
  YAML frontmatter is stripped first), so ``--search`` and the MCP
  ``vault_search`` tool find hand-written content.
- The first body paragraph becomes the note's snippet on concept-note
  source lines.
- ``knowledge/index/`` and ``knowledge/concepts/`` are excluded — they are
  *generated from* sources; indexing them would double-count archive
  content and feed concept notes back into the graph.
- ``metadata/index.jsonl`` is untouched: it remains the record of
  *ingested* sources only. Virtual records exist only in memory.

Notes written over MCP are reindexed automatically in the background
(``semantic.upsert_notes`` + a derived-notes rebuild when frontmatter
changed). Notes edited by hand join the indexes on the next
``--rebuild-search-index`` + ``--rebuild-concepts`` (in that order —
concept centroids read the embeddings).

## LLM provider for the summarizer

The summarizer dispatches to one of four providers, selected by
``BRAIN_LLM_PROVIDER`` env var (or auto-detected from whichever API
key is set):

| Provider | Env var | Default model |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` |
| `openai` | `OPENAI_API_KEY` | `gpt-5-mini` |
| `gemini` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini-2.5-flash` |
| `local` | `BRAIN_LOCAL_URL` + `BRAIN_LOCAL_MODEL` | (see below) |

The model name can be overridden with `BRAIN_LLM_MODEL`. The `local`
provider uses the OpenAI SDK with a custom `base_url`, so anything
that speaks the OpenAI Chat Completions API works: Ollama (≥0.5 for
structured outputs), LM Studio, llama.cpp's server, vLLM. Same Pydantic
schema across all four providers, so behaviour is consistent.

## Chat with your vault

Two paths, both offline-capable when paired with the `local` provider:

**Terminal (`scripts/ask.py`)** — single-shot, no plugins required:

```bash
uv run python scripts/ask.py "what does my vault say about X?"
uv run python scripts/ask.py --top-k 12 --provider local --model gemma4:31b "..."
```

Pipeline: question → embed query against the existing `metadata/embeddings.npy`
index → top-k chunks → LLM via the configured provider → citation-backed
answer in the terminal. Nothing new is written to disk.

**Obsidian (Copilot for Obsidian, or Smart Connections)** — chat panel
inside Obsidian:

1. Settings → Community plugins → Browse → install *Copilot* by Logan Yang.
2. Settings → Copilot → set the chat model to your provider. For
   offline use: provider Ollama, base URL `http://localhost:11434`,
   model `gemma4:31b` (or whatever you've pulled).
3. Open the Copilot panel, switch to "Vault QA" mode, ask.

The plugin builds its own retrieval index, separate from ours. Wasted
disk (~50–100 MB) but otherwise harmless. Smart Connections by Brian
Petro is a strong alternative with the same Ollama support.

## Autonomous curation (concept notes)

When summarization is enabled, the LLM emits 3-8 canonical **topic tags**
per document alongside the summary. The pipeline then writes one
`knowledge/concepts/<slug>.md` note per distinct topic, listing every
source in the vault that mentions it — that's the cross-source
auto-linking layer.

- Concept notes are auto-refreshed after every ingest run, and after every
  `--backfill-summaries` run. Use `--rebuild-concepts` to refresh manually.
- Topic canonicalisation: the prompt is given the current vault's topic
  list and asked to reuse exact strings when they fit. Topics that
  slugify identically (`Behaviour-Driven Development`,
  `behaviour-driven-development`) collapse into one note.
- Each concept note has an auto-generated block (sources list) and a
  `# Notes` block below the `<!-- AUTO-GENERATED-END -->` marker that is
  **preserved** across re-runs — that's where you write your own thoughts.
- Concept notes whose topics no longer appear in any source are removed
  on the next rebuild — but only if they still carry the
  `AUTO-GENERATED-START` marker (hand-written concept notes are never
  deleted).
- Rebuilds **skip unchanged notes**: a concept whose rendered content is
  already on disk byte-for-byte is counted `unchanged` and not rewritten,
  so `written` (and the `updated:` frontmatter stamp) now means *content
  actually changed*. Rebuilds are cheap and commit-clean — re-running on
  an unchanged vault touches nothing.
- Each concept note also carries a **Related concepts** block (see below)
  inside the auto-generated zone.

## Concept relationship graph

Concepts don't just index sources — they relate to each other. The graph is
derived (no database) from three deterministic signals:

- **Co-occurrence**: two concepts tagged on the same document. Weight is the
  document count; the contributing sources are recorded.
- **Semantic**: cosine between concept *centroids* (the mean of the embedding
  vectors of every chunk belonging to a concept's sources). Centroids are
  mean-centred to counter embedding anisotropy, then linked as a per-concept
  **k-nearest-neighbour** graph (`top_k=8`, cosine floor `0.30`) so the signal
  stays meaningful regardless of absolute cosine scale. Drops out cleanly when
  no search index exists — co-occurrence alone still carries the graph.
- **Typed**: explicit `relations:` frontmatter on knowledge entity notes
  (see below) lands as `kind: typed` edges carrying the relation name and
  its `valid_from`/`valid_until` interval.

- Edges land in `metadata/connections.jsonl` (derived/gitignored, atomic
  writes, deterministic ordering, no timestamps), one JSON object per edge:
  `{a, b, kind, weight, sources}`.
- A ranked **Related concepts** view (max 8 neighbours, multi-signal links
  first) is rendered into each concept note's auto-generated zone.
- Queryable over MCP via the **`vault_related`** tool (concept slug or display
  name → ranked neighbours with their co-occurrence/semantic strengths).
- Rebuilt after every ingest / `--backfill-summaries`, or manually:
  `--rebuild-connections` (graph only) or `--rebuild-concepts` (graph + notes).
  Free; no LLM calls.

## Typed entity relations & dashboards

`ingest_lib/relations.py` owns the entity-memory primitives — all
deterministic, LLM-free:

- **Parsing**: a tolerant reader of `relations:` frontmatter (malformed
  entries are skipped with a problem string, unknown rels reported, never
  stored) and a vault scanner that turns every hand-edited knowledge note
  into a graph node. Node ids are `knowledge/`-relative paths without
  extension (`people/anna-kowalska`); the closed relation vocabulary and
  the supersede-never-delete rule live in `AGENTS.md`.
- **Pure text editing**: `upsert_relation_in_text` (add/close/no-op one
  relation) and `append_fact_to_log` (one bullet under `## Log`) are
  `text -> text`, shared by the MCP entity tools and the consolidation
  pass.

`ingest_lib/dashboards.py` renders one auto-generated table per entity
group (people, organisations, projects, meetings) into
`knowledge/index/entities/`, same shape as concept notes (managed
frontmatter, AUTO-GENERATED zone, preserved user tail; unchanged
dashboards are skipped). They sit outside the enrichment scan, so they
never feed back into search or the graph. Rebuilt after every ingest and
`--rebuild-concepts`, or standalone with `--rebuild-dashboards`.

## Vault sweep (linter)

`scripts/sweep.py` lints the whole vault for consistency drift — archive
orphans (raw files vs `index.jsonl` records, both directions), dangling
wikilinks, relation problems (malformed entries, missing targets, bad
dates, inverted/overlapping intervals), near-duplicate concept slugs,
search-index drift (stale/missing/unindexed rows), and stale
unconsolidated assistant memory. Read-only unless `--write-report` is
given; always exits 0 (the per-category counts are the signal, and a
linter that fails the shell breaks cron pipelines). Checks live in
`ingest_lib/sweep.py`.

| Flag | Default | Meaning |
|---|---|---|
| `--as-of YYYY-MM-DD` | today (UTC) | Anchor date for the staleness check and the report's `updated:` stamp — pin it for a fully reproducible sweep |
| `--stale-days N` | 30 | Flag `knowledge/assistant/` notes left `memory_status: unconsolidated` longer than N days |
| `--write-report` | off | Also write findings to `knowledge/index/sweep-report.md` (atomic write) |

## Memory consolidation

`scripts/consolidate.py` is the deterministic "dream pass" over
`knowledge/assistant/inbox/` (counters and thresholds, no LLM — LLMs may
*propose* facts; only deterministic code or the human promotes them).
Facts with `approved: true` or enough confirmations are promoted into
their target entity notes (relations merged into frontmatter, the fact
line appended to `## Log`) and the originals moved — never deleted — to
`knowledge/assistant/archive/<YYYY-MM>/`. Facts that linger past the
staleness window are swept into monthly digests under
`knowledge/assistant/digests/`. The fact-note contract is
`knowledge/index/templates/memory-fact.md`; the pass itself lives in
`ingest_lib/consolidate.py`.

Once consolidated or digested, those notes are historical: only
`knowledge/assistant/archive/` is excluded from the semantic index, so
promoted facts don't resurface in search; digests remain searchable (and
are reindexed immediately after consolidation). Both `archive/` and
`digests/` are excluded from `sweep.py`'s stale-unconsolidated check, so
they never get re-flagged.

**Run `consolidate`/`sweep` when the MCP server is idle.** The server
serialises its own writes with an in-process lock only — there is no
cross-process lock between these CLIs and a running server, so a
concurrent MCP write could race the same note or the git index.

| Flag | Default | Meaning |
|---|---|---|
| `--as-of YYYY-MM-DD` | today (UTC) | Reference date for staleness, archive month and the `consolidated:` stamp |
| `--stale-days N` | 30 | Unconsolidated facts older than this are digested |
| `--min-confirmations N` | 3 | Promote unapproved facts at this confirmation count |
| `--dry-run` | off | Plan but write nothing (still creates a log file) |
| `--no-reindex` | off | Skip the post-run enrichment refresh (semantic upsert + connection graph + concept notes) |

## Source-grounded concept descriptions

`--describe-concepts` writes a synthesized description into each concept note's
**AI zone** — a short summary, a detailed H2/H3 explanation, and key
definitions — generated by the configured LLM from the *retrieved* text of the
concept's own sources (RAG over the search index). It never writes from
nothing: a concept with no retrievable context is skipped, not hallucinated.

- The AI zone (`<!-- AI-GENERATED-START/END -->`) sits **below**
  `AUTO-GENERATED-END`, so `--rebuild-concepts` preserves it like your
  hand-written notes — layout is: auto index → AI description → your `# Notes`.
- **Cached + idempotent + unattended.** Each zone carries an `ai-hash` keyed on
  the concept's source set + model + prompt version. A re-run regenerates only
  concepts whose sources changed; unchanged ones cost nothing. No approval step.
- **Cost control.** `--limit N` bounds how many are (re)generated per run;
  `--force` ignores the cache. Generation is *not* auto-run on ingest by default
  (it costs LLM calls) — run it explicitly, or set `BRAIN_AUTO_DESCRIBE=1`.
- Uses the same four-backend provider router as the summarizer
  (`BRAIN_LLM_PROVIDER` / auto-detected key; `claude-haiku-4-5` by default).

## Figure & table captioning

MinerU exports a PDF's figures and tables as `*_assets/<sha>.<ext>` images and
leaves bare `![](…)` links in the processed Markdown. `--caption-figures`
captions each with a vision-capable model and writes the caption **inline**
beneath the image, so figures show up in Obsidian and — crucially — get
embedded on the next `--rebuild-search-index`, becoming searchable.

- **Vision provider.** `anthropic` (`claude-haiku-4-5`, vision-capable),
  `openai`, or a local OpenAI-compatible vision model via `BRAIN_LOCAL_URL`.
  Same provider selection as the summarizer.
- **Cached + idempotent.** Each caption carries a `<!-- caption: <hash> -->`
  marker keyed on the image's content hash; re-runs don't re-caption. A durable
  cache (`metadata/captions.jsonl`, gitignored) means a caption is paid for once
  even if `archive/processed/` is later regenerated.
- **Bounded.** Tiny images (< 3 KB) are skipped as noise; `--limit N` caps the
  number of *new* vision calls per run (there can be thousands of figures).
  Never auto-run on ingest — it costs vision calls; run it explicitly.

## Optional: full PDF extraction with MinerU

Out of the box PDFs are extracted with `pypdf` (text only) and notes are
marked `status: partial`. For full extraction — including figures,
tables and formulas exported as separate image files — install MinerU:

```bash
# Pin 2.7.6. Do NOT install unpinned: the current latest (mineru 3.4.0) is
# broken — it requires transformers>=4.57.3, but its bundled UniMerNet
# imports `find_pruneable_heads_and_indices`, which was removed from
# transformers in 4.57, so every PDF fails to a pypdf fallback. 2.7.6 allows
# transformers>=4.49 (which still has the symbol). transformers==4.53.3 is
# pinned in pyproject.toml (so `uv sync` keeps it — no need to re-pin it
# here). `six` is a missing transitive dep of mineru's pytorchocr.
uv pip install --prerelease=allow "mineru[pipeline]==2.7.6" six
```

That's it. The `mineru` package (built on PaddleOCR's PP-Structure for
layout, PaddleOCR for OCR, and UniMerNet for formulas) auto-downloads
its model weights from Hugging Face on first run — about 14 GB into
`~/.cache/huggingface/`. No config file required.

> **Apple Silicon:** set `MINERU_DEVICE_MODE=mps` for an ~8× speedup over
> CPU (≈1 min/file vs ≈8 min/file in practice).
>
> **Office formats** (`.ppt`, `.pptx`, `.doc`) have no native MinerU path.
> Convert to PDF first and ingest the PDF for full figure/table extraction:
> `soffice --headless --convert-to pdf <file>` (LibreOffice). `.docx` is
> handled natively (text-only) by the docx extractor.

Knobs (env vars, all optional):

- `MINERU_DEVICE_MODE` — `cpu` (default), `mps` (Apple Silicon), or
  `cuda`. The extractor picks `mps` automatically when PyTorch reports
  it available.
- `MINERU_MODEL_SOURCE` — `huggingface` (default) or `modelscope` (use
  Alibaba's mirror if HF is blocked).
- `BRAIN_MINERU_LANG` — OCR language passed to MinerU (default `en`).
- `BRAIN_MINERU_FORMULA` — `true` (default) / `false`. Set `false` to
  disable MinerU's UniMerNet formula model, which **hallucinates dense
  fake LaTeX on handwriting** it misreads as math. Off = text + figures
  only (no fabricated equations). For handwriting prefer the VLM path
  below; this toggle is for printed docs whose formula output is noisy.

## Handwritten / scanned notes: vision-LLM extractor

MinerU's OCR is built for *printed* text. On handwriting it transcribes
prose only ~85% accurately and its formula model fabricates equations —
unacceptable for honest notes. For handwritten or scanned material, set:

```bash
BRAIN_PDF_EXTRACTOR=vlm uv run python scripts/ingest.py --inbox   # or scope with --path
```

This routes PDFs through `extractors/vlm.py`: each page is rendered to an
image and transcribed by a vision model (verbatim text, LaTeX for real
math incl. bra-ket, `[diagram: …]` for figures, `[illegible]` for
unreadable bits — it is prompted to **never invent** content). The
rendered page image is kept as an asset so the original stays viewable.

- Provider/model reuse the summarizer's config (`BRAIN_LLM_PROVIDER`,
  keys). Vision model defaults to `claude-sonnet-4-6`; override with
  `BRAIN_VLM_MODEL`. Render resolution via `BRAIN_VLM_SCALE` (default 2.0).
- Cost is ~one vision call per page (~cents). Set the env var only for
  handwritten modules — leave it unset so printed material keeps using
  MinerU.

MinerU is deliberately *not* in `pyproject.toml`'s lockfile because
some of its transitive deps are pre-releases. The ingestion script
checks whether the `mineru` CLI is on PATH; if it isn't, or if it
errors on a specific PDF, the script transparently falls back to
`pypdf` and records the MinerU error verbatim in the note's
`Processing notes` section.

## Internals

```
scripts/
├── ingest.py                       # argparse CLI (ingest + rebuilds + search)
├── ask.py                          # single-shot RAG chat with the vault
├── sweep.py                        # vault linter CLI
├── consolidate.py                  # memory-consolidation CLI
├── README.md                       # this file (you are here)
└── ingest_lib/
    ├── __init__.py                 # public re-exports
    ├── config.py                   # paths
    ├── hashing.py                  # SHA-256
    ├── logging_setup.py            # per-run log
    ├── metadata.py                 # IndexRecord + JSONL I/O
    ├── notes.py                    # processed + index note writers
    ├── pipeline.py                 # plan/run
    ├── summarize.py                # LLM summarizer (4-provider router)
    ├── knowledge.py                # curated notes as virtual records
    ├── concepts.py                 # concept-note rebuild (skip-unchanged)
    ├── connections.py              # concept/entity graph (co-occ + semantic + typed)
    ├── relations.py                # typed relations: parse + pure text edits
    ├── dashboards.py               # entity dashboards under knowledge/index/entities/
    ├── recency.py                  # memory_search re-ranking (recency × status)
    ├── sweep.py                    # vault-lint checks (CLI: scripts/sweep.py)
    ├── consolidate.py              # consolidation pass (CLI: scripts/consolidate.py)
    ├── semantic.py                 # embeddings index: build, search, upsert_notes
    ├── describe.py                 # AI concept descriptions (RAG)
    ├── caption.py                  # figure/table captioning (vision)
    ├── chat.py                     # RAG plumbing for ask.py
    └── extractors/
        ├── __init__.py             # extension → extractor registry
        ├── base.py                 # ExtractionResult dataclass
        ├── text.py                 # plain text + code
        ├── docx.py                 # python-docx
        ├── pptx.py                 # python-pptx
        ├── notebook.py             # nbformat
        ├── dataset.py              # CSV/TSV/JSONL schema-only
        ├── pdf.py                  # MinerU primary, pypdf fallback
        └── vlm.py                  # vision-LLM page transcription (BRAIN_PDF_EXTRACTOR=vlm)
```

## Adding a new file type

1. Create `ingest_lib/extractors/<name>.py` with a function

   ```python
   def extract(src: Path, assets_dir: Path) -> ExtractionResult: ...
   ```

   The function must:
   - never modify or delete `src`,
   - return one of `status="processed" | "partial" | "manual_review"`,
   - put any auxiliary files (extracted images, side-files) under
     `assets_dir`,
   - return them in `ExtractionResult.assets` so they get tracked.

2. Register it under the extensions it handles in
   `ingest_lib/extractors/__init__.py`.

3. Run the smoke test against a real file.

## Idempotency rules

A file is **skipped** when:

- the most recent `metadata/index.jsonl` record for that
  `relative_path` has `status: processed`, **and**
- the source's SHA-256 matches the recorded `source_hash`, **and**
- (cheap pre-check) the file size matches.

A file is **always re-processed** when its hash differs from the latest
recorded hash (replaces the previous note; new metadata line appended).

Files in `archive/raw/` whose content differs from an incoming
`inbox/` file with the same path are *not* overwritten — the run logs a
hash clash and surfaces the file as `manual_review`. This is on
purpose: raw is immutable.

## What this script will not do

- It will not OCR images outside of MinerU's pipeline. Install MinerU
  for OCR.
- It will not generate "summaries" for content it could not extract.
- It will not modify files in `archive/raw/` or `inbox/`.
- It will not delete or rename the user's hand-written notes.
