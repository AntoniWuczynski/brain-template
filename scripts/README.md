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
uv sync                                                   # one-time setup — prunes mineru, see below
uv run python scripts/ingest.py --dry-run --inbox         # see the plan
uv run python scripts/ingest.py --inbox                   # do it
uv run python scripts/ingest.py --raw                     # re-process archive
uv run python scripts/ingest.py --path inbox/foo.pdf      # single file
uv run python scripts/ingest.py --retry-partial           # re-extract partial notes, e.g. after installing MinerU
uv run python scripts/ingest.py --backfill-summaries      # fill missing summaries (LLM)
uv run python scripts/ingest.py --rebuild-concepts        # refresh concept index (free)
uv run python scripts/ingest.py --rebuild-connections     # rebuild concept graph (free)
uv run python scripts/ingest.py --rebuild-dashboards      # refresh entity dashboards (free)
uv run python scripts/ingest.py --rebuild-status          # refresh status dashboards (free; auto after each ingest)
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
``scripts/eval/retrieval_golden.jsonl``, 87 hand-labelled queries). Every
query that did not retrieve all of its expected sources gets a per-query
breakdown naming the absent path, its true rank, what outranked it and the
label's own justification (``--no-miss-breakdown`` suppresses it). Three extra
modes: ``--compare A --compare B`` A/Bs two configurations over the same set
and prints the per-query rank delta plus improved/regressed counts — each
argument is comma-separated ``key=value`` (``mode=dense|lexical|hybrid`` plus
any ``ENV_VAR=value`` the ranker reads, e.g.
``--compare mode=hybrid --compare mode=hybrid,BRAIN_QUERY_INSTRUCTION=0``), so
a ranking change gated behind an env var can be scored against the baseline in
one process; ``--compare-modes``
scores dense / lexical / hybrid side by side over the golden set, and
``--mine-log`` harvests the real query distribution from
``logs/mcp-access.jsonl`` into ``scripts/eval/mined_candidates.jsonl`` (with a
zero-hit report of queries that never returned a result). Mined candidates
carry ``expected: []`` — a human confirms relevance before promoting a line
into the golden set; the file is gitignored (it holds real query strings) and
never synced to the template. ``--golden PATH`` points ``eval_retrieval.py``
at a different query set (default ``scripts/eval/retrieval_golden.jsonl``).
**Golden-set contract.** A line is ``{"query", "expected": [source_paths],
"note"}``; ``expected`` paths are the retrieval layer's own source ids
(``university/<module>/<file>.pdf`` for archive sources,
``knowledge/<...>.md`` for curated notes), and ``note`` records *why* each
expected path is the right answer — labelled by reading the source, never by
trusting what the ranker returned. A query whose right answer the current
ranker misses is the valuable kind and belongs in the set. The set is only a
gate while it has headroom: at 87 queries the live baseline is recall@5 0.920,
recall@10 0.971, MRR 0.803, so a ranking change has something to win. The file
syncs to the public template, so queries and notes may name course codes and
topics but never private individuals or contact details
(``tests/test_eval_retrieval.py`` enforces the shape and that floor).

- First run downloads ~100 MB of model weights to ``~/.cache/huggingface/``.
- **Query instruction.** ``bge-small-en-v1.5`` is trained to prepend
  ``"Represent this sentence for searching relevant passages: "`` to the
  *query* (passages stay raw), which is how the index is built — so it is
  applied at query time only, needs **no** reindex, and is keyed to the model
  name (a model swap won't apply the wrong prefix). A/B it with the eval
  harness by setting ``BRAIN_QUERY_INSTRUCTION=0`` to disable.
- **Heading-path context (opt-in).** Set ``BRAIN_EMBED_HEADING_CONTEXT=1`` to
  embed each chunk with its ``{title} > {heading_path}`` prefix (a lecture
  slide then carries its section context). This changes the *passage* vectors,
  so it needs a full ``--rebuild-search-index``; measure with
  ``eval_retrieval.py --compare-modes`` before keeping it. The on-disk
  ``text`` (snippet + BM25 source) is unchanged.
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

### Optional ranking layers (all OFF by default)

``ingest_lib/rank.py`` adds four independent layers on top of the hybrid
fusion, each switched by an environment variable read at call time
(``config.ranking_config``). **With no environment set nothing runs and the
ranking is unchanged**, so the live MCP server behaves exactly as it did until
the owner flips a flag. All four apply to ``mode="hybrid"`` only — ``dense``
and ``lexical`` return raw cosine / BM25 scores, whose meaning an RRF-scale
boost or a cross-encoder reorder would silently change. Nothing here is
random: every ordering breaks ties on the candidate's row index.

| variable | default | what it does |
| --- | --- | --- |
| ``BRAIN_RANK_GRAPH`` | off | graph-adjacency boost from ``metadata/connections.jsonl`` |
| ``BRAIN_RANK_GRAPH_WEIGHT`` | ``0.5`` | boost ceiling, in units of the top RRF contribution ``1/(k+1)`` |
| ``BRAIN_RANK_SALIENCE`` | off | per-note salience (curated-vs-archive + typed-relation degree) |
| ``BRAIN_RANK_SALIENCE_WEIGHT`` | ``0.5`` | same units |
| ``BRAIN_RANK_RERANK`` | off | cross-encoder rerank of the top fused hits |
| ``BRAIN_RANK_RERANK_TOP_N`` | ``30`` | size of the reranked window |
| ``BRAIN_RANK_RERANK_MODEL`` | ``cross-encoder/ms-marco-MiniLM-L-6-v2`` | 22.7M params, ~88 MB, Apache-2.0 |
| ``BRAIN_QUERY_EXPANSION`` | off | union LLM paraphrases of the query into the fusion |
| ``BRAIN_QUERY_EXPANSION_VARIANTS`` | ``3`` | paraphrases requested (clamped to 1-5) |
| ``BRAIN_QUERY_EXPANSION_WEIGHT`` | ``0.5`` | a paraphrase ranking's weight in the fusion |

- **Graph adjacency** reads the connection graph the concept pass already
  writes. Co-occurrence edges carry their backing sources, which gives a
  source→concept map for free; a hit is boosted when its source's concepts
  overlap the profile of the query's own top hits (one hop of expansion, never
  two) and when it is corroborated by several distinct other top sources.
  Typed entity edges are a separate signal, used only to relate a
  knowledge-note hit to entity nodes the top hits already name — a hit's node
  is resolved through ``relations.canonical_entity_id``, never a path prefix.
  Leave-one-out: a seed never scores its own concepts back, so the boost
  cannot merely re-state the rank a hit already had. Cost is
  O(hits × concepts-per-source × 8), and the file is parsed once per
  generation and cached by mtime.
  The boost is a function of the QUERY, not of how many hits were asked for.
  Two things enforce that, and both matter because ``recency.memory_search``
  fetches 500 candidates on a filtered query while the eval measures at 10:
  the profile is seeded only from the top 10 fused rows (collecting ten
  distinct sources without that bound reached fused rank 34 on this index,
  where the order moves as soon as the pool widens), and the affinity is
  normalised against the seeds themselves rather than against the maximum
  over the candidate pool. Measured over 30 golden queries, the top ten now
  differs between ``top_k`` 10, 50 and 500 for 2 queries with the boost on,
  which is the baseline's own pool-widening variation.
- **Salience** adds ``0.5 × curated + 0.5 × log-saturated typed degree``. It
  only ever ADDS, and only to curated notes, so an archive chunk's score is
  untouched (measured: it still moves the mix, see below).
- **Rerank** scores ``(query, chunk)`` pairs with the cross-encoder and
  permutes the window. The window's own fused scores are re-assigned in the
  new order, so ``SearchHit.score`` stays the documented RRF-scale rank score
  and ``recency.memory_search``'s ``score × recency`` keeps its meaning.
  ``sentence_transformers`` is imported lazily behind ``try/except`` (CI
  installs neither it nor torch); an unavailable model logs and leaves the
  fused order untouched. No new dependency — ``CrossEncoder`` ships in the
  ``sentence-transformers`` package the embedder already needs. First use
  downloads ~88 MB to ``~/.cache/huggingface/`` and costs a ~20 s one-off
  model load (inside the MCP server that is paid once per process, on the
  first search after the flag is set).
- **Query expansion** asks the configured LLM router for 2-3 paraphrases and
  fuses each one's dense and lexical rankings at ``query_expansion_weight``.
  It costs one LLM call per uncached query, so paraphrases are cached in
  ``metadata/.cache/query-expansion.json`` (gitignored, regenerable). The key
  is the query prefixed by a digest of everything else that decides the
  answer — the prompt text, the number of paraphrases asked for, and the
  resolved provider and model — so editing the prompt, raising
  ``BRAIN_QUERY_EXPANSION_VARIANTS`` or switching provider re-asks instead of
  serving the previous prompt's answer forever. Entries are evicted oldest
  first by an insertion counter, not by position in the file, which is sorted
  by key. It needs an LLM key in the *process* environment —
  ``eval_retrieval.py`` does not load ``.env``.

**A/B results on the 87-query golden set** (live index, 2026-09-06 21:00,
re-measured after the graph boost was made pool-invariant; each feature
measured alone against the same baseline with ``eval_retrieval.py --compare``,
latency from 87 warm queries at ``top_k=10`` on an M-series laptop, ±1 ms run
to run):

| config | recall@5 | recall@10 | MRR | ms/query | archive share of top-10 |
| --- | --- | --- | --- | --- | --- |
| baseline (all off) | 0.937 | 0.971 | 0.817 | 19 | 0.814 |
| ``BRAIN_RANK_GRAPH=1`` | 0.960 | 0.971 | 0.700 | 18 | 0.817 |
| ``BRAIN_RANK_SALIENCE=1`` | 0.914 | 0.971 | 0.745 | 18 | 0.760 |
| ``BRAIN_RANK_RERANK=1`` | 0.954 | 0.960 | 0.822 | 133 | 0.806 |
| ``BRAIN_QUERY_EXPANSION=1``† | 0.902 | 0.971 | 0.749 | 73 (cached) | 0.803 |
| ``GRAPH=1`` + ``RERANK=1`` | 0.954 | 0.977 | 0.819 | 134 | 0.808 |

† Carried over from the 2026-09-06 03:20 run, where the baseline was 0.920 /
0.971 / 0.803, so read it as -0.017 / +0.000 / -0.054 against that baseline
rather than against the one above. It is the one row that cannot be re-run
without spending real LLM calls and writing the vault's paraphrase cache.
Absolute numbers move as the vault grows — the baseline was 0.920 in the
morning and 0.937 that evening on an unchanged code path — so only
same-run comparisons mean anything.

Verdicts, on the evidence above: **the reranker is the only layer that pays
for itself on its own** (+0.017 recall@5 at +0.005 MRR, for 7x the latency),
and **graph + rerank is the best combination measured** (+0.017 / +0.006 /
+0.002): the reranker repairs the MRR the graph boost costs. The graph boost
**alone trades MRR for recall**, and at the default weight of 0.5 it now
trades a lot more of it (+0.023 recall@5 for -0.117 MRR) than the pre-fix
code did on this same index (+0.017 / -0.041). That is not a regression,
it is the layer at its stated strength: dividing by the pool maximum used to
damp every boost by whatever the best-connected candidate in the pool happened
to be, so the effective weight fell as the fetch widened. At
``BRAIN_RANK_GRAPH_WEIGHT=0.25`` the same fix gives +0.017 recall@5 for -0.015
MRR, which beats the old default on both metrics — if the boost is ever
switched on, set the weight to 0.25 rather than leaving it at 0.5. It rewards
hub notes, and in this vault the hubs are dream digests, so e.g. "which phone
hardware refused to boot the generic system image" loses its exact answer note
from rank 1 to a project digest. **Salience is a clear loss** at every weight
tried (0.25/0.5/1.0 → MRR -0.034 / -0.072 / -0.147): curated notes are already
well ranked, and lifting them further costs 6% of the archive's share of the
top 10. **Query expansion** loses MRR at both weights tried (0.25 → -0.035,
0.5 → -0.054) even though its paraphrases read well. All four therefore stay
off; re-measure before enabling one, and A/B a single flag at a time.

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
| `local` | `BRAIN_LOCAL_URL` + `BRAIN_LOCAL_MODEL` | `llama3.1:8b` (via `BRAIN_LOCAL_MODEL`) |

The model name can be overridden with `BRAIN_LLM_MODEL`. The `local`
provider uses the OpenAI SDK with a custom `base_url`, so anything
that speaks the OpenAI Chat Completions API works: Ollama (≥0.5 for
structured outputs), LM Studio, llama.cpp's server, vLLM. Same Pydantic
schema across all four providers, so behaviour is consistent.

### Oversized documents (anthropic)

Documents are sent to the model **in full** — the summarizer never
truncates. `claude-haiku-4-5` has a 200K-token context window, so a very
large source (a 700-page textbook, a statistical-tables PDF) comes back
as a `400 prompt is too long`. On exactly that error the call is retried
**once** on a 1M-context model, `claude-sonnet-5` by default:

| Env var | Default | Effect |
|---|---|---|
| `BRAIN_LLM_FALLBACK_MODEL` | `claude-sonnet-5` | Model to retry an oversized document on. Set it to an empty string to disable the retry. |

The retry is triggered by the API's own error, not by a character-count
guess (a character estimate is wrong by a factor of several for slides,
code and CJK text), and it is **anthropic-only** — the other three
providers have their own model lineups, limits and error wording, so they
keep reporting the failure honestly instead. When it fires, the note
records which model actually did the work:

```
- summary: anthropic/claude-haiku-4-5
- summary: input exceeded claude-haiku-4-5's context window — retried on claude-sonnet-5
```

## Chat with your vault

Two paths, both offline-capable when paired with the `local` provider:

**Terminal (`scripts/ask.py`)** — single-shot, no plugins required:

```bash
uv run python scripts/ask.py "what does my vault say about X?"
uv run python scripts/ask.py --top-k 12 --provider local --model gemma4:31b "..."
uv run python scripts/ask.py -q "..."   # -q/--quiet: omit the trailing model/provider line
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
unconsolidated assistant memory. With `--check-integrity` it also re-hashes
every `archive/raw` file against its recorded `source_hash`
(`archive-corrupt`) to catch bit-rot or an accidental edit of the immutable
archive — off by default because it reads the whole archive (GBs). Read-only
unless `--write-report` is given; always exits 0 (the per-category counts are
the signal, and a linter that fails the shell breaks cron pipelines). Checks
live in `ingest_lib/sweep.py`.

| Flag | Default | Meaning |
|---|---|---|
| `--as-of YYYY-MM-DD` | today (UTC) | Anchor date for the staleness check and the report's `updated:` stamp — pin it for a fully reproducible sweep |
| `--stale-days N` | 30 | Flag `knowledge/assistant/` notes left `memory_status: unconsolidated` longer than N days |
| `--write-report` | off | Also write findings to `knowledge/index/sweep-report.md` (atomic write) |
| `--check-integrity` | off | Also re-hash every `archive/raw` file vs its recorded `source_hash` (`archive-corrupt`). Reads the whole archive, so opt-in |

## Meeting promotion (snapshots -> graph nodes)

`python -m ingest_lib.meetings [--dry-run]`
(`scripts/ingest_lib/meetings.py`, run by `maintain.sh` after the
duplicate pass, and called by `run_ingest` whenever a run processed a
snapshot) turns a processed Granola/justREC snapshot into the first-class
meeting note AGENTS.md describes — `knowledge/meetings/<YYYY>/<YYYY-MM-DD>-<slug>.md`,
the same shape and slug (`concepts.slugify`) the MCP `meeting_create` tool
writes — plus one proposed `attended` relation per attendee. Deterministic,
zero-LLM, driven by `metadata/index.jsonl` (the records whose `extractor` is
`meeting`), never by walking `archive/raw/`.

The write is split along the vault's defining line:

- **The meeting note is derived output**, so the pass writes it — but only
  when it is ABSENT. An existing note is never rewritten (a human or an
  agent may have filled in Agenda/Decisions/Actions). It carries
  `source_file:` and a `Links -> Source:` wikilink back to the snapshot per
  AGENTS.md rule 3, and `written_via: script` / `author: script:meeting-promotion`
  — the same honest third provenance value `ingest_lib/propose.py` stamps.
- **The `attended` edges on PEOPLE are graph writes**, so each one is
  PROPOSED through `ingest_lib/propose.py` as its own `approved: false`
  fact note in `knowledge/assistant/inbox/`. Only `consolidate.py` (a
  human's approval, or enough confirmations) ever puts one on a person.

**An unresolved attendee mints nothing.** Minting a node per attendee id is
how a calendar payload carrying an email address split one person into two
nodes, each collecting its own `attended` history — the `entity-duplicate`
pairs `sweep`/`duplicates` now report. A name is resolved only against the
`people/` notes that already exist, by the union of three matches, each
followed through `superseded_by` to the live survivor: the accent-folded
slug (`Antoni Wuczyński` -> `people/antoni-wuczynski`), the exact note title
(which is what catches an attendee named by their address, whose node is
slugged `people/alexasymmetricsecuritycom`), and an exact `aliases:` entry
(what AGENTS.md's merge leaves behind). The union must come out at exactly
one node: two people sharing a name is reported `ambiguous` and resolves to
nothing. Anything unresolved is listed by display name in the note's
`## Unresolved attendees` section and in the run's output, for a human to
create the person note and re-run.

Idempotent by construction: the note is written only when absent, a
proposal's filename is content-derived, and an attendee whose note already
declares the edge is not proposed at all — so a re-run over an unchanged
vault writes nothing and proposes nothing new. A snapshot with no canonical
`YYYY-MM-DD` date or no sluggable title names no node and is reported as
skipped rather than guessed at. Two DIFFERENT snapshots landing on one
`<date>-<slug>` id is reported as a conflict, with nothing written and
nothing proposed; a note carrying no `source_file` at all (every meeting
note `meeting_create` wrote) is treated as this same meeting and left
untouched, with the attendee proposals still run.

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

A fact may also carry an optional `promote.merge: {duplicate, survivor}`,
and on `approved: true` the pass EXECUTES the duplicate-entity merge
(`FOUNDER_DECISIONS.md` IMP-021) rather than just recording that one was
recommended: it copies the duplicate's relations onto the survivor, closes
every open relation on the duplicate with `valid_until` set to the run's
date, stamps `superseded_by` on the duplicate, and keeps the duplicate's
title as an alias on the survivor — the four steps AGENTS.md's "Merging a
duplicate entity" paragraph defines, and nothing else. The duplicate note
stays on disk (supersede, never delete), and `sweep`'s `entity-duplicate`
check goes quiet on the pair afterwards. Everything is validated and
computed before the first write, so a refusal — no survivor note, a
survivor that is itself superseded, one node named as both halves, an id
that is no graph node — changes nothing and leaves the fact in the inbox
with the reason in the run's problems. An already-superseded duplicate is
a no-op, so re-running is safe. `python -m ingest_lib.duplicates` (run by
`maintain.sh`) proposes these facts for email-slugged twins; it never
merges, and nothing merges without the human's `approved: true`.

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

### Scheduling

`scripts/maintain.sh` is the single entry point that runs the wikilink,
duplicate-entity and meeting-promotion passes, then `consolidate`,
`sweep --write-report` and `rotate_logs` — deterministic, exit 0, safe to
run by hand or on a schedule (`scripts/maintain.sh --dry-run` to plan only).
It prefers the repo `.venv` interpreter so a scheduler needs no `uv` on PATH.
Schedule it for a quiet hour (no cross-process lock vs a running MCP
server):

- **macOS** — `mcp_server/launchd/com.brain.maintenance.plist` (edit
  `REPO_ROOT`, `cp` to `~/Library/LaunchAgents/`, `launchctl load` it).
- **Linux** — `mcp_server/systemd/brain-maintenance.{service,timer}`
  (`systemctl enable --now brain-maintenance.timer`).

`scripts/rotate_logs.py` rotates the two MCP telemetry streams
(`logs/mcp-access.jsonl`, `logs/mcp-audit.jsonl` — see `mcp_server/audit.py`)
once a stream exceeds `--max-mb` (default 10, env `BRAIN_LOG_ROTATE_MB`):
the oversized file is renamed to a UTC-timestamped segment, gzipped, and the
uncompressed copy is dropped; the writer recreates the active path on its
next append. Rotated `.gz` segments are kept forever — this step never
touches an existing one — and are already excluded from git via
`logs/*.gz`.

## Dream pass

The LLM layer on top of the deterministic maintenance above: `consolidate.py`
and `sweep.py` promote and lint by counters and rules, but neither can read
two notes and say why they relate, write a digest, or ask a question. The
dream pass runs a headless coding-agent session — Claude Code or Codex, on
subscription auth, never API credits — that does. It only runs when a
deterministic gate decides enough new information has landed, so a quiet day
costs nothing. Design: `docs/superpowers/specs/2026-07-18-dream-pass-design.md`.
The pass itself is `.claude/skills/dream-pass/SKILL.md` — four jobs
(connections, digests, consolidation, questions), written so any agent, not
just Claude, can run it.

`scripts/dream_gate.py` is the deterministic half — pure Python, no LLM,
read-only except its own state file:

```bash
uv run python scripts/dream_gate.py                 # gate check: exit 0 dream / 1 skip / 2 git error
uv run python scripts/dream_gate.py --dry-run        # check only, no pending marker written
uv run python scripts/dream_gate.py --emit-packet    # print the session's worklist as JSON
uv run python scripts/dream_gate.py --mark-done      # after a successful dream session
```

State lives in `metadata/dream.json` (`last_run`, `last_commit`, advanced only
by `--mark-done`) and `metadata/dream.pending` (stamped the moment the gate
fires, cleared by `--mark-done`) — both gitignored, machine-local scheduler
state, unlike the committed `index.jsonl`. A pending marker two or more days
old means the dream keeps failing to complete: `sweep.py`'s `dream-stalled`
check flags it.

Each of the first three env vars also has a same-named CLI flag on
`dream_gate.py` (`--threshold`, `--stale-days`, `--pairs`) that overrides it
for one invocation — handy for tests and replays. `dream_gate.py` also takes
`--as-of ISO8601` (a full UTC timestamp, unlike `sweep`/`consolidate`'s
`--as-of YYYY-MM-DD`) to fix the reference time.

| Env var | Default | Meaning |
|---|---|---|
| `BRAIN_DREAM_THRESHOLD` | 5 | Changed notes/sources needed to dream |
| `BRAIN_DREAM_STALE_DAYS` | 7 | Dream on any change once this many days have passed since the last dream |
| `BRAIN_DREAM_PAIRS` | 10 | Candidate connection pairs included in the packet |
| `BRAIN_DREAM_MAX_TURNS` | 50 | Turn cap passed to the headless session |
| `BRAIN_DREAM_RUNNER` | `claude` | `claude` \| `codex` \| `noop` — which runner `dream.sh` launches |

### Scheduling

`scripts/dream.sh` is the entry point: gate, then — if warranted — launch the
headless session, capturing everything to `logs/dream-<UTC>.log`. It prefers
the repo `.venv` interpreter, same as `maintain.sh`. Always exits 0: a
skipped or failed dream must not look like a scheduler failure to
cron/launchd; the log and the `dream-stalled` sweep check are the signal.

- **macOS** — `mcp_server/launchd/com.brain.dream.plist` (edit `REPO_ROOT`,
  `cp` to `~/Library/LaunchAgents/`, `launchctl load` it).
  `StartCalendarInterval` 05:00 — a full hour after `com.brain.maintenance`'s
  04:00, so consolidate and sweep have finished before the LLM dreams over
  their output.

Codex is a first-class alternate runner (`codex exec` reading the same skill
document, no Claude-only constructs in the procedure):

```bash
BRAIN_DREAM_RUNNER=codex scripts/dream.sh
```

Codex needs the brain MCP server configured in its own config (every skill
write goes through `mcp__brain__*` tools, same as the Claude runner), and —
unlike the `claude` runner — carries no turn cap or tool allowlist; the
skill's own tripwires are the only bound on a Codex session.

## Connectors (pull external sources)

`scripts/pull.py` pulls an external source into the vault as archivable
snapshots. A **connector** is the one non-deterministic edge of the system:
it fetches new/changed items and writes each as a snapshot under
`inbox/<source_class>/`; the normal ingest pipeline then copies them to the
immutable `archive/raw/` and extracts them. The fetch is the only networked
step and happens *before* the archive boundary, so idempotency and honesty
hold downstream exactly as for a hand-dropped file.

```bash
uv run python scripts/pull.py --list                 # registered connectors
uv run python scripts/pull.py <name> --dry-run        # report, write nothing
uv run python scripts/pull.py <name> --then-ingest    # pull, then ingest inbox/
```

The SDK lives in `ingest_lib/connectors/`: a connector is a `name` plus a
`pull(state) -> Iterator[Snapshot]`, over a shared runner that skips
unchanged items (per-connector state in `metadata/connectors/<name>.json`,
keyed by native id + payload hash) and writes each snapshot atomically. A
connector carries **no** extraction logic — that lives in a matching
extractor registered by source-class prefix in
`extractors._SOURCE_CLASS_REGISTRY` (consulted before the file-extension
map, so a `.json` snapshot under `meetings/granola/` routes to the right
extractor instead of the generic text one). Exits 0, so it is cron/launchd-
safe like `sweep`/`consolidate`. Secrets come from `.env` — never a flag.

**Shipped connectors** (both meeting sources normalise to one snapshot schema
routed to `extractors/meeting.py` — title, date, attendees as `people/`
wikilinks, summary, transcript — and from there to the meeting-promotion
pass above, which makes the meeting a graph node):

- **`granola`** — pulls meetings from the Granola API (`GRANOLA_API_KEY`).
- **`justrec`** — reads justREC's local export folder (`BRAIN_JUSTREC_DIR`),
  no API/auth.

Two more connectors normalise into a second shared snapshot schema (one
JSON object per session/conversation, routed to `extractors/transcript.py`
— title, date, source context, the surviving user/assistant prose exchange,
and a processing-notes section recording exactly what was dropped or
redacted):

- **`claude_code`** — reads local Claude Code CLI session transcripts under
  `~/.claude/projects/<slugged-cwd>/*.jsonl` (override with
  `BRAIN_CLAUDE_CODE_DIR`, or `pull.py claude_code --path <dir>`, which sets
  the generic `BRAIN_PULL_PATH` and wins when both are set — a sessions
  DIRECTORY, same shape as `BRAIN_CLAUDE_CODE_DIR`). No API, no auth — the
  "fetch" is a local file read. Bounded by `BRAIN_CLAUDE_CODE_SINCE_DAYS`
  (default 14) and `BRAIN_CLAUDE_CODE_MAX_SESSIONS` (default 50). A session
  whose file was modified more recently than `BRAIN_CLAUDE_CODE_SETTLE_HOURS`
  (default 24) is skipped — it excludes the session doing the pulling, but
  the 24h default also lets a session spanning several days settle into ONE
  snapshot instead of being re-snapshotted whole (duplicated into the
  immutable archive) on every pull while it's still open. The title prefers
  the session's own `ai-title` line, then the first surviving user turn that
  isn't a collapsed bare-command marker (`_(ran `/dream-pass`)_` survives as
  a turn's text but is never used as a title), then the slugged project path
  and date.
- **`chat_export`** — reads a user-downloaded `conversations.json` archive
  (claude.ai's or ChatGPT's "Export data"), format auto-detected from shape.
  Pointed at by `BRAIN_CHAT_EXPORT_PATH`, or `pull.py <connector> --path
  <file>` (sets the generic `BRAIN_PULL_PATH`, which wins when both are
  set). Bounded to the `BRAIN_CHAT_EXPORT_MAX` most recent conversations
  (default 50, newest first). No API, no auth, no credential search.

Both connectors share `_transcript_common.py`: harness bookkeeping (tool
calls, "thinking" blocks, hook/queue events, injected wrapper tags like
`<system-reminder>`/`<task-notification>`/a bare slash-command invocation)
is dropped and counted in `stats`, never written — the wrapper-tag
allowlist is restricted to this harness's own tag vocabulary
(`local-command-*`/`task-*`/`bash-*`/`command-*`/`skill-*`/`system-*`), not
any tag-shaped text, so real pasted HTML/JSX/XML a turn happens to open
with (`<div>`, `<details>`, ...) survives instead of being mistaken for an
unrecognised wrapper and dropped whole. Obvious credential patterns — env
assignments in any casing (quoted or not, with or without a leading
`export`/`set`), JSON/YAML secret keys, `Authorization`/`Cookie`/
`X-Api-Key` headers, vendor-prefixed tokens (AWS, Anthropic, OpenAI,
GitHub, npm, GitLab, Google, Slack), a bare 40-hex/40-base64 high-entropy
value, private-key blocks, JWTs, and URL-embedded credentials (including an
empty username) — are redacted before a snapshot ever reaches `inbox/`
(`inbox/` and `archive/raw/` are immutable, so this must happen here, not
downstream; over-redacting a non-secret-looking value is an accepted
false positive, missing a real one is not). Each session/conversation's
total size is capped by `BRAIN_TRANSCRIPT_MAX_CHARS` (default 60,000
chars) on top of the 6,000-char per-turn cap — keeping BOTH the oldest and
newest turns (roughly a 25%/75% split of the budget) and eliding the
middle behind a visible `…[N turn(s) elided]…` marker turn, rather than
only ever dropping the oldest turns, so a note never opens mid-conversation
with the framing request gone; `dropped_overflow_turns` records how many
were elided, and a conversation left with no surviving user OR assistant
turn after capping is dropped rather than written as a one-sided note. A
source explicitly configured (a `--path`/dir/file env var) but not
resolving to a real file/dir is a loud, non-zero-exit failure, not a
silent zero-item pull.

A transcript pull that lands enough new snapshots (see
`BRAIN_DREAM_THRESHOLD` below) will make the next scheduled dream run fire —
by design, new sources are new evidence the dream gate has always counted,
same as any other connector's output; it is deliberately not excluded.

Adding another connector is a `pull()` + an extractor + an `.env` stanza.
Follow-up for meetings: promote each into a first-class `knowledge/meetings/`
note with typed `attended` relations (needs the attendee people notes to
exist first). See `IDEAS.md` §4.

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

- `MINERU_DEVICE_MODE` — `mps` (Apple Silicon), `cuda`, or `cpu`. Defaults
  to `mps` when PyTorch reports it available, else `cuda` if available,
  else `cpu`.
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
  keys). Vision model defaults to `claude-sonnet-5`; override with
  `BRAIN_VLM_MODEL`. Render resolution via `BRAIN_VLM_SCALE` (default 2.0).
- Cost is ~one vision call per page (~cents). Set the env var only for
  handwritten modules — leave it unset so printed material keeps using
  MinerU.

**Standalone images** (`.jpg` `.png` `.webp` `.gif` `.bmp` `.tiff`, and
`.heic`/`.heif` with the optional `pillow-heif` package) are ingested
automatically by `extractors/image.py` — no env var needed. Each photo,
whiteboard, screenshot or scan is transcribed/described by the same vision
model (reusing the handwriting extractor's dispatch and honest
`[illegible]`/never-invent rules); with no vision backend configured the
image is marked `manual_review`, never captioned from nothing.

**Audio & subtitles** (`extractors/audio.py`): `.vtt`/`.srt` transcripts parse
deterministically into a timestamped Markdown transcript (no model, no system
deps). Audio (`.m4a` `.mp3` `.wav` `.ogg` `.flac` `.m4b` `.aac`) is transcribed
with a local Whisper model when installed — activate with
`uv pip install faster-whisper` (it bundles audio decoding via PyAV; a system
`ffmpeg` is only needed for exotic codecs). Model via `BRAIN_WHISPER_MODEL`
(default `base`). Without the ASR backend the audio is marked `manual_review`
with the install command — never a fabricated transcript.

MinerU is deliberately *not* in `pyproject.toml`'s lockfile because
some of its transitive deps are pre-releases. The ingestion script
checks whether the `mineru` CLI is on PATH; if it isn't, or if it
errors on a specific PDF, the script transparently falls back to
`pypdf` and records the MinerU error verbatim in the note's
`Processing notes` section.

**Every `uv sync` prunes MinerU.** Because it isn't in the lockfile, `uv sync`
removes it (and its torch transitives) from the venv on every run, not just
the first. Re-run `uv pip install --prerelease=allow "mineru[pipeline]==2.7.6" six`
after each `uv sync` to restore full PDF extraction, or ingestion silently
falls back to `pypdf` and every PDF lands `partial`.

## Internals

```
scripts/
├── ingest.py                       # argparse CLI (ingest + rebuilds + search)
├── ask.py                          # single-shot RAG chat with the vault
├── sweep.py                        # vault linter CLI
├── consolidate.py                  # memory-consolidation CLI
├── rotate_logs.py                  # MCP telemetry log rotation CLI
├── dream_gate.py                   # deterministic dream-pass gate CLI
├── eval_retrieval.py               # retrieval eval CLI (recall@k / MRR)
├── pull.py                         # connector CLI: pull an external source into inbox/
├── maintain.sh                     # consolidate + sweep + rotate_logs, one entry point
├── dream.sh                        # dream-pass scheduler entry point
├── push_to_upstream.sh             # sync framework files to the public brain-template repo
├── pull_from_upstream.sh           # pull framework updates from brain-template
├── check-action-pins.sh            # verify workflow `uses:` lines are pinned to a commit SHA
├── eval/
│   └── retrieval_golden.jsonl      # golden query set for eval_retrieval.py
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
    ├── lexical.py                  # BM25 lexical retrieval over chunk text
    ├── describe.py                 # AI concept descriptions (RAG)
    ├── caption.py                  # figure/table captioning (vision)
    ├── chat.py                     # RAG plumbing for ask.py
    ├── dream.py                    # dream-pass gate: deterministic prep for the LLM session
    ├── evalmine.py                 # mine real queries from the MCP access log
    ├── evalret.py                  # recall@k / MRR scoring over the golden set
    ├── status.py                   # Processing Dashboard + Manual Review notes
    ├── connectors/
    │   ├── base.py                  # connector contract: source-native pull()
    │   ├── granola.py               # Granola meeting connector
    │   ├── justrec.py               # justREC meeting connector (local-first)
    │   ├── claude_code.py           # Claude Code session-transcript connector (local-first)
    │   ├── chat_export.py           # claude.ai/ChatGPT conversation-export connector (local-first)
    │   ├── _transcript_common.py    # shared normalisation for the two transcript connectors
    │   ├── runner.py                # drive a connector: pull, skip-unchanged, snapshot
    │   └── state.py                 # per-connector pull state (metadata/connectors/<name>.json)
    └── extractors/
        ├── __init__.py             # extension → extractor registry
        ├── base.py                 # ExtractionResult dataclass
        ├── text.py                 # plain text + code
        ├── docx.py                 # python-docx
        ├── pptx.py                 # python-pptx
        ├── notebook.py             # nbformat
        ├── dataset.py              # CSV/TSV/JSONL schema-only
        ├── pdf.py                  # MinerU primary, pypdf fallback
        ├── vlm.py                  # vision-LLM page transcription (BRAIN_PDF_EXTRACTOR=vlm)
        ├── image.py                # standalone-image extractor (vision LLM)
        ├── audio.py                # audio + subtitle/transcript extractor (faster-whisper)
        ├── meeting.py              # Granola/justREC meeting-snapshot extractor
        └── transcript.py           # claude_code/chat_export snapshot extractor
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

- It will not caption or transcribe a standalone image or handwritten PDF
  without a vision LLM configured (an API key, or `BRAIN_LOCAL_URL` for a
  local model) — with none configured it's marked `manual_review`, never
  described from nothing.
- It will not generate "summaries" for content it could not extract.
- It will not modify files in `archive/raw/` or `inbox/`.
- It will not delete or rename the user's hand-written notes.
