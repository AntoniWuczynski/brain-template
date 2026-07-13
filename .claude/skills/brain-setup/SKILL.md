---
name: brain-setup
description: >-
  Use when someone wants to set up, install, initialise, or get the "brain"
  knowledge vault running on their machine — e.g. "set up the brain", "get my
  vault working", "install brain", "help me configure this template", "onboard
  me", "set it up like yours". Runs from inside a brain repo checkout (a clone
  or a "Use this template" copy). Guides the person's Claude through the whole
  setup the maintainer runs: environment, LLM provider, optional MinerU and MCP
  server, an end-to-end check, and the global ~/.claude integration (the
  brain_memory_sync hooks, the skills, and the CLAUDE.md brain section) so the
  vault works as cross-session memory in every project.
---

# brain-setup

## Overview

Set up the brain vault **for a person** who has just cloned the repo or created
one from the template, taking them from an empty checkout to the same working
setup the maintainer runs. You run the shell steps, ask them the few decisions
that are genuinely theirs (which LLM provider, whether to install the heavy PDF
extractor, whether to host the MCP server, whether to wire the global
integration), prove ingestion works with a real round-trip, then — the part
that makes it "set up like theirs" — install the global `~/.claude` integration
so the vault behaves as durable memory across every project, not just this repo.

The reproducible pieces of the maintainer's global setup ship **alongside this
skill** in its `global/` directory: the `brain_memory_sync.py` hook, a
`settings.hooks.json` snippet, and a `CLAUDE.brain-vault.md` section. Find them
relative to this SKILL.md (the skill dir, whether that's the repo's
`.claude/skills/brain-setup/` or a global `~/.claude/skills/brain-setup/`).

**Core principles:**

- **Idempotent.** Every step checks current state first and is safe to re-run.
  Detect what's already done and skip it — never re-download or re-clobber.
- **Honest.** Verify each step by observing real output, not by assuming. If
  something fails, say so and stop with the error, don't paper over it.
- **Their machine, their choices.** Ask before anything slow, paid, or
  networked. Default to the cheapest working path.
- **Follow `AGENTS.md`.** No destructive ops on `archive/raw/` or `inbox/`.
  Confirm before multi-GB downloads. Installs are project-local via `uv` —
  never global, never raw `pip`.

## Preconditions

Confirm you're inside a brain checkout before doing anything: `scripts/ingest.py`,
`pyproject.toml`, and `AGENTS.md` should all exist at the repo root. If they
don't, STOP — tell the person to `cd` into their brain repo (or clone/create one
from the template first) and re-run you. This skill does not clone the repo for
them.

## Detect current state first

Before running steps, take stock so you can skip finished ones and report an
accurate plan:

- `uv --version` — is uv installed? (If not, point them at
  https://docs.astral.sh/uv/ — installing uv itself is a system-level action, so
  ask before running the installer.)
- `.venv/` present and `uv run python --version` → 3.12.x?
- `.env` present? Does it name a provider or a key (grep for a non-empty
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `GEMINI_API_KEY` /
  `BRAIN_LOCAL_URL`)? **Never print key values** — check presence only.
- `command -v mineru` — is full PDF extraction available?
- `~/.brain-mcp-token` present and `claude mcp list` showing `brain`? — is the
  MCP server already set up?
- `~/.claude/hooks/brain_memory_sync.py` present? `~/.claude/skills/brain-setup`
  and `~/.claude/skills/brain-project-note` present? Does `~/.claude/CLAUDE.md`
  already have a "Brain vault" section? — is the global integration already in
  place?

Report the plan (what you'll do, what you'll skip, what needs a decision) before
running the slow steps.

## Procedure

### 1. Environment (required)

```bash
uv sync
```

Creates `.venv` (Python 3.12) and installs the locked deps. Verify with
`uv run python -c "import sys; print(sys.version)"` → expect 3.12.x. If uv
resolves a different Python, say so — PaddlePaddle/MinerU have no 3.13+ wheels.

### 2. LLM provider (recommended — ask which)

Ingestion works with no provider, but index notes then show placeholders instead
of summaries, key points, and topic tags (which drive the concept layer). Ask
the person which they want:

- **anthropic** (`ANTHROPIC_API_KEY`, default model `claude-haiku-4-5`)
- **openai** (`OPENAI_API_KEY`, `gpt-5-mini`)
- **gemini** (`GOOGLE_API_KEY` or `GEMINI_API_KEY`, `gemini-2.5-flash`)
- **local** — any OpenAI-compatible server (Ollama, LM Studio, llama.cpp, vLLM)
  via `BRAIN_LOCAL_URL` + `BRAIN_LOCAL_MODEL`. Free and offline.
- **none for now** — set `BRAIN_SKIP_SUMMARY=1`; they can add a key later.

Then:

```bash
cp .env.example .env    # only if .env doesn't already exist — never clobber it
```

Set the chosen key in `.env`. Credentials come from `.env` only — never
hardcode a key anywhere else, and never echo it back. If the person pastes a
raw key into the chat, use it but remind them to rotate it afterwards (chat
history persists it). `.env` is git-ignored; confirm that's true before writing
a secret into it.

### 3. Full PDF extraction with MinerU (optional — ask, it's heavy)

Out of the box, PDFs extract text-only via `pypdf` and land as `status: partial`
(and `partial` notes are deliberately not searchable). For figures, tables and
formulas, MinerU is needed — but it auto-downloads **~14 GB** of weights from
Hugging Face on first use. Ask before installing.

```bash
uv pip install --prerelease=allow "mineru[pipeline]==2.7.6" six
```

Pin `2.7.6` — do not install unpinned. The current latest (mineru 3.4.0) needs
`transformers>=4.57.3` but imports a symbol removed in 4.57, so every PDF
silently falls back to `pypdf`. Tell the person the standing caveat: **`uv sync`
prunes MinerU** (it's not in the lockfile), so this line must be re-run after
every sync. See `scripts/README.md` for the full explanation. On Apple Silicon,
`MINERU_DEVICE_MODE=mps` gives roughly an 8× speedup.

If they skip MinerU, that's fine — note that PDFs will be `partial` until they
install it and re-ingest (`--retry-partial`).

### 4. MCP server (optional — ask; needed for agent access)

The MCP server is what lets agents (Claude Code, Codex, others) and the
`brain-project-note` skill reach the vault over the Model Context Protocol. Set
it up if the person wants that:

```bash
mcp_server/run-local.sh        # binds 127.0.0.1:8765, mints a token, prints the next command
claude mcp add --transport http --scope user \
  brain http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer $(cat ~/.brain-mcp-token)"
```

`run-local.sh` keeps git push off and stores the token at `~/.brain-mcp-token`
(0600). After registering, they reconnect with `/mcp`. Full contract:
`mcp/README.md`; remote/production deploy behind Cloudflare Access: `mcp/DEPLOY.md`.

### 5. Verify end-to-end (required)

Prove the pipeline actually runs. First a dry run:

```bash
uv run python scripts/ingest.py --dry-run --inbox
```

Then a real round-trip with a throwaway note (this is the honest check — it
exercises extraction, indexing, and the semantic index):

```bash
printf '# Brain setup smoke test\n\nRetrieval augmented generation lets an LLM answer from a vault.\n' \
  > inbox/_setup-smoke-test.md
uv run python scripts/ingest.py --path inbox/_setup-smoke-test.md
```

The run prints where it wrote. Confirm both outputs exist —
`archive/processed/_setup-smoke-test.md` and `knowledge/index/_setup-smoke-test.md`
(the extractor strips the source extension and writes `<stem>.md`; check the
run summary or the new `metadata/index.jsonl` record if unsure). If a provider
is configured, open the index note and check its `# Summary` section holds real
text, not a placeholder — that confirms the provider key works. Then search:

```bash
uv run python scripts/ingest.py --search "retrieval augmented generation" --top-k 3
```

The smoke-test note should come back. **Cleanup:** the raw copy now lives in
`archive/raw/` (immutable). Do not delete it silently — tell the person it's
there and offer to remove the three artifacts (`inbox/`, `archive/raw/`,
`archive/processed/`, `knowledge/index/` copies) only if they confirm, since
touching `archive/raw/` needs their say-so per `AGENTS.md`.

### 6. Global Claude Code integration (recommended — this makes it "like theirs")

Everything above sets up *this repo*. This step makes the vault behave as
cross-session memory in **every** project, the way the maintainer runs it: a
hook loads prior context at session start and nudges a distilled note at session
end, both skills are available everywhere, and a CLAUDE.md section sets the
standing "use the vault as memory" behaviour. Ask before touching `~/.claude` —
it's the person's global Claude config. Install the four pieces from this skill's
`global/` directory, each idempotently (skip what's already there):

1. **Skills, globally.** So `brain-setup` and `brain-project-note` work from any
   repo, not just this one:
   ```bash
   mkdir -p ~/.claude/skills
   cp -R .claude/skills/brain-setup ~/.claude/skills/
   cp -R .claude/skills/brain-project-note ~/.claude/skills/
   ```
2. **The hook.** Copy the bundled hook and make it executable:
   ```bash
   mkdir -p ~/.claude/hooks
   cp .claude/skills/brain-setup/global/brain_memory_sync.py ~/.claude/hooks/
   chmod +x ~/.claude/hooks/brain_memory_sync.py
   ```
3. **Wire the hook** in `~/.claude/settings.json` by merging the entries in
   `global/settings.hooks.json` (SessionStart → `session`, PostToolUse
   Write|Edit|MultiEdit → `mark`, Stop → `check`). **Merge, don't clobber** —
   if the file exists, read it, add these three under the matching event arrays
   (append, don't replace existing hooks), and write it back. Back it up first
   (`cp ~/.claude/settings.json ~/.claude/settings.json.bak`). If it doesn't
   exist, create it from the snippet (drop the `_comment` key). This is a global
   config edit — show the person the merged result before writing.
4. **The CLAUDE.md section.** If `~/.claude/CLAUDE.md` has no "Brain vault"
   section, append the body of `global/CLAUDE.brain-vault.md` (without its
   leading HTML comment). If a "Brain vault" section already exists, leave it —
   don't duplicate.

Notes:

- The hook probes the MCP server at `http://127.0.0.1:8765/health`. If the
  server runs elsewhere (a devcontainer, a remote host), set
  `BRAIN_MCP_HEALTH_URL` in the environment. The hook is fail-open — if the
  server is down it degrades to a one-line outage notice, never a block.
- Hooks and the CLAUDE.md section take effect in **new** sessions — tell the
  person to start a fresh Claude Code session (in another project) to see the
  SessionStart context-load nudge fire.
- The bundled hook is a copy. Its live home is `~/.claude/hooks/`; if the
  maintainer later changes the repo copy, re-run this step to update.

## Report

Summarise: Python version, provider configured (name only, never the key),
MinerU installed or skipped (and the re-pin caveat if installed), MCP server
registered or skipped, that the smoke test passed with the search hit, and what
of the global integration was installed (skills, hook, settings, CLAUDE.md
section) versus already present. Then the everyday loop: drop files into
`inbox/`, run `uv run python scripts/ingest.py --inbox`, and open the repo root
as an Obsidian vault (`knowledge/index/Home.md` is the entry point). If you wired
the global integration, tell them to start a fresh session in another project to
see the SessionStart nudge. Point deeper questions at the README,
`scripts/README.md`, and `mcp/README.md`.

## Common mistakes

| Mistake | Do instead |
|---|---|
| Running steps blind | Detect state first; skip what's done; report the plan |
| Installing MinerU without asking | It's ~14 GB — confirm first (`AGENTS.md`) |
| Installing MinerU unpinned | Pin `==2.7.6`; unpinned 3.4.0 silently breaks every PDF |
| Forgetting the re-pin caveat | `uv sync` prunes MinerU — it must be reinstalled after each sync |
| `pip install` outside the venv | Project-local via `uv` only; never global |
| Clobbering an existing `.env` | Only `cp .env.example .env` when `.env` is absent |
| Echoing or hardcoding a key | Keys live in `.env` (git-ignored) only; never print them; remind to rotate a pasted key |
| Claiming success without checking | Verify by real output — the smoke-test note must actually ingest and return in search |
| Deleting the smoke-test raw file silently | `archive/raw/` is immutable — remove only on the person's confirmation |
| Treating `partial` PDFs as searchable | Only `processed` notes are indexed; install MinerU + `--retry-partial` for PDFs |
| Overwriting `~/.claude/settings.json` | Merge additively into the existing `hooks` object; back up first; show the result before writing |
| Duplicating the CLAUDE.md brain section | Append only if no "Brain vault" section already exists |
| Expecting hooks to fire this session | Hooks apply to NEW sessions — have them restart Claude Code in another project |
| Editing `~/.claude` without asking | It's the person's global config — confirm before the hook/settings/CLAUDE.md changes |

---

> **Sync note:** this whole skill directory is the source of truth, including
> `global/` (`brain_memory_sync.py`, `settings.hooks.json`,
> `CLAUDE.brain-vault.md`). The bundled public copy at
> `_template/.claude/skills/brain-setup/` (shipped to the template by
> `scripts/push_to_upstream.sh`) and any globally-installed copy at
> `~/.claude/skills/brain-setup/` must be re-synced from here after edits — and
> `global/brain_memory_sync.py` must be kept in step with the live
> `~/.claude/hooks/brain_memory_sync.py`.
